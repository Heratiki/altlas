"""
Code generator that creates new code attempts based on task and history.
Now uses a PyTorch RNN model.
"""

import random
# import json # Removed as not used in this context
from pathlib import Path
import configparser
import logging
import collections

import torch
import torch.nn.functional as F
import torch.optim as optim

# Assuming tokenizer.py and model.py are sibling modules or correctly in path
try:
    # Need to adjust import path relative to runner.py execution context
    from tokenizer import Tokenizer
    from agent_core.model import AltLAS_RNN
except ImportError as e:
    logging.error(f"Error importing Tokenizer or AltLAS_RNN: {e}. Ensure they are in the correct path relative to the execution root.")
    # Attempt relative import if run as module? (Less likely scenario here)
    try:
        from .model import AltLAS_RNN
        # Assuming tokenizer is at the root level
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from tokenizer import Tokenizer
    except ImportError:
         logging.error(f"Secondary import attempt failed: {e}")
         raise # Re-raise original error if secondary fails

# Configure basic logging

class CodeGenerator:
    """Generates code attempts using a PyTorch RNN model and learns via RL."""

    def __init__(self, config_path="config.ini", device=torch.device("cpu")):
        """
        Initializes the CodeGenerator with Tokenizer, Model, and Optimizer.

        Args:
            config_path (str): Path to the configuration file (relative to project root).
            device (torch.device): The PyTorch device to use (cpu or cuda).
        """
        # --- Read Config ---
        config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        # config_path is expected relative to project root where runner.py is
        abs_config_path = Path(config_path).resolve()
        if not abs_config_path.exists():
             logging.error(f"Config file not found at resolved path: {abs_config_path}")
             # Fallback attempt relative to this file's parent's parent
             abs_config_path = Path(__file__).parent.parent / config_path
             if not abs_config_path.exists():
                  raise FileNotFoundError(f"Config file not found at {config_path} or {abs_config_path}")
        config.read(abs_config_path)
        logging.info(f"CodeGenerator reading config from: {abs_config_path}")

        try:
            gen_config = config['Generator']
            paths_config = config['Paths']
            optimizer_config = config['Optimizer']
            model_config = config['Model'] # Added section for model params
            
            # Store optimizer config for token penalties
            self.token_imbalance_penalty = optimizer_config.getfloat('TokenImbalancePenalty', 0.2)

            self.max_gen_length = gen_config.getint('MaxGenerationLength', 50)
            self.early_stop_entropy_threshold = gen_config.getfloat('EarlyStopEntropyThreshold', 0.0)
            self.early_stop_repetition_window = gen_config.getint('EarlyStopRepetitionWindow', 10)
            self.early_stop_repetition_threshold = gen_config.getint('EarlyStopRepetitionThreshold', 3)

            # Model Hyperparameters
            self.embedding_dim = model_config.getint('EmbeddingDim', 64)
            self.hidden_dim = model_config.getint('HiddenDim', 128)
            self.num_layers = model_config.getint('NumLayers', 1)

            # Optimizer Hyperparameters
            self.learning_rate = optimizer_config.getfloat('LearningRate', 0.001)
            # Read initial/max entropy coefficient
            self.max_entropy_coefficient = optimizer_config.getfloat('EntropyCoefficient', 0.1) 
            # Read minimum entropy coefficient for annealing
            self.min_entropy_coefficient = optimizer_config.getfloat('MinEntropyCoefficient', 0.001)
            self.gradient_clip_norm = optimizer_config.getfloat('GradientClipNorm', 1.0)    
            self.baseline_ema_alpha = optimizer_config.getfloat('BaselineEMAAlpha', 0.1)
            self.repetition_penalty = optimizer_config.getfloat('RepetitionPenalty', 1.0) # 1.0 means no penalty
            self.initial_grammar_boost = optimizer_config.getfloat('InitialGrammarBoost', 2.0)
            self.grammar_boost_decay = optimizer_config.getfloat('GrammarBoostDecay', 1.0) # 1.0 means no decay
            
            # Experience replay buffer configuration
            self.experience_buffer_size = optimizer_config.getint('ExperienceBufferSize', 10)
            self.replay_prob = optimizer_config.getfloat('ReplayProbability', 0.3)
            
            # Dynamic learning rate parameters
            self.enable_dynamic_lr = optimizer_config.getboolean('EnableDynamicLR', True)
            self.min_learning_rate = optimizer_config.getfloat('MinLearningRate', 0.0001)
            self.lr_patience = optimizer_config.getint('LRPatience', 50)

            # Paths
            # Paths should be relative to project root defined in config
            self.vocab_path = Path(paths_config.get('VocabFile', 'memory/vocab.json'))
            self.model_state_path = Path(paths_config.get('ModelStateFile', 'memory/model_state.pth'))
            self.optimizer_state_path = Path(paths_config.get('OptimizerStateFile', 'memory/optimizer_state.pth'))

        except KeyError as e:
            logging.error(f"Missing section or key in config file {abs_config_path}: {e}")
            raise
        except ValueError as e:
             logging.error(f"Invalid value type in config file {abs_config_path}: {e}")
             raise
        # --- End Read Config ---

        # Parse ModelFlags from config if present
        try:
            self.use_attention = config.getboolean('ModelFlags', 'UseAttention', fallback=True)
            self.use_layernorm = config.getboolean('ModelFlags', 'UseLayerNorm', fallback=True)
            self.use_residual = config.getboolean('ModelFlags', 'UseResidual', fallback=True)
            self.use_positional_encoding = config.getboolean('ModelFlags', 'UsePositionalEncoding', fallback=True)
        except Exception:
            # Fallback defaults if section missing or error
            self.use_attention = True
            self.use_layernorm = True
            self.use_residual = True
            self.use_positional_encoding = True

        self.device = device
        logging.info(f"CodeGenerator using device: {self.device}")

        # Initialize Tokenizer
        # Ensure path is absolute or relative to project root
        if not self.vocab_path.is_absolute():
            self.vocab_path = Path(__file__).parent.parent / self.vocab_path
        self.tokenizer = Tokenizer(vocab_path=str(self.vocab_path))

        # Initialize Model
        self.model = AltLAS_RNN(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_attention=self.use_attention,
            use_layernorm=self.use_layernorm,
            attention_residual=self.use_residual,
            positional_encoding_type="sinusoidal" if self.use_positional_encoding else "none"
        ).to(self.device)
        logging.info(f"Initialized AltLAS_RNN model with {sum(p.numel() for p in self.model.parameters())} parameters.")

        # Validate Initial Weights
        self._validate_initial_weights()

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logging.info(f"Initialized Adam optimizer with learning rate: {self.learning_rate}")

        # Initialize EMA baseline for REINFORCE
        self.baseline = 0.0
        logging.info(f"Initialized EMA baseline with alpha: {self.baseline_ema_alpha}")

        # Initialize token frequency counter
        self.token_frequency = {} 
        
        # Initialize experience replay buffer
        self.experience_buffer = collections.deque(maxlen=self.experience_buffer_size)
        
        # Initialize learning rate scheduler state
        self.no_improvement_count = 0
        self.best_reward = 0.0
        self.current_lr = self.learning_rate

        # Attempt to load previous state
        self._load_state() # Call load state during initialization

    def reset_weights(self):
        """
        Reset the model weights to break out of local minima.
        This method reinitializes the model with fresh weights and resets the optimizer.
        """
        logging.info("Resetting model weights to break out of potential local minimum...")
        
        # Create a new model instance with the same architecture but fresh weights
        self.model = AltLAS_RNN(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_attention=self.use_attention,
            use_layernorm=self.use_layernorm,
            attention_residual=self.use_residual,
            positional_encoding_type="sinusoidal" if self.use_positional_encoding else "none"
        ).to(self.device)
        
        # Reinitialize the optimizer with the new model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Reset learning related state variables
        self.baseline = 0.0
        self.current_lr = self.learning_rate
        self.no_improvement_count = 0
        
        # Clear the experience buffer to prevent reinforcement of old patterns
        self.experience_buffer.clear()
        
        # Re-validate the new weights
        self._validate_initial_weights()
        
        logging.info(f"Model weights reset successfully. New parameter count: {sum(p.numel() for p in self.model.parameters())}")
        return True

    def _validate_initial_weights(self, std_threshold=1e-6, max_abs_mean=1.0, entropy_threshold=0.5):
        """Performs basic sanity checks on initial model weights and output distribution."""
        logging.info("Validating initial model weights and output distribution...")
        all_ok = True
        with torch.no_grad():
            # --- Weight Parameter Checks ---
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # ... (existing checks for std, mean, NaN/Inf) ...
                    mean_val = param.data.mean().item()
                    std_val = param.data.std().item()
                    if std_val < std_threshold:
                        logging.warning(f"  VALIDATION WARNING: Parameter '{name}' has near-zero std ({std_val:.2e}). Weights might be constant.")
                        all_ok = False
                    if abs(mean_val) > max_abs_mean:
                        logging.warning(f"  VALIDATION WARNING: Parameter '{name}' has large absolute mean ({mean_val:.4f}). Potential instability.")
                        all_ok = False
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        logging.error(f"  VALIDATION ERROR: Parameter '{name}' contains NaN or Inf values!")
                        all_ok = False
            # --- End Weight Parameter Checks ---

            # --- Initial Output Distribution Check ---
            try:
                self.model.eval() # Ensure model is in eval mode for this check
                # Create a dummy input (e.g., SOS token)
                dummy_input = torch.LongTensor([[self.tokenizer.sos_token_id]]).to(self.device)
                dummy_hidden = self.model.init_hidden(batch_size=1, device=self.device)
                
                # Perform a forward pass
                logits, _ = self.model(dummy_input, dummy_hidden)
                probabilities = F.softmax(logits.squeeze(), dim=-1)
                
                # Calculate entropy: - sum(p * log(p))
                # Add epsilon for numerical stability
                log_probabilities = torch.log(probabilities + 1e-9)
                entropy = -torch.sum(probabilities * log_probabilities).item()
                
                # Normalize entropy by log(vocab_size) for a value between 0 and 1
                # High entropy (near 1) is expected for random initialization
                max_entropy = torch.log(torch.tensor(self.tokenizer.vocab_size, dtype=torch.float)).item()
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                logging.info(f"  Initial Output Normalized Entropy: {normalized_entropy:.4f} (Threshold: > {entropy_threshold})")
                
                if normalized_entropy < entropy_threshold:
                    logging.warning(f"  VALIDATION WARNING: Initial output distribution has low entropy ({normalized_entropy:.4f}). Model might be poorly initialized or vocabulary too small.")
                    all_ok = False
                if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                    logging.error(f"  VALIDATION ERROR: Initial output probabilities contain NaN or Inf!")
                    all_ok = False
                    
            except Exception as e:
                logging.error(f"  VALIDATION ERROR: Failed to check initial output distribution: {e}")
                all_ok = False
            # --- End Initial Output Distribution Check ---
                        
        if all_ok:
            logging.info("Initial weight and output validation passed.")
        else:
            logging.warning("Initial weight and output validation completed with warnings/errors. Review initialization logs.")

    def save_weights(self):
        """Saves the state dictionaries of the model and optimizer."""
        try:
            # Ensure parent directories exist
            self.model_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.optimizer_state_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state
            torch.save(self.model.state_dict(), self.model_state_path)
            logging.info(f"Saved model state to {self.model_state_path}")

            # Save optimizer state
            torch.save(self.optimizer.state_dict(), self.optimizer_state_path)
            logging.info(f"Saved optimizer state to {self.optimizer_state_path}")

        except Exception as e:
            logging.error(f"Error saving model/optimizer state: {e}")

    def _load_state(self):
        """Loads the state dictionaries for the model and optimizer if files exist."""
        try:
            if self.model_state_path.exists():
                # Load state dict, ensuring it's mapped to the correct device
                self.model.load_state_dict(torch.load(self.model_state_path, map_location=self.device))
                self.model.to(self.device) # Ensure model is on the correct device after loading
                logging.info(f"Loaded model state from {self.model_state_path}")
            else:
                logging.info("Model state file not found. Initializing new model.")

            if self.optimizer_state_path.exists():
                self.optimizer.load_state_dict(torch.load(self.optimizer_state_path, map_location=self.device))
                # Manually move optimizer states to the correct device if needed (sometimes necessary)
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                logging.info(f"Loaded optimizer state from {self.optimizer_state_path}")
            else:
                logging.info("Optimizer state file not found. Initializing new optimizer.")

        except Exception as e:
            logging.error(f"Error loading model/optimizer state: {e}. Starting with fresh state.")

    def _parse_hint(self, hint: str) -> set:
        """
        Parses a hint string to find relevant token IDs from the vocabulary.
        Maps common instruction words to abstract tokens.
        """
        if not hint:
            return set()

        hinted_token_ids = set()
        hint_lower = hint.lower()
        # Simple word extraction (lowercase)
        hint_words = set(word.strip('.,!?"\'`') for word in hint_lower.split())

        # Mapping from common hint words/concepts to abstract tokens
        agnostic_hint_map = {
            "output": "OUTPUT_OP", "print": "OUTPUT_OP", "display": "OUTPUT_OP", "show": "OUTPUT_OP",
            "function": "FUNC_DEF", "define": "FUNC_DEF", "method": "FUNC_DEF",
            "if": "CONDITIONAL_IF", "condition": "CONDITIONAL_IF",
            "else": "CONDITIONAL_ELSE",
            "loop": ["LOOP_FOR", "LOOP_WHILE"], "iterate": ["LOOP_FOR", "LOOP_WHILE"], "repeat": ["LOOP_FOR", "LOOP_WHILE"],
            "for": "LOOP_FOR",
            "while": "LOOP_WHILE",
            "return": "RETURN_STMT",
            "add": "+", "sum": "+",
            "subtract": "-", "difference": "-",
            "multiply": "*", "product": "*",
            "divide": "/", "quotient": "/",
            "assign": "=", "variable": "VAR_GENERIC", "store": "=",
            "compare": ["COMP_EQ", "COMP_NEQ", "<", ">", "COMP_LTE", "COMP_GTE"],
            "equal": "COMP_EQ", "same": "COMP_EQ",
            "not equal": "COMP_NEQ", "different": "COMP_NEQ",
            "less than": "<",
            "greater than": ">",
            "less or equal": "COMP_LTE",
            "greater or equal": "COMP_GTE",
            "true": "BOOL_TRUE",
            "false": "BOOL_FALSE",
            "none": "NULL_VALUE", "null": "NULL_VALUE",
            "string": "STRING_LITERAL_PLACEHOLDER", "text": "STRING_LITERAL_PLACEHOLDER",
            "number": "NUMBER_LITERAL_PLACEHOLDER", "integer": "NUMBER_LITERAL_PLACEHOLDER", "float": "NUMBER_LITERAL_PLACEHOLDER"
        }

        # 1. Check for direct keyword mappings
        import re
        for keyword, abstract_token_or_list in agnostic_hint_map.items():
            escaped_keyword = re.escape(keyword)
            if re.search(r'\b' + escaped_keyword + r'\b', hint_lower):
                tokens_to_add = abstract_token_or_list if isinstance(abstract_token_or_list, list) else [abstract_token_or_list]
                for token_str in tokens_to_add:
                    if token_str in self.tokenizer.token_to_id:
                        token_id = self.tokenizer.token_to_id[token_str]
                        hinted_token_ids.add(token_id)
                        logging.debug(f"Hint '{hint}' mapped keyword '{keyword}' to abstract token: '{token_str}' (ID: {token_id})")
                    else:
                        logging.warning(f"Hint keyword map: Abstract token '{token_str}' not found in vocabulary.")

        # 2. Check if any vocabulary tokens (non-special) are directly mentioned
        for token, token_id in self.tokenizer.token_to_id.items():
             # Skip special tokens like <PAD>, <SOS>, etc.
            if token.startswith('<') and token.endswith('>'):
                continue
                
            # Check if the token itself (case-insensitive for matching hint words) is in the hint words
            token_text_lower = token.strip('"').lower()
            if token_text_lower in hint_words:
                hinted_token_ids.add(token_id)
                logging.debug(f"Hint '{hint}' directly matched token: '{token}' (ID: {token_id})")
            # Additionally check if the token text appears as a substring in the hint
            elif token_text_lower and token_text_lower in hint_lower:
                 hinted_token_ids.add(token_id)
                 logging.debug(f"Hint '{hint}' contained token text: '{token_text_lower}' (ID: {token_id})")

        return hinted_token_ids
    # Abstract grammar constraints using the new vocabulary
    ABSTRACT_GRAMMAR_RULES = {
        # After function definition, expect function name or opening parenthesis
        "FUNC_DEF": ["FUNC_GENERIC", "("],
        # After conditional, expect opening parenthesis (for condition)
        "CONDITIONAL_IF": ["("],
        "LOOP_WHILE": ["("],
        # After loop keyword, expect variable or range-like construct
        "LOOP_FOR": ["VAR_GENERIC"],
        # After opening parenthesis, expect value, variable, or closing parenthesis
        "(": ["NUMBER_LITERAL_PLACEHOLDER", "STRING_LITERAL_PLACEHOLDER", "VAR_GENERIC", "FUNC_GENERIC", ")", "BOOL_TRUE", "BOOL_FALSE", "NULL_VALUE"],
        # After assignment, expect value, variable, or expression start
        "=": ["NUMBER_LITERAL_PLACEHOLDER", "STRING_LITERAL_PLACEHOLDER", "VAR_GENERIC", "FUNC_GENERIC", "(", "[", "{", "BOOL_TRUE", "BOOL_FALSE", "NULL_VALUE"],
        # After binary operators, expect value or variable
        "+": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC", "("],
        "-": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC", "("],
        "*": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC", "("],
        "/": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC", "("],
        "COMP_EQ": ["NUMBER_LITERAL_PLACEHOLDER", "STRING_LITERAL_PLACEHOLDER", "VAR_GENERIC", "BOOL_TRUE", "BOOL_FALSE", "NULL_VALUE"],
        "COMP_NEQ": ["NUMBER_LITERAL_PLACEHOLDER", "STRING_LITERAL_PLACEHOLDER", "VAR_GENERIC", "BOOL_TRUE", "BOOL_FALSE", "NULL_VALUE"],
        "<": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC"],
        ">": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC"],
        "COMP_LTE": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC"],
        "COMP_GTE": ["NUMBER_LITERAL_PLACEHOLDER", "VAR_GENERIC"],
        # After return, expect value or variable
        "RETURN_STMT": ["NUMBER_LITERAL_PLACEHOLDER", "STRING_LITERAL_PLACEHOLDER", "VAR_GENERIC", "BOOL_TRUE", "BOOL_FALSE", "NULL_VALUE"],
        # After colon (often end of block header), expect newline
        ":": ["\\n"],
        # After newline, expect potential keywords, variables, or output
        "\\n": ["OUTPUT_OP", "CONDITIONAL_IF", "LOOP_FOR", "LOOP_WHILE", "FUNC_DEF", "RETURN_STMT", "PASS_STMT", "VAR_GENERIC"]
        # Add more rules as needed for brackets, commas, etc.
    }
    def generate(self, task=None, history=None, hint=None, temperature=0.7, last_feedback=None, feedback_history_for_fp=None):
        # Configurable constants
        min_tokens_before_stop = getattr(self, 'min_tokens_before_stop', 3)
        enable_logit_noise = getattr(self, 'enable_logit_noise', True)
        logit_noise_std = getattr(self, 'logit_noise_std', 0.1)
        enable_entropy_adaptation = getattr(self, 'enable_entropy_adaptation', True)
        entropy_increase_factor = getattr(self, 'entropy_increase_factor', 1.1)
        max_entropy_temperature = getattr(self, 'max_entropy_temperature', 1.5)
        repetition_overuse_threshold = getattr(self, 'repetition_overuse_threshold', 0.5)  # 50% of recent outputs
        repetition_window_size = getattr(self, 'repetition_window_size', 50)
        if not hasattr(self, 'token_overuse_counter'):
            self.token_overuse_counter = collections.Counter()
        if not hasattr(self, 'recent_generations'):
            self.recent_generations = collections.deque(maxlen=repetition_window_size)
        try:
            self.model.eval() # Set model to evaluation mode (disables dropout etc.)

            # --- MODEL HEALTH CHECK ---
            # Track number of consecutive distribution collapses
            if not hasattr(self, 'distribution_collapse_counter'):
                self.distribution_collapse_counter = 0
                self.last_collapse_reset = 0
                self.consecutive_failures = 0
            
            # Reset model weights if consecutive collapses exceed threshold
            # This is more aggressive than our regular learning-rate adjustments
            attempt_count = history.attempt_count if history and hasattr(history, 'attempt_count') else 0
            if (self.distribution_collapse_counter >= 5 and 
                attempt_count - self.last_collapse_reset > 20):
                logging.warning(f"Model has experienced {self.distribution_collapse_counter} distribution collapses. " 
                              f"Resetting model weights to break out of pathological state.")
                self.model.reinitialize_weights()
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                self.distribution_collapse_counter = 0
                self.last_collapse_reset = attempt_count
                self.baseline = 0.0
                self.current_lr = self.learning_rate
                
                # Temporarily increase temperature significantly right after reset
                temperature = min(2.0, temperature * 2)
                logging.info(f"Increased temperature to {temperature} after model reset")
            
            # Track consecutive failures to help identify stuck model
            if last_feedback and hasattr(last_feedback, 'severity') and last_feedback.severity > 0.3:
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0
                
            # If model is producing many consecutive failures, gradually increase temperature
            if self.consecutive_failures > 10:
                temp_boost = min(1.0, self.consecutive_failures * 0.05)
                temperature = min(1.5, temperature + temp_boost)
                logging.info(f"Boosting temperature to {temperature} after {self.consecutive_failures} consecutive failures")
            # --- END MODEL HEALTH CHECK ---

            # --- Feedback-Guided Exploration Adjustment ---
            feedback_penalty_factor = 0.8 # How much to reduce probability of error-related tokens (e.g., 0.8 means 20% reduction)
            feedback_relevant_token_ids = set()
            adjusted_temperature = temperature # Start with base temperature

            if last_feedback and hasattr(last_feedback, 'severity') and last_feedback.severity > 0.1:
                # If last attempt had an error, slightly increase temperature for more exploration
                adjusted_temperature = min(1.0, temperature + 0.1 * last_feedback.severity)
                logging.info(f"Adjusting temperature based on last feedback severity ({last_feedback.severity:.2f}): {temperature:.2f} -> {adjusted_temperature:.2f}")
                
                # Get relevant tokens from last feedback to penalize them slightly
                if hasattr(last_feedback, 'relevant_tokens') and last_feedback.relevant_tokens:
                    relevant_texts = last_feedback.relevant_tokens
                    for token_id, token_text in self.tokenizer.id_to_token.items():
                        if token_text in relevant_texts:
                            feedback_relevant_token_ids.add(token_id)
                    if feedback_relevant_token_ids:
                        logging.debug(f"Identified {len(feedback_relevant_token_ids)} tokens to penalize based on last feedback.")
            # --- End Feedback-Guided Exploration Adjustment ---

            # --- Analyze Feedback History for Persistent Problem Tokens ---
            persistent_penalty_factor = 0.6 # Stronger penalty for persistent problems
            persistent_problem_token_ids = set()
            if feedback_history_for_fp:
                token_error_counts = collections.Counter()
                error_count_for_fp = 0
                for feedback_item in feedback_history_for_fp:
                    if feedback_item.get('severity', 0) > 0.1:
                        error_count_for_fp += 1
                        for token_text in feedback_item.get('relevant_tokens', []):
                            token_error_counts[token_text] += 1
                
                # If this fingerprint has failed multiple times, identify consistently relevant tokens
                if error_count_for_fp >= 2:
                    for token_text, count in token_error_counts.items():
                        # If a token was relevant in >50% of the errors for this fingerprint
                        if count > error_count_for_fp * 0.5:
                            # Find the token ID
                            for token_id, t_text in self.tokenizer.id_to_token.items():
                                if t_text == token_text:
                                    persistent_problem_token_ids.add(token_id)
                                    break
                    if persistent_problem_token_ids:
                         logging.info(f"Identified {len(persistent_problem_token_ids)} persistent problem tokens for fingerprint based on history. Applying stronger penalty ({persistent_penalty_factor}).")
            # --- End Feedback History Analysis ---

            # --- Success Token Boosting --- 
            successful_token_ids = set()
            success_boost_factor = 1.2 # Smaller boost than hints
            if self.experience_buffer:
                # Get tokens from the highest reward experience in the buffer
                best_experience = max(self.experience_buffer, key=lambda x: x[0]) # x[0] is reward
                if best_experience[0] > 0.7: # Only boost if reward was high
                    successful_token_ids = set(best_experience[1].tolist()) # x[1] is generated_ids tensor
                    logging.debug(f"Identified {len(successful_token_ids)} tokens to boost from successful experience (reward {best_experience[0]:.2f}).")
            # --- End Success Token Boosting ---

            # Determine max generation length from task or config
            max_gen_length = task.max_tokens if (task and task.max_tokens is not None) else self.max_gen_length
            logging.debug(f"Using max_gen_length={max_gen_length} {'(from task)' if task and task.max_tokens else '(from config)'}")

            # Check if the no_improvement_count is high and we're still at low rewards
            template_guided = False
            if hasattr(self, 'no_improvement_count') and self.no_improvement_count > 50 and self.best_reward < 0.5:
                # Use template-based generation for this attempt
                if random.random() < 0.7:  # 70% chance to use template-based generation when struggling
                    template_guided = True
                    logging.info(f"Using template-guided generation (no_improvement_count: {self.no_improvement_count})")
                    
                    # Removed Python-specific template generation for 'add_two_numbers'
                    # A new agnostic template system would be needed here if desired.
            
            # Parse hint if provided
            hinted_token_ids = self._parse_hint(hint)
            hint_boost_factor = 1.5 # How much to increase probability mass for hinted tokens
            
            # Get benchmark-specific template tokens if available
            if task and not template_guided:
                template_token_ids, template_boost = self.generate_agnostic_template(task) # Call new function
                if template_token_ids:
                    # Add template tokens to hinted tokens with higher boost
                    for token_id in template_token_ids:
                        hinted_token_ids.add(token_id)
                    hint_boost_factor = template_boost  # Use stronger boost for template tokens
                    logging.info(f"Added {len(template_token_ids)} template tokens with boost {template_boost}")
            
            # Removed Python-specific hint boosting for 'add_two_numbers'

            generated_ids = [] # Full sequence for final output
            generated_in_sequence = set() # Track tokens within this specific generation
            log_probs_list = [] # Store log probs of chosen tokens

            # Start sequence generation with SOS token
            current_token_id = self.tokenizer.sos_token_id
            current_token_str = self.tokenizer.id_to_token.get(current_token_id, "<UNK>")
            # Input tensor needs batch dimension: (batch_size=1, seq_len=1)
            input_tensor = torch.LongTensor([[current_token_id]]).to(self.device)

            # Initialize hidden state for the batch
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
            current_grammar_boost = self.initial_grammar_boost # Initialize grammar boost for this generation

            with torch.no_grad(): # Disable gradient calculation during generation
                for _ in range(max_gen_length):
                    # Get output logits and updated hidden state from the model
                    # Output shape: (batch=1, seq=1, vocab_size)
                    logging.debug(f"Loop {_}: Input shape {input_tensor.shape}")
                    output_logits, hidden = self.model(input_tensor, hidden)

                    logging.debug(f"Loop {_}: Model output logits shape {output_logits.shape}")
                    # Get logits for the last token: shape (vocab_size)
                    # Squeeze removes batch and sequence dimensions (both are 1)
                    last_logits = output_logits.squeeze(0).squeeze(0)
                    
                    # Apply ADJUSTED temperature to logits before softmax
                    if adjusted_temperature != 1.0:
                        last_logits = last_logits / adjusted_temperature

                    # Optionally inject noise into logits to encourage exploration
                    if enable_logit_noise:
                        noise = torch.randn_like(last_logits) * logit_noise_std
                        last_logits = last_logits + noise

                    # EMERGENCY RECOVERY: Reset collapsed distributions when entropy is too low
                    # Calculate entropy before softmax to check for collapse
                    max_val = last_logits.max().item()
                    min_val = last_logits.min().item()
                    # Detect if the distribution has collapsed (one value much higher than all others)
                    if max_val - min_val > 20.0:  # Extremely skewed logits
                        # Detect if we're in a stuck loop (same token repeatedly)
                        if len(generated_ids) >= 2 and all(x == generated_ids[-1] for x in generated_ids[-2:]):
                            logging.warning(f"DISTRIBUTION COLLAPSE DETECTED - Emergency logit reset at step {_}")
                            # Apply a much stronger intervention - reset logits to small random values
                            last_logits = torch.rand_like(last_logits) * 0.1
                            # Boost template tokens more aggressively
                            if hinted_token_ids:
                                for token_id in hinted_token_ids:
                                    if 0 <= token_id < last_logits.shape[0]:
                                        last_logits[token_id] += 3.0  # Strong bias, not just multiplier
                            # Force recovery by temporarily boosting common useful tokens
                            useful_tokens = [self.tokenizer.token_to_id.get(t, -1) for t in 
                                           ["def", "print", "return", "=", "+", "(", ")", ":", "\n"]]
                            for token_id in useful_tokens:
                                if token_id >= 0 and token_id < last_logits.shape[0]:
                                    last_logits[token_id] += 2.0  # Strong bias

                    # Apply softmax to get probabilities for sampling
                    probabilities = F.softmax(last_logits, dim=-1)
                    
                    # DIAGNOSTIC: Log distribution details for debugging low-entropy issues
                    if _ < 5:  # Only log for first few steps to avoid log bloat
                        # Get top-k probability values and indices
                        top_k = 5
                        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[0]))
                        top_tokens = [self.tokenizer.id_to_token.get(idx.item(), "<UNK>") for idx in top_indices]
                        
                        # Format probability details for logging
                        prob_details = ", ".join([f"{token}({idx.item()}): {prob.item():.4f}" 
                                               for token, idx, prob in zip(top_tokens, top_indices, top_probs)])
                        
                        # Calculate statistics about the distribution
                        prob_mean = probabilities.mean().item()
                        prob_std = probabilities.std().item()
                        prob_max = probabilities.max().item()
                        prob_min = probabilities.min().item()
                        
                        logging.debug(f"Loop {_}: Probability stats - Mean: {prob_mean:.6f}, Std: {prob_std:.6f}, "
                                    f"Max: {prob_max:.6f}, Min: {prob_min:.6f}")
                        logging.debug(f"Loop {_}: Top {len(top_indices)} tokens: {prob_details}")
                    
                    # Get token imbalance penalties from tokenizer
                    token_penalties = self.tokenizer.get_token_penalties(
                        penalty_factor=self.token_imbalance_penalty
                    )
                    if token_penalties:
                        penalty_applied = False
                        penalty_count = 0
                        for token_id, penalty in token_penalties.items():
                            if 0 <= token_id < probabilities.shape[0]:
                                probabilities[token_id] *= (1.0 - penalty)
                                penalty_count += 1
                                penalty_applied = True
                        # Re-normalize probabilities after applying penalties
                        if penalty_applied:
                            logging.debug(f"Applied imbalance penalties to {penalty_count} tokens in generate()")
                        probabilities = probabilities / (probabilities.sum() + 1e-9)
                    
                    # --- Apply Hint Bias ---
                    if hinted_token_ids:
                        for token_id in hinted_token_ids:
                            if 0 <= token_id < probabilities.shape[0]: # Check bounds
                                probabilities[token_id] *= hint_boost_factor
                        # Re-normalize probabilities after boosting
                        # Add a small epsilon to prevent division by zero if all probabilities become zero
                        probabilities = probabilities / (probabilities.sum() + 1e-9)
                    # --- End Hint Bias ---
                    
                    # --- Apply Feedback Penalty (from last step AND persistent history) ---
                    if feedback_relevant_token_ids or persistent_problem_token_ids:
                        penalty_applied_count = 0
                        persistent_penalty_applied_count = 0
                        for token_id in range(probabilities.shape[0]):
                            applied_penalty = 1.0
                            is_persistent = False
                            if token_id in persistent_problem_token_ids:
                                applied_penalty *= persistent_penalty_factor
                                is_persistent = True
                                persistent_penalty_applied_count += 1
                            # Apply standard feedback penalty only if not already penalized as persistent
                            elif token_id in feedback_relevant_token_ids:
                                applied_penalty *= feedback_penalty_factor
                                penalty_applied_count += 1
                                
                            if applied_penalty < 1.0:
                                probabilities[token_id] *= applied_penalty
                                
                        if penalty_applied_count > 0 or persistent_penalty_applied_count > 0:
                            # Re-normalize probabilities after penalizing
                            probabilities = probabilities / (probabilities.sum() + 1e-9)
                            # logging.debug(f"Applied penalties: Standard={penalty_applied_count}, Persistent={persistent_penalty_applied_count}")
                    # --- End Feedback Penalty ---
                    
                    # --- Apply Success Token Boost ---
                    if successful_token_ids:
                        boost_applied_count = 0
                        for token_id in successful_token_ids:
                            if 0 <= token_id < probabilities.shape[0]: # Check bounds
                                probabilities[token_id] *= success_boost_factor
                                boost_applied_count += 1
                        if boost_applied_count > 0:
                            # Re-normalize probabilities after boosting
                            probabilities = probabilities / (probabilities.sum() + 1e-9)
                            # logging.debug(f"Applied success boost factor {success_boost_factor} to {boost_applied_count} tokens.")
                    # --- End Success Token Boost ---
                    
                    # --- Apply Abstract Grammar Rules ---
                    if current_token_str in self.ABSTRACT_GRAMMAR_RULES:
                        allowed_next_tokens = self.ABSTRACT_GRAMMAR_RULES[current_token_str]
                        # Use the current, potentially decayed, grammar boost factor
                        grammar_boost_factor = current_grammar_boost
                        
                        boost_applied = False
                        # Boost probabilities of tokens that match the grammar rules
                        for token_id, token in self.tokenizer.id_to_token.items():
                            # Check if the token itself is allowed, or if it's a placeholder type
                            if token in allowed_next_tokens:
                                probabilities[token_id] *= grammar_boost_factor
                                boost_applied = True
                                # logging.debug(f"Grammar boost: '{current_token_str}' -> '{token}'")
                        
                        # Re-normalize after grammar boosting if any boost was applied
                        if boost_applied:
                            probabilities = probabilities / (probabilities.sum() + 1e-9)
                    # --- End Abstract Grammar Rules ---
                    # --- Apply Repetition Penalty (for tokens generated in *this* sequence) ---
                    if self.repetition_penalty < 1.0 and generated_in_sequence:
                        penalty_applied_count = 0
                        for token_id in generated_in_sequence:
                             if 0 <= token_id < probabilities.shape[0]:
                                probabilities[token_id] *= self.repetition_penalty
                                penalty_applied_count += 1
                        if penalty_applied_count > 0:
                            # Re-normalize probabilities after applying penalty
                            probabilities = probabilities / (probabilities.sum() + 1e-9)
                            # logging.debug(f"Applied repetition penalty {self.repetition_penalty} to {penalty_applied_count} tokens.")
                    # --- End Repetition Penalty ---

                    # Apply log_softmax to get log probabilities (numerically stable) - Use original logits for loss
                    log_probabilities = F.log_softmax(last_logits, dim=-1)
                    
                    # --- Top-k Sampling ---
                    # Limit to top-k tokens (k=10 is a reasonable default)
                    k = 10
                    top_k_probs, top_k_indices = torch.topk(probabilities, k=min(k, len(probabilities)))
                    # Create a new probability distribution with only top-k tokens
                    limited_probabilities = torch.zeros_like(probabilities)
                    limited_probabilities.scatter_(0, top_k_indices, top_k_probs)
                    # Re-normalize the limited distribution
                    limited_probabilities = limited_probabilities / limited_probabilities.sum()
                    # --- End Top-k Sampling ---

                    # Sample the next token ID based on the probability distribution
                    next_token_id = torch.multinomial(limited_probabilities, num_samples=1).item()

                    # --- Token Frequency Tracking ---
                    token_str = self.tokenizer.id_to_token.get(next_token_id, "<UNK>")
                    self.token_frequency[token_str] = self.token_frequency.get(token_str, 0) + 1
                    # --- End Token Frequency Tracking ---

                    logging.debug(f"Loop {_}: Sampled next token ID: {next_token_id} ('{token_str}')")
                    # --- Early Stopping Checks ---
                    # 1. Entropy Check
                    if self.early_stop_entropy_threshold > 0.0:
                        # Calculate normalized entropy
                        log_probs_dist = torch.log(probabilities + 1e-9)
                        entropy = -torch.sum(probabilities * log_probs_dist).item()
                        max_entropy = torch.log(torch.tensor(self.tokenizer.vocab_size, dtype=torch.float)).item()
                        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        
                        if normalized_entropy < self.early_stop_entropy_threshold and _ + 1 >= min_tokens_before_stop:
                            logging.info(f"Early stopping generation due to low entropy ({normalized_entropy:.4f} < {self.early_stop_entropy_threshold:.4f}) at step {_ + 1}.")
                            break # Exit generation loop
                        elif normalized_entropy < self.early_stop_entropy_threshold:
                            logging.debug(f"Entropy {normalized_entropy:.4f} below threshold but continuing (step {_ + 1} < min_tokens_before_stop={min_tokens_before_stop})")

                    # 2. Repetition Check
                    if self.early_stop_repetition_window > 0 and len(generated_ids) >= self.early_stop_repetition_window:
                        last_n_tokens = generated_ids[-self.early_stop_repetition_window:]
                        # Check for repeating subsequences of length >= threshold
                        for length in range(self.early_stop_repetition_threshold, self.early_stop_repetition_window // 2 + 1):
                            # Example: window=10, length=3. Check seq[7:10] == seq[4:7]
                            # Example: window=10, length=4. Check seq[6:10] == seq[2:6]
                            # Example: window=10, length=5. Check seq[5:10] == seq[0:5]
                            idx = self.early_stop_repetition_window - length
                            if last_n_tokens[idx:] == last_n_tokens[idx-length:idx]:
                                repeating_sequence = self.tokenizer.decode(torch.LongTensor(last_n_tokens[idx:]))
                                logging.info(f"Early stopping generation due to repeating sequence (length {length}): '{repeating_sequence}' at step {_ + 1}.")
                                break # Exit inner loop (length check)
                        else: # If inner loop completed without break
                            continue # Continue generation loop
                        break # Exit generation loop if inner loop broke (repetition found)
                    # --- End Early Stopping Checks ---

                    # --- Token Handling ---
                    # 1. Stop if EOS token is generated
                    if next_token_id == self.tokenizer.eos_token_id:
                        # Track generated token for overuse detection
                        self.token_overuse_counter[next_token_id] += 1
                        generated_ids.append(next_token_id)
                        generated_in_sequence.add(next_token_id)

                    # Add to recent generations window
                    self.recent_generations.append(next_token_id)

                    # After generation, check for overuse of any token
                    if _ == max_gen_length - 1 or next_token_id == self.tokenizer.eos_token_id:
                        overused_tokens = []
                        for token_id, count in self.token_overuse_counter.items():
                            recent_count = sum(1 for t in self.recent_generations if t == token_id)
                            freq = recent_count / max(1, len(self.recent_generations))
                            if freq > repetition_overuse_threshold:
                                overused_tokens.append(token_id)
                        if overused_tokens and enable_entropy_adaptation:
                            old_temp = adjusted_temperature
                            adjusted_temperature = min(max_entropy_temperature, adjusted_temperature * entropy_increase_factor)
                            logging.info(f"Detected overused tokens {overused_tokens} (freq>{repetition_overuse_threshold}). Increasing temperature {old_temp:.2f} -> {adjusted_temperature:.2f} for next generation.")

                        break

                    # 2. Store generated ID (excluding SOS, implicitly excluding EOS by breaking)
                    generated_ids.append(next_token_id)
                    generated_in_sequence.add(next_token_id) # Add to set for repetition penalty check

                    # 3. Store log probability of the *chosen* token for RL
                    log_probs_list.append(log_probabilities[next_token_id])

                    # 4. Prepare input for the next iteration: the chosen token
                    current_token_id = next_token_id
                    current_token_str = token_str  # Update current token string for grammar rules
                    input_tensor = torch.LongTensor([[current_token_id]]).to(self.device)

                    # Decay grammar boost for the next step
                    if self.grammar_boost_decay < 1.0:
                        current_grammar_boost *= self.grammar_boost_decay
                        # Optional: Add a minimum boost floor?
                        # current_grammar_boost = max(1.0, current_grammar_boost)

            # Decode the generated sequence of IDs back to a string (outside the loop)
            generated_tensor = torch.LongTensor(generated_ids)
            generated_code = self.tokenizer.decode(generated_tensor) # decode handles filtering special tokens

            # Convert generated IDs list to tensor
            generated_ids_tensor = torch.LongTensor(generated_ids).to(self.device)
    
            logging.debug(f"Returning from generate. Code type: {type(generated_code)}, Generated IDs shape: {generated_ids_tensor.shape}")
            if getattr(self, 'debug_mode', False):
                logging.debug(f"[DEBUG] Abstract token IDs: {generated_ids}")
                try:
                    tokens = [self.tokenizer.id_to_token.get(tid, "<UNK>") for tid in generated_ids]
                    logging.debug(f"[DEBUG] Abstract tokens: {tokens}")
                except Exception:
                    pass
                logging.debug(f"[DEBUG] Decoded source code:\n{generated_code}")

            # Check for empty or invalid generation
            penalize_empty_generations = getattr(self, 'penalize_empty_generations', True)
            if penalize_empty_generations:
                if generated_ids_tensor.numel() == 0 or not generated_code.strip():
                    logging.warning("Empty or invalid generation detected. Returning penalty signal.")
                    # Optionally, set a penalty reward or error flag here
                    # For now, return None to indicate failure
                    return None, None

            return generated_code, generated_ids_tensor

        except Exception as e:
            import traceback
            logging.error(f"!!! EXCEPTION INSIDE generate method: {type(e).__name__}: {e}")
            logging.error(traceback.format_exc())
            # Return None to indicate failure, handled by runner.py
            return None, None

    def generate_with_beam_search(self, task=None, history=None, hint=None, beam_width=3, temperature=0.7, last_feedback=None, feedback_history_for_fp=None):
        """
        Generates code using beam search to maintain multiple candidate sequences.
        This helps the model produce more coherent code by considering multiple possibilities.
        
        Args:
            task (Task, optional): The current task being attempted
            history (list, optional): List of previous attempts
            hint (str, optional): Optional hint to guide generation
            beam_width (int): Number of candidate sequences to maintain
            temperature (float): Controls randomness in token selection
            last_feedback (ToolFeedback, optional): Feedback from the previous execution attempt.
            feedback_history_for_fp (list, optional): History of feedback for fingerprint analysis.
            
        Returns:
            tuple: (code, token_ids) - The generated code and token IDs
        """
        try:
            self.model.eval()  # Set model to evaluation mode
            
            # --- Feedback-Guided Exploration Adjustment ---
            feedback_penalty_factor = 0.8 # How much to reduce probability of error-related tokens
            feedback_relevant_token_ids = set()
            adjusted_temperature = temperature # Start with base temperature

            if last_feedback and hasattr(last_feedback, 'severity') and last_feedback.severity > 0.1:
                # If last attempt had an error, slightly increase temperature for more exploration
                adjusted_temperature = min(1.0, temperature + 0.1 * last_feedback.severity)
                logging.info(f"Adjusting beam search temperature based on last feedback severity ({last_feedback.severity:.2f}): {temperature:.2f} -> {adjusted_temperature:.2f}")
                
                # Get relevant tokens from last feedback to penalize them slightly
                if hasattr(last_feedback, 'relevant_tokens') and last_feedback.relevant_tokens:
                    relevant_texts = last_feedback.relevant_tokens
                    for token_id, token_text in self.tokenizer.id_to_token.items():
                        if token_text in relevant_texts:
                            feedback_relevant_token_ids.add(token_id)
                    if feedback_relevant_token_ids:
                        logging.debug(f"Identified {len(feedback_relevant_token_ids)} tokens to penalize in beam search based on last feedback.")
            # --- End Feedback-Guided Exploration Adjustment ---
            
            # --- Analyze Feedback History for Persistent Problem Tokens ---
            persistent_penalty_factor = 0.6 # Stronger penalty for persistent problems
            persistent_problem_token_ids = set()
            if feedback_history_for_fp:
                token_error_counts = collections.Counter()
                error_count_for_fp = 0
                for feedback_item in feedback_history_for_fp:
                    if feedback_item.get('severity', 0) > 0.1:
                        error_count_for_fp += 1
                        for token_text in feedback_item.get('relevant_tokens', []):
                            token_error_counts[token_text] += 1
                
                if error_count_for_fp >= 2:
                    for token_text, count in token_error_counts.items():
                        if count > error_count_for_fp * 0.5:
                            for token_id, t_text in self.tokenizer.id_to_token.items():
                                if t_text == token_text:
                                    persistent_problem_token_ids.add(token_id)
                                    break
                    if persistent_problem_token_ids:
                         logging.info(f"Beam Search: Identified {len(persistent_problem_token_ids)} persistent problem tokens. Applying stronger penalty ({persistent_penalty_factor}).")
            # --- End Feedback History Analysis ---
            
            # --- Success Token Boosting --- 
            successful_token_ids = set()
            success_boost_factor = 1.2 # Smaller boost than hints
            if self.experience_buffer:
                # Get tokens from the highest reward experience in the buffer
                best_experience = max(self.experience_buffer, key=lambda x: x[0]) # x[0] is reward
                if best_experience[0] > 0.7: # Only boost if reward was high
                    successful_token_ids = set(best_experience[1].tolist()) # x[1] is generated_ids tensor
                    logging.debug(f"Beam Search: Identified {len(successful_token_ids)} tokens to boost from successful experience (reward {best_experience[0]:.2f}).")
            # --- End Success Token Boosting ---
            
            # Determine max generation length from task or config
            max_gen_length = task.max_tokens if (task and task.max_tokens is not None) else self.max_gen_length
            
            # Check if we should use template-based generation
            # Removed Python-specific template generation for 'add_two_numbers' in beam search
            
            # Parse hint if provided
            hinted_token_ids = self._parse_hint(hint)
            hint_boost_factor = 1.5
            
            # Start with a single beam containing the start token
            # Each beam is (token_ids, score, hidden_state)
            beams = [
                (
                    [self.tokenizer.sos_token_id],  # Token IDs (starting with SOS)
                    0.0,  # Score (log probability sum)
                    self.model.init_hidden(batch_size=1, device=self.device),  # Hidden state
                    self.initial_grammar_boost # Store initial boost with the beam state
                )
            ]
            
            # Keep track of the current grammar boost factor for each beam
            # Beams now: (token_ids, score, hidden_state, current_grammar_boost)
            
            for step in range(max_gen_length):
                # Skip beam search if we only have one beam left
                if len(beams) == 1 and beams[0][0][-1] == self.tokenizer.eos_token_id:
                    break
                    
                candidates = []
                
                # Process each beam
                for token_ids, score, hidden, current_grammar_boost in beams:
                    # Skip completed sequences (ending with EOS)
                    if token_ids[-1] == self.tokenizer.eos_token_id:
                        # Pass the current boost factor along for completed sequences
                        candidates.append((token_ids, score, hidden, current_grammar_boost))
                        continue
                        
                    # Create input tensor from last token
                    input_tensor = torch.LongTensor([[token_ids[-1]]]).to(self.device)
                    
                    # Get model predictions
                    with torch.no_grad():
                        output_logits, new_hidden = self.model(input_tensor, hidden)
                        
                        # Get logits for next token prediction
                        last_logits = output_logits.squeeze(0).squeeze(0)
                        
                        # Apply ADJUSTED temperature
                        if adjusted_temperature != 1.0:
                            last_logits = last_logits / adjusted_temperature
                            
                        # Get probabilities
                        probabilities = F.softmax(last_logits, dim=-1)
                        
                        # Get token imbalance penalties from tokenizer for beam search
                        token_penalties = self.tokenizer.get_token_penalties(
                        penalty_factor=self.token_imbalance_penalty
                        )
                        if token_penalties:
                            penalty_applied = False
                            penalty_count = 0
                            for token_id, penalty in token_penalties.items():
                                if 0 <= token_id < probabilities.shape[0]:
                                    probabilities[token_id] *= (1.0 - penalty)
                                    penalty_count += 1
                                    penalty_applied = True
                            # Re-normalize probabilities after applying penalties
                            if penalty_applied:
                                logging.debug(f"Applied imbalance penalties to {penalty_count} tokens in beam search")
                            probabilities = probabilities / (probabilities.sum() + 1e-9)
                        
                        # Apply hint bias
                        if hinted_token_ids:
                            for token_id in hinted_token_ids:
                                if 0 <= token_id < probabilities.shape[0]:
                                    probabilities[token_id] *= hint_boost_factor
                            probabilities = probabilities / (probabilities.sum() + 1e-9)
                        
                        # --- Apply Feedback Penalty (from last step AND persistent history) ---
                        if feedback_relevant_token_ids or persistent_problem_token_ids:
                            penalty_applied_count = 0
                            persistent_penalty_applied_count = 0
                            for token_id in range(probabilities.shape[0]):
                                applied_penalty = 1.0
                                is_persistent = False
                                if token_id in persistent_problem_token_ids:
                                    applied_penalty *= persistent_penalty_factor
                                    is_persistent = True
                                    persistent_penalty_applied_count += 1
                                elif token_id in feedback_relevant_token_ids:
                                    applied_penalty *= feedback_penalty_factor
                                    penalty_applied_count += 1
                                    
                                if applied_penalty < 1.0:
                                    probabilities[token_id] *= applied_penalty
                                    
                            if penalty_applied_count > 0 or persistent_penalty_applied_count > 0:
                                # Re-normalize probabilities after penalizing
                                probabilities = probabilities / (probabilities.sum() + 1e-9)
                                # logging.debug(f"Beam Step: Applied penalties: Standard={penalty_applied_count}, Persistent={persistent_penalty_applied_count}")
                        # --- End Feedback Penalty ---
                        
                        # --- Apply Success Token Boost ---
                        if successful_token_ids:
                            boost_applied_count = 0
                            for token_id in successful_token_ids:
                                if 0 <= token_id < probabilities.shape[0]: # Check bounds
                                    probabilities[token_id] *= success_boost_factor
                                    boost_applied_count += 1
                            if boost_applied_count > 0:
                                # Re-normalize probabilities after boosting
                                probabilities = probabilities / (probabilities.sum() + 1e-9)
                                # logging.debug(f"Beam Step: Applied success boost factor {success_boost_factor} to {boost_applied_count} tokens.")
                        # --- End Success Token Boost ---
                        
                        # --- Apply Abstract Grammar Rules ---
                        current_token_str = self.tokenizer.id_to_token.get(token_ids[-1], "<UNK>")
                        if current_token_str in self.ABSTRACT_GRAMMAR_RULES:
                            allowed_next_tokens = self.ABSTRACT_GRAMMAR_RULES[current_token_str]
                            # Use the current grammar boost for this beam
                            grammar_boost_factor = current_grammar_boost
                            
                            boost_applied = False
                            for token_id, token in self.tokenizer.id_to_token.items():
                                if token in allowed_next_tokens:
                                     if 0 <= token_id < probabilities.shape[0]:
                                        probabilities[token_id] *= grammar_boost_factor
                                        boost_applied = True
                            
                            if boost_applied:
                                probabilities = probabilities / (probabilities.sum() + 1e-9)
                        # --- End Abstract Grammar Rules ---
                        # --- Apply Repetition Penalty (for tokens generated in *this* beam) ---
                        if self.repetition_penalty < 1.0:
                            current_beam_tokens = set(token_ids[1:]) # Exclude SOS token
                            if current_beam_tokens:
                                penalty_applied_count = 0
                                for token_id in current_beam_tokens:
                                    if 0 <= token_id < probabilities.shape[0]:
                                        probabilities[token_id] *= self.repetition_penalty
                                        penalty_applied_count += 1
                                if penalty_applied_count > 0:
                                    # Re-normalize probabilities after applying penalty
                                    probabilities = probabilities / (probabilities.sum() + 1e-9)
                                    # logging.debug(f"Beam Step: Applied repetition penalty {self.repetition_penalty} to {penalty_applied_count} tokens.")
                        # --- End Repetition Penalty ---

                        # Get top-k next tokens
                        log_probs = torch.log(probabilities + 1e-9)
                        topk_log_probs, topk_indices = torch.topk(log_probs, k=min(beam_width, len(probabilities)))
                        
                        # Add candidates for each top-k token
                        for i in range(len(topk_indices)):
                            next_token_id = topk_indices[i].item()
                            next_score = score + topk_log_probs[i].item()
                            next_token_ids = token_ids + [next_token_id]
                            
                            # Calculate decayed boost for the next step
                            next_grammar_boost = current_grammar_boost * self.grammar_boost_decay if self.grammar_boost_decay < 1.0 else current_grammar_boost
                            
                            candidates.append((next_token_ids, next_score, new_hidden, next_grammar_boost))
                
                # Sort candidates by score and keep top beam_width
                # Sort candidates by score and keep top beam_width
                # x[1] is the score
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Select the best beam
            # Select the best beam based on score (index 1)
            best_beam = max(beams, key=lambda x: x[1])
            best_token_ids = best_beam[0]
            
            # Remove SOS token at beginning if present
            if best_token_ids[0] == self.tokenizer.sos_token_id:
                best_token_ids = best_token_ids[1:]
                
            # Convert to tensor and decode to string
            best_tensor = torch.LongTensor(best_token_ids)
            generated_code = self.tokenizer.decode(best_tensor)
            generated_ids_tensor = torch.LongTensor(best_token_ids).to(self.device)
            
            logging.info(f"Beam search generated code: '{generated_code}'")
            return generated_code, generated_ids_tensor
            
        except Exception as e:
            import traceback
            logging.error(f"Exception in beam search generation: {type(e).__name__}: {e}")
            logging.error(traceback.format_exc())
            return None, None

    def learn(self, reward: float, generated_ids: torch.Tensor, current_entropy_coef: float, tool_feedback=None):
        """
        Updates the model weights using REINFORCE with baseline, entropy regularization
        (with dynamic coefficient), and gradient clipping. Now uses structured tool feedback.

        Args:
            reward (float): The reward received for the generated sequence.
            generated_ids (torch.Tensor): The tensor of token IDs for the generated sequence.
            current_entropy_coef (float): The dynamically calculated entropy coefficient for this step.
            tool_feedback (ToolFeedback, optional): Structured feedback from tool execution.

        Returns:
            float: The calculated total gradient norm before clipping.
        """
        if generated_ids.numel() == 0:
            logging.warning("Learn called with empty generated_ids tensor. Skipping update.")
            return # Cannot learn from an empty sequence

        # Reset weights if model is stuck in a bad local minimum for too long
        if self.no_improvement_count > 150 and reward < 0.30:
            logging.warning(f"Model stuck in bad local minimum for {self.no_improvement_count} steps. Reinitializing weights.")
            self.model.reinitialize_weights()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.baseline = 0.0
            self.no_improvement_count = 0
            self.current_lr = self.learning_rate
            return  # Skip this learning step after reinitialization

        # Apply stronger penalties for syntax errors
        if tool_feedback and hasattr(tool_feedback, 'feedback_type') and tool_feedback.feedback_type == "syntax_error":
            logging.info("Detected syntax error - applying stronger penalty for learning.")
            # Reduce reward further for syntax errors to strongly discourage them
            reward = reward * 0.5
            
        # Update dynamic learning rate based on reward progress AND tool feedback
        self._update_learning_rate(reward, tool_feedback)
        
        # Add experience to replay buffer if above threshold 
        if reward >= 0.5:
            self._add_to_experience_buffer(reward, generated_ids, tool_feedback)
            
        # Possibly replay a past successful experience
        replay_experience = self._should_replay_experience()
        if replay_experience:
            logging.info(f"Replaying previous successful experience (reward: {replay_experience[0]:.4f})")
            # Use the replayed experience for this learning step
            reward, generated_ids, tool_feedback = replay_experience

        self.model.train() # Set model to training mode

        # --- Recalculate log probabilities and get action probabilities ---
        # Prepend SOS token for model input
        sos_tensor = torch.LongTensor([[self.tokenizer.sos_token_id]]).to(self.device)
        # Input sequence for forward pass: [SOS, token1, token2, ..., tokenN-1]
        # Target sequence for loss:       [token1, token2, ..., tokenN] (which is generated_ids)
        # Ensure generated_ids is 2D (batch_size, seq_len-1) for slicing
        if generated_ids.dim() == 1:
            generated_ids_batched = generated_ids.unsqueeze(0)
        else:
            generated_ids_batched = generated_ids
            
        # Handle case where generated_ids has only one token (e.g., just EOS)
        if generated_ids_batched.shape[1] > 0:
            input_seq = torch.cat((sos_tensor, generated_ids_batched[:, :-1]), dim=1)
        else: # If only EOS was generated (or empty sequence somehow), input is just SOS
            input_seq = sos_tensor
            # If input_seq and generated_ids are empty, we should have returned earlier
            # but handle defensively
            if generated_ids_batched.numel() == 0:
                 logging.warning("Learn called with effectively empty sequence after SOS. Skipping update.")
                 return

        logits, _ = self.model(input_seq) # Shape: (batch=1, seq_len, vocab_size)
        logits = logits.squeeze(0) # Shape: (seq_len, vocab_size)

        # Get action probabilities (for entropy) and log probabilities (for policy loss)
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = F.log_softmax(logits, dim=-1)

        # Select the log probabilities corresponding to the actual generated tokens
        # Ensure generated_ids is 1D for indexing
        if generated_ids.dim() > 1:
            generated_ids_flat = generated_ids.squeeze()
        else:
            generated_ids_flat = generated_ids
            
        # Check if generated_ids_flat is empty after potential squeeze
        if generated_ids_flat.numel() == 0:
            logging.warning("Learn called with empty sequence after processing. Skipping update.")
            return
            
        # Ensure indices are within bounds
        if generated_ids_flat.max() >= log_probabilities.shape[1] or generated_ids_flat.min() < 0:
            logging.error(f"Token index out of bounds. Max index: {generated_ids_flat.max()}, Vocab size: {log_probabilities.shape[1]}")
            # Handle error appropriately, e.g., skip update or clamp indices
            return # Skip update for now
            
        # Ensure log_probabilities has the correct shape (seq_len, vocab_size)
        if log_probabilities.shape[0] != generated_ids_flat.shape[0]:
             logging.error(f"Shape mismatch: log_probabilities ({log_probabilities.shape[0]}) vs generated_ids ({generated_ids_flat.shape[0]})")
             # This might indicate an issue with input_seq generation or model output
             return # Skip update

        log_prob_actions = log_probabilities[range(generated_ids_flat.shape[0]), generated_ids_flat]
        # --- End Recalculation ---

        # --- Calculate Advantage and Update Baseline ---
        advantage = float(reward) - self.baseline
        # Update baseline using EMA
        self.baseline = self.baseline_ema_alpha * float(reward) + (1 - self.baseline_ema_alpha) * self.baseline
        logging.debug(f"  Reward: {reward:.4f}, Baseline: {self.baseline:.4f}, Advantage: {advantage:.4f}")

        # --- Calculate Token-wise Weights using Tool Feedback ---
        token_weights = torch.ones_like(log_prob_actions) # Start with weight 1.0 for all tokens
        
        if tool_feedback is not None and hasattr(tool_feedback, 'relevant_tokens') and tool_feedback.relevant_tokens:
            if tool_feedback.severity > 0.1: # Only apply significant weighting for actual errors
                relevant_token_texts = tool_feedback.relevant_tokens # This is a set of strings
                
                # Decode generated IDs to strings
                generated_tokens_text = [self.tokenizer.id_to_token.get(tid.item(), "<UNK>") for tid in generated_ids_flat]
                
                adjustment_applied = False
                for i, token_text in enumerate(generated_tokens_text):
                    # Check if this token is considered relevant to the error
                    is_relevant = False
                    if token_text in relevant_token_texts:
                        is_relevant = True
                    else:
                        # Check if token is part of a relevant multi-token phrase (e.g., variable name)
                        for relevant in relevant_token_texts:
                            if token_text in relevant or relevant in token_text:
                                is_relevant = True
                                break
                                
                    if is_relevant:
                        # Amplify the weight based on severity, especially for negative advantage
                        weight_increase = tool_feedback.severity
                        if advantage < 0:
                            # Make penalty stronger for relevant tokens if advantage is negative
                            token_weights[i] = 1.0 + weight_increase * 1.5 # Stronger amplification for penalties
                        else:
                            # Slightly increase weight even for positive advantage if token caused error
                            token_weights[i] = 1.0 + weight_increase * 0.5 # Weaker amplification for rewards
                        adjustment_applied = True
                        logging.debug(f"  Adjusting weight for token '{token_text}' at index {i} to {token_weights[i]:.2f} (Severity: {tool_feedback.severity:.2f}, Advantage: {advantage:.2f})")
                
                if adjustment_applied:
                    logging.info(f"Applied differential token weighting based on tool feedback ({tool_feedback.feedback_type})")
        # --- End Token-wise Weights Calculation ---

        # --- Calculate Loss ---
        # 1. Policy Gradient Loss (REINFORCE with baseline and differential weighting)
        # Apply token-specific weights to the advantage for each token
        weighted_advantage = advantage * token_weights
        policy_loss = -torch.sum(log_prob_actions * weighted_advantage)
        
        # 2. Entropy Bonus Calculation
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
        entropy_bonus = -torch.sum(entropy) # Sum entropy over the sequence
        
        # 3. Total Loss
        loss = policy_loss + current_entropy_coef * entropy_bonus
        loss = loss.to(self.device)
        # --- End Calculate Loss ---

        current_loss = loss.item()
        policy_loss_val = policy_loss.item()
        entropy_bonus_val = entropy_bonus.item()
        logging.info(f"Learn Step: Reward={reward:.4f}, Baseline={self.baseline:.4f}, Advantage={advantage:.4f}")
        # Log the dynamic entropy coefficient used
        logging.info(f"  Total Loss={current_loss:.4f} (Policy={policy_loss_val:.4f}, EntropyBonus={entropy_bonus_val:.4f} * {current_entropy_coef:.4f})") 
        if tool_feedback:
            logging.info(f"  Tool Feedback: {tool_feedback.feedback_type}, Severity: {tool_feedback.severity:.2f}")

        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # --- Gradient Monitoring & Clipping ---
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logging.info(f"  Gradient Norm (Before Clip): {total_norm:.4f}")
        
        # Apply gradient clipping if norm > 0
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
        self.optimizer.step()
        logging.debug("Optimizer step performed.")

        return total_norm # Return the calculated norm
        
    def _add_to_experience_buffer(self, reward, generated_ids, tool_feedback):
        """Add successful experience to the buffer for future replay"""
        # Clone the tensor to avoid reference issues
        ids_clone = generated_ids.clone().detach()
        self.experience_buffer.append((reward, ids_clone, tool_feedback))
        logging.debug(f"Added experience with reward {reward:.4f} to buffer (size: {len(self.experience_buffer)})")
    
    def _should_replay_experience(self):
        """Determine if we should replay a successful experience"""
        if not self.experience_buffer or random.random() > self.replay_prob:
            return None
            
        # Sample from buffer with probabilities weighted by reward
        rewards = [exp[0] for exp in self.experience_buffer]
        # Normalize rewards to probabilities
        total = sum(rewards)
        if total == 0:
            return random.choice(self.experience_buffer)
            
        probs = [r/total for r in rewards]
        return random.choices(self.experience_buffer, weights=probs, k=1)[0]
        
    def _update_learning_rate(self, reward, tool_feedback=None):
        """Dynamically adjust learning rate based on reward progress and tool feedback."""
        if not self.enable_dynamic_lr:
            return
            
        # Track if we're making progress
        if reward > self.best_reward:
            self.best_reward = reward
            self.no_improvement_count = 0
            # Optional: Slightly increase LR on significant improvement?
            # new_lr = min(self.learning_rate, self.current_lr * 1.05)
            # ... update logic ...
        else:
            self.no_improvement_count += 1
            
        # Reduce learning rate if stuck at plateau for a while
        # Modify the condition to be less aggressive if the error was minor
        severity_factor = 1.0
        is_severe_error = False
        if tool_feedback and hasattr(tool_feedback, 'severity'):
            # Reduce impact of patience counter for less severe errors
            severity_factor = tool_feedback.severity 
            if tool_feedback.feedback_type in ['syntax_error', 'execution_timeout', 'import_error']:
                is_severe_error = True

        # Effective patience: higher patience for less severe errors
        # Example: severity 0.3 -> effective_patience = 50 / (0.3 + 0.1) = 125
        # Example: severity 0.9 -> effective_patience = 50 / (0.9 + 0.1) = 50
        effective_patience = self.lr_patience / max(0.1, severity_factor + 0.1) 
        
        # Only reduce LR if score is somewhat reasonable (avoid early collapse)
        # OR if the error is very severe (like syntax error)
        should_consider_reduction = (reward >= 0.3) or is_severe_error

        if self.no_improvement_count >= effective_patience and should_consider_reduction:
            # Make reduction factor dependent on severity - reduce less for minor errors
            reduction_factor = 0.8 * (severity_factor * 0.5 + 0.5) # Scale between 0.4 (severity 0) and 0.8 (severity 1)
            new_lr = max(self.min_learning_rate, self.current_lr * reduction_factor)
            
            # Only update if the change is significant
            if new_lr < self.current_lr * 0.95:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
                logging.info(f"Reducing learning rate: {self.current_lr:.6f} -> {new_lr:.6f} (no improvement for {self.no_improvement_count} steps, effective patience {effective_patience:.0f}, severity {severity_factor:.2f})")
                self.current_lr = new_lr
                self.no_improvement_count = 0  # Reset counter after adjustment
            else:
                 # Log even if change wasn't applied due to minimum LR or small diff
                 logging.debug(f"LR reduction condition met, but new LR {new_lr:.6f} not applied (current: {self.current_lr:.6f})")
                 # Still reset counter to avoid rapid successive reduction checks
                 self.no_improvement_count = 0

    def generate_agnostic_template(self, task):
        """
        Generates a sequence of abstract token IDs based on keywords in the task description.
        This provides a basic structural hint for the generator.

        Args:
            task: The benchmark task object

        Returns:
            tuple: (template_token_ids, template_boost_factor)
                template_token_ids: List of token IDs for template tokens to boost
                template_boost_factor: How much to boost template tokens
        """
        template_tokens = []
        template_boost_factor = 2.5 # Slightly lower boost than before, more general
        
        task_desc = task.description.lower() if hasattr(task, 'description') else ""
        task_name = task.name.lower() if hasattr(task, 'name') else ""
        combined_text = task_name + " " + task_desc

        # Simple keyword mapping to abstract tokens
        # Import re locally if not already imported globally
        import re 
        keyword_map = {
            "output": "OUTPUT_OP", "print": "OUTPUT_OP", "display": "OUTPUT_OP", "show": "OUTPUT_OP",
            "function": "FUNC_DEF", "define": "FUNC_DEF", "method": "FUNC_DEF",
            "if": "CONDITIONAL_IF", "condition": "CONDITIONAL_IF",
            "else": "CONDITIONAL_ELSE",
            "loop": ["LOOP_FOR", "LOOP_WHILE"], "iterate": ["LOOP_FOR", "LOOP_WHILE"], "repeat": ["LOOP_FOR", "LOOP_WHILE"],
            "for": "LOOP_FOR",
            "while": "LOOP_WHILE",
            "return": "RETURN_STMT",
            "add": "+", "sum": "+",
            "subtract": "-", "difference": "-",
            "multiply": "*", "product": "*",
            "divide": "/", "quotient": "/",
            "assign": "=", "variable": "VAR_GENERIC", "store": "=",
            "compare": ["COMP_EQ", "COMP_NEQ", "<", ">", "COMP_LTE", "COMP_GTE"], 
            "equal": "COMP_EQ", "same": "COMP_EQ",
            "not equal": "COMP_NEQ", "different": "COMP_NEQ",
            "less than": "<",
            "greater than": ">",
            "less or equal": "COMP_LTE",
            "greater or equal": "COMP_GTE",
            "true": "BOOL_TRUE",
            "false": "BOOL_FALSE",
            "none": "NULL_VALUE", "null": "NULL_VALUE"
        }

        found_abstract_tokens = set()

        # Check for keywords in the combined task text
        for keyword, abstract_token_or_list in keyword_map.items():
            # Use regex to find whole words or phrases to avoid partial matches (e.g., 'if' in 'different')
            # Handle keywords that might be regex special characters (like '+', '*')
            escaped_keyword = re.escape(keyword)
            if re.search(r'\b' + escaped_keyword + r'\b', combined_text):
                if isinstance(abstract_token_or_list, list):
                    for token in abstract_token_or_list:
                         found_abstract_tokens.add(token)
                else:
                    found_abstract_tokens.add(abstract_token_or_list)

        # Convert found abstract token strings to IDs
        template_token_ids = []
        for token_str in found_abstract_tokens:
            if token_str in self.tokenizer.token_to_id:
                template_token_ids.append(self.tokenizer.token_to_id[token_str])
            else:
                 logging.warning(f"Agnostic template keyword '{token_str}' not found in vocabulary.")

        if template_token_ids:
             logging.info(f"Generated agnostic template tokens based on task description: {found_abstract_tokens}")

        return template_token_ids, template_boost_factor

    def generate_agnostic_template(self, task):
        """
        Generates a sequence of abstract token IDs based on keywords in the task description.
        This provides a basic structural hint for the generator.

        Args:
            task: The benchmark task object

        Returns:
            tuple: (template_token_ids, template_boost_factor)
                template_token_ids: List of token IDs for template tokens to boost
                template_boost_factor: How much to boost template tokens
        """
        template_tokens = []
        template_boost_factor = 2.5 # Slightly lower boost than before, more general
        
        task_desc = task.description.lower() if hasattr(task, 'description') else ""
        task_name = task.name.lower() if hasattr(task, 'name') else ""
        combined_text = task_name + " " + task_desc

        # Simple keyword mapping to abstract tokens
        # Import re locally if not already imported globally
        import re
        keyword_map = {
            "output": "OUTPUT_OP", "print": "OUTPUT_OP", "display": "OUTPUT_OP", "show": "OUTPUT_OP",
            "function": "FUNC_DEF", "define": "FUNC_DEF", "method": "FUNC_DEF",
            "if": "CONDITIONAL_IF", "condition": "CONDITIONAL_IF",
            "else": "CONDITIONAL_ELSE",
            "loop": ["LOOP_FOR", "LOOP_WHILE"], "iterate": ["LOOP_FOR", "LOOP_WHILE"], "repeat": ["LOOP_FOR", "LOOP_WHILE"],
            "for": "LOOP_FOR",
            "while": "LOOP_WHILE",
            "return": "RETURN_STMT",
            "add": "+", "sum": "+",
            "subtract": "-", "difference": "-",
            "multiply": "*", "product": "*",
            "divide": "/", "quotient": "/",
            "assign": "=", "variable": "VAR_GENERIC", "store": "=",
            "compare": ["COMP_EQ", "COMP_NEQ", "<", ">", "COMP_LTE", "COMP_GTE"],
            "equal": "COMP_EQ", "same": "COMP_EQ",
            "not equal": "COMP_NEQ", "different": "COMP_NEQ",
            "less than": "<",
            "greater than": ">",
            "less or equal": "COMP_LTE",
            "greater or equal": "COMP_GTE",
            "true": "BOOL_TRUE",
            "false": "BOOL_FALSE",
            "none": "NULL_VALUE", "null": "NULL_VALUE"
        }

        found_abstract_tokens = set()

        # Check for keywords in the combined task text
        for keyword, abstract_token_or_list in keyword_map.items():
            # Use regex to find whole words or phrases to avoid partial matches (e.g., 'if' in 'different')
            # Handle keywords that might be regex special characters (like '+', '*')
            escaped_keyword = re.escape(keyword)
            if re.search(r'\b' + escaped_keyword + r'\b', combined_text):
                if isinstance(abstract_token_or_list, list):
                    for token in abstract_token_or_list:
                         found_abstract_tokens.add(token)
                else:
                    found_abstract_tokens.add(abstract_token_or_list)

        # Convert found abstract token strings to IDs
        template_token_ids = []
        for token_str in found_abstract_tokens:
            if token_str in self.tokenizer.token_to_id:
                template_token_ids.append(self.tokenizer.token_to_id[token_str])
            else:
                 logging.warning(f"Agnostic template keyword '{token_str}' not found in vocabulary.")

        if template_token_ids:
             logging.info(f"Generated agnostic template tokens based on task description: {found_abstract_tokens}")

        return template_token_ids, template_boost_factor

    def get_state(self):
        """Returns a dictionary containing the current state of the generator for reporting."""
        return {
            'current_learning_rate': self.current_lr,
            'max_entropy_coefficient': self.max_entropy_coefficient,
            'min_entropy_coefficient': self.min_entropy_coefficient,
            'baseline': self.baseline,
            'no_improvement_count': self.no_improvement_count,
            'best_reward': self.best_reward,
            'token_frequency': self.token_frequency.copy(), # Return a copy
            'experience_buffer_size': len(self.experience_buffer)
        }
