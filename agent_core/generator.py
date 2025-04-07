"""
Code generator that creates new code attempts based on task and history.
Now uses a PyTorch RNN model.
"""

import random
import json
from pathlib import Path
import configparser
import logging

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        config = configparser.ConfigParser()
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

            self.max_gen_length = gen_config.getint('MaxGenerationLength', 50)

            # Model Hyperparameters
            self.embedding_dim = model_config.getint('EmbeddingDim', 64)
            self.hidden_dim = model_config.getint('HiddenDim', 128)
            self.num_layers = model_config.getint('NumLayers', 1)

            # Optimizer Hyperparameters
            self.learning_rate = optimizer_config.getfloat('LearningRate', 0.001)

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
            num_layers=self.num_layers
        ).to(self.device)
        logging.info(f"Initialized AltLAS_RNN model with {sum(p.numel() for p in self.model.parameters())} parameters.")

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logging.info(f"Initialized Adam optimizer with learning rate: {self.learning_rate}")

        # Attempt to load previous state
        self._load_state() # Call load state during initialization

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
        """Parses a hint string to find relevant token IDs from the vocabulary."""
        if not hint:
            return set()

        hinted_token_ids = set()
        # Simple word extraction (lowercase)
        hint_words = set(word.strip('.,!?"\'`)') for word in hint.lower().split())

        for token, token_id in self.tokenizer.token_to_id.items():
            # Check if the token (stripped of quotes for literals) is in the hint words
            token_text = token.strip('"')
            if token_text in hint_words:
                hinted_token_ids.add(token_id)
                logging.debug(f"Hint '{hint}' matched token: '{token}' (ID: {token_id})")
            # Additionally check if parts of multi-word tokens match (e.g., 'hello' in '"hello"')
            elif token_text and token_text in hint: # Avoid matching empty strings
                 hinted_token_ids.add(token_id)
                 logging.debug(f"Hint '{hint}' contained token text: '{token_text}' (ID: {token_id})")

        return hinted_token_ids

    def generate(self, task=None, history=None, hint=None):
        """
        Generates a code attempt using the RNN model by sampling token by token.

        Args:
            task: The current task definition (unused in this basic version).
            history: History of previous attempts (unused in this basic version).
            hint: Hint from advisor (unused in this basic version).

        Returns:
            tuple: (generated_code_string, generated_ids_tensor)
                   - generated_code_string (str): The decoded code attempt.
                   - generated_ids_tensor (torch.Tensor): Tensor containing the IDs
                     of the generated tokens (excluding SOS/EOS).
                     Shape: (sequence_length,)
        """
        try:
            self.model.eval() # Set model to evaluation mode (disables dropout etc.)

            # Parse hint if provided
            hinted_token_ids = self._parse_hint(hint)
            hint_boost_factor = 1.5 # How much to increase probability mass for hinted tokens

            generated_ids = []
            log_probs_list = [] # Store log probs of chosen tokens

            # Start sequence generation with SOS token
            current_token_id = self.tokenizer.sos_token_id
            # Input tensor needs batch dimension: (batch_size=1, seq_len=1)
            input_tensor = torch.LongTensor([[current_token_id]]).to(self.device)

            # Initialize hidden state for the batch
            hidden = self.model.init_hidden(batch_size=1, device=self.device)

            with torch.no_grad(): # Disable gradient calculation during generation
                for _ in range(self.max_gen_length):
                    # Get output logits and updated hidden state from the model
                    # Output shape: (batch=1, seq=1, vocab_size)
                    logging.debug(f"Loop {_}: Input shape {input_tensor.shape}")
                    output_logits, hidden = self.model(input_tensor, hidden)

                    logging.debug(f"Loop {_}: Model output logits shape {output_logits.shape}")
                    # Get logits for the last token: shape (vocab_size)
                    # Squeeze removes batch and sequence dimensions (both are 1)
                    last_logits = output_logits.squeeze(0).squeeze(0)

                    # Apply softmax to get probabilities for sampling
                    probabilities = F.softmax(last_logits, dim=-1)
                    # --- Apply Hint Bias ---
                    if hinted_token_ids:
                        for token_id in hinted_token_ids:
                            if 0 <= token_id < probabilities.shape[0]: # Check bounds
                                probabilities[token_id] *= hint_boost_factor
                        # Re-normalize probabilities after boosting
                        # Add a small epsilon to prevent division by zero if all probabilities become zero
                        probabilities = probabilities / (probabilities.sum() + 1e-9)
                    # --- End Hint Bias ---

                    # Apply log_softmax to get log probabilities (numerically stable) - Use original logits for loss
                    log_probabilities = F.log_softmax(last_logits, dim=-1)

                    # Sample the next token ID based on the probabilities
                    next_token_id = torch.multinomial(probabilities, num_samples=1).item()

                    logging.debug(f"Loop {_}: Sampled next token ID: {next_token_id}")
                    # --- Token Handling ---
                    # 1. Stop if EOS token is generated
                    if next_token_id == self.tokenizer.eos_token_id:
                        break

                    # 2. Store generated ID (excluding SOS, implicitly excluding EOS by breaking)
                    generated_ids.append(next_token_id)

                    # 3. Store log probability of the *chosen* token for RL
                    log_probs_list.append(log_probabilities[next_token_id])

                    # 4. Prepare input for the next iteration: the chosen token
                    current_token_id = next_token_id
                    input_tensor = torch.LongTensor([[current_token_id]]).to(self.device)

            # Decode the generated sequence of IDs back to a string (outside the loop)
            generated_tensor = torch.LongTensor(generated_ids)
            generated_code = self.tokenizer.decode(generated_tensor) # decode handles filtering special tokens

            # Convert generated IDs list to tensor
            generated_ids_tensor = torch.LongTensor(generated_ids).to(self.device)
    
            logging.debug(f"Returning from generate. Code type: {type(generated_code)}, Generated IDs shape: {generated_ids_tensor.shape}")
            logging.debug(f"Generated code: '{generated_code}'")
    
            return generated_code, generated_ids_tensor

        except Exception as e:
            import traceback
            logging.error(f"!!! EXCEPTION INSIDE generate method: {type(e).__name__}: {e}")
            logging.error(traceback.format_exc())
            # Return None to indicate failure, handled by runner.py
            return None, None

    def learn(self, reward: float, generated_ids: torch.Tensor):
        """
        Updates the model weights using the REINFORCE algorithm based on the
        reward received for a generated sequence. Recalculates log probabilities
        with gradient tracking enabled.

        Args:
            reward (float): The scalar reward obtained for the generated sequence.
                            (In this simple case, the final score is used as the return).
            generated_ids (torch.Tensor): A tensor containing the IDs of the
                                          actions (tokens) taken in the generated sequence.
                                          Shape: (sequence_length,)
        """
        if generated_ids.numel() == 0:
            logging.warning("Learn called with empty generated_ids tensor. Skipping update.")
            return # Cannot learn from an empty sequence

        self.model.train() # Set model to training mode

        # --- Recalculate log probabilities with gradient tracking ---
        # Prepend SOS token for model input
        sos_tensor = torch.LongTensor([[self.tokenizer.sos_token_id]]).to(self.device)
        # Input sequence for forward pass: [SOS, token1, token2, ..., tokenN-1]
        # Target sequence for loss:       [token1, token2, ..., tokenN] (which is generated_ids)
        input_seq = torch.cat((sos_tensor, generated_ids.unsqueeze(0)[:, :-1]), dim=1)

        # Perform forward pass to get logits with gradients
        # We don't need the hidden state output here
        logits, _ = self.model(input_seq) # Shape: (batch=1, seq_len, vocab_size)
        logits = logits.squeeze(0) # Shape: (seq_len, vocab_size)

        # Calculate log probabilities from logits
        log_probabilities = F.log_softmax(logits, dim=-1)

        # Select the log probabilities corresponding to the actual generated tokens
        # generated_ids shape: (seq_len,)
        # log_probabilities shape: (seq_len, vocab_size)
        # We need log_prob_actions shape: (seq_len,)
        log_prob_actions = log_probabilities[range(generated_ids.shape[0]), generated_ids]
        # --- End Recalculation ---

        # REINFORCE loss: - (sum of log_prob_actions * reward)
        loss = -torch.sum(log_prob_actions) * reward
        loss = loss.to(self.device) # Ensure loss is on the correct device

        logging.debug(f"Calculated loss: {loss.item()} for reward: {reward}")

        # Perform optimization step
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()           # Calculate gradients
        # Optional: Gradient clipping can help stabilize training
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()      # Update model parameters

        logging.debug("Optimizer step performed.")


# Removed leftover return statement and comment