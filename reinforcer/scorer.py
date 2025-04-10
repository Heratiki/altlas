"""
Module for scoring code execution results against task criteria.
"""

import configparser
from pathlib import Path
import difflib
import ast
import logging
import re
import json
import os
from typing import Dict, Optional, List, Union, Any

from reinforcer.tool_feedback import ToolFeedback

class AttemptScorer:
    """Scores execution results against task criteria."""

    def __init__(self, config_path="config.ini"):
        config = configparser.ConfigParser()
        # Use absolute path for reliability within modules
        abs_config_path = Path(__file__).parent.parent / config_path
        if not abs_config_path.exists():
             # Fallback if not found relative to parent
             abs_config_path = Path(config_path).resolve()
             if not abs_config_path.exists():
                  raise FileNotFoundError(f"Config file not found at {config_path} or {abs_config_path}")
        config.read(abs_config_path)
        # No scorer-specific config needed for now
        # scorer_config = config['Scorer']
        
        # Initialize language maps cache
        self._language_maps_cache = {}
        self._language_maps_dir = Path(__file__).parent.parent / "language_maps"
        if not self._language_maps_dir.exists():
            logging.warning(f"Language maps directory not found at {self._language_maps_dir}")
            # Create the directory if it doesn't exist
            os.makedirs(self._language_maps_dir, exist_ok=True)
    
    def load_language_map(self, language_name: str) -> Dict[str, str]:
        """
        Load a language-specific mapping file.
        
        Args:
            language_name (str): The name of the language to load the map for
                                (e.g., "python", "javascript")
                                
        Returns:
            Dict[str, str]: Mapping from abstract tokens to language-specific tokens
        """
        # Check if already cached
        if language_name in self._language_maps_cache:
            return self._language_maps_cache[language_name]
        
        # Try to load the language-specific map
        language_map_path = self._language_maps_dir / f"{language_name}.json"
        default_map_path = self._language_maps_dir / "default.json"
        
        try:
            if language_map_path.exists():
                with open(language_map_path, 'r') as f:
                    language_map = json.load(f)
                    logging.info(f"Loaded language map for '{language_name}'")
            elif default_map_path.exists():
                # Fall back to default.json if language-specific map doesn't exist
                with open(default_map_path, 'r') as f:
                    language_map = json.load(f)
                    logging.warning(f"Language map for '{language_name}' not found. Using default map.")
            else:
                # Fall back to hardcoded defaults if no map files exist
                logging.warning(f"No language maps found. Using hardcoded default mappings.")
                language_map = self._get_default_fallback_map()
            
            # Cache the loaded map
            self._language_maps_cache[language_name] = language_map
            return language_map
            
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing language map for '{language_name}': {e}")
            # Fall back to hardcoded defaults on error
            language_map = self._get_default_fallback_map()
            self._language_maps_cache[language_name] = language_map
            return language_map
        except Exception as e:
            logging.error(f"Unexpected error loading language map for '{language_name}': {e}")
            # Fall back to hardcoded defaults on error
            language_map = self._get_default_fallback_map()
            self._language_maps_cache[language_name] = language_map
            return language_map

    def _get_default_fallback_map(self) -> Dict[str, str]:
        """Return a hardcoded default mapping as a last resort fallback."""
        return {
            # Core Operations / Keywords
            "OUTPUT_OP": "print",
            "CONDITIONAL_IF": "if",
            "CONDITIONAL_ELSE": "else",
            "LOOP_FOR": "for",
            "LOOP_WHILE": "while",
            "FUNC_DEF": "def",
            "RETURN_STMT": "return",
            "PASS_STMT": "pass",

            # Operators
            "=": "=",
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "COMP_EQ": "==",
            "COMP_NEQ": "!=",
            "<": "<",
            ">": ">",
            "COMP_LTE": "<=",
            "COMP_GTE": ">=",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",

            # Literals
            "BOOL_TRUE": "True",
            "BOOL_FALSE": "False",
            "NULL_VALUE": "None",
            "NUMBER_LITERAL_PLACEHOLDER": "0",
            "STRING_LITERAL_PLACEHOLDER": '""',

            # Generic Identifiers
            "VAR_GENERIC": "var",
            "FUNC_GENERIC": "func",

            # Newline
            "\\n": "\n"
        }

    def normalize_for_scoring(self, decoded_code: str, task=None) -> str:
        """
        Replaces abstract token representations with concrete literals for scoring,
        using language-specific mappings based on the task's target language.
        
        Args:
            decoded_code (str): The code to normalize
            task (Task, optional): The task object, which may specify target_language
            
        Returns:
            str: Normalized code for scoring
        """
        # Determine the target language
        target_language = "python"  # Default to Python for backward compatibility
        if task and hasattr(task, 'target_language'):
            target_language = task.target_language
            
        # Load the appropriate language map
        replacements = self.load_language_map(target_language)
        
        normalized_code = decoded_code
        # Use regex to replace whole words only to avoid partial matches (e.g., VAR_GENERIC_2)
        for abstract, literal in replacements.items():
            # Use word boundaries (\b) to match whole tokens
            normalized_code = re.sub(r'\b' + re.escape(abstract) + r'\b', literal, normalized_code)
            
        return normalized_code

    def score(self, code_attempt: str, result, task) -> float:
        """Score the execution result against the task's success criteria,
        using a modular, adaptive reward shaping strategy.
        
        Args:
            code_attempt (str): The code string that was generated and executed.
            result (ExecutionResult): The result of code execution.
            task (Task): The task definition.
            
        Returns:
            float: Score between 0.0 and 1.0.
        """
        # Initialize the tool feedback analyzer
        tool_feedback = ToolFeedback(code_attempt, result.__dict__ if hasattr(result, '__dict__') else result)
        feedback_type = tool_feedback.feedback_type
        
        # Initialize various scoring components
        scores = {
            'syntax': self._evaluate_syntax(code_attempt, task),
            'execution': self._evaluate_execution(result, feedback_type),
            'output': self._evaluate_output_match(result, task),
            'structural': self._evaluate_structural_match(code_attempt, task),
            'constraints': self._evaluate_constraints(code_attempt, task),
            'semantic': self._evaluate_semantic_similarity(code_attempt, result, task)
        }
        
        # Log all individual score components for debugging and analysis
        logging.debug(f"Score components: {', '.join([f'{k}: {v:.2f}' for k, v in scores.items()])}")
        
        # Calculate progressive final score with adaptive weighting
        if result.status == 'success' and scores['output'] > 0.9:
            # Full reward for correct output with successful execution
            final_score = 1.0
        else:
            # Calculate weighted score with progressive scaling
            final_score = self._calculate_weighted_score(scores, feedback_type)
            
            # Apply smooth, non-linear scaling to break through learning plateaus
            if final_score > 0.4:
                # Increase granularity in mid-to-high range
                final_score = 0.4 + ((final_score - 0.4) * 1.2)
                final_score = min(1.0, final_score)  # Cap at 1.0
        
        logging.debug(f"Scorer: Final score: {final_score:.2f}, Feedback type: {feedback_type}")
        return final_score
    
    def _evaluate_syntax(self, code_attempt: str, task=None) -> float:
        """
        Evaluate the syntactic validity of the code after normalization.
        
        The validation method depends on the task's target language:
        - For Python, uses ast.parse()
        - For other languages, returns a neutral score (syntax validation not implemented)
        """
        # Determine the target language
        target_language = "python"  # Default to Python for backward compatibility
        if task and hasattr(task, 'target_language'):
            target_language = task.target_language
        
        # Normalize the code using the appropriate language map
        normalized_attempt = self.normalize_for_scoring(code_attempt, task)
        
        # Choose validation method based on language
        if target_language.lower() == "python":
            try:
                ast.parse(normalized_attempt)
                # Give a moderate score for valid syntax
                return 0.25
            except SyntaxError:
                logging.debug("Scorer: Python syntax check failed (SyntaxError)")
                return 0.05
            except Exception as e:
                logging.warning(f"Scorer: Python syntax check failed ({type(e).__name__})")
                return 0.05
        else:
            # For non-Python languages, return a neutral score (0.1) since we can't validate syntax
            logging.debug(f"Scorer: Syntax check for {target_language} not implemented. Using neutral score.")
            return 0.1
    
    def _evaluate_execution(self, result, feedback_type: str) -> float:
        """Evaluate the execution outcome based on feedback type."""
        # Use the feedback analyzer to classify the execution result
        feedback_scores = {
            'execution_success': 0.8,           # Successful execution with output
            'execution_success_no_output': 0.5, # Successful execution but no output
            'syntax_error': 0.05,               # Syntax errors are serious
            'name_error': 0.15,                 # Name errors (undefined variables)
            'type_error': 0.2,                  # Type errors (often close but wrong type)
            'value_error': 0.2,                 # Value errors (often close but wrong value)
            'index_error': 0.25,                # Index errors (collection handling issues)
            'key_error': 0.25,                  # Key errors (dict access issues)
            'attribute_error': 0.2,             # Attribute errors (object usage issues)
            'import_error': 0.15,               # Import errors (missing modules)
            'zero_division_error': 0.3,         # Division errors (math problems)
            'assertion_error': 0.4,             # Assertion errors (test failures but executable)
            'execution_timeout': 0.05,          # Timeouts are serious problems
        }
        
        # If it's a generic error, use severity to determine a more dynamic reward
        if feedback_type == 'generic_error':
            # Extract error severity if available (default to 0.5 if not)
            severity = getattr(result, 'error_severity', 0.5)
            
            # Calculate a graduated reward based on severity:
            # - Low severity (0.0-0.3): Better reward (0.2-0.3)
            # - Medium severity (0.3-0.7): Medium reward (0.1-0.2)
            # - High severity (0.7-1.0): Lower reward (0.05-0.1)
            if severity < 0.3:
                # Less severe errors get better rewards
                base_score = 0.3 - (severity * 0.33)  # 0.3 down to 0.2
            elif severity < 0.7:
                # Medium severity errors
                base_score = 0.2 - ((severity - 0.3) * 0.25)  # 0.2 down to 0.1
            else:
                # Most severe errors
                base_score = 0.1 - ((severity - 0.7) * 0.167)  # 0.1 down to 0.05
                
            # Add a small random variation to avoid getting stuck at exact same values
            variation = (hash(str(result)) % 100) / 1000  # ±0.05 variation
            return max(0.05, min(0.35, base_score + variation))
        
        return feedback_scores.get(feedback_type, 0.1)  # Still return 0.1 as absolute fallback
    
    def _evaluate_output_match(self, result, task) -> float:
        """Evaluate how well the normalized output matches the expected output."""
        # Default score if no expected output
        if 'success_criteria' not in task.__dict__ or 'expected_output' not in task.success_criteria:
            return 0.0

        expected_output = task.success_criteria.get('expected_output', '').strip()
        # Normalize the actual output using task-specific language map
        actual_output = self.normalize_for_scoring(result.stdout.strip(), task)
        
        if not expected_output:  # No expected output defined
            return 0.0
            
        # Exact match gives full points
        if actual_output == expected_output:
            return 1.0
            
        # Partial match using string similarity
        similarity = difflib.SequenceMatcher(None, actual_output, expected_output).ratio()
        
        # Scale similarity - we want high scores only for very close matches
        # 0.8+ similarity gives 0.6-0.9 score, anything lower is scaled down significantly
        if similarity > 0.9:
            return 0.9  # Very close but not exact
        elif similarity > 0.8:
            return 0.7  # Close match
        elif similarity > 0.6:
            return 0.4  # Moderate match
        elif similarity > 0.4:
            return 0.2  # Weak match
        else:
            return 0.1  # Poor match
    
    def _evaluate_structural_match(self, code_attempt: str, task) -> float:
        """Evaluate structural similarity to expected patterns in the task, including code structure patterns."""
        if 'success_criteria' not in task.__dict__ or 'valid_patterns' not in task.success_criteria:
            return 0.0

        # Check for exact pattern matches first
        if self._check_exact_pattern_match(code_attempt, task):
            return 0.9  # High score but not 1.0 (reserved for correct output)

        # Additional: Check for structural pattern matches (regex or AST-based)
        structure_score = 0.0
        if 'structure_patterns' in task.success_criteria:
            for pattern in task.success_criteria['structure_patterns']:
                try:
                    if re.search(pattern, code_attempt, re.MULTILINE):
                        structure_score = max(structure_score, 0.2)  # Boost if any pattern matches
                except Exception:
                    continue  # Ignore invalid regex

        # Combine similarity score with structure score (capped at 0.9)
        similarity_score = self._calculate_pattern_similarity(code_attempt, task)
        return min(0.9, similarity_score + structure_score)
    
    def _check_exact_pattern_match(self, code_attempt: str, task) -> bool:
        """Check if normalized code exactly matches any of the valid patterns."""
        if 'valid_patterns' not in task.success_criteria:
            return False

        # First, normalize abstract tokens using task-specific language map, then normalize whitespace/case
        normalized_for_scoring = self.normalize_for_scoring(code_attempt, task)
        normalized_attempt = self._normalize_code(normalized_for_scoring, task)
        
        for pattern_group in task.success_criteria['valid_patterns']:
            # Check main pattern
            if self._matches_pattern(normalized_attempt, pattern_group, task):
                return True
                
            # Check variations if they exist
            if 'variations' in pattern_group:
                for variation in pattern_group['variations']:
                    normalized_variation = self._normalize_pattern(variation, task)
                    if normalized_attempt == normalized_variation:
                        return True
        
        return False
    
    def _normalize_code(self, code: str, task) -> str:
        """Normalize code based on task sensitivity settings."""
        # Default to case-sensitive and whitespace-insensitive
        case_sensitive = task.success_criteria.get('case_sensitive', True)
        whitespace_sensitive = task.success_criteria.get('whitespace_sensitive', False)
        
        normalized = code.strip()
        
        if not case_sensitive:
            normalized = normalized.lower()
            
        if not whitespace_sensitive:
            # Normalize whitespace while preserving line breaks
            lines = normalized.split('\n')
            normalized_lines = [' '.join(line.split()) for line in lines]
            normalized = '\n'.join(normalized_lines)
            
        return normalized
    
    def _normalize_pattern(self, pattern: Union[str, List[str]], task) -> str:
        """Normalize a pattern for comparison."""
        if isinstance(pattern, list):
            pattern = '\n'.join(pattern)
            
        return self._normalize_code(pattern, task)
    
    def _matches_pattern(self, normalized_code: str, pattern_group: Dict[str, Any], task) -> bool:
        """Check if normalized code matches a pattern group."""
        pattern = pattern_group.get('pattern', '')
        normalized_pattern = self._normalize_pattern(pattern, task)
        return normalized_code == normalized_pattern
    
    def _calculate_pattern_similarity(self, code_attempt: str, task) -> float:
        """Calculate similarity between normalized code and valid patterns."""
        max_similarity = 0.0
        # First, normalize abstract tokens using task-specific language map, then normalize whitespace/case
        normalized_for_scoring = self.normalize_for_scoring(code_attempt, task)
        normalized_attempt = self._normalize_code(normalized_for_scoring, task)

        
        for pattern_group in task.success_criteria.get('valid_patterns', []):
            # Check main pattern
            pattern = pattern_group.get('pattern', '')
            normalized_pattern = self._normalize_pattern(pattern, task)
            similarity = difflib.SequenceMatcher(None, normalized_attempt, normalized_pattern).ratio()
            max_similarity = max(max_similarity, similarity)
            
            # Check variations
            for variation in pattern_group.get('variations', []):
                normalized_variation = self._normalize_pattern(variation, task)
                var_similarity = difflib.SequenceMatcher(None, normalized_attempt, normalized_variation).ratio()
                max_similarity = max(max_similarity, var_similarity)
        
        # Scale similarity to an appropriate reward range
        # Map the raw similarity (0.0-1.0) to a score range (0.2-0.7)
        # This creates a smoother progression from syntax validity to full solution
        scaled_similarity = 0.2 + (max_similarity * 0.5)
        
        logging.debug(f"Pattern similarity: Raw {max_similarity:.2f}, Scaled {scaled_similarity:.2f}")
        return scaled_similarity
    
    def _evaluate_constraints(self, code_attempt: str, task) -> float:
        """Evaluate if normalized code meets task-specific constraints."""
        normalized_attempt = self.normalize_for_scoring(code_attempt, task)
        constraints = getattr(task, 'constraints', {})
        if not constraints:
            return 0.0
            
        score = 0.0
        
        # Check for required operators in normalized code
        if 'required_operators' in constraints:
            operators_found = all(op in normalized_attempt for op in constraints['required_operators'])
            if operators_found:
                score += 0.05
                
        # Check for required numbers/values in normalized code
        if 'required_numbers' in constraints:
            # Convert numbers to strings for checking presence in normalized code
            numbers_found = all(str(num) in normalized_attempt for num in constraints['required_numbers'])
            if numbers_found:
                score += 0.05
                
        # Check for required keywords/identifiers in normalized code
        if 'required_keywords' in constraints:
            keywords_found = all(kw in normalized_attempt for kw in constraints['required_keywords'])
            if keywords_found:
                score += 0.05
                
        # Check for token limit constraints (using original code for token count is okay)
        if 'max_tokens' in constraints:
            # Simple approximation of token count using split on original code
            token_count = len(code_attempt.split()) # Use original code for token count
            if token_count <= constraints['max_tokens']:
                score += 0.05
                
        return score
    
    def _evaluate_semantic_similarity(self, code_attempt: str, result, task) -> float:
        """
        Evaluate semantic similarity of the normalized code to the task requirements.
        This looks at deeper meaning rather than just surface patterns.
        """
        normalized_attempt = self.normalize_for_scoring(code_attempt, task)
        # Initialize semantic score
        semantic_score = 0.0

        # Check if code contains key concepts from the task description
        if hasattr(task, 'description'):
            # Extract likely keywords from task description
            description_keywords = self._extract_keywords(task.description)
            
            # Check for keyword presence in normalized code
            keyword_matches = sum(1 for kw in description_keywords if kw in normalized_attempt)
            if keyword_matches > 0:
                semantic_score += min(0.2, keyword_matches * 0.05)  # Cap at 0.2
        
        # Check for expected function/class names if specified
        if 'success_criteria' in task.__dict__:
            criteria = task.success_criteria
            
            # Function name matching
            if 'function_name' in criteria:
                # Check against normalized code
                function_pattern = f"def {criteria['function_name']}"
                if function_pattern in normalized_attempt:
                    semantic_score += 0.2
            
            # Class name matching
            if 'class_name' in criteria:
                # Check against normalized code
                class_pattern = f"class {criteria['class_name']}"
                if class_pattern in normalized_attempt:
                    semantic_score += 0.2
        
        # Look at AST structure of normalized code to evaluate if it has the right elements
        # This is language-specific, so only do it for Python
        if task and hasattr(task, 'target_language') and task.target_language.lower() == "python":
            try:
                tree = ast.parse(normalized_attempt)
                
                # Check for specific AST node types that might be required
                # This is a generic check that can be enhanced for specific tasks
                has_function_def = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                has_class_def = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
                has_control_flow = any(isinstance(node, (ast.If, ast.For, ast.While)) for node in ast.walk(tree))
                
                # Award points for having appropriate code structures
                if 'success_criteria' in task.__dict__ and 'required_structures' in task.success_criteria:
                    required = task.success_criteria['required_structures']
                    if 'function' in required and has_function_def:
                        semantic_score += 0.1
                    if 'class' in required and has_class_def:
                        semantic_score += 0.1
                    if 'control_flow' in required and has_control_flow:
                        semantic_score += 0.1
                else:
                    # Generic scoring if no specific requirements
                    if has_function_def:
                        semantic_score += 0.05
                    if has_class_def:
                        semantic_score += 0.05
                    if has_control_flow:
                        semantic_score += 0.05
            except:
                # If code can't be parsed, no semantic points from AST analysis
                pass
        
        return min(0.5, semantic_score)  # Cap at 0.5 to leave room for output matching
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key programming concepts and identifiers from text."""
        # Extract words that are likely to be code-related
        # This is a simple extraction - could be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        return [w for w in words if len(w) > 2 and w.lower() not in {
            'the', 'and', 'for', 'that', 'with', 'this', 'from', 'task', 
            'generate', 'code', 'create', 'write', 'benchmark'
        }]
    
    def _calculate_weighted_score(self, scores: Dict[str, float], feedback_type: str) -> float:
        """
        Calculate a weighted score from component scores, with adaptive weighting
        based on the execution status and feedback type.
        """
        # Base weights - will be adjusted based on feedback and scores
        weights = {
            'syntax': 0.1,
            'execution': 0.2, 
            'output': 0.35,
            'structural': 0.15,
            'constraints': 0.1,
            'semantic': 0.1
        }
        
        # Adjust weights based on execution feedback type
        if feedback_type.startswith('execution_success'):
            # Successful execution - emphasize output matching and constraints
            weights['output'] = 0.4
            weights['structural'] = 0.2
            weights['syntax'] = 0.05  # Lower weight for syntax when execution is successful
        elif 'error' in feedback_type:
            # Execution error - emphasize structural and semantic similarity
            weights['structural'] = 0.25
            weights['semantic'] = 0.2
            weights['execution'] = 0.15  # Lower execution score weight for errors
        
        # Calculate weighted sum of scores
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        
        # Ensure minimum score is at least 0.05 to avoid zero rewards
        return max(0.05, weighted_sum)
    
    def get_tool_feedback(self, code_attempt: str, result) -> ToolFeedback:
        """
        Create a ToolFeedback object for the execution result.
        
        Args:
            code_attempt (str): The code that was executed
            result: The execution result
            
        Returns:
            ToolFeedback: Analysis of the execution result
        """
        return ToolFeedback(code_attempt, result.__dict__ if hasattr(result, '__dict__') else result)

    def get_state(self):
        """Returns a dictionary containing the current state of the scorer for reporting."""
        # Return the base weights used for scoring components
        base_weights = {
            'syntax': 0.1,
            'execution': 0.2, 
            'output': 0.35,
            'structural': 0.15,
            'constraints': 0.1,
            'semantic': 0.1
        }
        return {
            'component_weights': base_weights
        }

    def calculate_reward(self, code_output, success, syntax_valid, execution_valid, has_output, has_structure, has_correct_ops, has_almost_correct_result, correct_format, config):
        """Calculates reward based on code output and execution status."""
        reward = 0.0

        # Access reward weights from config
        # Use config.get('Runner', 'RewardWeights', fallback=...) if stored in config.ini
        # Or access directly if passed as a dictionary
        reward_weights = config.get('reward_weights', {
            "unsafe_code": 0.0,
            "compiles": 0.1,
            "produces_output": 0.2,
            "structured_output": 0.3,
            "correct_ops": 0.5,
            "almost_correct": 0.7,
            "correct_format": 0.9,
            "correct_simplistic": 0.95,
            "correct_compiled": 1.0
        })

        if not syntax_valid:
            # Penalize unsafe/empty code implicitly by returning 0
            # Could add specific check if needed: reward = reward_weights["unsafe_code"]
            return reward_weights.get("unsafe_code", 0.0)

        # Base reward for compiling
        reward = reward_weights.get("compiles", 0.1)

        # Incremental rewards based on achievements
        if execution_valid:
            if has_output:
                reward = max(reward, reward_weights.get("produces_output", 0.2))
                if has_structure:
                    reward = max(reward, reward_weights.get("structured_output", 0.3))
                    if has_correct_ops:
                        reward = max(reward, reward_weights.get("correct_ops", 0.5))
                        if has_almost_correct_result:
                            reward = max(reward, reward_weights.get("almost_correct", 0.7))
                            if correct_format:
                                reward = max(reward, reward_weights.get("correct_format", 0.9))
                                if success: # Simplistic correct answer (might need refinement)
                                     reward = max(reward, reward_weights.get("correct_simplistic", 0.95))

        # Full reward for perfect success after compile/run
        if success and execution_valid:
            reward = reward_weights.get("correct_compiled", 1.0)

        return reward