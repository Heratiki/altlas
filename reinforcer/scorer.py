"""
Module for scoring code execution results against task criteria.
"""

import configparser
from pathlib import Path
import difflib # Import difflib for SequenceMatcher
import ast # <--- Import ast for syntax checking
import logging # <--- Import logging
from typing import Dict, Optional

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
    
    def score(self, code_attempt: str, result, task):
        """Score the execution result against the task's success criteria,
        incorporating a base score for syntactic validity.
        
        Args:
            code_attempt (str): The code string that was generated and executed.
            result (ExecutionResult): The result of code execution.
            task (Task): The task definition.
            
        Returns:
            float: Score between 0.0 and 1.0.
        """
        # --- Create Tool Feedback object to analyze the result ---
        tool_feedback = ToolFeedback(code_attempt, result.__dict__ if hasattr(result, '__dict__') else result)
        
        # --- Check Syntax First ---
        base_syntax_score = 0.05 # Base score if syntax is invalid
        try:
            ast.parse(code_attempt)
            base_syntax_score = 0.25 # Higher base score for valid syntax
            logging.debug(f"Scorer: Syntax check passed for attempt. Base score: {base_syntax_score}")
        except SyntaxError:
            logging.debug(f"Scorer: Syntax check failed (SyntaxError). Base score: {base_syntax_score}")
            pass # Keep base_syntax_score at 0.05
        except Exception as e: # Catch other potential parsing errors like MemoryError
            logging.warning(f"Scorer: Syntax check failed ({type(e).__name__}). Base score: {base_syntax_score}")
            pass # Keep base_syntax_score at 0.05
        # --- End Syntax Check ---

        # --- Score based on Error Types from Tool Feedback ---
        feedback_type = tool_feedback.feedback_type
        feedback_score = self._get_score_from_feedback_type(feedback_type)
        
        # --- Execution Success Score ---
        execution_score = 0.0
        if result.status == 'success':
            if 'exact_output' in task.success_criteria:
                execution_score = self._score_exact_output(result, task)
            elif 'function_name' in task.success_criteria:
                # Future: implement function-based scoring
                execution_score = 0.05
            else:
                execution_score = 0.05
        
        # --- Final Score: Use the highest of the different scoring methods ---
        final_score = max(base_syntax_score, feedback_score, execution_score)
        
        # Log scoring details
        logging.debug(f"Scorer: Base syntax score: {base_syntax_score:.2f}, Feedback score: {feedback_score:.2f}, Execution score: {execution_score:.2f}")
        logging.debug(f"Scorer: Final score: {final_score:.2f}, Feedback type: {feedback_type}")
        
        return final_score

    def _get_score_from_feedback_type(self, feedback_type: str) -> float:
        """Get a score based on the feedback type."""
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
            'generic_error': 0.1,               # Default for unclassified errors
        }
        
        return feedback_scores.get(feedback_type, 0.1)
    
    def _score_exact_output(self, result, task):
        """Score based on pattern matching and output validation."""
        if task.success_criteria['type'] == 'code_pattern':
            # Check output first
            output_score = 0.0
            if result.stdout.strip() == task.success_criteria['expected_output']:
                output_score = 0.8  # Base score for correct output
            
            # Check code patterns
            code_score = 0.0
            code = result.code.strip()
            
            # Normalize whitespace if not sensitive
            if not task.success_criteria.get('whitespace_sensitive', True):
                code = ' '.join(code.split())
            
            # Check each pattern type
            for pattern_group in task.success_criteria['valid_patterns']:
                # Convert pattern to canonical form for comparison
                if isinstance(pattern_group['pattern'], list):
                    pattern = '\n'.join(pattern_group['pattern'])
                else:
                    pattern = pattern_group['pattern']
                
                # Check main pattern
                if self._matches_pattern(code, pattern):
                    code_score = 0.2
                    break
                
                # Check variations if they exist
                if 'variations' in pattern_group:
                    for variation in pattern_group['variations']:
                        if isinstance(variation, list):
                            variation = '\n'.join(variation)
                        if self._matches_pattern(code, variation):
                            code_score = 0.2
                            break
                    if code_score > 0:
                        break
            
            # Check constraints if they exist
            constraint_score = 0.0
            if 'constraints' in task:
                constraints_met = True
                if 'required_operators' in task['constraints']:
                    for op in task['constraints']['required_operators']:
                        if op not in code:
                            constraints_met = False
                if 'required_numbers' in task['constraints']:
                    for num in task['constraints']['required_numbers']:
                        if str(num) not in code:
                            constraints_met = False
                if constraints_met:
                    constraint_score = 0.1
            
            return min(1.0, output_score + code_score + constraint_score)
            
        else:  # Fall back to simple output matching
            expected = task.success_criteria['exact_output']
            actual = result.stdout.strip()
            return 1.0 if actual == expected else 0.0
            
    def _matches_pattern(self, code: str, pattern: str) -> bool:
        """Check if code matches a pattern, handling whitespace normalization."""
        code = ' '.join(code.split())
        pattern = ' '.join(pattern.split())
        return code == pattern
            
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