"""
Module for scoring code execution results against task criteria.
"""

import configparser
from pathlib import Path
import difflib # Import difflib for SequenceMatcher

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
    
    def score(self, result, task):
        """Score the execution result against the task's success criteria.
        
        Args:
            result (ExecutionResult): The result of code execution.
            task (Task): The task definition.
            
        Returns:
            float: Score between 0.0 and 1.0.
        """
        # Base score if execution failed
        if result.status != 'success':
            # Give slightly higher score for execution errors vs timeouts/other non-execution
            return 0.1 if result.status == 'error' else 0.0 
        
        # --- Execution Succeeded --- 
        
        # Process based on task type defined in success_criteria
        if 'exact_output' in task.success_criteria:
            return self._score_exact_output(result, task)
        elif 'function_name' in task.success_criteria:
            # FUTURE: Implement scoring for function-based tasks.
            return 0.05 # Give a minimal score for successful execution even if task type not fully scored
        else:
            # FUTURE: Implement scoring for other task types.
            return 0.05 # Give a minimal score for successful execution for unknown task types
    
    def _score_exact_output(self, result, task):
        """Score based on exact output matching, with partial credit using SequenceMatcher."""
        expected = task.success_criteria['exact_output']
        actual = result.stdout.strip() # Strip whitespace from actual output
        expected = expected.strip()   # Strip whitespace from expected output
        
        # Handle case sensitivity
        if not task.success_criteria.get('case_sensitive', True):
            expected = expected.lower()
            actual = actual.lower()
        
        # Exact match - return 1.0 for perfect match
        if actual == expected:
            return 1.0
            
        # Partial match using SequenceMatcher for similarity ratio
        # SequenceMatcher provides a ratio from 0.0 (no match) to 1.0 (perfect match)
        # We scale this ratio slightly to ensure it's less than the perfect score of 1.0
        # but higher than the base error score (0.1)
        try:
            # Handle potential empty strings
            if not actual and not expected:
                return 1.0 # Both empty could be considered a match depending on task
            if not actual or not expected:
                return 0.1 # One empty, one not - low score
                
            similarity_ratio = difflib.SequenceMatcher(None, actual, expected).ratio()
            # Scale the similarity score (e.g., map 0-1 ratio to 0.1-0.9 range)
            partial_score = 0.1 + similarity_ratio * 0.8 
            return max(0.1, partial_score) # Ensure score is at least the base error score
        except Exception as e:
            # Log error during scoring? 
            print(f"Error during SequenceMatcher scoring: {e}")
            return 0.1 # Fallback score if similarity calculation fails