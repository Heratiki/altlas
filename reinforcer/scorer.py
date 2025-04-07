"""
Module for scoring code execution results against task criteria.
"""

import configparser
from pathlib import Path

class AttemptScorer:
    """Scores execution results against task criteria."""

    def __init__(self, config_path="config.ini"):
        config = configparser.ConfigParser()
        # Use absolute path for reliability within modules
        abs_config_path = Path(__file__).parent.parent / config_path
        config.read(abs_config_path)
        scorer_config = config['Scorer']
        # SuccessThreshold is primarily used by the runner to decide if the task is complete.
        # The scorer itself should aim to return a score between 0.0 and 1.0 reflecting quality.
        # We might not need to read it here unless we use it for scaling internal scores.
        # For now, let's keep the scorer focused on calculating the raw score.
        # self.success_threshold = scorer_config.getfloat('SuccessThreshold', 0.99)
    
    def score(self, result, task):
        """Score the execution result against the task's success criteria.
        
        Args:
            result (ExecutionResult): The result of code execution.
            task (Task): The task definition.
            
        Returns:
            float: Score between 0.0 and 1.0.
        """
        # If execution failed, give a very low score
        if result.status != 'success':
            return 0.01 if result.status == 'error' else 0.0
        
        # Process based on task type
        if 'exact_output' in task.success_criteria:
            return self._score_exact_output(result, task)
        elif 'function_name' in task.success_criteria:
            # FUTURE INTENT: Implement scoring for function-based tasks.
            # This requires more complex analysis than simple stdout matching:
            # 1.  Code Parsing/Execution: Safely parse the generated code (e.g., using `ast`) 
            #     or execute it within a restricted namespace (`exec`) to find the required function.
            # 2.  Function Validation: Check if a function with the correct name (`task.success_criteria['function_name']`)
            #     exists and potentially check its signature (number of arguments).
            # 3.  Test Case Execution: Iterate through `task.success_criteria['test_cases']`, call the 
            #     user's function with the provided inputs, and compare the actual output against the expected output.
            # 4.  Partial Scoring: Award partial points for defining the function correctly, passing some tests, etc.
            # 5.  Error Handling: Gracefully handle errors during function execution within the test cases.
            return 0.0  # Not implemented yet
        else:
            # FUTURE INTENT: As more complex task types are added (e.g., class implementation, 
            # algorithm efficiency, interaction with APIs), this section will need corresponding
            # scoring logic. The scorer needs to evolve alongside the task complexity.
            return 0.0  # Unknown task type
    
    def _score_exact_output(self, result, task):
        """Score based on exact output matching."""
        expected = task.success_criteria['exact_output']
        actual = result.stdout
        
        if not task.success_criteria.get('case_sensitive', True):
            expected = expected.lower()
            actual = actual.lower()
        
        # Exact match - return 1.0 for perfect match
        if actual == expected:
            return 1.0
            
        # Partial match
        # FUTURE INTENT: The partial matching logic is currently very simple (substring/word overlap).
        # More sophisticated partial scoring could involve:
        # - Sequence alignment algorithms (e.g., Levenshtein distance) to measure closeness.
        # - Semantic similarity if comparing more complex outputs.
        # - Scoring based on partial fulfillment of structured output requirements.
        if expected in actual:
            return 0.7
            
        # Contains some of the words
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        common_words = expected_words.intersection(actual_words)
        
        if common_words:
            return 0.3 * (len(common_words) / len(expected_words))
            
        # No match
        return 0.0