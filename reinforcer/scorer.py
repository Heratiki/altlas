"""
Module for scoring code execution results against task criteria.
"""

class AttemptScorer:
    """Scores execution results against task criteria."""
    
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
            return 0.0  # Not implemented yet
        else:
            return 0.0  # Unknown task type
    
    def _score_exact_output(self, result, task):
        """Score based on exact output matching."""
        expected = task.success_criteria['exact_output']
        actual = result.stdout
        
        if not task.success_criteria.get('case_sensitive', True):
            expected = expected.lower()
            actual = actual.lower()
        
        # Exact match
        if actual == expected:
            return 1.0
            
        # Partial match
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