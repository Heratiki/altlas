"""
Logger for storing attempt data.
"""

import json
import time
from pathlib import Path
import os

class AttemptLogger:
    """Logs code attempts and their results."""
    
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = Path(__file__).parent / 'logs'
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.attempts = []
        self.best_score = 0.0
        self.best_attempt_number = 0 # Track which attempt got the best score
        self.best_attempt = None
        
    def log_attempt(self, attempt_number: int, code: str, result, score: float, fingerprint: str):
        """Log a code attempt with its result and score.

        Args:
            attempt_number (int): The current attempt number.
            code (str): The generated code.
            result (ExecutionResult): The execution result.
            score (float): The attempt's score.
            fingerprint (str): The attempt's fingerprint.
        """
        # FUTURE INTENT: The logging mechanism can be significantly enhanced:
        # 1.  Structured Logging: Use a more structured format (e.g., JSON lines) for easier parsing and analysis.
        # 2.  Selective Logging: Implement more sophisticated criteria for what constitutes a "valuable" attempt
        #     worth logging in detail vs. just summarizing. This could be based on score thresholds, novelty,
        #     specific error types encountered, or reduction in error complexity.
        # 3.  Metadata: Log richer metadata, such as the generator state (e.g., weights used), hint received (if any),
        #     task parameters, execution resource usage (CPU/memory if available from executor).
        # 4.  Database Backend: For larger scale runs, logging to files might become inefficient. Consider using
        #     a database (like SQLite initially, or potentially a NoSQL DB) for better querying and management.
        # 5.  Error Classification: Log categorized error types (SyntaxError, NameError, TypeError, etc.) 
        #     extracted from `result.stderr` to provide more specific feedback for learning.
        attempt = {
            'timestamp': time.time(),
            'code': code,
            'attempt_number': attempt_number, # Store attempt number
            'fingerprint': fingerprint,
            # FUTURE INTENT: Store the full ExecutionResult object or its dict representation 
            # instead of manually copying fields, ensuring all execution details are captured.
            'result': {
                'status': result.status if result else 'unsafe_or_duplicate',
                'stdout': result.stdout if result else '',
                'stderr': result.stderr if result else '',
                'runtime': result.runtime if result else 0,
                'exit_code': result.exit_code if result else -1
            },
            'score': score
        }
        
        self.attempts.append(attempt)
        
        # Update best score
        if score > self.best_score:
            self.best_score = score
            self.best_attempt_number = attempt_number # Update best attempt number
            self.best_attempt = attempt
            
            # Save the best attempt to disk
            self._save_best_attempt()
    
    def _save_best_attempt(self):
        """Save the best attempt to disk."""
        if self.best_attempt:
            best_path = self.log_dir / 'best_attempt.json'
            with open(best_path, 'w') as f:
                json.dump(self.best_attempt, f, indent=2)
    
    def get_history(self, limit=10):
        """Get the most recent attempts.
        
        Args:
            limit (int): Number of attempts to return.
            
        Returns:
            list: The most recent attempts.
        """
        # FUTURE INTENT: Retrieving history might become more complex.
        # - Need efficient ways to retrieve relevant history for specific learning algorithms 
        #   (e.g., getting attempts with similar code structures, or attempts that failed with specific errors).
        # - May involve querying a database if the backend changes.
        # - Could implement different history views (e.g., only successful attempts, only recent failures).
        return self.attempts[-limit:] if self.attempts else []
    
    def get_best_score(self):
        """Get the best score achieved so far."""
        return self.best_score

    def get_best_score_info(self):
        """Get the best score achieved so far and the attempt number."""
        return self.best_score, self.best_attempt_number