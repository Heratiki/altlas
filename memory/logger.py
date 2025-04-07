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
        self.best_attempt = None
        
    def log_attempt(self, code, result, score, fingerprint):
        """Log a code attempt with its result and score.
        
        Args:
            code (str): The generated code.
            result (ExecutionResult): The execution result.
            score (float): The attempt's score.
            fingerprint (str): The attempt's fingerprint.
        """
        attempt = {
            'timestamp': time.time(),
            'code': code,
            'fingerprint': fingerprint,
            'result': {
                'status': result.status,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'runtime': result.runtime,
                'exit_code': result.exit_code
            },
            'score': score
        }
        
        self.attempts.append(attempt)
        
        # Update best score
        if score > self.best_score:
            self.best_score = score
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
        return self.attempts[-limit:] if self.attempts else []
    
    def get_best_score(self):
        """Get the best score achieved so far."""
        return self.best_score