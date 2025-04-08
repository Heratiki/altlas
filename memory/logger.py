"""
Logger for storing attempt data.
"""

import json
import time
from pathlib import Path
import os
import logging

class AttemptLogger:
    """Logs code attempts and their results."""
    
    def __init__(self, log_dir=None):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to store log files. Defaults to memory/logs.
        """
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.attempts = []
        self.best_score = 0.0
        self.best_attempt_number = None
        self.best_attempt = None
        
        # Load historical attempts if they exist
        self._load_history()

    def _load_history(self):
        """Load historical attempts from disk."""
        history_file = self.log_dir / "attempt_history.json"
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.attempts = data.get('attempts', [])
                    self.best_score = data.get('best_score', 0.0)
                    self.best_attempt_number = data.get('best_attempt_number')
                    self.best_attempt = data.get('best_attempt')
                logging.info(f"Loaded {len(self.attempts)} historical attempts.")
        except Exception as e:
            logging.error(f"Error loading attempt history: {e}", exc_info=True)
            # Start fresh if there's an error
            self.attempts = []
            self.best_score = 0.0
            self.best_attempt_number = None
            self.best_attempt = None

    def _save_history(self):
        """Save the full attempt history to disk."""
        history_file = self.log_dir / "attempt_history.json"
        try:
            data = {
                'attempts': self.attempts,
                'best_score': self.best_score,
                'best_attempt_number': self.best_attempt_number,
                'best_attempt': self.best_attempt
            }
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving attempt history: {e}", exc_info=True)

    def log_attempt(self, attempt_number: int, code: str, result, score: float, fingerprint: str):
        """Log a code attempt and its results."""
        attempt = {
            'attempt_number': attempt_number,
            'code': code,
            'fingerprint': fingerprint,
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
            self.best_attempt_number = attempt_number
            self.best_attempt = attempt
            
            # Save the best attempt separately
            self._save_best_attempt()
        
        # Save full history periodically (every 100 attempts)
        if attempt_number % 100 == 0:
            self._save_history()
    
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

    def get_best_score_info(self):
        """Get the best score achieved so far and the attempt number."""
        return self.best_score, self.best_attempt_number
        
    def get_attempt(self, attempt_number: int):
        """Get a specific attempt by its number.
        
        Args:
            attempt_number (int): The attempt number to retrieve.
            
        Returns:
            dict: The attempt data, or None if not found.
        """
        # First check if it's the best attempt since we track that specially
        if attempt_number == self.best_attempt_number:
            return self.best_attempt
            
        # Otherwise search through attempts
        for attempt in self.attempts:
            if attempt['attempt_number'] == attempt_number:
                return attempt
                
        return None