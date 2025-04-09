#!/usr/bin/env python3
"""
AltLAS AttemptManager - Module for managing attempts, state persistence, and fingerprinting.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

log = logging.getLogger(__name__)

class AttemptManager:
    """
    Manages attempts, state persistence, and fingerprinting for AltLAS.
    
    This class combines functionality previously spread across:
    - State persistence functions in runner.py
    - AttemptLogger in memory/logger.py
    - AttemptFingerprinter in memory/fingerprints.py
    """
    def __init__(self, project_root=None):
        """
        Initialize the attempt manager.
        
        Args:
            project_root (Path, optional): Path to the project root directory.
                If None, it will be inferred from the current file's location.
        """
        if project_root is None:
            # Default to the parent directory of the directory containing this file
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.state_file = self.project_root / "memory" / "state.json"
        self.metrics_state_file = self.project_root / "memory" / "reports" / "metrics_state.json"
        self.fingerprint_file = self.project_root / "memory" / "fingerprints.json"
        self.best_attempt_file = self.project_root / "memory" / "logs" / "best_attempt.json"
        self.attempt_history_file = self.project_root / "memory" / "logs" / "attempt_history.json"
        
        # Ensure directories exist
        self._ensure_dirs_exist()
        
        # Initialize attempt history and fingerprints
        self.attempts = []
        self.fingerprints = self._load_fingerprints()
        self._load_attempts()
        self.failed_fingerprints = {} # Initialize failed fingerprints tracker
    
    def _ensure_dirs_exist(self):
        """Ensure that all necessary directories exist."""
        (self.project_root / "memory").mkdir(exist_ok=True)
        (self.project_root / "memory" / "logs").mkdir(exist_ok=True)
        (self.project_root / "memory" / "reports").mkdir(exist_ok=True)
    
    def _load_fingerprints(self) -> Dict[str, int]:
        """
        Load fingerprints from the fingerprint file.
        
        Returns:
            dict: Dictionary mapping fingerprints to their occurrence count.
        """
        if not self.fingerprint_file.exists():
            return {}
        
        try:
            with open(self.fingerprint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Error loading fingerprints: {e}")
            return {}
    
    def _save_fingerprints(self):
        """Save fingerprints to the fingerprint file."""
        try:
            with open(self.fingerprint_file, 'w') as f:
                json.dump(self.fingerprints, f, indent=2)
        except Exception as e:
            log.error(f"Error saving fingerprints: {e}")
    
    def _load_attempts(self):
        """Load attempt history from file."""
        if not self.attempt_history_file.exists():
            self.attempts = []
            return
        
        try:
            with open(self.attempt_history_file, 'r') as f:
                self.attempts = json.load(f)
            log.debug(f"Loaded {len(self.attempts)} attempts from history file")
        except Exception as e:
            log.error(f"Error loading attempt history: {e}")
            self.attempts = []
    
    def _save_attempts(self):
        """Save attempt history to file."""
        try:
            with open(self.attempt_history_file, 'w') as f:
                json.dump(self.attempts, f, indent=2)
            log.debug(f"Saved {len(self.attempts)} attempts to history file")
        except Exception as e:
            log.error(f"Error saving attempt history: {e}")
    
    def get_fingerprint(self, code: str) -> str:
        """
        Get fingerprint for a code snippet.
        
        Args:
            code (str): The code to fingerprint.
            
        Returns:
            str: The fingerprint hash for the code.
        """
        if not code:
            return "empty_code"
        
        # Create a normalized version of the code (preserving case but normalizing whitespace)
        normalized_code = ""
        
        # Handle newlines and space normalization but preserve case
        for line in code.split('\n'):
            # Remove leading/trailing whitespace from each line
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                # Normalize multiple spaces within a line
                normalized_line = ' '.join(word for word in stripped_line.split(' ') if word)
                normalized_code += normalized_line + '\n'
        
        # Use a simple hash as the fingerprint
        import hashlib
        return hashlib.md5(normalized_code.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, fingerprint: str) -> bool:
        """
        Check if a fingerprint is a duplicate.
        
        Args:
            fingerprint (str): The fingerprint to check.
            
        Returns:
            bool: True if the fingerprint is a duplicate, False otherwise.
        """
        # Note the fingerprint and increment its count
        if fingerprint in self.fingerprints:
            self.fingerprints[fingerprint] += 1
            self._save_fingerprints()
            return True
        else:
            self.fingerprints[fingerprint] = 1
            self._save_fingerprints()
            return False
    
    def log_attempt(self, attempt_count: int, code: str, result: Dict[str, Any], 
                   score: float, fingerprint: Optional[str] = None):
        """
        Log an attempt.
        
        Args:
            attempt_count (int): The attempt number.
            code (str): The generated code.
            result (dict): The execution result.
            score (float): The score for this attempt.
            fingerprint (str, optional): The fingerprint for this code.
        """
        if fingerprint is None:
            fingerprint = self.get_fingerprint(code)
        
        # Convert ExecutionResult objects to dicts if needed
        if hasattr(result, '__dict__') and not isinstance(result, dict):
            try:
                result = vars(result)
            except Exception:
                # fallback: convert to string if __dict__ fails
                result = str(result)
        
        # Create attempt record
        attempt = {
            'attempt': attempt_count,
            'timestamp': time.time(),
            'code': code,
            'result': result,
            'score': score,
            'fingerprint': fingerprint,
        }
        
        # Add to attempts list
        self.attempts.append(attempt)
        
        # Save to history file (periodically)
        if attempt_count % 10 == 0:  # Save every 10 attempts
            self._save_attempts()
        
        # Check if this is the best attempt
        best_score, _ = self.get_best_score_info()
        if score > best_score:
            # Save this as the best attempt
            try:
                with open(self.best_attempt_file, 'w') as f:
                    json.dump(attempt, f, indent=2)
                log.info(f"New best score: {score:.4f} (Attempt {attempt_count})")
            except Exception as e:
                log.error(f"Error saving best attempt: {e}") # Correctly indented log message

        # Track failed fingerprints (Moved outside the try/except block)
        if score < 0.5:
            self.failed_fingerprints[fingerprint] = self.failed_fingerprints.get(fingerprint, 0) + 1
            # Optionally save failed fingerprints periodically if needed
            # if attempt_count % 50 == 0: self._save_failed_fingerprints() 
    
    def get_attempt(self, attempt_num: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific attempt by number.
        
        Args:
            attempt_num (int): The attempt number to retrieve.
            
        Returns:
            dict or None: The attempt data, or None if not found.
        """
        for attempt in self.attempts:
            if attempt.get('attempt') == attempt_num:
                return attempt
        return None
    
    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.
        
        Returns:
            float: The best score.
        """
        best_score = 0.0
        for attempt in self.attempts:
            score = attempt.get('score', 0.0)
            if score > best_score:
                best_score = score
        return best_score
    
    def get_best_score_info(self) -> Tuple[float, int]:
        """
        Get the best score and its attempt number.
        
        Returns:
            tuple: (best_score, attempt_number)
        """
        best_score = 0.0
        best_attempt = 0
        for attempt in self.attempts:
            score = attempt.get('score', 0.0)
            if score > best_score:
                best_score = score
                best_attempt = attempt.get('attempt', 0)
        return best_score, best_attempt
    
    def get_history(self, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the attempt history.
        
        Args:
            max_entries (int, optional): Maximum number of history entries to return.
                If None, returns all entries.
                
        Returns:
            list: List of attempt records.
        """
        if max_entries is None:
            return self.attempts
        return self.attempts[-max_entries:]
    
    def save_run_state(self, attempt_count: int, task_name: str, success: bool, 
                      best_score: Optional[float] = None, best_attempt: Optional[int] = None):
        """
        Save the current run state to a file to allow resuming after interruption.
        
        Args:
            attempt_count (int): Current attempt count
            task_name (str): Name of the current task
            success (bool): Whether a successful solution has been found
            best_score (float, optional): Current best score
            best_attempt (int, optional): Attempt number of the best solution
            
        Returns:
            bool: True if state was saved successfully, False otherwise
        """
        try:
            state = {
                "attempt_count": attempt_count,
                "task_name": task_name,
                "success": success,
                "best_score": best_score,
                "best_attempt": best_attempt,
                "timestamp": time.time(),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Load and merge existing metrics state if it exists
            if self.metrics_state_file.exists():
                try:
                    with open(self.metrics_state_file, 'r') as f:
                        state["report_metrics"] = json.load(f)
                except Exception as e:
                    log.warning(f"Could not load metrics state: {e}")
                    state["report_metrics"] = {}
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            log.debug(f"Run state saved: attempt_count={attempt_count}, task={task_name}, success={success}")
            return True
        except Exception as e:
            log.error(f"Failed to save run state: {e}", exc_info=True)
            return False
    
    def load_run_state(self, task_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load the previous run state from file.
        
        Args:
            task_name (str, optional): If provided, only load state if it matches this task name
            
        Returns:
            dict or None: The loaded state or None if no state exists or on error
        """
        try:
            if not self.state_file.exists():
                log.debug("No previous run state found.")
                return None
                
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Validate state has required fields
            required_fields = ["attempt_count", "task_name", "success"]
            if not all(field in state for field in required_fields):
                log.warning("Previous run state file is invalid (missing required fields).")
                return None
                
            # Check if task name matches if specified
            if task_name and state["task_name"] != task_name:
                log.info(f"Previous run was for a different task ('{state['task_name']}'), starting fresh.")
                return None
                
            log.info(f"Loaded previous run state: attempt_count={state['attempt_count']}, task={state['task_name']}, success={state['success']}")
            return state
        except Exception as e:
            log.error(f"Error loading run state: {e}", exc_info=True)
            return None
    
    def clear_run_state(self) -> bool:
        """
        Delete the run state file.
        
        Returns:
            bool: True if state was cleared successfully, False otherwise
        """
        try:
            if self.state_file.exists():
                os.remove(self.state_file)
                log.info("Run state file cleared.")
            return True
        except Exception as e:
            log.error(f"Error clearing run state: {e}", exc_info=True)
            return False
    
    def reset_training_state(self) -> bool:
        """
        Fully reset all training state files to ensure a fresh start.
        
        Returns:
            bool: True if at least one file was deleted, False otherwise
        """
        log.info("🔄 Resetting all training state files...")
    
        # Clear run state JSON
        self.clear_run_state()
    
        # Reset metrics_state.json
        try:
            self.metrics_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_state_file, 'w') as f:
                json.dump({}, f)
            log.info("Reset memory/reports/metrics_state.json")
        except Exception as e:
            log.warning(f"Failed to reset metrics_state.json: {e}")
        
        # Define all paths to clear (using absolute paths)
        paths_to_clear = [
            self.project_root / "memory" / "model_state.pth",
            self.project_root / "memory" / "optimizer_state.pth",
            self.project_root / "memory" / "logs" / "best_attempt.json",
            self.project_root / "memory" / "logs" / "attempt_history.json",
            self.project_root / "memory" / "fingerprints.json",
            self.metrics_state_file
        ]
        
        # Add any additional state files found in memory directory
        memory_dir = self.project_root / "memory"
        for file in memory_dir.glob("*.pth"):
            if file not in paths_to_clear:
                paths_to_clear.append(file)
        
        # Count of files deleted
        deleted_count = 0
        
        # Delete each file if it exists
        for path in paths_to_clear:
            if path.exists():
                try:
                    os.remove(path)
                    log.info(f"✅ Deleted {path}")
                    deleted_count += 1
                except Exception as e:
                    log.error(f"⚠️ Failed to delete {path}: {e}")
        
        # Check logs directory for any other files
        logs_dir = self.project_root / "memory" / "logs"
        if logs_dir.exists():
            for file in logs_dir.glob("*.json"):
                try:
                    os.remove(file)
                    log.info(f"✅ Deleted additional log file: {file}")
                    deleted_count += 1
                except Exception as e:
                    log.error(f"⚠️ Failed to delete log file {file}: {e}")
        
        # Reset instance variables
        self.attempts = []
        self.fingerprints = {}
        
        if deleted_count > 0:
            log.info(f"✅ Training state reset complete. {deleted_count} files were deleted.")
        else:
            log.warning("⚠️ No training state files were found to delete.")
        
    def get_failure_penalty(self, fingerprint: str) -> float:
        """
        Return penalty factor for repeated failed fingerprints.
        """
        fail_count = self.failed_fingerprints.get(fingerprint, 0)
        if fail_count >= 3:
            return min(0.5, 0.1 * fail_count)  # Cap penalty
        return 0.0

        return deleted_count > 0