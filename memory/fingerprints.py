"""
Fingerprinting system to prevent repeating the same failures.
"""

import hashlib
import re
from pathlib import Path
import json

class AttemptFingerprinter:
    """Creates and tracks fingerprints for code attempts."""
    
    def __init__(self, fingerprints_file=None):
        if fingerprints_file is None:
            fingerprints_file = Path(__file__).parent / 'fingerprints.json'
            
        self.fingerprints_file = Path(fingerprints_file)
        self.seen_fingerprints = self._load_fingerprints()
        
    def _load_fingerprints(self):
        """Load fingerprints from disk."""
        if self.fingerprints_file.exists():
            try:
                with open(self.fingerprints_file, 'r') as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()
    
    def _save_fingerprints(self):
        """Save fingerprints to disk."""
        with open(self.fingerprints_file, 'w') as f:
            json.dump(list(self.seen_fingerprints), f)
    
    def get_fingerprint(self, code):
        """Generate a fingerprint for a code attempt.
        
        Args:
            code (str): The code to fingerprint.
            
        Returns:
            str: The fingerprint.
        """
        # Normalize the code
        normalized = self._normalize_code(code)
        
        # Create a hash
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _normalize_code(self, code):
        """Normalize code to ignore non-semantic differences."""
        # FUTURE INTENT: This normalization is very basic (removing comments/blanks, lowercasing).
        # More sophisticated normalization could involve:
        # 1.  AST-based Normalization: Parsing the code into an Abstract Syntax Tree (AST) and then 
        #     serializing it back into a canonical string format. This ignores variations in whitespace,
        #     comment content, and potentially variable naming (if alpha-renaming is applied).
        # 2.  Semantic Hashing: Techniques like SimHashing could be used to generate fingerprints where
        #     similar code (semantically) results in similar hashes, allowing detection of near-duplicates,
        #     not just exact duplicates after basic normalization.
        # 3.  Considering Execution Behavior: Fingerprinting could potentially incorporate aspects of the
        #     code's execution trace or output for certain inputs, although this is more complex.
        
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove blank lines
        code = re.sub(r'\n\s*\n', '\n', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Convert to lowercase (simplistic but helps for now)
        # FUTURE INTENT: Lowercasing is too aggressive as it loses case sensitivity which matters in Python.
        # AST-based normalization would be a much better approach.
        return code.lower()
    
    def is_duplicate(self, fingerprint):
        """Check if a fingerprint has been seen before.
        
        Args:
            fingerprint (str): The fingerprint to check.
            
        Returns:
            bool: True if the fingerprint has been seen before.
        """
        is_dup = fingerprint in self.seen_fingerprints
        
        if not is_dup:
            self.seen_fingerprints.add(fingerprint)
            self._save_fingerprints()
            
        return is_dup