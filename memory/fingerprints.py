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
        """Normalize code to ignore non-semantic differences like comments and whitespace.
        Adds extraction of key syntax patterns for better duplicate detection."""
        # Remove comments
        code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

        # Extract key syntax patterns (function defs, class defs, imports)
        patterns = []
        try:
            # Function definitions
            patterns += re.findall(r'def\s+\w+\s*\(', code_no_comments)
            # Class definitions
            patterns += re.findall(r'class\s+\w+\s*[:(]', code_no_comments)
            # Imports
            patterns += re.findall(r'import\s+\w+', code_no_comments)
            patterns += re.findall(r'from\s+\w+\s+import', code_no_comments)
        except Exception:
            pass  # Defensive: don't break normalization if regex fails

        # Basic normalization: strip whitespace lines
        lines = code_no_comments.splitlines()
        processed_lines = [line.strip() for line in lines if line.strip()]
        normalized_code = "\n".join(processed_lines)

        # Append extracted patterns to normalized string
        syntax_summary = "|".join(sorted(set(patterns)))
        combined = normalized_code + "\n#PATTERNS:" + syntax_summary
        return combined
    
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