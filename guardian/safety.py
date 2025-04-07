"""
Safety module to filter unsafe or redundant code attempts.
"""

import re

class SafetyChecker:
    """Checks if generated code is safe to execute."""
    
    def __init__(self):
        # Define patterns for unsafe code
        self.unsafe_patterns = [
            r'import\s+os', 
            r'import\s+sys',
            r'import\s+subprocess',
            r'__import__',
            r'eval\(',
            r'exec\(',
            r'open\(',
            r'file\(',
            r'socket\(',
            r'subprocess\.',
            r'os\.',
            r'sys\.',
        ]
        
    def is_safe(self, code):
        """Check if the provided code is safe to execute.
        
        Returns:
            bool: True if the code is safe, False otherwise.
        """
        # Simple pattern matching for unsafe code
        for pattern in self.unsafe_patterns:
            if re.search(pattern, code):
                return False
        
        # For now, everything else is considered safe
        # This will evolve to be more sophisticated
        return True