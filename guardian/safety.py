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
        # FUTURE INTENT: This pattern matching is extremely basic and easily bypassed.
        # A more robust safety checker would involve:
        # 1.  Static Analysis: Using tools like Bandit or custom AST analysis to detect potentially
        #     dangerous patterns (e.g., arbitrary code execution, file system access beyond
        #     a designated scratch space, network calls).
        # 2.  Dynamic Analysis/Sandboxing: Executing the code within a heavily restricted environment
        #     (e.g., a minimal Docker container with no network access, limited filesystem visibility,
        #     resource constraints via cgroups) and monitoring its behavior for disallowed system calls
        #     or resource exhaustion. This is partially handled by the CodeExecutor but could be enhanced.
        # 3.  Capability Limiting: Modifying the execution environment itself to remove access to
        #     dangerous modules or functions (e.g., using restricted Python modes if applicable, or
        #     overriding built-ins within the execution context).
        # 4.  Learned Safety Models: Potentially training a model to predict the likelihood of code
        #     being unsafe based on patterns observed during training, although this is complex and
        #     requires careful validation.
        for pattern in self.unsafe_patterns:
            if re.search(pattern, code):
                return False
        
        # For now, everything else is considered safe
        # This will evolve to be more sophisticated
        # FUTURE INTENT: The default assumption should eventually shift towards caution.
        # Code should perhaps be considered unsafe unless proven otherwise by passing rigorous checks.
        return True