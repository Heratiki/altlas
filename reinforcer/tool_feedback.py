"""
Tool Feedback module for extracting structured learning signals from execution results.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

class ToolFeedback:
    """
    Analyzes execution results to provide structured feedback for learning.
    This class extracts error types, patterns, and relevant tokens to help
    the model understand what went wrong in code execution.
    """
    
    # Error type classification patterns
    ERROR_PATTERNS = {
        'syntax_error': [r'SyntaxError', r'IndentationError', r'TabError'],
        'name_error': [r'NameError.*not defined', r"name '([^']*)' is not defined"],
        'type_error': [r'TypeError', r'unsupported operand type', r'object is not'],
        'value_error': [r'ValueError'],
        'index_error': [r'IndexError', r'index out of range'],
        'key_error': [r'KeyError'],
        'attribute_error': [r'AttributeError', r"has no attribute '([^']*)'"],
        'import_error': [r'ImportError', r'ModuleNotFoundError'],
        'zero_division_error': [r'ZeroDivisionError', r'division by zero'],
        'assertion_error': [r'AssertionError'],
        'execution_timeout': [r'TimeoutError', r'execution timed out'],
    }
    
    # Token patterns that often cause errors
    PROBLEMATIC_TOKEN_PATTERNS = {
        'syntax_error': [r'"([^"]*)"', r"'([^']*)'", r'def\s+(\w+)', r'class\s+(\w+)',
                       r'import\s+(\w+)', r'from\s+(\w+)', r'\w+\s*\(', r'\w+\s*='],
        'name_error': [r"'([^']*)'", r'"([^"]*)"', r'([a-zA-Z_][a-zA-Z0-9_]*)'],
        'type_error': [r"'([^']*)'", r'"([^"]*)"', r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('],
        'default': [r'[a-zA-Z_][a-zA-Z0-9_]*', r'"[^"]*"', r"'[^']*'", r'\d+']
    }
    
    def __init__(self, code_attempt: str, execution_result: Dict):
        """
        Initialize with the code attempt and its execution result.
        
        Args:
            code_attempt (str): The generated code string that was executed
            execution_result (Dict): The result from the code executor
        """
        self.code = code_attempt
        self.result = execution_result
        
        # Extract core feedback attributes
        self.feedback_type = self._classify_feedback()
        self.error_line = self._extract_error_line()
        self.error_message = self._extract_error_message()
        self.severity = self._determine_severity()
        self.relevant_tokens = self._identify_relevant_tokens()
        
        logging.debug(f"Tool Feedback: Type={self.feedback_type}, Severity={self.severity:.2f}")
        if self.error_line:
            logging.debug(f"  Error on line {self.error_line}: {self.error_message}")
        if self.relevant_tokens:
            logging.debug(f"  Relevant tokens: {', '.join(list(self.relevant_tokens)[:5])}" + 
                         (f" and {len(self.relevant_tokens)-5} more" if len(self.relevant_tokens) > 5 else ""))
    
    def _classify_feedback(self) -> str:
        """
        Classify the type of feedback based on execution result.
        
        Returns:
            str: The classified error type or 'success' for successful execution
        """
        # Check for successful execution first
        if self.result.get('status') == 'success':
            output = self.result.get('stdout', '').strip()
            if output:
                return 'execution_success'
            else:
                return 'execution_success_no_output'
        
        # Get error message
        error = self.result.get('error', '')
        
        # Check for timeout
        if self.result.get('status') == 'timeout':
            return 'execution_timeout'
        
        # Check for specific error types
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error, re.IGNORECASE):
                    return error_type
        
        # Default error type if no specific match
        return 'generic_error'
    
    def _extract_error_line(self) -> Optional[int]:
        """
        Extract the line number from the error message if available.
        
        Returns:
            Optional[int]: Line number or None if not found
        """
        if self.result.get('status') != 'error':
            return None
            
        error = self.result.get('error', '')
        
        # Common patterns for error line reporting
        patterns = [
            r'line\s+(\d+)', 
            r'(\d+):\s+',
            r', line (\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def _extract_error_message(self) -> str:
        """
        Extract the clean error message without the traceback.
        
        Returns:
            str: The core error message
        """
        if self.result.get('status') != 'error':
            return ""
            
        error = self.result.get('error', '')
        
        # Try to extract just the last line of the error message
        lines = error.strip().split('\n')
        if lines:
            # Look for the actual error message (typically after the traceback)
            for line in reversed(lines):
                if ':' in line and not line.lstrip().startswith('File "'):
                    return line.strip()
            
            # Fallback to the last line
            return lines[-1].strip()
        
        return error.strip()
    
    def _determine_severity(self) -> float:
        """
        Determine how severe the error is (0.0-1.0 scale).
        Higher severity means more fundamental problems.
        
        Returns:
            float: Severity score between 0.0 and 1.0
        """
        # Successful execution has 0 severity
        if self.feedback_type.startswith('execution_success'):
            return 0.0
        
        # Base severity by error type
        base_severity = {
            'syntax_error': 0.9,        # Very severe - code won't even parse
            'name_error': 0.7,          # Severe - undefined variable
            'type_error': 0.6,          # Moderately severe - wrong type
            'value_error': 0.5,         # Moderately severe - invalid value
            'index_error': 0.5,         # Moderately severe - invalid index
            'key_error': 0.5,           # Moderately severe - missing key
            'attribute_error': 0.6,     # Moderately severe - missing attribute
            'import_error': 0.7,        # Severe - missing module
            'zero_division_error': 0.4, # Less severe - runtime math error
            'assertion_error': 0.3,     # Less severe - failed test condition
            'execution_timeout': 0.8,   # Very severe - infinite loop/hanging
            'generic_error': 0.5,       # Default moderate severity
        }.get(self.feedback_type, 0.5)
        
        # Adjust severity based on error location if available
        if self.error_line is not None:
            # Earlier line errors are often more fundamental
            line_count = len(self.code.split('\n'))
            if line_count > 1:
                # Normalized position in code (0-1)
                position_factor = self.error_line / line_count
                # Errors at the beginning are slightly more severe
                position_adjustment = 0.1 * (1 - position_factor)
                base_severity += position_adjustment
        
        # Cap severity between 0 and 1
        return min(1.0, max(0.0, base_severity))
    
    def _identify_relevant_tokens(self) -> Set[str]:
        """
        Identify which tokens likely contributed to the error.
        
        Returns:
            Set[str]: Set of tokens that are likely relevant to the error
        """
        relevant_tokens = set()
        
        # For successful execution, no error tokens
        if self.feedback_type.startswith('execution_success'):
            return relevant_tokens
            
        error_msg = self.result.get('error', '')
        
        # Extract specific patterns based on error type
        patterns = self.PROBLEMATIC_TOKEN_PATTERNS.get(
            self.feedback_type, 
            self.PROBLEMATIC_TOKEN_PATTERNS['default']
        )
        
        # Look in error message first
        for pattern in patterns:
            for match in re.finditer(pattern, error_msg):
                token = match.group(0).strip()
                if token and len(token) > 1:  # Avoid single-character tokens
                    relevant_tokens.add(token)
        
        # If we found specific tokens in the error message, return them
        if relevant_tokens:
            return relevant_tokens
            
        # Otherwise, fall back to code analysis
        # If we have an error line, focus on that line
        if self.error_line is not None and self.error_line > 0:
            lines = self.code.split('\n')
            if 0 <= self.error_line - 1 < len(lines):  # Convert to 0-indexed
                error_line_content = lines[self.error_line - 1]
                
                # Extract tokens from the error line
                for pattern in patterns:
                    for match in re.finditer(pattern, error_line_content):
                        token = match.group(0).strip()
                        if token and len(token) > 1:  # Avoid single-character tokens
                            relevant_tokens.add(token)
        
        # If still no tokens found, extract some from the whole code
        if not relevant_tokens:
            for pattern in patterns:
                for match in re.finditer(pattern, self.code):
                    token = match.group(0).strip()
                    if token and len(token) > 1:  # Avoid single-character tokens
                        relevant_tokens.add(token)
                        
                        # Limit to a reasonable number of tokens
                        if len(relevant_tokens) >= 10:
                            break
        
        return relevant_tokens
    
    def get_feedback_summary(self) -> Dict:
        """
        Get a summary of the feedback in dictionary form.
        
        Returns:
            Dict: Summary of the feedback analysis
        """
        return {
            'feedback_type': self.feedback_type,
            'severity': self.severity,
            'error_line': self.error_line,
            'error_message': self.error_message,
            'relevant_tokens': list(self.relevant_tokens),
        }
    
    def get_penalty_factors(self) -> Dict[str, float]:
        """
        Get penalty factors to adjust learning for specific tokens.
        
        Returns:
            Dict[str, float]: Mapping of tokens to their penalty factors
        """
        penalties = {}
        
        # Only assign penalties for errors
        if self.feedback_type.startswith('execution_success'):
            return penalties
            
        # Base penalty factor based on severity
        base_penalty = self.severity * 0.5  # Scale to 0-0.5 range
        
        # Assign penalties to relevant tokens
        for token in self.relevant_tokens:
            penalties[token] = base_penalty
            
        return penalties