"""
Code executor that runs the generated code in a sandbox.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
import time

@dataclass
class ExecutionResult:
    """Results from executing a code attempt."""
    status: str  # 'success', 'error', 'timeout'
    stdout: str
    stderr: str
    runtime: float
    exit_code: int
    exception: str = None

class CodeExecutor:
    """Executes code attempts safely and returns results."""
    
    def __init__(self, timeout=5):
        self.timeout = timeout  # seconds
        
    def execute(self, code):
        """Execute the provided code and return the results.
        
        Args:
            code (str): The Python code to execute.
            
        Returns:
            ExecutionResult: The results of the execution.
        """
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_path = temp_file.name
        
        start_time = time.time()
        
        try:
            # Execute the code in a subprocess
            process = subprocess.Popen(
                ['python', temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                exit_code = process.returncode
                status = 'success' if exit_code == 0 else 'error'
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                status = 'timeout'
                
        except Exception as e:
            status = 'error'
            stdout = ''
            stderr = str(e)
            exit_code = -1
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        runtime = time.time() - start_time
        
        return ExecutionResult(
            status=status,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            runtime=runtime,
            exit_code=exit_code
        )