"""
Code executor that runs the generated code in a sandbox.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
import time
import configparser
from pathlib import Path

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
    
    def __init__(self, config_path="config.ini"):
        config = configparser.ConfigParser()
        # Use absolute path for reliability within modules
        abs_config_path = Path(__file__).parent.parent / config_path
        config.read(abs_config_path)
        exec_config = config['Executor']
        self.timeout = exec_config.getint('Timeout', 5)
        
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
            # FUTURE INTENT: The execution environment needs careful consideration for safety and consistency.
            # 1.  Enhanced Sandboxing: While using a subprocess is a start, true sandboxing might involve:
            #     - Running inside dedicated, minimal Docker containers per execution.
            #     - Using technologies like `nsjail` or `firecracker` for stronger isolation.
            #     - Applying resource limits (CPU, memory, time, network access, filesystem visibility)
            #       more granularly than just a simple timeout.
            # 2.  Environment Consistency: Ensure the execution environment is identical across all runs
            #     and matches the target environment if the agent is intended to produce code for a specific platform.
            # 3.  State Management: For tasks requiring state across multiple executions (e.g., interacting
            #     with a simulated API or filesystem), the executor needs to manage this state safely.
            # 4.  Observability: Potentially capture more detailed execution traces (e.g., using `sys.settrace`
            #     or external tools) if needed for advanced scoring or debugging, while being mindful of overhead.
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