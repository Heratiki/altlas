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
import logging
import json
import re

try:
    import resource
except ImportError:
    resource = None

@dataclass
class ExecutionResult:
    """Results from executing a code attempt."""
    status: str  # 'success', 'error', 'timeout'
    stdout: str
    stderr: str
    runtime: float
    exit_code: int
    code: str = ""  # Store the code that was executed
    exception: str = None
    error: str = ""
    error_type: str = ""

class CodeExecutor:
    """Executes code attempts safely and returns results."""
    
    def __init__(self, config_path="config.ini"):
        config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        # Use absolute path for reliability within modules
        abs_config_path = Path(__file__).parent.parent / config_path
        config.read(abs_config_path)
        exec_config = config['Executor']
        self.timeout = exec_config.getint('Timeout', 5)
        
        # Initialize language-specific command maps
        self._language_command_map = {
            "python": ["python"],
            "javascript": ["node"],
            "js": ["node"],  # Alias for javascript
            # Add other languages as needed
        }
        
        # Initialize language-specific file extensions
        self._language_extension_map = {
            "python": ".py",
            "javascript": ".js",
            "js": ".js",  # Alias for javascript
            # Add other languages as needed
        }
        
        # Load custom language maps if available
        self._load_language_maps()
    
    def _load_language_maps(self):
        """Load language-specific command and extension maps from config if available."""
        language_map_path = Path(__file__).parent.parent / "language_maps" / "execution_config.json"
        
        if language_map_path.exists():
            try:
                with open(language_map_path, 'r') as f:
                    execution_config = json.load(f)
                    
                # Update command and extension maps with values from config
                if 'command_map' in execution_config:
                    self._language_command_map.update(execution_config['command_map'])
                    
                if 'extension_map' in execution_config:
                    self._language_extension_map.update(execution_config['extension_map'])
                    
                logging.info(f"Loaded language execution configuration from {language_map_path}")
            except Exception as e:
                logging.error(f"Error loading language execution configuration: {e}")
        
    def _extract_error_type(self, error_text: str, fallback: str = "ExecutionError") -> str:
        """Extract a stable error type label from stderr or an exception message."""
        if not error_text:
            return fallback

        for line in reversed(error_text.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*)", stripped)
            if match:
                return match.group(1)

        return fallback

    def _build_preexec_fn(self, resource_limits):
        """Create a POSIX pre-exec function for CPU and memory limits."""
        if not resource_limits or os.name == "nt" or resource is None:
            return None

        memory_limit_mb = resource_limits.get("memory_limit")
        cpu_limit_seconds = resource_limits.get("cpu_limit")

        if memory_limit_mb is None and cpu_limit_seconds is None:
            return None

        def _apply_limits():
            if memory_limit_mb is not None:
                memory_bytes = int(memory_limit_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            if cpu_limit_seconds is not None:
                cpu_seconds = max(1, int(cpu_limit_seconds))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        return _apply_limits

    def execute(self, code, language="python", timeout=None, resource_limits=None):
        """Execute the provided code and return the results.
        """
        if getattr(self, 'debug_mode', False):
            logging.debug(f"[DEBUG] Executing code in language: {language}")
        if getattr(self, 'debug_mode', False):
            logging.debug(f"[DEBUG] Executing code in language: {language}")
            logging.debug(f"[DEBUG] Code to execute:\n{code}")

        # Normalize language name to lowercase
        language = language.lower()
        effective_timeout = timeout if timeout is not None else self.timeout
        
        # Get the appropriate file extension for the language
        file_extension = self._language_extension_map.get(language, ".txt")
        
        # Get the appropriate command for the language
        command = self._language_command_map.get(language)
        
        # If language not supported, log an error and fall back to Python
        if not command:
            logging.error(f"Unsupported language: {language}. Falling back to Python.")
            language = "python"
            file_extension = ".py"
            command = ["python"]
        
        # Create a temporary file for the code with the appropriate extension
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_path = temp_file.name
        
        start_time = time.time()
        
        try:
            # Execute the code in a subprocess with the appropriate command
            process = subprocess.Popen(
                command + [temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._build_preexec_fn(resource_limits),
            )
            
            try:
                stdout, stderr = process.communicate(timeout=effective_timeout)
                exit_code = process.returncode
                status = 'success' if exit_code == 0 else 'error'
                error = stderr.strip() if status == 'error' else ""
                error_type = self._extract_error_type(error, fallback="ExecutionError") if error else ""
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                status = 'timeout'
                error = f"Execution timed out after {effective_timeout} seconds."
                error_type = "TimeoutError"
                
        except Exception as e:
            status = 'error'
            stdout = ''
            stderr = str(e)
            exit_code = -1
            error = str(e)
            error_type = type(e).__name__
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
            exit_code=exit_code,
            code=code,
            exception=stderr if status == "error" else None,
            error=error if 'error' in locals() else "",
            error_type=error_type if 'error_type' in locals() else "",
        )

        # After execution, log results
        if getattr(self, 'debug_mode', False):
            try:
                logging.debug(f"[DEBUG] Execution stdout: {stdout.strip()}")
                logging.debug(f"[DEBUG] Execution stderr: {stderr.strip()}")
                logging.debug(f"[DEBUG] Execution return code: {exit_code}")
            except Exception:
                pass
