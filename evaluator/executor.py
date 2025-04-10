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

class CodeExecutor:
    """Executes code attempts safely and returns results."""
    
    def __init__(self, config_path="config.ini"):
        config = configparser.ConfigParser()
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
        
    def execute(self, code, language="python"):
        """Execute the provided code and return the results.
        
        Args:
            code (str): The code to execute.
            language (str): The programming language of the code (e.g., "python", "javascript").
                           Defaults to "python" for backward compatibility.
            
        Returns:
            ExecutionResult: The results of the execution.
        """
        # Normalize language name to lowercase
        language = language.lower()
        
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
            exit_code=exit_code,
            code=code  # Include the original code
        )