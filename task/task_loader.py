"""
Task loader module that loads and manages task definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import json
import logging # Import logging

# Configure logging - REMOVED basicConfig call
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Task:
    """Represents a programming task to be solved by AltLAS."""
    name: str
    description: str
    success_criteria: dict
    max_tokens: Optional[int] = None  # New field with default None
    
    def __str__(self):
        return f"Task: {self.name} - {self.description}"

class TaskLoader:
    """Loads task definitions from JSON files within the tasks directory and subdirectories."""
    
    def __init__(self):
        # Base directory for tasks is the directory containing this script
        self.tasks_dir = Path(__file__).parent.absolute()
        logging.info(f"TaskLoader initialized. Searching for tasks in: {self.tasks_dir}")
        
    def find_task_file(self, task_name: str) -> Optional[Path]:
        """Search for task_name.json recursively within the tasks directory."""
        target_filename = f"{task_name}.json"
        for filepath in self.tasks_dir.rglob('*.json'): # rglob searches recursively
            if filepath.name == target_filename:
                logging.info(f"Found task file for '{task_name}' at: {filepath}")
                return filepath
        logging.warning(f"Task file '{target_filename}' not found in {self.tasks_dir} or subdirectories.")
        return None

    def load_task(self, task_name: str) -> Task:
        """Load a task definition from its JSON file by name."""
        task_file_path = self.find_task_file(task_name)
        
        if not task_file_path:
            raise ValueError(f"Task '{task_name}' definition file not found.")
            
        try:
            with open(task_file_path, 'r') as f:
                task_data = json.load(f)
            
            # Validate required fields (can be expanded)
            if not all(k in task_data for k in ["name", "description", "success_criteria"]):
                raise ValueError(f"Task file {task_file_path} is missing required fields (name, description, success_criteria).")
            
            # Ensure the name in the file matches the requested task_name
            if task_data.get("name") != task_name:
                logging.warning(f"Task name in file '{task_data.get('name')}' does not match requested name '{task_name}'. Using requested name.")

            # Create Task object with optional max_tokens
            task = Task(
                name=task_name,  # Use the requested name
                description=task_data["description"],
                success_criteria=task_data["success_criteria"],
                max_tokens=task_data.get("max_tokens")  # Will be None if not specified
            )
            
            if task.max_tokens is not None:
                logging.info(f"Task '{task_name}' loaded with max_tokens={task.max_tokens}")
            else:
                logging.info(f"Task '{task_name}' loaded without max_tokens specification")
            
            return task
            
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from task file {task_file_path}: {e}")
            raise ValueError(f"Invalid JSON in task file: {task_file_path}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred loading task '{task_name}' from {task_file_path}: {e}")
            raise