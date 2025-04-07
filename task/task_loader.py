"""
Task loader module that loads and manages task definitions.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import json

@dataclass
class Task:
    """Represents a programming task to be solved by AltLAS."""
    name: str
    description: str
    success_criteria: dict
    
    def __str__(self):
        return f"Task: {self.name} - {self.description}"

class TaskLoader:
    """Loads task definitions from the tasks directory."""
    
    def __init__(self):
        self.tasks_dir = Path(__file__).parent.absolute()
        
    def load_task(self, task_name):
        """Load a task by name."""
        # For now, return hardcoded tasks for simplicity
        tasks = {
            "hello_world": Task(
                name="hello_world",
                description="Output the string 'hello world'",
                success_criteria={
                    "exact_output": "hello world",
                    "case_sensitive": False
                }
            ),
            "add_numbers": Task(
                name="add_numbers",
                description="Create a function that adds two numbers",
                success_criteria={
                    "function_name": "add",
                    "test_cases": [
                        {"inputs": [1, 2], "expected": 3},
                        {"inputs": [5, 7], "expected": 12}
                    ]
                }
            )
        }
        
        if task_name not in tasks:
            raise ValueError(f"Task '{task_name}' not found")
        
        return tasks[task_name]