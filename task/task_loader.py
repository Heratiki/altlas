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
        # FUTURE INTENT: Task loading and management will become more dynamic:
        # 1.  Loading from Files: Define tasks in separate files (e.g., JSON, YAML, or even Python modules)
        #     within the `tasks/` directory instead of hardcoding them here.
        # 2.  Task Generation: Potentially generate tasks programmatically based on difficulty parameters
        #     or observed agent capabilities (curriculum learning).
        # 3.  Task Dependencies: Define dependencies between tasks (e.g., Task B requires concepts learned in Task A).
        # 4.  Complex Task Types: Support richer task definitions beyond simple input/output or function checks,
        #     including tasks involving state, interaction, resource constraints, or specific library usage.
        # 5.  Task Validation: Add validation to ensure task definitions are well-formed.
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