"""
Code generator that creates new code attempts based on task and history.
"""

import random
import string

class CodeGenerator:
    """Generates code attempts for the AltLAS system."""
    
    def __init__(self):
        # Start with very simple generation strategies
        self.strategies = [
            self._generate_random_print,
            self._generate_simple_hello,
            self._generate_python_hello,
        ]
        
    def generate(self, task, history=None):
        """Generate a code attempt for the given task."""
        # Start with completely random generation
        # As the system evolves, this will become more sophisticated
        strategy = random.choice(self.strategies)
        return strategy(task)
    
    def _generate_random_print(self, task):
        """Generate a random print statement."""
        words = ["hello", "world", "python", "programming", "code", "learning"]
        selected_words = random.sample(words, k=random.randint(1, 3))
        return f'print("{" ".join(selected_words)}")'
    
    def _generate_simple_hello(self, task):
        """Generate variations of hello world."""
        templates = [
            'print("hello world")',
            'print("Hello World")',
            'print("HELLO WORLD")',
            'print("hello") \nprint("world")',
            'message = "hello world"\nprint(message)',
        ]
        return random.choice(templates)
    
    def _generate_python_hello(self, task):
        """Generate slightly more complex hello world programs."""
        templates = [
            'for word in ["hello", "world"]:\n    print(word, end=" ")',
            'words = ["hello", "world"]\nprint(" ".join(words))',
            'print("hel" + "lo" + " " + "wor" + "ld")',
        ]
        return random.choice(templates)