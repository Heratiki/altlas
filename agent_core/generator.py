"""
Code generator that creates new code attempts based on task and history.
"""

import random
import string
from collections import Counter
import re 
import json # Added for saving/loading weights
from pathlib import Path # Added for path handling

class CodeGenerator:
    """Generates code attempts for the AltLAS system, learning from history."""
    
    def __init__(self, weights_file="memory/generator_weights.json"): # Added weights_file parameter
        # Define basic Python building blocks
        self.keywords = ["print", "if", "else", "for", "while", "def", "return", "pass"]
        self.operators = ["=", "+", "-", "*", "/", "==", "!=", "<", ">"]
        self.literals = [f'"{word}"' for word in ["hello", "world", "a", "b", "x", "y", "result"]] + \
                   [str(i) for i in range(10)] + ["True", "False", "None"]
        self.common_vars = [f"var_{c}" for c in string.ascii_lowercase[:5]]
        
        self.all_elements = self.keywords + self.operators + self.literals + self.common_vars
        
        # Learning parameters (can be tuned)
        self.learning_rate = 0.05 
        self.history_window = 20 

        # --- Weight Loading ---
        self.weights_file = Path(weights_file)
        self.element_weights = self._load_weights()
        if not self.element_weights:
             # Initialize weights if file doesn't exist or is invalid
             print("Initializing new generator weights.")
             self.element_weights = {element: 1.0 for element in self.all_elements}
             # Normalize initial weights
             total_weight = sum(self.element_weights.values())
             if total_weight > 0:
                 for element in self.element_weights:
                     self.element_weights[element] /= total_weight
        else:
             print(f"Loaded generator weights from {self.weights_file}")
        # --- End Weight Loading ---

    def _load_weights(self):
        """Load element weights from the specified file."""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    weights = json.load(f)
                    # Basic validation: check if loaded keys match current elements
                    if set(weights.keys()) == set(self.all_elements):
                        return weights
                    else:
                        print("Warning: Loaded weights keys mismatch current elements. Re-initializing.")
                        return None
            except json.JSONDecodeError:
                print(f"Warning: Error decoding weights file {self.weights_file}. Re-initializing.")
                return None
            except Exception as e:
                print(f"Warning: Error loading weights file {self.weights_file}: {e}. Re-initializing.")
                return None
        return None

    def save_weights(self):
        """Save the current element weights to the specified file."""
        try:
            # Ensure the directory exists
            self.weights_file.parent.mkdir(parents=True, exist_ok=True) 
            with open(self.weights_file, 'w') as f:
                json.dump(self.element_weights, f, indent=2)
            print(f"Saved generator weights to {self.weights_file}")
        except Exception as e:
            print(f"Error saving weights to {self.weights_file}: {e}")

    def _update_weights(self, history):
        """Adjust element weights based on recent attempt scores."""
        if not history:
            return

        # Consider only the recent history
        recent_history = history[-self.history_window:]
        
        # Calculate adjustments based on scores
        adjustments = Counter()
        total_score_effect = 0

        for attempt in recent_history:
            score = attempt.get('score', 0.0)
            code = attempt.get('code', '')
            
            # Simple tokenization (split by space) - very basic!
            tokens_in_attempt = set(code.split(' ')) 
            
            # Find which known elements were present
            elements_present = {elem for elem in self.all_elements if elem in tokens_in_attempt}

            # Adjust weights based on score (reward good, penalize bad)
            # Give higher weight to higher scores
            reward = score * score # Square score to emphasize high scores more
            penalty = (1.0 - score) * 0.1 # Penalize failures less harshly initially

            for element in elements_present:
                 adjustments[element] += reward 
                 adjustments[element] -= penalty
                 total_score_effect += abs(reward - penalty) # Track magnitude of changes

        if total_score_effect == 0: return # Avoid division by zero if no effective history

        # Apply adjustments with learning rate
        for element, adj in adjustments.items():
             # Scale adjustment by learning rate and relative magnitude
             scaled_adj = self.learning_rate * (adj / total_score_effect) * len(self.all_elements)
             self.element_weights[element] += scaled_adj
             # Ensure weights don't go below a small positive value
             self.element_weights[element] = max(0.01, self.element_weights[element])

        # Normalize weights (optional, but keeps them manageable)
        total_weight = sum(self.element_weights.values())
        if total_weight > 0:
            for element in self.element_weights:
                self.element_weights[element] /= total_weight
        else: # Reset if all weights somehow became zero
             self.element_weights = {element: 1.0 for element in self.all_elements}

    def _apply_hint(self, hint):
        """Temporarily boost weights based on keywords found in the hint."""
        if not hint:
            return

        print(f"Applying hint: {hint}")
        # Very basic hint processing: look for known elements mentioned
        hint_keywords = re.findall(r'\b\w+\b', hint.lower()) # Extract words
        
        boost_factor = 1.5 # How much to boost hinted elements (tuneable)

        for element in self.element_weights:
            # Check if the element itself (or part of it, e.g., 'print' in 'printing') is in the hint
            if any(element.strip('"') in h_word for h_word in hint_keywords):
                 print(f"Boosting weight for '{element}' based on hint.")
                 self.element_weights[element] *= boost_factor

        # Re-normalize weights after boosting
        total_weight = sum(self.element_weights.values())
        if total_weight > 0:
            for element in self.element_weights:
                self.element_weights[element] /= total_weight

    def generate(self, task, history=None, hint=None): # Added hint parameter
        """Generate a code attempt, learning from history and potentially using a hint."""
        
        # 1. Learn from history first
        if history:
            self._update_weights(history)

        # 2. Apply hint if provided (modifies weights temporarily)
        if hint:
            self._apply_hint(hint) # Apply hint *after* normal weight updates
            
        # 3. Generate code using current (potentially hinted) weights
        return self._generate_basic_code_sequence(task)

    def _generate_basic_code_sequence(self, task):
        """Generate a short sequence of basic Python elements using learned weights."""
        
        # Get current elements and their weights
        elements = list(self.element_weights.keys())
        weights = list(self.element_weights.values())

        # Normalize weights to ensure they sum to 1 for random.choices
        total_weight = sum(weights)
        if total_weight <= 0: # Handle edge case where all weights are zero or negative
            weights = [1.0] * len(elements) # Fallback to uniform
            total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Generate a short sequence of random elements based on weights
        code_length = random.randint(1, 5) 
        
        # Add a small chance (epsilon) of picking a random element regardless of weight
        # This encourages exploration
        epsilon = 0.1 
        code_parts = []
        for _ in range(code_length):
            if random.random() < epsilon:
                code_parts.append(random.choice(elements))
            else:
                code_parts.append(random.choices(elements, weights=normalized_weights, k=1)[0])

        # Very basic formatting attempt (same as before, could be improved)
        if "print" in code_parts and any(lit in code_parts for lit in self.literals):
             printable = random.choice([lit for lit in code_parts if lit in self.literals])
             return f"print({printable})"
        elif "=" in code_parts and any(var in code_parts for var in self.common_vars) and any(lit in code_parts for lit in self.literals):
             var = random.choice([v for v in code_parts if v in self.common_vars])
             val = random.choice([lit for lit in code_parts if lit in self.literals])
             return f"{var} = {val}"
        else:
            return " ".join(code_parts)