#!/usr/bin/env python3
"""
Test script to verify language map loading and normalization for different languages.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task.task_loader import TaskLoader
from reinforcer.scorer import AttemptScorer

def test_language_maps():
    """Test language map loading for different languages."""
    print("Testing language map loading...\n")
    
    # Initialize the scorer
    scorer = AttemptScorer()
    
    # Test loading maps directly
    print("1. Testing direct language map loading:")
    python_map = scorer.load_language_map("python")
    js_map = scorer.load_language_map("javascript")
    default_map = scorer.load_language_map("unknown")
    
    print(f"Python map loaded: {len(python_map)} tokens")
    print(f"JavaScript map loaded: {len(js_map)} tokens")
    print(f"Default map loaded for unknown language: {len(default_map)} tokens")
    
    # Compare some key tokens
    print("\nComparing key tokens across language maps:")
    compare_tokens = ["OUTPUT_OP", "CONDITIONAL_IF", "BOOL_TRUE", "NULL_VALUE", "FUNC_DEF"]
    for token in compare_tokens:
        print(f"{token}: Python='{python_map.get(token, 'N/A')}', JavaScript='{js_map.get(token, 'N/A')}'")
    
    # Test with actual tasks
    print("\n2. Testing with actual tasks:")
    task_loader = TaskLoader()
    
    # Load Python and JavaScript tasks
    py_task = task_loader.load_task("benchmark_add_two_numbers")
    js_task = task_loader.load_task("benchmark_js_add_two_numbers")
    
    print(f"Python task language: {py_task.target_language}")
    print(f"JavaScript task language: {js_task.target_language}")
    
    # Test normalization
    print("\n3. Testing code normalization:")
    
    # Abstract code with tokens that should be replaced
    abstract_code = "OUTPUT_OP(NUMBER_LITERAL_PLACEHOLDER + NUMBER_LITERAL_PLACEHOLDER)"
    py_normalized = scorer.normalize_for_scoring(abstract_code, py_task)
    js_normalized = scorer.normalize_for_scoring(abstract_code, js_task)
    
    print(f"Abstract code: {abstract_code}")
    print(f"Python normalized: {py_normalized}")
    print(f"JavaScript normalized: {js_normalized}")
    
    # Test syntax validation
    print("\n4. Testing language-specific syntax validation:")
    
    valid_py_code = "print(5 + 3)"
    valid_js_code = "console.log(5 + 3);"
    
    py_score = scorer._evaluate_syntax(valid_py_code, py_task)
    js_score = scorer._evaluate_syntax(valid_js_code, js_task)
    cross_score_py_to_js = scorer._evaluate_syntax(valid_py_code, js_task)
    cross_score_js_to_py = scorer._evaluate_syntax(valid_js_code, py_task)
    
    print(f"Python code syntax score in Python context: {py_score}")
    print(f"JavaScript code syntax score in JavaScript context: {js_score}")
    print(f"Python code syntax score in JavaScript context: {cross_score_py_to_js}")
    print(f"JavaScript code syntax score in Python context: {cross_score_js_to_py}")

if __name__ == "__main__":
    test_language_maps()