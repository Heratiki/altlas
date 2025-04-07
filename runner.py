#!/usr/bin/env python3
"""
AltLAS Runner - Main orchestrator loop for the AltLAS system.
"""

import os
import sys
import time
from pathlib import Path

# Ensure all module directories are in the path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import components
try:
    from agent_core.generator import CodeGenerator
    from guardian.safety import SafetyChecker
    from evaluator.executor import CodeExecutor
    from reinforcer.scorer import AttemptScorer
    from memory.logger import AttemptLogger
    from memory.fingerprints import AttemptFingerprinter
    from task.task_loader import TaskLoader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are created.")
    sys.exit(1)

def main():
    """Main runner function that orchestrates the AltLAS workflow."""
    print("🧠 Starting AltLAS - Learning Agent")
    
    # 1. Load task
    task_loader = TaskLoader()
    current_task = task_loader.load_task("hello_world")  # Start with a simple task
    print(f"📋 Loaded task: {current_task.name}")
    
    # 2. Initialize components
    generator = CodeGenerator()
    safety_checker = SafetyChecker()
    executor = CodeExecutor()
    scorer = AttemptScorer()
    logger = AttemptLogger()
    fingerprinter = AttemptFingerprinter()
    
    max_attempts = 100
    success = False
    attempt_count = 0
    
    # 3. Main learning loop
    while attempt_count < max_attempts and not success:
        attempt_count += 1
        print(f"\n🔄 Attempt {attempt_count}/{max_attempts}")
        
        # Generate code
        code_attempt = generator.generate(current_task, logger.get_history())
        print(f"📝 Generated code attempt ({len(code_attempt)} chars)")
        
        # Check safety and novelty
        if not safety_checker.is_safe(code_attempt):
            print("⚠️ Unsafe code detected, skipping execution")
            continue
            
        fingerprint = fingerprinter.get_fingerprint(code_attempt)
        if fingerprinter.is_duplicate(fingerprint):
            print("🔄 Duplicate attempt detected, skipping execution")
            continue
        
        # Execute code
        result = executor.execute(code_attempt)
        print(f"🏃 Execution complete: {result.status}")
        
        # Score result
        score = scorer.score(result, current_task)
        print(f"📊 Attempt score: {score:.2f}")
        
        # Log if valuable
        if score > 0.0:
            logger.log_attempt(code_attempt, result, score, fingerprint)
            
        # Check for success
        if score >= 0.99:
            success = True
            print(f"🎉 Success achieved on attempt {attempt_count}!")
    
    # 4. Summarize results
    if success:
        print("✅ Task completed successfully!")
    else:
        print("⏱️ Maximum attempts reached without success")
    
    print(f"📈 Best score achieved: {logger.get_best_score():.2f}")
    
if __name__ == "__main__":
    main()