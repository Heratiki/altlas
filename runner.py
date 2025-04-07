#!/usr/bin/env python3
"""
AltLAS Runner - Main orchestrator loop for the AltLAS system.
"""

import os
import sys
import time
from pathlib import Path
import random # Added for hint simulation
import configparser # Added

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

# --- Advisor Placeholder ---
def get_hint_from_advisor(task, history):
    """
    Placeholder function to simulate getting a hint from an external LLM advisor.
    Replace this with actual API calls to LM-Studio or similar.
    FUTURE INTENT: This function will eventually be replaced by a robust mechanism 
    for interacting with external advisors (LLMs, static analysis tools, etc.).
    Key aspects to develop:
    1.  Real API Integration: Connect to LM-Studio or other chosen advisor endpoints.
    2.  Contextual Prompts: Construct much richer prompts for the advisor, including:
        - Full task definition.
        - More extensive history (successful/failed attempts, errors encountered).
        - Current generator state (e.g., element weights, learned patterns).
        - Specific problem (e.g., error message, lack of score improvement).
    3.  Hint Parsing & Translation: Process the advisor's response (which might be natural language)
        into a format the CodeGenerator can understand and utilize (see `_apply_hint` intent).
    4.  Adaptive Hinting Strategy: AltLAS might learn *when* to ask for hints, *which* advisor
        to consult (if multiple are available), and how much weight to give advisor suggestions.
        It should avoid over-reliance and prioritize its own learned strategies when possible.
    """
    print("🤔 Agent seems stuck. Requesting a hint from advisor...")
    # Simulate getting a hint - replace with real logic
    # FUTURE INTENT: The simulation logic will be removed. Real API calls and error handling
    # for network issues or advisor unavailability will be implemented.
    possible_hints = [
        "Maybe try using a loop?",
        "Consider printing the variable.",
        "What if you assign the value first?",
        "Think about the 'print' keyword.",
        "Try combining 'hello' and 'world'.",
        None, # Simulate advisor having no hint
    ]
    # Basic context for the advisor (can be expanded)
    # FUTURE INTENT: Replace this minimal context with the richer prompt described above.
    context = f"Task: {task.description}\nRecent Scores: {[h.get('score', 0) for h in history[-5:]]}"
    print(f"Sent context to advisor:\n{context}")
    
    hint = random.choice(possible_hints)
    if hint:
        print(f"💡 Advisor Hint: {hint}")
    else:
        print("🤷 Advisor had no hint this time.")
    return hint
# --- End Advisor Placeholder ---


def main():
    """Main runner function that orchestrates the AltLAS workflow."""
    print("🧠 Starting AltLAS - Learning Agent")

    # --- Read Config ---
    config = configparser.ConfigParser()
    # Define config path relative to the runner script
    config_path = Path(__file__).parent / "config.ini" 
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    config.read(config_path)
    
    try:
        runner_config = config['Runner']
        max_attempts = runner_config.getint('MaxAttempts', 1000)
        stuck_check_window = runner_config.getint('StuckCheckWindow', 15)
        stuck_threshold = runner_config.getfloat('StuckThreshold', 0.01)

        scorer_config = config['Scorer']
        success_threshold = scorer_config.getfloat('SuccessThreshold', 0.99)
    except KeyError as e:
        print(f"Error: Missing section in {config_path}: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid value type in {config_path}: {e}")
        sys.exit(1)
    # --- End Read Config ---
    
    # 1. Load task
    task_loader = TaskLoader()
    current_task = task_loader.load_task("hello_world")  
    print(f"📋 Loaded task: {current_task.name}")
    
    # 2. Initialize components, passing config path as string
    generator = CodeGenerator(config_path=str(config_path))
    safety_checker = SafetyChecker() # Assuming no config needed yet
    executor = CodeExecutor(config_path=str(config_path))
    scorer = AttemptScorer(config_path=str(config_path))
    logger = AttemptLogger() # Assuming no config needed yet
    fingerprinter = AttemptFingerprinter() # Uses its own file path logic

    success = False
    attempt_count = 0
    
    # Stuck detection parameters
    last_best_score_at_check = 0.0
    attempts_since_last_check = 0
    current_hint = None

    try: 
        # 3. Main learning loop
        while attempt_count < max_attempts and not success:
            attempt_count += 1
            attempts_since_last_check += 1
            print(f"\n🔄 Attempt {attempt_count}/{max_attempts}")

            # --- Stuck Detection & Hinting ---
            if attempts_since_last_check >= stuck_check_window:
                current_best_score = logger.get_best_score()
                if current_best_score - last_best_score_at_check < stuck_threshold:
                    current_hint = get_hint_from_advisor(current_task, logger.get_history())
                else:
                     current_hint = None 
                last_best_score_at_check = current_best_score
                attempts_since_last_check = 0
            # --- End Stuck Detection ---

            # Generate code, potentially using a hint
            code_attempt = generator.generate(current_task, logger.get_history(), hint=current_hint)
            print(f"📝 Generated code attempt ({len(code_attempt)} chars)")
            
            # Check safety and novelty
            if not safety_checker.is_safe(code_attempt):
                print("⚠️ Unsafe code detected, skipping execution")
                logger.log_attempt(code_attempt, None, 0.0, "unsafe") # Log unsafe attempts differently
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
            
            # Log attempt (even if score is low, for learning)
            # Pass the actual result object now
            logger.log_attempt(code_attempt, result, score, fingerprint) 
                
            # Check for success using configured threshold
            if score >= success_threshold: # Use configured threshold
                success = True
                print(f"🎉 Success achieved on attempt {attempt_count}!")
                
            # Clear hint after one use to avoid over-reliance
            current_hint = None 
    
    finally: 
        # --- Save Learned Weights ---
        # Ensure generator exists before saving (in case of early error)
        if 'generator' in locals(): 
            generator.save_weights()
        # --- End Save Weights ---

        # 4. Summarize results
        if success:
            print("✅ Task completed successfully!")
        elif attempt_count >= max_attempts:
            print("⏱️ Maximum attempts reached.")
        else:
             print("⏹️ Run ended.") # Handle cases like Ctrl+C or early errors
        
        # Ensure logger exists before getting best score
        if 'logger' in locals():
            print(f"📈 Best score achieved: {logger.get_best_score():.2f}")

if __name__ == "__main__":
    main()