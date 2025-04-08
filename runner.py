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
import argparse # Import argparse
import logging # Import logging
from logging.handlers import RotatingFileHandler # Use RotatingFileHandler
from rich.logging import RichHandler # Import RichHandler
from rich.console import Console # Ensure Console is imported
import psutil

# Check if the script is already running
current_script_name = "runner.py"
current_pid = os.getpid()
for proc in psutil.process_iter(['pid', 'name']):
    if proc.info['name'] == current_script_name and proc.info['pid'] != current_pid:
        print(f"Another instance of {current_script_name} is already running. Exiting.")
        sys.exit(1)

# Rich UI components
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

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
    from utils import get_pytorch_device # <-- Add import
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are created.")
    sys.exit(1)

# --- Centralized Logging Setup ---
LOG_FILENAME = "altlas_run.log"
LOG_LEVEL_FILE = logging.DEBUG # Log DEBUG and higher to file
LOG_LEVEL_CONSOLE = logging.CRITICAL # Only log CRITICAL to console, preventing UI flickering

# Create separate console objects for logging and for the Live display
log_console = Console(stderr=True)  # Use stderr for logging to avoid conflicts
ui_console = Console()  # Use stdout for the Live UI display

# Configure root logger
log = logging.getLogger() # Get root logger
log.setLevel(LOG_LEVEL_FILE) # Set root logger level to the lowest level needed (DEBUG)

# Formatter for file logs
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler (Rotating)
# Rotates logs, keeping 5 backups of 5MB each
file_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8') # Added encoding
file_handler.setFormatter(file_formatter)
file_handler.setLevel(LOG_LEVEL_FILE) # File handler captures DEBUG+ 

# Console Handler (Rich) - Only for CRITICAL messages
console_handler = RichHandler(console=log_console, rich_tracebacks=True, show_path=False, level=LOG_LEVEL_CONSOLE) # Console handler captures CRITICAL only

# Remove existing handlers if any (important for re-running in interactive sessions)
for handler in log.handlers[:]:
    log.removeHandler(handler)

# Add handlers to the root logger
log.addHandler(file_handler)
log.addHandler(console_handler)

# --- End Logging Setup ---

# --- Advisor Implementation ---
import json
import requests
from typing import List, Dict, Any, Optional

def get_hint_from_advisor(task, history, max_history_entries=5):
    """
    Get a hint from an external LLM advisor based on the current task and execution history.
    
    Args:
        task: The current task object with name, description, etc.
        history: List of previous attempt records
        max_history_entries: Maximum number of history entries to include
    
    Returns:
        str or None: A hint to guide the agent, or None if no hint could be generated
    """
    try:
        # Extract the most relevant information from history
        recent_history = history[-max_history_entries:] if history else []
        
        # Get the most recent errors and scores
        errors = [
            h.get('result', {}).get('error', '') 
            for h in recent_history 
            if h.get('result', {}).get('status') == 'error' and h.get('result', {}).get('error')
        ]
        recent_scores = [h.get('score', 0) for h in recent_history]
        
        # Get code samples from recent attempts
        code_samples = [h.get('code', '') for h in recent_history if h.get('code')]
        
        # Construct a prompt for the LLM
        prompt = f"""You are an AI programming advisor helping a learning agent solve a coding task.

TASK DESCRIPTION:
{task.name}
{task.description}

RECENT SCORES (higher is better):
{recent_scores}

RECENT ERRORS:
{errors[:3]}  # Limiting to most recent 3 errors

MOST RECENT CODE ATTEMPT:
```python
{code_samples[-1] if code_samples else 'No attempts yet'}
```

Based on this information, provide ONE specific, concise hint that would help the agent improve. 
Focus on ONE problem at a time. Limit your hint to one or two sentences.
"""
        
        # Call the OpenAI API or other LLM service
        hint = call_llm_api(prompt)
        
        if hint:
            log.info(f"💡 Advisor Hint: {hint}")
            return hint
        else:
            log.info("🤷 Advisor could not generate a hint")
            return None
            
    except Exception as e:
        log.error(f"⚠️ Error getting hint from advisor: {str(e)}")
        # Fallback to a generic hint if the API call fails
        return "Try a different approach."

def call_llm_api(prompt: str) -> Optional[str]:
    """
    Call an external LLM API to get a response.
    
    This function can be configured to use different LLM APIs:
    - OpenAI API
    - Local model via LM Studio
    - Hugging Face API
    - etc.
    
    Returns:
        str or None: The LLM's response or None if the call failed
    """
    # Check for API key in environment or config
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # If no API key, try using a local LLM via LM Studio
    if not api_key:
        return call_local_llm(prompt)
    
    try:
        # OpenAI API call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful programming advisor giving concise hints."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=10
        )
        
        if response.status_code == 200:
            response_data = response.json()
            hint = response_data['choices'][0]['message']['content'].strip()
            return hint
        else:
            log.error(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        log.error(f"Error calling LLM API: {str(e)}")
        return None

def call_local_llm(prompt: str) -> Optional[str]:
    """
    Call a local LLM using LM Studio running on the host machine.
    Uses localhost to connect with --network=host container mode.
    """
    try:
        # With --network=host, localhost in the container is the same as on the host
        base_url = "http://localhost:1234/v1"
        log.info("🌐 Using localhost with network=host to connect to LM Studio")
        
        # First, get the list of available models
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            models_response = requests.get(
                f"{base_url}/models",
                headers=headers,
                timeout=5
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [model["id"] for model in models_data.get("data", [])]
                log.info(f"🤖 Available LM Studio models: {available_models}")
                
                # Choose an appropriate model (prefer code-oriented models if available)
                chosen_model = None
                preferred_models = [
                    "wizardcoder", "codellama", "code-llama", "stable-code", 
                    "starcoder", "olympiccoder", "qwen2.5-coder"
                ]
                
                # Look for preferred models
                for preferred in preferred_models:
                    for model in available_models:
                        if preferred.lower() in model.lower():
                            chosen_model = model
                            break
                    if chosen_model:
                        break
                
                # If no preferred model is found, use the first available model
                if not chosen_model and available_models:
                    chosen_model = available_models[0]
                
                if not chosen_model:
                    log.warning("⚠️ No models available in LM Studio")
                    return generate_static_hint(prompt)
                    
                log.info(f"🧠 Using LM Studio model: {chosen_model}")
            else:
                log.warning(f"⚠️ Failed to retrieve models from LM Studio: {models_response.status_code} - {models_response.text}")
                return generate_static_hint(prompt)
        except Exception as model_e:
            log.warning(f"⚠️ Error querying available models: {str(model_e)}")
            return generate_static_hint(prompt)
            
        # Now make the actual API call for completion
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful programming advisor giving concise hints."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        # Add the model if we found one
        if chosen_model:
            data["model"] = chosen_model
        
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=10
        )
        
        if response.status_code == 200:
            response_data = response.json()
            hint = response_data['choices'][0]['message']['content'].strip()
            return hint
        else:
            log.warning(f"⚠️ Local LLM API error: {response.status_code} - {response.text}")
            return generate_static_hint(prompt)
            
    except Exception as e:
        log.warning(f"⚠️ Error calling local LLM: {str(e)}")
        return generate_static_hint(prompt)

def generate_static_hint(prompt: str) -> str:
    """Generate a static hint based on the task and error patterns."""
    # Check for task-specific patterns first
    if "benchmark_add_two_numbers" in prompt:
        if "print(8)" in prompt.lower():
            return "Try using addition (5 + 3) instead of printing the number directly."
        if "print('8')" in prompt.lower() or 'print("8")' in prompt.lower():
            return "Use addition of numbers, not just printing the string '8'."
        if "+" not in prompt:
            return "Remember to use the + operator to add the numbers."
        return "Try adding the numbers 5 and 3 using the + operator and print the result."

    # Extract code from the prompt if available
    code_sample = ""
    if "```python" in prompt and "```" in prompt.split("```python", 1)[1]:
        code_sample = prompt.split("```python", 1)[1].split("```", 1)[0].strip()
    
    # Extract errors from the prompt if available
    errors = []
    if "RECENT ERRORS:" in prompt:
        errors_section = prompt.split("RECENT ERRORS:", 1)[1].split("\n\n", 1)[0].strip()
        errors = [err.strip() for err in errors_section.split("\n") if err.strip()]
    
    # Check for specific error patterns and provide relevant hints
    if errors:
        for error in errors:
            if "NameError" in error or "not defined" in error:
                return "Check that all variables are properly defined before use."
            elif "TypeError" in error:
                return "Verify that you're using compatible data types in your operations."
            elif "IndexError" in error or "out of range" in error:
                return "Make sure your list indices are within the valid range."
            elif "KeyError" in error:
                return "Ensure the dictionary key you're trying to access exists."
            elif "SyntaxError" in error:
                return "Fix the syntax error in your code - check for missing parentheses, quotes, or colons."
            elif "IndentationError" in error:
                return "Correct the indentation in your code."
            elif "AttributeError" in error:
                return "Verify that the object has the attribute or method you're trying to access."
    
    # Check for task-specific patterns
    if "hello world" in prompt.lower():
        return "Remember to use the print() function to output text."
    
    # Look for clues in the code sample
    if code_sample:
        if "def " in code_sample and "return" not in code_sample:
            return "Make sure your function returns a value."
        if "for " in code_sample and "range" not in code_sample:
            return "Consider using range() for numeric loops."
        if "if " in code_sample and ":" not in code_sample:
            return "Check that your if statement has a colon at the end."
    
    # General programming hints as ultimate fallback
    general_hints = [
        "Try breaking down the problem into smaller steps.",
        "Use print statements to debug your code.",
        "Consider edge cases in your solution.",
        "Check that your function returns the correct data type.",
        "Make sure your loops have the correct termination condition.",
        "Consider using a different algorithm approach.",
        "Try writing pseudocode before implementing the actual code.",
        "Review the core requirements of the task again.",
        "Simplify your solution - complex code often leads to more bugs.",
        "Double-check variable scopes and ensure they're accessible where needed."
    ]
    
    return random.choice(general_hints)
# --- End Advisor Implementation ---


def reset_training_state():
    """
    Fully reset all training state files to ensure a fresh start.
    This function deletes all files that store state between runs.
    """
    log.info("🔄 Resetting all training state files...")
    
    # Get the project root directory (where runner.py is)
    project_root = Path(__file__).parent.absolute()
    
    # Define all paths to clear (using absolute paths)
    paths_to_clear = [
        project_root / "memory" / "model_state.pth",
        project_root / "memory" / "optimizer_state.pth",
        project_root / "memory" / "logs" / "best_attempt.json",
        project_root / "memory" / "fingerprints.json"
    ]
    
    # Add any additional state files found in memory directory
    memory_dir = project_root / "memory"
    for file in memory_dir.glob("*.pth"):
        if file not in paths_to_clear:
            paths_to_clear.append(file)
    
    # Count of files deleted
    deleted_count = 0
    
    # Delete each file if it exists
    for path in paths_to_clear:
        if path.exists():
            try:
                path.unlink()
                log.info(f"✅ Deleted {path}")
                deleted_count += 1
            except Exception as e:
                log.error(f"⚠️ Failed to delete {path}: {e}")
    
    # Check logs directory for any other files
    logs_dir = project_root / "memory" / "logs"
    if logs_dir.exists():
        for file in logs_dir.glob("*.json"):
            try:
                file.unlink()
                log.info(f"✅ Deleted additional log file: {file}")
                deleted_count += 1
            except Exception as e:
                log.error(f"⚠️ Failed to delete log file {file}: {e}")
    
    if deleted_count > 0:
        log.info(f"✅ Training state reset complete. {deleted_count} files were deleted.")
    else:
        log.warning("⚠️ No training state files were found to delete.")
    
    return deleted_count > 0


def main():
    """Main runner function that orchestrates the AltLAS workflow."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the AltLAS Learning Agent.")
    parser.add_argument("--task", type=str, default="hello_world", 
                        help="Name of the task JSON file to load (without .json extension). Defaults to hello_world.")
    parser.add_argument("--reset", action="store_true", 
                        help="Reset all training state files and start from attempt 1.")
    args = parser.parse_args()
    task_to_load = args.task
    reset_training = args.reset
    # --- End Argument Parsing ---

    # Use the standard logger. RichHandler will format INFO+ messages for the console.
    log.info(f"🧠 Starting AltLAS - Task: {task_to_load}")
    log.debug("Debug logging is active and directed to file.")

    # Reset training state if requested
    if reset_training:
        log.info("🔄 Reset flag detected - clearing all training state files...")
        reset_successful = reset_training_state()
        if reset_successful:
            log.info("✅ Training state reset complete - starting from attempt 1")
        else:
            log.warning("⚠️ No training state files were found to delete")
            
    # Additional attempt count reset
    # Create a fresh logger to ensure attempt count starts at 1
    logger = AttemptLogger()
    fingerprinter = AttemptFingerprinter()
    
    # Ensure attempt count starts at 0 (will increment to 1 on first iteration)
    attempt_count = 0

    # --- Read Config ---
    config = configparser.ConfigParser()
    # Define config path relative to the runner script
    config_path = Path(__file__).parent / "config.ini" 
    if not config_path.exists():
        log_console.print(f"[bold red]Error: Configuration file not found at {config_path}[/bold red]")
        sys.exit(1)
    config.read(config_path)
    
    try:
        runner_config = config['Runner']
        max_attempts = runner_config.getint('MaxAttempts', 1000)
        stuck_check_window = runner_config.getint('StuckCheckWindow', 15)
        stuck_threshold = runner_config.getfloat('StuckThreshold', 0.01)
        hint_probability_on_stuck = runner_config.getfloat('HintProbabilityOnStuck', 1.0)
        max_consecutive_stuck_checks = runner_config.getint('MaxConsecutiveStuckChecks', 3) 
        log_frequency = runner_config.getint('LogFrequency', 500) 
        top_tokens_to_log = runner_config.getint('TopTokensToLog', 10) 

        scorer_config = config['Scorer']
        success_threshold = scorer_config.getfloat('SuccessThreshold', 0.99)
    except KeyError as e:
        log_console.print(f"[bold red]Error: Missing section or key in {config_path}: {e}[/bold red]")
        sys.exit(1)
    except ValueError as e:
        log_console.print(f"[bold red]Error: Invalid value type in {config_path}: {e}[/bold red]")
        sys.exit(1)
    # --- End Read Config ---
    
    # --- Determine PyTorch Device ---
    device = get_pytorch_device() # <-- Determine device
    # --- End Determine Device ---
    
    # 1. Load task using the name from command-line args or default
    task_loader = TaskLoader()
    try:
        current_task = task_loader.load_task(task_to_load)
        # Log task loading info
        log.info(f"📋 Loaded task: {current_task.name} - {current_task.description}")
    except ValueError as e:
        # Log error and exit
        log.error(f"Error loading task '{task_to_load}': {e}", exc_info=True)
        # log_console.print(f"[bold red]Error loading task '{task_to_load}': {e}[/bold red]") # Keep direct print for fatal startup error
        sys.exit(1)
    
    # 2. Initialize components, passing config path as string
    generator = CodeGenerator(config_path=str(config_path), device=device) # <-- Pass device
    safety_checker = SafetyChecker() # Assuming no config needed yet
    executor = CodeExecutor(config_path=str(config_path))
    scorer = AttemptScorer(config_path=str(config_path))
    logger = AttemptLogger() # Assuming no config needed yet
    fingerprinter = AttemptFingerprinter() # Uses its own file path logic
    # Reset attempt count and clear previous log file
    log_file_path = Path("memory/logs/best_attempt.json")
    if log_file_path.exists():
        log_file_path.unlink()
    attempt_count = 0

    success = False
    attempt_count = 0
    
    # Stuck detection parameters
    last_best_score_at_check = 0.0
    attempts_since_last_check = 0
    consecutive_stuck_count = 0 # Initialize consecutive stuck counter
    current_hint = None
    
    # Keep track of status messages for display
    status_messages = []
    duplicate_count = 0
    unsafe_count = 0
    error_count = 0
    success_count = 0
    hints_requested = 0 # Add counter for hints requested
    hints_provided = 0  # Add counter for hints provided
    
    # Add timing tracking
    start_time = time.time()
    last_progress_update = start_time
    attempt_times = []  # Keep track of recent attempt times for rate calculation
    RATE_WINDOW = 100  # Number of attempts to average for rate calculation
    
    # Create the Rich layout
    layout = Layout()
    layout.split(
        Layout(name="header", size=5),  # Increased size for progress and timing info
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="status", ratio=2)
    )
    
    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        TaskProgressColumn(),
        TextColumn("[bold]{task.fields[percentage]:.2f}%")
    )
    task_id = progress.add_task("[blue]Running attempts...", total=max_attempts, percentage=0)
    
    user_interrupted = False # Flag to track user interrupt
    run_exception = None     # Store any unexpected exception
    
    try:
        # Use the UI console object for Live display with reduced refresh rate
        with Live(layout, refresh_per_second=4, console=ui_console, transient=True) as live:
            # 3. Main learning loop
            while attempt_count < max_attempts and not success:
                try:
                    # --- Inner Try Block for Attempt Logic --- 
                    attempt_count += 1
                    attempts_since_last_check += 1
                    
                    # Update timing statistics
                    current_time = time.time()
                    attempt_times.append(current_time)
                    if len(attempt_times) > RATE_WINDOW:
                        attempt_times.pop(0)
                    
                    # Calculate timing metrics
                    elapsed_time = current_time - start_time
                    if len(attempt_times) >= 2:
                        current_rate = len(attempt_times) / (attempt_times[-1] - attempt_times[0])
                    else:
                        current_rate = 0
                    
                    remaining_attempts = max_attempts - attempt_count
                    if current_rate > 0:
                        estimated_time_remaining = remaining_attempts / current_rate
                    else:
                        estimated_time_remaining = float('inf')
                    
                    # Create timing panel content
                    def format_time(seconds):
                        if seconds == float('inf'):
                            return "∞"
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        if hours > 0:
                            return f"{hours}h {minutes}m {secs}s"
                        elif minutes > 0:
                            return f"{minutes}m {secs}s"
                        else:
                            return f"{secs}s"
                    
                    timing_table = Table(show_header=False, box=None)
                    timing_table.add_row("Elapsed Time", format_time(elapsed_time))
                    timing_table.add_row("Est. Remaining", format_time(estimated_time_remaining))
                    timing_table.add_row("Current Rate", f"{current_rate:.1f} attempts/sec")
                    timing_table.add_row("Avg Time/Attempt", f"{(elapsed_time/attempt_count):.2f}s")
                    
                    # Update progress and timing panel
                    progress.update(task_id, completed=attempt_count, percentage=100*attempt_count/max_attempts)
                    
                    # Create a new layout just for the header content
                    header_content = Layout()
                    header_content.split(
                        Layout(progress, size=3),
                        Layout(timing_table)
                    )
                    
                    header_panel = Panel(
                        header_content,
                        title="AltLAS Progress",
                        border_style="blue"
                    )
                    layout["header"].update(header_panel)
                    
                    # ... (Periodic Logging) ...
                    if log_frequency > 0 and attempt_count % log_frequency == 0:
                        if hasattr(generator, 'token_frequency') and generator.token_frequency:
                            sorted_freq = sorted(generator.token_frequency.items(), key=lambda item: item[1], reverse=True)
                            top_n = sorted_freq[:top_tokens_to_log]
                            freq_str = ", ".join([f"'{token}': {count}" for token, count in top_n])
                            # Log this as INFO to appear on console via RichHandler
                            log.info(f"[Attempt {attempt_count}] Top {top_tokens_to_log} generated tokens: {freq_str}") 
                        # Log detailed info to file only
                        log.debug(f"Attempt {attempt_count} full token frequencies: {generator.token_frequency}")

                    # ... (Stuck Detection & Hinting) ...
                    if attempts_since_last_check >= stuck_check_window:
                        current_best_score = logger.get_best_score()
                        if current_best_score - last_best_score_at_check < stuck_threshold:
                            consecutive_stuck_count += 1
                            # Add to status messages instead of logging to console
                            status_messages.insert(0, f"📉 [{time.strftime('%H:%M:%S')}] Stuck detected (Check {consecutive_stuck_count}/{max_consecutive_stuck_checks})")
                            # Still log to file
                            log.info(f"📉 Stuck detected (Check {consecutive_stuck_count}/{max_consecutive_stuck_checks}). Score hasn't improved enough.")
                            
                            if consecutive_stuck_count >= max_consecutive_stuck_checks:
                                if random.random() < hint_probability_on_stuck:
                                    current_hint = get_hint_from_advisor(current_task, logger.get_history())
                                    if current_hint:
                                        # Add to status messages instead of logging to console
                                        status_messages.insert(0, f"🤔 [{time.strftime('%H:%M:%S')}] Hint requested: {current_hint[:50]}...")
                                        # Still log to file
                                        log.info(f"🤔 Hint requested (Prob: {hint_probability_on_stuck:.2f}): {current_hint}")
                                        hints_requested += 1
                                        hints_provided += 1 # Assuming hint was provided if not None
                                    else:
                                        # Add to status messages instead of logging to console
                                        status_messages.insert(0, f"🤷 [{time.strftime('%H:%M:%S')}] Advisor couldn't generate hint")
                                        # Still log to file
                                        log.warning(f"🤷 Advisor couldn't generate hint (Prob: {hint_probability_on_stuck:.2f})")
                                        hints_requested += 1
                                    consecutive_stuck_count = 0 
                                else:
                                    # Only log to file
                                    log.info(f"🚫 Hint skipped due to probability ({hint_probability_on_stuck:.2f}) despite meeting consecutive stuck threshold.")
                                    current_hint = None
                            else:
                                current_hint = None
                        else:
                            if consecutive_stuck_count > 0:
                                 # Only log to file
                                 log.info(f"📈 Progress detected. Resetting consecutive stuck counter.")
                            consecutive_stuck_count = 0 
                            current_hint = None 
                        last_best_score_at_check = current_best_score
                        attempts_since_last_check = 0

                    # ... (Generate Code) ...
                    # Determine which generation method to use based on stuck detection
                    if consecutive_stuck_count >= 2:
                        # Use beam search when struggling to make progress
                        code_attempt, generated_ids = generator.generate_with_beam_search(current_task, logger.get_history(), hint=current_hint)
                        status_messages.insert(0, f"🔄 [{time.strftime('%H:%M:%S')}] Using beam search generation (stuck count: {consecutive_stuck_count})")
                        log.info(f"🔄 Using beam search generation due to stuck detection (count: {consecutive_stuck_count})")
                    else:
                        # Use standard generation with dynamic temperature
                        # Get the current best score to determine temperature
                        current_best_score = logger.get_best_score()
                        
                        # Adjust temperature based on best score so far
                        if current_best_score > 0.3:
                            temperature = 0.6  # More focused sampling for higher scores
                        else:
                            temperature = 0.8  # More exploration for lower scores
                            
                        code_attempt, generated_ids = generator.generate(current_task, logger.get_history(), hint=current_hint, temperature=temperature)
                    
                    if code_attempt is None or generated_ids is None:
                         status_messages.insert(0, f"⚠️ [{time.strftime('%H:%M:%S')}] Generator returned None unexpectedly. Skipping attempt.")
                         continue

                    # ... (Safety/Novelty Check) ...
                    if not safety_checker.is_safe(code_attempt):
                        # Add to status messages instead of logging to console
                        status_messages.insert(0, f"⚠️ [{time.strftime('%H:%M:%S')}] Unsafe code attempt detected. Skipping.")
                        # Log to file
                        log.warning(f"⚠️ Unsafe code attempt detected at attempt {attempt_count}. Skipping.")
                        unsafe_count += 1
                        continue
                    
                    fingerprint = fingerprinter.get_fingerprint(code_attempt)
                    if fingerprinter.is_duplicate(fingerprint):
                        # Add to status messages instead of logging to console  
                        status_messages.insert(0, f"⚠️ [{time.strftime('%H:%M:%S')}] Duplicate code attempt detected. Skipping.")
                        # Log to file
                        log.warning(f"⚠️ Duplicate code attempt detected at attempt {attempt_count}. Skipping.")
                        duplicate_count += 1
                        continue

                    # ... (Execute Code) ...
                    result = executor.execute(code_attempt)
                    
                    # ... (Score Result) ...
                    # Pass code_attempt to scorer for syntax check
                    score = scorer.score(code_attempt, result, current_task)
                    
                    # ... (Update status message) ...
                    status_messages.insert(0, f"📝 [{time.strftime('%H:%M:%S')}] Attempt {attempt_count} scored {score:.2f}")
                    if score >= success_threshold:
                        success_count += 1
                    else:
                        error_count += 1
                    
                    # ... (Log Attempt) ...
                    logger.log_attempt(attempt_count, code_attempt, result, score, fingerprint)
                    
                    # ... (Perform Learning Step) ...
                    # Calculate dynamic entropy coefficient
                    if attempt_count > 0: # Avoid division by zero
                        success_rate = success_count / attempt_count
                        # Anneal entropy: start high, decrease as success rate increases
                        annealing_factor = 1.0 - success_rate
                        current_entropy_coef = (
                            generator.min_entropy_coefficient + 
                            (generator.max_entropy_coefficient - generator.min_entropy_coefficient) * annealing_factor
                        )
                        # Ensure it doesn't go below the minimum
                        current_entropy_coef = max(generator.min_entropy_coefficient, current_entropy_coef)
                    else:
                        current_entropy_coef = generator.max_entropy_coefficient # Start with max entropy
                    
                    # Create tool feedback from execution result
                    tool_feedback = scorer.get_tool_feedback(code_attempt, result)
                    
                    # Pass the dynamic coefficient and tool feedback to the learn method
                    generator.learn(score, generated_ids, current_entropy_coef, tool_feedback=tool_feedback)
                    
                    # --- Update Rich UI --- 
                    best_score_val, best_attempt_num = logger.get_best_score_info()
                    stats_table = Table(show_header=False, box=None)
                    stats_table.add_row("Total Attempts", str(attempt_count))
                    stats_table.add_row("Success Attempts", str(success_count))
                    stats_table.add_row("Error Attempts", str(error_count))
                    stats_table.add_row("Duplicate Attempts", str(duplicate_count))
                    stats_table.add_row("Unsafe Attempts", str(unsafe_count))
                    stats_table.add_row("Hints Requested", str(hints_requested))
                    stats_table.add_row("Hints Provided", str(hints_provided))
                    stats_table.add_row("Current Entropy Coef", f"{current_entropy_coef:.4f}")
                    stats_table.add_row("Highest Score", f"{best_score_val:.2f}")
                    stats_table.add_row("Best Attempt", str(best_attempt_num))
                    # Add best attempt's code instead of result
                    best_attempt_info = logger.get_attempt(best_attempt_num)
                    if best_attempt_info:
                        stats_table.add_row("Best Code", str(best_attempt_info.get('code', 'N/A')))
                    layout["stats"].update(Panel(stats_table, title="Statistics", border_style="green"))
                    status_panel = Text("\n".join(status_messages[:15]))
                    layout["status"].update(Panel(status_panel, title="Status Messages", border_style="yellow"))
                    if current_hint:
                        hint_text = Text(f"🤔 Current Hint: {current_hint}", style="italic yellow")
                        layout["footer"].update(Panel(hint_text, title="Advisor Hint", border_style="yellow"))
                    else:
                        layout["footer"].update(Panel("No active hints", title="Advisor Hint", border_style="dim"))
                    
                    # ... (Check for Success) ...
                    if score >= success_threshold:
                        success = True
                        # ... (update UI for success) ...
                        live.update(layout)
                        time.sleep(1)
                        break # Exit the while loop on success
                    # --- End Inner Try Block ---
                
                except Exception as inner_e:
                    # Log errors using the configured logger
                    log.error(f"❌ Error during attempt {attempt_count}: {type(inner_e).__name__} - {inner_e}", exc_info=True)
                    status_messages.insert(0, f"❌ [{time.strftime('%H:%M:%S')}] Error in attempt {attempt_count}. See logs. Skipping attempt.")
                    continue 

            # --- End Main Loop ---
            
    except KeyboardInterrupt:
        log.warning("🏃 User interrupted the run (Ctrl+C).")
        user_interrupted = True
    except Exception as e:
        # Catch unexpected errors outside the inner attempt loop
        log.error(f"💥 Unhandled exception during run: {type(e).__name__} - {e}", exc_info=True)
        run_exception = e # Store exception to prevent saving state
    finally: 
        log.info("Finishing run...")
        # --- Save Learned Weights (Only on Clean Exit) ---
        # Save if run completed normally (success or max attempts) OR if user interrupted cleanly
        # DO NOT save if an unexpected exception occurred
        if run_exception is None and (
               success or attempt_count >= max_attempts or user_interrupted
           ):
            if 'generator' in locals(): 
                log.info("Attempting to save model and optimizer state...")
                generator.save_weights()
            else:
                log.warning("Generator not initialized, cannot save weights.")
        elif run_exception is not None:
             log.warning(f"Run terminated due to an error ({type(run_exception).__name__}). Model state will NOT be saved.")
        else: 
             # Should not happen if logic is correct, but log just in case
             log.warning("Unknown run termination state. Model state will NOT be saved.")
        # --- End Save Weights ---

        # --- Log Final Token Frequencies --- 
        if hasattr(generator, 'token_frequency') and generator.token_frequency:
            ui_console.print("[bold blue]Final Generated Token Frequencies:[/bold blue]")
            # Sort for consistent output
            sorted_final_freq = sorted(generator.token_frequency.items(), key=lambda item: item[1], reverse=True)
            # Use Rich Table for better formatting
            freq_table = Table(title="Token Frequencies", show_header=True, header_style="bold magenta")
            freq_table.add_column("Token", style="dim")
            freq_table.add_column("Count", justify="right")
            for token, count in sorted_final_freq:
                 # Escape special characters like newlines for display
                 display_token = repr(token).strip("'")
                 freq_table.add_row(display_token, str(count))
            ui_console.print(freq_table)
        else:
            ui_console.print("[yellow]No token frequency data collected.[/yellow]")
        # --- End Log Final Token Frequencies ---

        # 4. Summarize results
        summary = []
        if success:
            summary.append("[bold green]✅ Task completed successfully![/bold green]")
        elif user_interrupted:
            summary.append("[bold yellow]🏃 Run interrupted by user (Ctrl+C).[/bold yellow]")
        elif attempt_count >= max_attempts:
            summary.append("[bold yellow]⏱️ Maximum attempts reached.[/bold yellow]")
        elif run_exception is not None:
             summary.append(f"[bold red]💥 Run terminated due to error: {type(run_exception).__name__}[/bold red]")
        else:
            summary.append("[bold red]⏹️ Run ended unexpectedly.[/bold red]")
        
        # Ensure logger exists before getting best score
        if 'logger' in locals():
            best_score_val, best_attempt_num = logger.get_best_score_info()
            summary.append(f"[bold blue]📈 Best score achieved: {best_score_val:.2f}[/bold blue] (Attempt {best_attempt_num})")
        
        # Add final run statistics
        summary.append("\n[bold]Run Statistics:[/bold]")
        summary.append(f"  - Total Attempts: {attempt_count}")
        summary.append(f"  - Success Attempts: {success_count}")
        summary.append(f"  - Error Attempts: {error_count}")
        summary.append(f"  - Duplicate Attempts: {duplicate_count}")
        summary.append(f"  - Unsafe Attempts: {unsafe_count}")
        summary.append(f"  - Hints Requested: {hints_requested}")
        summary.append(f"  - Hints Provided: {hints_provided}")
        
        # Print summary
        ui_console.print(Panel("\n".join(summary), title="Run Summary", border_style="bold"))

if __name__ == "__main__":
    main()