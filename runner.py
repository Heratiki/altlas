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
            print(f"💡 Advisor Hint: {hint}")
            return hint
        else:
            print("🤷 Advisor could not generate a hint")
            return None
            
    except Exception as e:
        print(f"⚠️ Error getting hint from advisor: {str(e)}")
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
            print(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return None

def call_local_llm(prompt: str) -> Optional[str]:
    """
    Call a local LLM using LM Studio running on the host machine.
    Uses host.docker.internal to connect from within the container.
    """
    try:
        # Always use host.docker.internal in this Docker-based setup
        base_url = "http://host.docker.internal:1234/v1"
        print("🐳 Using host.docker.internal to connect to LM Studio")
        
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
                print(f"🤖 Available LM Studio models: {available_models}")
                
                # Choose an appropriate model (prefer code-oriented models if available)
                chosen_model = None
                preferred_models = [
                    "codellama", "code-llama", "wizardcoder", "stable-code", "phi-2",
                    "mistral", "mixtral", "llama3", "llama-3", "llama2", "llama-2"
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
                    print("⚠️ No models available in LM Studio")
                    return generate_static_hint(prompt)
                    
                print(f"🧠 Using LM Studio model: {chosen_model}")
            else:
                print(f"⚠️ Failed to retrieve models from LM Studio: {models_response.status_code} - {models_response.text}")
                return generate_static_hint(prompt)
        except Exception as model_e:
            print(f"⚠️ Error querying available models: {str(model_e)}")
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
            print(f"⚠️ Local LLM API error: {response.status_code} - {response.text}")
            return generate_static_hint(prompt)
            
    except Exception as e:
        print(f"⚠️ Error calling local LLM: {str(e)}")
        return generate_static_hint(prompt)

def generate_static_hint(prompt: str) -> str:
    """
    Generate a helpful static hint based on the context in the prompt.
    This function provides more intelligent fallback hints when external LLMs are unavailable.
    
    Args:
        prompt: The original prompt that would have been sent to the LLM
    
    Returns:
        str: A relevant programming hint
    """
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


def main():
    """Main runner function that orchestrates the AltLAS workflow."""
    # Initialize Rich console
    console = Console()
    console.print("[bold blue]🧠 Starting AltLAS - Learning Agent[/bold blue]")

    # --- Read Config ---
    config = configparser.ConfigParser()
    # Define config path relative to the runner script
    config_path = Path(__file__).parent / "config.ini" 
    if not config_path.exists():
        console.print(f"[bold red]Error: Configuration file not found at {config_path}[/bold red]")
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
        console.print(f"[bold red]Error: Missing section in {config_path}: {e}[/bold red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Error: Invalid value type in {config_path}: {e}[/bold red]")
        sys.exit(1)
    # --- End Read Config ---
    
    # --- Determine PyTorch Device ---
    device = get_pytorch_device() # <-- Determine device
    # --- End Determine Device ---
    
    # 1. Load task
    task_loader = TaskLoader()
    current_task = task_loader.load_task("hello_world")
    console.print(f"[bold green]📋 Loaded task: {current_task.name}[/bold green]")
    
    # 2. Initialize components, passing config path as string
    generator = CodeGenerator(config_path=str(config_path), device=device) # <-- Pass device
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
    
    # Keep track of status messages for display
    status_messages = []
    duplicate_count = 0
    unsafe_count = 0
    error_count = 0
    success_count = 0
    
    # Create the Rich layout
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
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
    
    try:
        # Start live display
        with Live(layout, refresh_per_second=10, console=console) as live:
            # 3. Main learning loop
            while attempt_count < max_attempts and not success:
                attempt_count += 1
                attempts_since_last_check += 1
                
                # Update progress
                progress.update(task_id, completed=attempt_count, percentage=100*attempt_count/max_attempts)
                
                # Update header with progress
                layout["header"].update(Panel(progress, title="AltLAS Progress", border_style="blue"))
                
                # --- Stuck Detection & Hinting ---
                if attempts_since_last_check >= stuck_check_window:
                    current_best_score = logger.get_best_score()
                    if current_best_score - last_best_score_at_check < stuck_threshold:
                        current_hint = get_hint_from_advisor(current_task, logger.get_history())
                        if current_hint:
                            status_messages.insert(0, f"🤔 [{time.strftime('%H:%M:%S')}] Hint requested: {current_hint}")
                    else:
                         current_hint = None 
                    last_best_score_at_check = current_best_score
                    attempts_since_last_check = 0
                # --- End Stuck Detection ---

                # Generate code, potentially using a hint
                try:
                    code_attempt, generated_ids = generator.generate(current_task, logger.get_history(), hint=current_hint) # Now returns generated_ids
                    # Check if generate somehow returned None despite not having a path for it
                    if code_attempt is None or generated_ids is None: # Check generated_ids
                         status_messages.insert(0, f"⚠️ [{time.strftime('%H:%M:%S')}] Generator returned None unexpectedly. Skipping attempt.")
                         continue
                except Exception as gen_e:
                    status_messages.insert(0, f"❌ [{time.strftime('%H:%M:%S')}] Exception during generation: {type(gen_e).__name__}: {gen_e}")
                    # Decide how to handle - skip attempt
                    continue # Skip to next attempt
                
                # Check safety and novelty
                if not safety_checker.is_safe(code_attempt):
                    status_messages.insert(0, f"⚠️ [{time.strftime('%H:%M:%S')}] Unsafe code detected in attempt {attempt_count}, skipping execution")
                    unsafe_count += 1
                    logger.log_attempt(attempt_count, code_attempt, None, 0.0, "unsafe", status="unsafe") # Log unsafe attempts differently
                    continue
                    
                fingerprint = fingerprinter.get_fingerprint(code_attempt)
                if fingerprinter.is_duplicate(fingerprint):
                    status_messages.insert(0, f"🔄 [{time.strftime('%H:%M:%S')}] Duplicate attempt {attempt_count} detected, skipping execution")
                    duplicate_count += 1
                    continue
                
                # Execute code
                result = executor.execute(code_attempt)
                
                # Score result
                score = scorer.score(result, current_task)
                
                # Update status message
                status_text = f"Attempt {attempt_count}: Score {score:.2f}, Status: {result.status}"
                if result.status == "error":
                    error_count += 1
                    status_messages.insert(0, f"❗ [{time.strftime('%H:%M:%S')}] {status_text}")
                elif result.status == "success":
                    success_count += 1
                    status_messages.insert(0, f"✅ [{time.strftime('%H:%M:%S')}] {status_text}")
                else:
                    status_messages.insert(0, f"ℹ️ [{time.strftime('%H:%M:%S')}] {status_text}")
                
                # Log attempt (even if score is low, for learning)
                logger.log_attempt(attempt_count, code_attempt, result, score, fingerprint) # Pass attempt_count
                
                # --- Perform learning step --- 
                generator.learn(score, generated_ids) # Pass generated_ids
                # --- End learning step ---
                
                # Update stats display
                best_score_val, best_attempt_num = logger.get_best_score_info()
                
                stats_table = Table(show_header=False, box=None)
                stats_table.add_column("Category", style="cyan")
                stats_table.add_column("Value", style="green")
                
                stats_table.add_row("Current Attempt", f"{attempt_count}/{max_attempts}")
                stats_table.add_row("Best Score", f"{best_score_val:.2f} (Attempt {best_attempt_num})")
                stats_table.add_row("Latest Score", f"{score:.2f}")
                stats_table.add_row("Duplicate Attempts", str(duplicate_count))
                stats_table.add_row("Unsafe Attempts", str(unsafe_count))
                stats_table.add_row("Error Attempts", str(error_count))
                stats_table.add_row("Success Attempts", str(success_count))
                
                # Task details
                stats_table.add_row("", "")  # Empty row for spacing
                stats_table.add_row("Task", current_task.name)
                stats_table.add_row("Description", current_task.description)
                
                layout["stats"].update(Panel(stats_table, title="Statistics", border_style="green"))
                
                # Update status messages (keep the most recent ones)
                status_panel = Text("\n".join(status_messages[:15]))
                layout["status"].update(Panel(status_panel, title="Status Messages", border_style="yellow"))
                
                # Update footer with hints if available
                if current_hint:
                    hint_text = Text(f"🤔 Current Hint: {current_hint}", style="italic yellow")
                    layout["footer"].update(Panel(hint_text, title="Advisor Hint", border_style="yellow"))
                else:
                    layout["footer"].update(Panel("No active hints", title="Advisor Hint", border_style="dim"))
                
                # Check for success using configured threshold
                if score >= success_threshold: # Use configured threshold
                    success = True
                    message = f"🎉 Success achieved on attempt {attempt_count}!"
                    status_messages.insert(0, f"🎉 [{time.strftime('%H:%M:%S')}] {message}")
                    layout["header"].update(Panel(Text(message, style="bold green"), title="Success!", border_style="green"))
                    # Update the display one last time and wait briefly for user to see the success
                    live.update(layout)
                    time.sleep(1)
        
        # Clear hint after one use to avoid over-reliance (outside loop)
        current_hint = None
    
    finally: 
        # --- Save Learned Weights ---
        # Ensure generator exists before saving (in case of early error)
        if 'generator' in locals(): 
            generator.save_weights()
        # --- End Save Weights ---

        # 4. Summarize results
        summary = []
        if success:
            summary.append("[bold green]✅ Task completed successfully![/bold green]")
        elif attempt_count >= max_attempts:
            summary.append("[bold yellow]⏱️ Maximum attempts reached.[/bold yellow]")
        else:
            summary.append("[bold red]⏹️ Run ended.[/bold red]") # Handle cases like Ctrl+C or early errors
        
        # Ensure logger exists before getting best score
        if 'logger' in locals():
            summary.append(f"[bold blue]📈 Best score achieved: {logger.get_best_score():.2f}[/bold blue]")
        
        # Print summary
        console.print(Panel("\n".join(summary), title="Run Summary", border_style="bold"))

if __name__ == "__main__":
    main()