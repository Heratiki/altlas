#!/usr/bin/env python3
"""
AltLAS Runner - Main orchestrator for the AltLAS system.

Initializes components and starts the training loop.
"""

import os
import sys
import time
import argparse
import logging
import signal
import configparser
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console

# Ensure all module directories are in the path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import core components
from core.config_loader import ConfigLoader
from core.ui_display import UIDisplay
from core.attempt_manager import AttemptManager
from core.training_loop import TrainingLoop

# Import other necessary components
from agent_core.generator import CodeGenerator
from guardian.safety import SafetyChecker
from evaluator.executor import CodeExecutor
from reinforcer.scorer import AttemptScorer
from memory.report_generator import TrainingReportGenerator
from task.task_loader import TaskLoader
from utils import get_pytorch_device

# --- Centralized Logging Setup ---
LOG_FILENAME = "altlas_run.log"

# Load config.ini early to get logging level
config = configparser.ConfigParser()
config.read(Path(__file__).parent / 'config.ini')

file_log_level_str = 'INFO'
try:
    file_log_level_str = config.get('Logging', 'FileLogLevel', fallback='INFO').upper()
except Exception:
    pass

log_level_map = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
}
file_log_level = log_level_map.get(file_log_level_str, logging.INFO)

# Create console for logging (UI display uses its own consoles)
log_console = Console(stderr=True)

# Configure root logger
log = logging.getLogger()
log.setLevel(file_log_level)  # Set root logger level based on file_log_level

# Formatter for file logs
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler (Rotating)
file_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(file_log_level)  # Set from config

# Console Handler (Rich)
console_handler = RichHandler(console=log_console, rich_tracebacks=True, show_path=False)
console_handler.setLevel(logging.INFO)  # Console shows INFO+

# Remove existing handlers if any
for handler in log.handlers[:]:
    log.removeHandler(handler)

# Add handlers
log.addHandler(file_handler)
log.addHandler(console_handler)
# --- End Logging Setup ---

# Global variable to hold the training loop instance for signal handling
training_loop_instance = None

# Define signal handler for Ctrl+C
def signal_handler(sig, frame):
    global training_loop_instance
    if training_loop_instance:
        training_loop_instance.set_user_interrupted()
    else:
        # Handle cases where interruption happens before loop starts
        log.warning("🏃 User interrupted before training loop started. Exiting.")
        sys.exit(0)

# Install signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def main():
    """Main runner function that orchestrates the AltLAS workflow."""
    global training_loop_instance

    # Add console handler at the start of main
    if console_handler not in log.handlers:
        log.addHandler(console_handler)

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

    log.info(f"🧠 Starting AltLAS - Task: {task_to_load}")
    log.debug("Debug logging is active and directed to file.")

    try:
        # --- Initialize Core Components ---
        project_root = Path(__file__).parent.absolute()
        config_loader = ConfigLoader(config_path=project_root / "config.ini")
        attempt_manager = AttemptManager(project_root=project_root)
        ui_display = UIDisplay()
        # --- End Core Component Initialization ---

        # Reset training state if requested
        if reset_training:
            log.info("🔄 Reset flag detected - clearing all training state files...")
            reset_successful = attempt_manager.reset_training_state()
            if reset_successful:
                log.info("✅ Training state reset complete - starting from attempt 1")
            else:
                log.warning("⚠️ No training state files were found to delete")
        
        # --- Determine PyTorch Device ---
        device = get_pytorch_device()
        # --- End Determine Device ---
        
        # --- Load Task ---
        task_loader = TaskLoader()
        try:
            current_task = task_loader.load_task(task_to_load)
            log.info(f"📋 Loaded task: {current_task.name} - {current_task.description}")
        except ValueError as e:
            # Ensure console handler is added back before logging the error
            if console_handler not in log.handlers:
                log.addHandler(console_handler)
            log.error(f"Error loading task '{task_to_load}': {e}", exc_info=True)
            sys.exit(1)
        # --- End Load Task ---
        
        # --- Initialize AltLAS Components ---
        # Pass config path string for components that need it directly
        config_path_str = str(config_loader.get_config_path())
        
        generator = CodeGenerator(config_path=config_path_str, device=device)
        safety_checker = SafetyChecker() 
        executor = CodeExecutor(config_path=config_path_str)
        scorer = AttemptScorer(config_path=config_path_str)
        report_generator = TrainingReportGenerator()
        # --- End AltLAS Component Initialization ---

        # --- Initialize Training Loop ---
        training_loop_instance = TrainingLoop(
            config_loader=config_loader,
            attempt_manager=attempt_manager,
            ui_display=ui_display,
            generator=generator,
            safety_checker=safety_checker,
            executor=executor,
            scorer=scorer,
            report_generator=report_generator,
            current_task=current_task,
            device=device
        )
        # --- End Training Loop Initialization ---

        # --- Run Training Loop ---
        # Remove console handler *before* starting Live display run
        if console_handler in log.handlers:
            log.removeHandler(console_handler) 
            log.debug("Console logging handler removed for Live display.") # Log to file

        training_loop_instance.run() # This contains the Live display context
        # --- End Run Training Loop ---

    except FileNotFoundError as e:
        # Ensure console handler is added back before logging the error
        if console_handler not in log.handlers:
            log.addHandler(console_handler)
        log.error(f"Initialization failed: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        # Ensure console handler is added back before logging the error
        if console_handler not in log.handlers:
            log.addHandler(console_handler)
        log.error(f"Configuration error: {e}")
        sys.exit(1)
    except ImportError as e:
        # Ensure console handler is added back before logging the error
        if console_handler not in log.handlers:
            log.addHandler(console_handler)
        log.error(f"Error importing modules: {e}")
        log.error("Please ensure all required modules are available and paths are correct.")
        sys.exit(1)
    except Exception as e:
        # Ensure console handler is added back before logging the error
        if console_handler not in log.handlers:
            log.addHandler(console_handler)
        log.exception(f"An unexpected error occurred during initialization or execution: {e}")
        sys.exit(1)
    finally:
        # Ensure console handler is added back after the loop finishes or if an error occurs
        if console_handler not in log.handlers:
            log.addHandler(console_handler)
            log.debug("Console logging handler re-enabled.") # Log to console and file

if __name__ == "__main__":
    main()