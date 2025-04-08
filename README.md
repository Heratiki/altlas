# AltLAS - Autonomous Learning Task Agent System

## Overview

AltLAS is an experimental system designed to explore emergent learning in AI agents. The goal is to create an agent that learns to solve programming tasks through reinforcement learning, starting with minimal prior knowledge and developing its own coding approach.

This system uses a PyTorch-based RNN (LSTM) model trained with the REINFORCE algorithm (with baseline and entropy regularization) to generate code attempts.

Each task is defined in a JSON file that includes not only the expected output but also valid solution patterns and constraints. This allows the system to recognize multiple valid approaches to solving a problem while ensuring core requirements (like specific operators or values) are met.

## Setup

This project is designed to run within a development container.

1.  **Prerequisites:** Docker and VS Code with the "Dev Containers" extension.
2.  **Open in Container:** Open the project folder in VS Code and use the command palette (`Ctrl+Shift+P`) to run "Dev Containers: Reopen in Container". This will build the Docker image defined in `.devcontainer/Dockerfile` (which includes Python, PyTorch with CUDA support if available) and start the container.
3.  **Verification:** Once the container is running, you can verify the PyTorch installation by opening a terminal in VS Code (`Ctrl+\``) and running:
    ```bash
    python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
    ```

## Configuration

Key hyperparameters, paths, and settings are managed in the `config.ini` file. This includes:

*   `[Runner]`: Max attempts, stuck detection parameters, hint probability, logging frequency.
*   `[Model]`: RNN/LSTM dimensions (embedding, hidden layers).
*   `[Optimizer]`: Learning rate, entropy coefficient, gradient clipping norm, baseline EMA alpha.
*   `[Executor]`: Code execution timeout.
*   `[Scorer]`: Task success threshold.
*   `[Paths]`: Locations for vocabulary, model state, optimizer state.

Review and adjust these parameters as needed before running experiments.

## Running the Agent

The main script to run the agent is `runner.py`.

1.  **Open Terminal:** Use the integrated terminal in VS Code (`Ctrl+\``).
2.  **Run Command:** Execute the runner script, optionally specifying a task:
    ```bash
    python runner.py [--task <task_name>]
    ```
    *   `--task <task_name>`: Specifies the task to run. The `<task_name>` corresponds to a JSON file in the `task/` directory or its subdirectories (e.g., `task/benchmarks/`). Do not include the `.json` extension.
    *   If `--task` is omitted, it defaults to `hello_world`.

3.  **Available Tasks:**
    *   `hello_world`: A simple task requiring the agent to print "hello world". Found in `task/hello_world.json` (implicitly, needs to be created or loaded dynamically by TaskLoader improvements).
    *   `benchmark_add_two_numbers`: A benchmark task demonstrating pattern-based task definition. It requires the agent to print the sum of 5 and 3 (output "8") while accepting multiple valid solution approaches like direct addition, variable-based addition, or function-based solutions. Found in `task/benchmarks/benchmark_add_two_numbers.json`.

## Task Definition

Tasks are defined in JSON files with the following structure:
* `name`: Task identifier
* `description`: Human-readable task description
* `success_criteria`:
  * `type`: Type of validation (e.g., "code_pattern", "exact_output")
  * `expected_output`: Expected program output
  * `valid_patterns`: Array of acceptable solution patterns and variations
  * `case_sensitive`: Whether output matching is case-sensitive
  * `whitespace_sensitive`: Whether whitespace matters in pattern matching
* `constraints`: Optional requirements like:
  * `required_operators`: Operators that must be used
  * `required_numbers`: Numbers that must appear in the solution
  * `max_tokens`: Maximum tokens allowed in solution
* `difficulty`: Task difficulty level (e.g., "beginner", "intermediate")

## Understanding the Output

*   **Rich UI:** The script uses the `rich` library to display a live dashboard in the terminal, showing:
    *   Overall progress.
    *   Current statistics (attempt count, scores, duplicates, errors, hints).
    *   Recent status messages.
    *   Current advisor hint (if active).
*   **Logs:** Detailed information, including weight initialization, validation checks, learning steps (loss, gradient norm), periodic token frequencies, and errors, are printed to the console/standard output.
*   **Final Summary:** At the end of the run, a summary panel shows overall results and statistics. It will also indicate if the run was interrupted early by the user (Ctrl+C).
*   **Final Token Frequencies:** A table showing the frequency of each token generated during the run is printed.
*   **Saved State:** Model and optimizer states are saved to files specified in `config.ini` (default: `memory/model_state.pth`, `memory/optimizer_state.pth`). **Note:** State is saved upon successful completion, reaching max attempts, or clean user interruption (Ctrl+C). State is *not* saved if the run terminates due to an unexpected error.
*   **Best Attempt:** The code corresponding to the best score achieved during the run is saved to `memory/logs/best_attempt.json`.
*   **Interrupting:** You can stop a run early by pressing `Ctrl+C` in the terminal. The current progress (model/optimizer state) will be saved if the interruption is handled cleanly.

## Development

*   **Development Plan:** See `dev_plan.md` for the ongoing development roadmap, completed tasks, and future plans.
*   **PyTorch Transition:** See `pytorch_transition_plan.md` for details on the transition to the PyTorch framework.
*   **Project Guide:** See `AltLAS_Project_Guide.md` for overarching project goals and philosophies.