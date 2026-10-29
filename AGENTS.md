# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives at the repository root and in focused packages. Use `runner.py` for the main training loop and `altlas_ui.py` for the terminal launcher. Shared orchestration and config logic live in `core/`; model and generation code in `agent_core/`; execution, scoring, and safety checks in `evaluator/`, `reinforcer/`, and `guardian/`. Tasks are JSON fixtures under `task/`, especially `task/benchmarks/`. Persisted runtime state and generated reports go under `memory/` and should usually not be committed.

## Build, Test, and Development Commands
This project is designed for the devcontainer in `.devcontainer/`. Install with `uv sync --extra cpu`; run commands with `uv run --extra cpu` (a bare `uv run` removes torch).

```bash
uv run --extra cpu python runner.py --task benchmark_add_two_numbers
uv run --extra cpu python runner.py --task benchmark_add_two_numbers --debug
uv run --extra cpu python runner.py --task benchmark_add_two_numbers --reset
./start_altlas.sh
uv run --extra cpu python -m py_compile runner.py altlas_ui.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py
```

`runner.py` starts a training run for a specific task, `--debug` increases console logging, and `--reset` clears saved state before training. `start_altlas.sh` launches the terminal UI directly in-container or via Docker outside the container. Use `py_compile` as a lightweight syntax check before opening a PR.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and short module docstrings where helpful. Keep imports grouped as standard library, third-party, then local packages. Match current naming patterns such as `TrainingLoop`, `CodeGenerator`, and `load_task`. There is no repo-level formatter config, so keep changes consistent with surrounding files and avoid broad style-only rewrites.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with targeted benchmark runs in `task/benchmarks/` and include the exact command used in your PR notes. For changes to task loading, execution, or scoring, run at least one relevant benchmark and a syntax pass with `uv run --extra cpu python -m py_compile`. If you add automated tests later, place them in a new `tests/` package and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects, often with Conventional Commit prefixes such as `feat:` and `refactor:`. Prefer `type: concise summary`, for example `feat: improve duplicate attempt tracking`. Keep commits focused. PRs should describe the affected module(s), task or benchmark used for validation, any `config.ini` changes, and terminal output or screenshots when UI behavior changes. Do not commit logs, `.pth` model state, or generated files from `memory/`.

## AMD Navi 10 GPU (optional)

The default is CPU (`uv sync --extra cpu`). For an RX 5700 XT (gfx1010) use the other extra instead; the two extras conflict, so use one venv per extra:

```bash
uv sync --extra gfx101x          # about 5 GB download
ALTLAS_DEVICE=cuda uv run --extra gfx101x python runner.py --task benchmark_add_two_numbers
```

- The build is an unmaintained June 2026 AMD nightly (`torch 2.12.0+rocm7.14.0a20260612`, TheRock gfx101X index).
- MIOpen fails on LSTM training on gfx101x, so `get_pytorch_device()` sets `torch.backends.cudnn.enabled = False` there. Set `ALTLAS_MIOPEN=1` to keep MIOpen on (to test newer stacks).
- Without `ALTLAS_DEVICE=cuda` the run stays on CPU.
