# AltLAS

AltLAS is an experimental autonomous coding system. The repository now combines two ideas that were previously mixed together: a research loop for bottom-up learning, and a practical agent stack for generation, execution, scoring, reporting, and provider-backed assistance.

This refresh reflects the current 2026 model landscape. Strong local and open models, longer context windows, multimodal inputs, tool use, and cheaper hosted reasoning all change what “bootstrapping” should mean. AltLAS should be designed as a hybrid system, not as a single isolated learner.

## Current State

Implemented in this repository today:

- PyTorch-based sequence generator and training loop
- benchmark-style task definitions under `task/`
- execution, scoring, safety, duplicate detection, and reporting
- terminal UI and resumable runs
- provider hooks for local inference via LM Studio and vLLM
- language execution maps for Python-first, multi-language experimentation

Not mature yet:

- rigorous automated eval harness
- robust resource enforcement in the executor
- dependable model routing across local and hosted providers
- reliable task solving beyond basic experiments

## Quick Start

The project is designed for the devcontainer in `.devcontainer/`. Install with `uv sync --extra cpu`; run commands with `uv run --extra cpu` (a bare `uv run` removes torch).

```bash
uv run --extra cpu python runner.py --task benchmark_add_two_numbers
uv run --extra cpu python runner.py --task benchmark_add_two_numbers --debug
uv run --extra cpu python runner.py --task benchmark_add_two_numbers --reset
./start_altlas.sh
```

Primary entrypoints:

- `runner.py`: orchestration and training run startup
- `altlas_ui.py`: terminal task picker and launcher
- `config.ini`: runtime, model, optimizer, logging, and provider settings

## AMD Navi 10 GPU (optional)

The default is CPU (`uv sync --extra cpu`). For an RX 5700 XT (gfx1010) use the other extra instead; the two extras conflict, so use one venv per extra:

```bash
uv sync --extra gfx101x          # about 5 GB download
ALTLAS_DEVICE=cuda uv run --extra gfx101x python runner.py --task benchmark_add_two_numbers
```

- The build is an unmaintained June 2026 AMD nightly (`torch 2.12.0+rocm7.14.0a20260612`, TheRock gfx101X index).
- MIOpen fails on LSTM training on gfx101x, so `get_pytorch_device()` sets `torch.backends.cudnn.enabled = False` there. Set `ALTLAS_MIOPEN=1` to keep MIOpen on (to test newer stacks).
- Without `ALTLAS_DEVICE=cuda` the run stays on CPU.

## Repository Layout

- `agent_core/`: generation model and code synthesis logic
- `core/`: config loading, UI, attempt management, training loop
- `evaluator/`: code execution
- `guardian/`: safety checks
- `reinforcer/`: scoring and tool feedback
- `task/`: benchmark and task definitions
- `memory/`: state, reports, and persisted artifacts
- `language_maps/`: language execution metadata

## Design Direction

AltLAS now targets a capability-based architecture:

- local models for cheap iteration and private execution
- stronger hosted models for planning, critique, repair, and distillation
- deterministic tools for execution, verification, and measurement
- eval-driven routing instead of hard-coding one “best” model path

See the updated design docs:

- `AltLAS_Project_Guide.md`
- `dev_plan.md`
- `docs/architecture-2026.md`
- `docs/model-strategy.md`
- `docs/evals.md`

## Development Priorities

The next practical work is:

1. build a real benchmark and regression harness
2. harden execution limits and error reporting
3. separate research-mode learning from production-style agent workflows
4. introduce a model router with explicit local, hosted, and specialist roles
