# T-001: Python 3.13 packaging with uv

Status: landed (worktree-agent-a10f73a65275d38a2, 2026-10-29)
Opened: 2026-10-29 by the orchestrator, from maintainer rulings R1-R4 after the project's 18-month idle.
Depends on: none (T-002 and T-003 are independent)

## What happens today

- The repo has no requirements file and no pyproject. `.devcontainer/Dockerfile` (python:3.11-slim) runs `pip install onnx onnxruntime numpy rich`, then `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`. Nothing is pinned.
- Code imports torch, numpy, requests, aiohttp, rich, matplotlib (plus stdlib). It imports neither onnx nor onnxruntime. requests, aiohttp and matplotlib are not installed by the Dockerfile; `.devcontainer/devcontainer.json:35` patches that with `"postCreateCommand": "pip install requests ipykernel"`. aiohttp and matplotlib are missing entirely.
- The host has Python 3.12.10 and uv 0.6.4. The host torch is 2.9.1+rocm6.4, which has no gfx1010 kernels (see T-002).
- Command lines in `AGENTS.md:10-14` and `README.md:30-32` use bare `python`; `start_altlas.sh:11,33` and `start_altlas.ps1:18,43` run `python altlas_ui.py`.

## Rulings (2026-10-29)

- R1: Torch runs on CPU by default everywhere. Reason: the RX 5700 XT (gfx1010) has no kernels in ROCm torch, and the model is about 683k parameters. Rejected: a ROCm torch build.
- R2: Packaging is `pyproject.toml` + `uv.lock` with the latest compatible versions; the Dockerfile installs from the lock; onnx and onnxruntime are dropped. Rejected: requirements.txt; keeping the unused onnx packages.
- R3: Python 3.13 on the host and in the devcontainer. Rejected: staying on 3.11/3.12.

## Settled design

- `pyproject.toml`: project `altlas`, `requires-python = ">=3.13,<3.14"`, dependencies torch, numpy, requests, aiohttp, rich, matplotlib (add any other third-party import the lane finds by grepping imports). Not a package to build: use `[tool.uv] package = false`. No onnx, onnxruntime, torchvision, torchaudio.
- torch comes from the PyTorch CPU index: `[[tool.uv.index]] name = "pytorch-cpu" url = "https://download.pytorch.org/whl/cpu" explicit = true` and `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]`.
- `.python-version` contains `3.13`.
- `.devcontainer/Dockerfile`: base `python:3.13-slim`; keep the apt packages and the vscode user block; install uv (copy the binary from `ghcr.io/astral-sh/uv`); copy `pyproject.toml`, `uv.lock`, `.python-version` to `/workspace` and run `uv sync --frozen`; put the venv on PATH. Remove the pip lines for onnx, onnxruntime and torch.
- `.devcontainer/devcontainer.json`: remove `pip install requests ipykernel` from `postCreateCommand` (or replace with `uv sync --frozen`); keep ipykernel only if added as a dev dependency group.
- `AGENTS.md`, `README.md`, `start_altlas.sh`, `start_altlas.ps1`: every `python ...` command becomes `uv run python ...`, including the py_compile line.

## Owns

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `.devcontainer/**`
- `start_altlas.sh`
- `start_altlas.ps1`
- `AGENTS.md`
- `README.md`

## Out of scope

- `utils.py` device logic (T-002) and the LLM provider (T-003).
- `CLAUDE.md` gates (they stay bare `python`; the orchestrator updates them after landing).
- Any ROCm/CUDA wheel, Vulkan torch, or a requirements.txt.
- Running `docker build`: the orchestrator checks it.

## Acceptance

- [x] `uv lock` then `uv sync` exit 0 on the host.
- [x] `uv run python -c "import torch,sys; print(sys.version_info[:2], torch.__version__)"` prints `(3, 13)` and a version ending `+cpu`.
- [x] `uv run python -c "import numpy, requests, aiohttp, rich, matplotlib"` exits 0; `grep -n "onnx" pyproject.toml uv.lock .devcontainer/Dockerfile` prints nothing.
- [x] `uv run python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py` exits 0.
- [x] `grep -nE "^[^#]*\bpython (runner|altlas_ui|-m py_compile)" AGENTS.md README.md start_altlas.sh start_altlas.ps1` prints no line lacking `uv run`.
- [x] Dockerfile first line is `FROM python:3.13-slim`, contains `uv sync --frozen` and `USER $USERNAME` for vscode (grep).
- [ ] `docker build` of `.devcontainer` is checked by the orchestrator, not the lane.
- [x] Ticket Status set to `landed (<branch>, YYYY-MM-DD)` in the lane's commit.
- [x] Gates pass: `python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py` and `node scripts/check-tickets.mjs`.

Do NOT run the test harness. Stage paths explicitly.
