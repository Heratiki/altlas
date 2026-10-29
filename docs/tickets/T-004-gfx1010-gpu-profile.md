# T-004: Optional AMD Navi 10 (gfx1010) GPU setup

Status: open
Opened: 2026-10-29 by the orchestrator, from maintainer ruling R8 and host tests of 2026-10-29.
Depends on: T-001 (uv packaging), T-002 (safe device selection; CPU stays the default, R1)

## What happens today

- `pyproject.toml:4-11` lists `torch` as a plain dependency. `pyproject.toml:14-22` defines one explicit index, `pytorch-cpu` (`https://download.pytorch.org/whl/cpu`), and `[tool.uv.sources]` sends `torch` to it on Linux. There is no way to choose a ROCm build. `uv.lock` (header: `version = 1`, `revision = 1`, `requires-python = "==3.13.*"`) pins only the CPU build.
- `.devcontainer/Dockerfile:13` copies uv 0.6.4; `Dockerfile:21` runs `uv sync --frozen`; `.devcontainer/devcontainer.json:36` runs `uv sync --frozen` as `postCreateCommand`. The host uv is also 0.6.4.
- `utils.py:10-56` `get_pytorch_device()` returns CPU unless `ALTLAS_DEVICE` is `cuda` or `auto` (`utils.py:30-39`); then it needs `torch.cuda.is_available()` (`utils.py:41`) and a one-op kernel probe (`utils.py:45-53`). It never touches `torch.backends.cudnn`. Callers: `runner.py:38,181` and `benchmark_runner.py:26,185` call it and pass the device on; `agent_core/generator.py:40` takes `device` (default CPU) and does `.to(self.device)` (e.g. lines 147, 192).
- The docstring at `utils.py:22-25` already notes that `HSA_OVERRIDE_GFX_VERSION=10.3.0` is unreliable on gfx1010.
- Evidence from tests on this host on 2026-10-29 (RX 5700 XT, Navi 10, gfx1010):
  - AMD TheRock per-family index `https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/` holds `torch==2.12.0+rocm7.14.0a20260612`, cp313 linux_x86_64, sha256 `55f101603f2dc0e0feeb5c1ccb1c88c4271c9d88b970f10ba600212899bb1e8b`. It depends on the pre-release `rocm[libraries]==7.14.0a20260612`, `rocm-sdk-core`, `rocm-sdk-libraries-gfx101x-dgpu`, and a matching `triton` from the same index. Install needed `--prerelease=allow` and `--index-strategy unsafe-best-match` with PyPI as an extra index. The install is about 5 GB.
  - Per-family builds stop at 2026-06-12 (AMD moved to multi-arch releases). Whether a multi-arch release covers gfx1010 is UNCONFIRMED.
  - REA static inspection: `libtorch_hip.so` holds 625 code objects for gfx1010/gfx1011/gfx1012.
  - In an isolated venv: rocBLAS matmul, conv2d, LayerNorm, Embedding and LSTM forward match CPU.
  - Training (LSTM backward + Adam) FAILS with MIOpen on: `miopenStatusUnknownError`, from a runtime CK reduction kernel (`gridwise_generic_reduction_first_call_warpwise_reduce_partial_dims.cpp`: `No CK_AMD_GPU_GFX* macro defined`).
  - With `torch.backends.cudnn.enabled = False`, 20 training steps on GPU match CPU (loss 4.5864 -> 0.5628, final diff 1.8e-7).
  - `benchmark_runner.py` with `ALTLAS_DEVICE=cuda` runs on cuda and T-002's kernel probe passes.

## Rulings (2026-10-29)

- R8: Add an optional gfx1010 GPU setup: a uv extra that uses the TheRock gfx101X index, and set `torch.backends.cudnn.enabled = False` when the device is gfx1010. Reason: the only failure found is the MIOpen path, and turning it off makes training match CPU. Rejected: `HSA_OVERRIDE_GFX_VERSION` (unreliable, see `utils.py:22-25`); making the GPU the default (R1: CPU default).

## Settled design

1. `pyproject.toml`: two mutually exclusive extras.
   - `cpu`: the current CPU index (`pytorch-cpu`). It is what the Dockerfile and docs use.
   - `gfx101x`: a new explicit index `https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/`, with torch pinned to `2.12.0+rocm7.14.0a20260612`.
   - Declare them with `[tool.uv] conflicts = [[{ extra = "cpu" }, { extra = "gfx101x" }]]`, and give each extra its own `[tool.uv.sources]` torch entry (using `extra = ...` in the source).
   - The pre-release `rocm` packages must resolve for `gfx101x` only (a `prerelease` setting or explicit pre-release pins; the lane picks the least global option). The `gfx101x` resolution also needs `index-strategy = "unsafe-best-match"` (as a uv setting) with PyPI available.
   - Re-lock `uv.lock`.
   - uv 0.6.4: the lane first checks that this version supports `conflicts`, per-extra sources and the settings above. If not, the lane STOPS, reports which uv version is needed, and asks for a ruling. It must not upgrade uv.
   - With extras, `uv sync` with no extra installs no torch. Then `.devcontainer/Dockerfile:21`, `.devcontainer/devcontainer.json:36`, `AGENTS.md` and `README.md` must pass `--extra cpu`. Whatever the lane changes, a plain devcontainer build must still give CPU torch.
2. `utils.py` `get_pytorch_device()`: after the kernel probe passes (after `utils.py:53`), read `torch.cuda.get_device_properties(device).gcnArchName`. If it starts with `gfx101`, set `torch.backends.cudnn.enabled = False` and log one INFO line: MIOpen is off for RNN training on gfx101x. Env `ALTLAS_MIOPEN=1` keeps MIOpen on (for testing newer stacks). The CPU path and the fallback paths are unchanged. If `gcnArchName` is missing (non-ROCm torch), do nothing.
3. Docs: `AGENTS.md` and `README.md` each get a short "AMD Navi 10 GPU (optional)" section: `uv sync --extra gfx101x`, `ALTLAS_DEVICE=cuda`, the MIOpen note and `ALTLAS_MIOPEN=1`, and that the build is an unmaintained June 2026 nightly (about 5 GB download).

## Confirm before implementing

- Whether a multi-arch AMD release covers gfx1010 is UNCONFIRMED. This ticket pins the 2026-06-12 per-family build; a later multi-arch pin is a separate ruling.
- If uv 0.6.4 cannot express the design, which uv version to use (stop and ask; see Settled design 1).
- Whether the gfx101x pin should be reachable without the user passing `--prerelease`/index flags is left to the lane's reading of uv settings; if it needs global `prerelease = "allow"` (which affects the cpu extra), stop and ask.

## Owns

- `pyproject.toml`
- `uv.lock`
- `utils.py`
- `AGENTS.md`
- `README.md`
- `.devcontainer/**` (only if item 1 requires it)
- `docs/tickets/T-004-gfx1010-gpu-profile.md` (Status line only)

## Out of scope

- Making the GPU the default, or changing `ALTLAS_DEVICE` semantics (T-002).
- Multi-arch ROCm releases, other GPU families, `HSA_OVERRIDE_GFX_VERSION`.
- Upgrading uv.
- Changes to `runner.py`, `benchmark_runner.py`, `agent_core/generator.py` or the model.
- ROCm in the Dockerfile or GPU passthrough in the devcontainer.
- Running the full benchmark suite.

## Acceptance

Use `UV_PROJECT_ENVIRONMENT=.scratch/venv-gfx` for the gfx101x venv (keep `.scratch/` untracked). The gfx101x sync downloads about 5 GB.

- [ ] `uv sync --extra cpu` gives `torch.__version__` ending `+cpu`.
- [ ] `UV_PROJECT_ENVIRONMENT=.scratch/venv-gfx uv sync --extra gfx101x` gives `torch.__version__ == "2.12.0+rocm7.14.0a20260612"`.
- [ ] `uv sync --extra cpu --extra gfx101x` is rejected by uv (conflict).
- [ ] With the gfx venv and `ALTLAS_DEVICE=cuda`, `get_pytorch_device()` returns `cuda` and `torch.backends.cudnn.enabled` is `False`.
- [ ] Same venv, a 20-step LSTM + Adam train on GPU matches the same train on CPU within 1e-4 (final loss), seeded identically.
- [ ] Same venv with `ALTLAS_MIOPEN=1` and `ALTLAS_DEVICE=cuda`: `torch.backends.cudnn.enabled` stays `True`.
- [ ] With the cpu venv and no env, `get_pytorch_device()` returns `cpu` and `cudnn.enabled` is untouched (CPU path unchanged).
- [ ] `.devcontainer/Dockerfile` and `devcontainer.json` install CPU torch (`--extra cpu` or equivalent); `uv sync --frozen --extra cpu` succeeds against the new `uv.lock`.
- [ ] `AGENTS.md` and `README.md` carry the "AMD Navi 10 GPU (optional)" section.
- [ ] Existing behaviour of T-002's probe and fallback stays unchanged (`utils.py` lines for the unset/cpu/unrecognised/no-GPU/probe-failure paths).
- [ ] Ticket Status set to `landed (<branch>, YYYY-MM-DD)` in the lane's commit.
- [ ] Gates pass: `uv run python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py` and `node scripts/check-tickets.mjs`.

Do NOT run the test harness. Stage paths explicitly.
