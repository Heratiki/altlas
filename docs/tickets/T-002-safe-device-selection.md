# T-002: Safe PyTorch device selection

Status: open
Opened: 2026-10-29 by the orchestrator, from maintainer ruling R1 after the project's 18-month idle.
Depends on: none

## What happens today

- `utils.py:7-20` `get_pytorch_device()` returns `cuda` whenever `torch.cuda.is_available()` is true, with no test that a kernel runs.
- On this host (AMD RX 5700 XT, gfx1010) the installed torch 2.9.1+rocm6.4 reports CUDA (HIP) available but has no gfx1010 kernels. The first op crashes with `torch.AcceleratorError: HIP error: invalid device function` at `agent_core/generator.py:220` (`param.data.mean().item()`), reached from `benchmark_runner.py:186`.
- Callers: `runner.py:38,181` and `benchmark_runner.py:26,185`; both take the result as `device` and pass it to `CodeGenerator(..., device=device)`. `utils.py:24-25` also calls it under `__main__`.

## Rulings (2026-10-29)

- R1: Torch runs on CPU by default everywhere. A GPU is opt-in, with the ROCm override `HSA_OVERRIDE_GFX_VERSION=10.3.0` documented as unreliable on Navi 10. Reason: the model is about 683k parameters and CPU is fine. Rejected: auto-selecting the GPU when available; Vulkan torch (unmaintained, inference-only).

## Settled design

- `get_pytorch_device()` reads env `ALTLAS_DEVICE` (case-insensitive, stripped). Unset or any value other than `cuda` or `auto` (including `cpu`) returns `torch.device("cpu")`; an unrecognised non-empty value logs a warning.
- For `cuda` or `auto`: if `torch.cuda.is_available()` is false, log and return CPU. Otherwise, inside try/except Exception, run a tiny kernel on the device (e.g. `torch.ones(8, device=dev).mean().item()`, and `torch.cuda.synchronize()`); on success return the cuda device; on any exception log a warning that names the error and return CPU.
- Docstring documents `ALTLAS_DEVICE` values, the default, and that `HSA_OVERRIDE_GFX_VERSION=10.3.0` is a ROCm override that is unreliable on gfx1010 (Navi 10).
- Return type and signature are unchanged, so `runner.py` and `benchmark_runner.py` need no edit.

## Owns

- `utils.py`

## Out of scope

- `runner.py`, `benchmark_runner.py`, `agent_core/generator.py`, any config.ini key for the device.
- Packaging (T-001), the LLM provider (T-003).
- Installing a ROCm torch.

## Acceptance

- [ ] `env -u ALTLAS_DEVICE python -c "from utils import get_pytorch_device as g; print(g())"` prints `cpu`.
- [ ] `ALTLAS_DEVICE=cpu` prints `cpu`.
- [ ] `ALTLAS_DEVICE=auto python -c "from utils import get_pytorch_device as g; print(g())"` on this host exits 0, prints `cpu`, and the log contains a WARNING naming the failed kernel (not a traceback).
- [ ] `ALTLAS_DEVICE=cuda` behaves the same as `auto` on this host.
- [ ] `ALTLAS_DEVICE=bogus` prints `cpu` with a warning.
- [ ] `grep -n "ALTLAS_DEVICE\|HSA_OVERRIDE_GFX_VERSION" utils.py` shows both in the docstring.
- [ ] `grep -rn get_pytorch_device --include=*.py .` shows the callers listed above and none requiring a signature change.
- [ ] Ticket Status set to `landed (<branch>, YYYY-MM-DD)` in the lane's commit.
- [ ] Gates pass: `python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py` and `node scripts/check-tickets.mjs`.

Do NOT run the test harness. Stage paths explicitly.
