# T-005: Move the gfx101x extra to AMD's multi-arch gfx1010 build

Status: open
Opened: 2026-10-29 by the orchestrator, from maintainer ruling R9 and host tests of 2026-10-29.
Depends on: T-004 (gfx101x extra, MIOpen-off logic), T-001 (uv packaging)

## What happens today

- `pyproject.toml:17-23` defines the `gfx101x` extra with the June 2026 per-family pins (`torch==2.12.0+rocm7.14.0a20260612`, `rocm[libraries]`, `rocm-sdk-core`, `rocm-sdk-libraries-gfx101x-dgpu`, `triton==3.7.0+gitb4e20bbe.rocm7.14.0a20260612`). `pyproject.toml:36-39` defines the explicit index `therock-gfx101x` (`https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/`). `pyproject.toml:42-49` routes torch (both extras) and the four ROCm packages to it. `pyproject.toml:29` declares `conflicts` between `cpu` and `gfx101x`; `pyproject.toml:16` is the `cpu` extra.
- `AGENTS.md:28-39` and `README.md:42-53` ("AMD Navi 10 GPU (optional)") say the build is "an unmaintained June 2026 AMD nightly (`torch 2.12.0+rocm7.14.0a20260612`, TheRock gfx101X index)" (`AGENTS.md:37`, `README.md:51`) and give a "about 5 GB download" (`AGENTS.md:33`, `README.md:47`).
- T-004 ("Confirm before implementing", `docs/tickets/T-004-gfx1010-gpu-profile.md`) left open whether a multi-arch AMD release covers gfx1010, and said AMD's per-family builds stop at 2026-06-12. That is now answered (Rulings below). The index in use is therefore frozen and will not get newer builds.
- `utils.py:60-69` turns `torch.backends.cudnn.enabled` off when `gcnArchName` starts with `gfx101` and `ALTLAS_MIOPEN` is not `1`.
- Evidence (2026-10-29, from the working reference project `/home/heratiki/Source/CUDA`: its `pyproject.toml`, `uv.lock`, `README.md`):
  - Index `https://rocm.nightlies.amd.com/whl-multi-arch/` holds multi-arch gfx1010 builds up to 2026-08-22 (torch 2.8.0 to 2.15.0a0). It ships a base torch wheel plus a separate gfx1010 device wheel (`amd-torch-device-gfx1010`) and a gfx1010 ROCm device wheel (`rocm-sdk-device-gfx1010`).
  - Pins, all from that index, cp313 linux: `torch==2.14.0+rocm10.1.0a20260822`, `amd-torch-device-gfx1010==2.14.0+rocm10.1.0a20260822`, `rocm==10.1.0a20260822`, `rocm-sdk-core==10.1.0a20260822`, `rocm-sdk-libraries==10.1.0a20260822`, `rocm-sdk-device-gfx1010==10.1.0a20260822`, `triton==3.8.0+git675c5987.rocm10.1.0a20260822`. `rocm-bootstrap` 0.3.0 also comes from this index.
  - Tests on the RX 5700 XT with this build: matmul (rocBLAS), conv2d, LayerNorm, Embedding and LSTM forward match CPU. LSTM backward + Adam FAILS with MIOpen on (the same CK `No CK_AMD_GPU_GFX* macro defined` error as the June build) and matches CPU with MIOpen off (loss 4.5864 -> 0.5628, diff 1.8e-7). `gcnArchName` is still `gfx1010:xnack-`.
  - The download is about 1.4 GB. Saved copies of the wheels with `SHA256SUMS` are in `/home/heratiki/Source/CUDA/wheels/multi-arch-2026-08-22/` (offline fallback; not part of this repo).

## Rulings (2026-10-29)

- R9: Move AltLAS's `gfx101x` extra to AMD's multi-arch gfx1010 build (index `https://rocm.nightlies.amd.com/whl-multi-arch/`, the pins above). Reason: AMD stopped per-family builds after 2026-06-12; the multi-arch build is newer, about 1.4 GB instead of about 5 GB, and passes the same tests with MIOpen off on the target GPU. Rejected: staying on the frozen per-family index; a global `prerelease = "allow"` (pin each pre-release explicitly, as in T-004).
- T-004's `utils.py` MIOpen-off logic stays (maintainer evidence above: the CK failure persists on this build). Rejected: removing or loosening it.
- The extra keeps its name `gfx101x` and its `conflicts` with `cpu`; the `cpu` extra is unchanged.

## Settled design

1. `pyproject.toml`:
   - Replace the `therock-gfx101x` index with an explicit index `therock-multi-arch` at `https://rocm.nightlies.amd.com/whl-multi-arch/`.
   - Replace the `gfx101x` extra pins with the seven pins above (`torch`, `amd-torch-device-gfx1010`, `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1010`, `triton`). Keep the comment that transitive pre-releases are pinned so no global prerelease setting is needed.
   - `[tool.uv.sources]`: `torch` keeps its two entries (`pytorch-cpu` with `extra = "cpu"` and the Linux marker; `therock-multi-arch` with `extra = "gfx101x"`). Every other multi-arch package (`amd-torch-device-gfx1010`, `triton`, `rocm`, `rocm-bootstrap`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1010`) gets a source with `index = "therock-multi-arch"` and `extra = "gfx101x"`. Remove the `rocm-sdk-libraries-gfx101x-dgpu` entry.
   - Keep `index-strategy = "unsafe-best-match"`, `conflicts` and the `cpu` extra as they are.
2. Re-lock `uv.lock`. The cpu-path entries (torch 2.15.0+cpu and its dependencies) must be unchanged against main.
3. `AGENTS.md` and `README.md` "AMD Navi 10 GPU (optional)" sections: state the build date (2026-08-22 AMD multi-arch nightly), `torch 2.14.0+rocm10.1.0a20260822` on ROCm 10.1.0a20260822, that the gfx1010 kernels come in a separate device wheel (`amd-torch-device-gfx1010`), the download (about 1.4 GB), that these are AMD nightlies without official support, and keep the MIOpen note and `ALTLAS_MIOPEN=1` unchanged. Remove "unmaintained June 2026", "TheRock gfx101X index" and "about 5 GB".
4. `utils.py` does not change unless a check proves it must.

## Confirm before implementing

- `rocm-bootstrap` (0.3.0) comes from the multi-arch index in the reference project, which gives it a source but no pin. Whether AltLAS also needs no pin for it is for the lane to check against the resolution; if it forces a global prerelease setting, stop and ask.
- If a check shows `utils.py` must change (for example `gcnArchName` no longer starts with `gfx101` on some host), stop and report with the measured value; do not edit `utils.py` without a ruling.
- Which wheels the `uv sync --extra gfx101x` resolution needs beyond the seven pins is the lane's reading of uv; add a pin only for a pre-release that uv demands.

## Owns

- `pyproject.toml`
- `uv.lock`
- `AGENTS.md`
- `README.md`
- `docs/tickets/T-005-gfx101x-multi-arch.md` (Status line only)
- `utils.py` (only if a check proves it must; see "Confirm before implementing")

## Out of scope

- Newer multi-arch builds than 2026-08-22, other GPU families, other Python versions than 3.13.
- Changes to the `cpu` extra, the cpu index, `.devcontainer/**` or the Dockerfile.
- Vendoring or committing wheels; the offline copies in `/home/heratiki/Source/CUDA/wheels/` stay outside the repo.
- Making the GPU the default or changing `ALTLAS_DEVICE` semantics (T-002).
- Changing the MIOpen-off logic (`utils.py:60-69`).
- Running the full benchmark suite.

## Acceptance

Use `UV_PROJECT_ENVIRONMENT=/home/heratiki/Source/altlas/.scratch/venv-gfx` for the gfx101x venv (keep `.scratch/` untracked). The gfx101x sync downloads about 1.4 GB.

- [ ] `uv sync --frozen --extra cpu` gives `torch.__version__ == "2.15.0+cpu"`.
- [ ] The cpu-path entries of `uv.lock` are unchanged against main (the diff of `uv.lock` shows no change to the `torch` 2.15.0+cpu entry and its cpu-path dependencies).
- [ ] `UV_PROJECT_ENVIRONMENT=.scratch/venv-gfx uv sync --extra gfx101x` gives `torch.__version__ == "2.14.0+rocm10.1.0a20260822"` and `amd-torch-device-gfx1010` installed at `2.14.0+rocm10.1.0a20260822`.
- [ ] With that venv and `ALTLAS_DEVICE=cuda`, `get_pytorch_device()` returns `cuda` and `torch.backends.cudnn.enabled` is `False`.
- [ ] Same venv: a 20-step LSTM + Adam train on GPU matches the same train on CPU within 1e-4 (final loss), seeded identically.
- [ ] Same venv with `ALTLAS_MIOPEN=1` and `ALTLAS_DEVICE=cuda`: `torch.backends.cudnn.enabled` stays `True`.
- [ ] `uv sync --extra cpu --extra gfx101x` is rejected by uv (conflict).
- [ ] `grep -n "therock-gfx101x\|v2-staging" pyproject.toml AGENTS.md README.md` finds nothing (checked against a known positive: the same grep on `git show main:pyproject.toml` finds lines).
- [ ] `AGENTS.md` and `README.md` carry the updated section (build date, versions, separate device wheel, about 1.4 GB, no official support, MIOpen note unchanged).
- [ ] `utils.py` is unchanged (`git diff main -- utils.py` is empty), or the change is listed under a ruling.
- [ ] Ticket Status set to `landed (<branch>, YYYY-MM-DD)` in the lane's commit.
- [ ] Gates pass: `uv run --extra cpu python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py` and `node scripts/check-tickets.mjs`.

Do NOT run the test harness. Stage paths explicitly.
