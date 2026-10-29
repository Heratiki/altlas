# AltLAS handoff

## Last Updated

2026-10-29 (session 2). Landed T-001..T-005 (wave 1), pushed at aeed2d1. Open
tickets: none. Gates (py_compile, ticket linter) pass on main. Full benchmark
run: no crash, 0/10 pass (untrained learner). No deploy: the devcontainer image
was built as `altlas-image:t005-test` only.

2026-10-29 (session 1). Orchestration set up with /bootstrap-orchestration.

## Next session, first three things

1. Run `/session-start`. Expect a clean main at aeed2d1 (or later) and no lanes.
2. Ask the maintainer the AltLAS direction ruling in `docs/agents/work-ledger.md`
   ("Rulings to ask"). The current learner does not learn (see section 2 below).
   The next wave depends on that answer; do not start one before it.
3. Before any torch work, sync the venv you need: `uv sync --extra cpu` (default)
   or `uv sync --extra gfx101x` (RX 5700 XT GPU, then `ALTLAS_DEVICE=cuda`). The
   repo `.venv` was left on the gfx101x extra by a GPU training run.

## 1. What landed (wave 1)

- **Python 3.13 + uv** (T-001): `pyproject.toml`, `uv.lock`, `.python-version`,
  devcontainer installs from the lock. Every command is `uv run --extra cpu ...`
  (a bare `uv run` removes torch). Gates and harness in CLAUDE.md use that form.
- **Safe device** (T-002, `utils.py`): CPU unless `ALTLAS_DEVICE=cuda|auto`; a GPU
  kernel probe falls back to CPU instead of crashing.
- **llama.cpp provider** (T-003): `Provider = llamacpp`, router at
  `http://localhost:8080/v1`; advisor/analyst Qwen3-4B-Instruct, planner/critic
  Qwen3-8B then gemma-4-12B; all roles local; no vLLM default URL (port 8000 is
  another project on this host).
- **GPU** (T-004, T-005): extras `cpu` and `gfx101x` conflict. gfx101x = AMD
  TheRock multi-arch nightly `torch 2.14.0+rocm10.1.0a20260822` +
  `amd-torch-device-gfx1010`. On gfx101x, `utils.py` turns MIOpen off (RNN
  backward fails in a runtime Composable Kernel build); `ALTLAS_MIOPEN=1` keeps it.
  GPU training matches CPU within 1e-7 on the test LSTM.

## 2. Findings and open issues

- **The learner does not learn.** The model is a 683k-parameter LSTM with no task
  input (generation starts from `<SOS>` only), trained from random weights by
  single-sample REINFORCE. Code that runs and prints nothing gets 0.5 reward
  (`reinforcer/scorer.py:260`), so every attempt is `9`. A 30-minute GPU run went
  from attempt 21,201 to 29,797 with 0 successes; 0 LLM hints were used. The
  2025 state before that run is in `.scratch/memory-backup-2026-10-29/` (this
  host only, git-ignored).
- `benchmark_runner.py` tests only the local generator; it never calls the LLM roles.
- `/ship` needs a green full run; 0/10 cannot be green until the learner changes,
  or the maintainer redefines "green" for this harness.
- Unverified: the devcontainer opened in VS Code (only `docker build` was run);
  the async LLM provider path against a live server.
- Unlazy scopes with unmet gates (verifier ledgers, superseded, safe to archive on
  the maintainer's yes): `.unlazy/verify-T-001-r2` (G2, G8, G11 replaced by the
  verifier; G9 docker accepted on orchestrator evidence) and `.unlazy/verify-T-002`
  (V2 failed only on a re-run quirk of the lane's G11; checked by hand).
- Lane note: codex lanes ran in a sandbox with no network (DNS failed). Give
  research lanes that need the web another route, or check the sandbox first.

## 3. Outside this repo (this host only)

- `~/Source/CUDA`: a uv project for gfx1010 PyTorch (multi-arch build), with
  `gfx1010.setup()`, `check_gpu.py`, and saved wheels plus SHA256SUMS in
  `wheels/` (multi-arch 2026-08-22 and per-family 2026-06-12) for an offline install.
- `~/Source/rl-games`: single-file PPO; CartPole (mean 482.5) and LunarLander
  (mean 208.6) solved on the CPU.
- Removed this session: `~/Source/rocm_sdk_builder` (local changes saved in
  `~/Source/CUDA/old-attempts/`), `~/Source/ROCm-rocm-5.2.0`, `/opt/rocm_sdk_612`.

## What Is Currently In-Flight

Nothing. No lanes, worktrees or background runs.
