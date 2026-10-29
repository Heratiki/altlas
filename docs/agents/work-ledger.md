# Work ledger

The orchestrator's queue of work. Update it when a lane starts, lands or
fails, and when the maintainer rules. Procedure:
`docs/agents/external-lanes.md`. Models: `docs/agents/model-register.md`.

## Before any wave

1. `~/.claude/lane-usage.sh`; log the line (memory `lane-usage-tally`,
   "Pool readings"). Baseline 2026-10-29T15:10Z: claude 5h 5% wk 62%, codex 5h 2% wk 12%,
   gemini 5h 100% wk 86% (agy full), zai wk 31%, go wk 11%.
2. Check the model lists (`codex debug models`, `agy models`, the Claude
   models in the session) against the register.
3. If this repo uses external lanes: the `lanes` tool `list` (and the tmux
   windows `codex`, `agy` where tmux is installed and `tmux_session` is set).

## Next waves

| Ticket | Route | State | Next |
|---|---|---|---|
| (none) | | | No ticket is open. The maintainer chooses the next direction (see "Rulings to ask"). |

Wave order: a file two tickets need is changed by the first ticket's lane,
merged, then the second lane starts.

## Rulings to ask

One line each, with the recommended option. Ask 1-2 per round
(AskUserQuestion).

- AltLAS direction (2026-10-29 review: the from-scratch LSTM + single-sample
  REINFORCE design gave 0 successes in ~8,600 GPU attempts). Options:
  (a) rebuild as an LLM-guided evolution loop (FunSearch/AlphaEvolve style:
  local llama-server mutator, existing executor/scorer as checker, MAP-Elites
  archive), (b) pretrained small code LLM + GRPO/LoRA, (c) keep the LSTM and
  fix task input, batching and the no-output reward. The maintainer said
  "not now" to a plan for (a). Recommended when asked: (a).

## Play / manual checks for the maintainer

One line each: what to check, where, and which ticket it closes.

- Open the devcontainer in VS Code once (`.devcontainer`, context `..`) to
  confirm `uv sync --frozen --extra cpu` in postCreate works (T-001/T-004).
  The orchestrator built the image with docker; VS Code itself was not tried.

## Done waves

Newest first: `## Wave N — <date>, <pushed at sha | NOT pushed>` and one line
per ticket.

## Wave 1 — 2026-10-29, pushed at aeed2d1

- T-001 Python 3.13 + uv packaging (pyproject, uv.lock, devcontainer, docs). Landed 9973486.
- T-002 safe device selection: CPU default, ALTLAS_DEVICE=cuda|auto with kernel probe. Landed 6fbfa27.
- T-003 llama.cpp provider (router at localhost:8080), split role models, all roles local,
  token-budget truncation reported as an error. Landed 54126f7.
- T-004 optional gfx101x GPU extra (cpu/gfx101x conflicting extras), MIOpen off on gfx101x. Landed 27db339.
- T-005 gfx101x extra moved to AMD multi-arch build (torch 2.14.0+rocm10.1.0a20260822). Landed aeed2d1.
- Full benchmark run at 54126f7: no crash, 0/10 pass (untrained learner; benchmark_runner
  never calls the LLM). Not a green run; /ship stopped at step 2 earlier in the session.
