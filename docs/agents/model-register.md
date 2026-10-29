# Model register

Every model the lanes use (Claude, codex, agy): what it can do here, and how
fast. The orchestrator owns this file and keeps it current from outcomes.
Pool costs come from `~/.claude/lane-usage.sh` readings before and after a
lane, never from token counts.

## When to update

- **After every lane:** one Evidence line under the model: date, task class,
  effort, wall time, pool delta, result (first time / N follow-ups / failed).
- **New model seen** (`codex debug models`, `agy models`, the session's
  Claude models): one web search for its release notes, a row marked `?`
  (UNPROVEN), one low-risk lane before any routing rule uses it.
- **Whose fault:** a model-fault failure (clear brief, working environment)
  re-runs at once on the next better model and moves that class up now. An
  orchestrator-fault failure is fixed in the brief and re-run on the same
  model, with no rating change. Two cheap successes move a class down.
- **Stark contradiction:** one web search for known issues, then change the
  rating and `docs/agents/external-lanes.md`; tell the maintainer in the wave
  report.

Ratings: **A** lands the class first time, **B** lands with follow-ups,
**C** fails, **?** unproven.

## Claude (in-process subagents)

| Model | Good at | Rating by class | Speed | Notes |
|---|---|---|---|---|
| Opus | Design tickets; the orchestrator | ? | | Never the default lane. |
| Sonnet | Bounded tickets, ticket prose, verification | ? | | Default lane model. |
| Haiku | Short read-only checks | ? | | Not for long procedures. |

### Evidence

## codex (lanes runner `codex`)

| Model | Good at | Rating | Speed | Notes |
|---|---|---|---|---|
| | | | | |

### Evidence

## agy (lanes runner `agy`)

| Model | Good at | Rating | Speed | Notes |
|---|---|---|---|---|
| Gemini Flash (high) | Research, audits, drafts (Flash first for simple work) | ? | | Ends turns early; expect a nudge. |

### Evidence

## opencode (lanes runner `opencode`)

Z.ai models only (`zai-coding-plan/`) until 2026-11-21 (maintainer ruling
2026-10-29). Pool: Z.ai.

| Model | Good at | Rating | Speed | Notes |
|---|---|---|---|---|
| | | | | |

### Evidence

| Ended | Lane | Kind | Model | Effort | Task | Pool | Outcome | Wall (min) | 5h (pp) | Week (pp) | Marks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-29T13:00Z | agent-a8dd436c | claude | claude-sonnet-5-5 | low | verify | claude | completed | 0.8 | 0 | 0 | split |
| 2026-10-29T13:06Z | agent-acb4dcfc | claude | claude-sonnet-5-5 | low | write | claude | completed | 1.7 | 0 | 0 | split |
| 2026-10-29T13:06Z | rocm-gfx1010 | codex | gpt-6.1-terra | medium | ? | codex | done | 1.2 | 1 | 0 | partial |
| 2026-10-29T13:12Z | agent-a10f73a6 | claude | claude-sonnet-5-5 | medium | build | claude | completed | 5.8 | 0.3 | 0 | split, lane=agent-a10f73a65275d38a2, gates=16 |
| 2026-10-29T13:09Z | agent-a32c877f | claude | claude-sonnet-5-5 | medium | build | claude | completed | 2.3 | 0.2 | 0 | split, gates=11 |
| 2026-10-29T13:11Z | agent-ad14b95f | claude | claude-sonnet-5-5 | low | verify | claude | completed | 0.5 | 0 | 0 | split, lane=agent-a10f73a65275d38a2. |
| 2026-10-29T13:11Z | therock-gfx101x | codex | gpt-6.1-terra | medium | ? | codex | done | 0.6 | 1 | 0 |  |
| 2026-10-29T13:13Z | agent-a7181d50 | claude | claude-sonnet-5-5 | low | verify | claude | completed | 2.6 | 0.2 | 0 | split, lane=agent-a32c877f7ea9f595f. |
| 2026-10-29T13:17Z | agent-a13e30e7 | claude | claude-sonnet-5-5 | medium | build | claude | completed | 6.8 | 0.3 | 0 | split, gates=11 |
| 2026-10-29T13:15Z | agent-a3fc1a92 | claude | claude-sonnet-5-5 | low | verify | claude | completed | 1 | 0 | 0 | split, lane=agent-a13e30e7aa3033006. |
| 2026-10-29T13:16Z | agent-aea50b1e | claude | claude-sonnet-5-5 | low | verify | claude | completed | 1.9 | 0 | 0 | split, lane=agent-a10f73a65275d38a2. |
| 2026-10-29T13:18Z | agent-a5fc3c1a | claude | claude-sonnet-5-5 | low | verify | claude | completed | 0.5 | 0 | 0 | split |
| 2026-10-29T13:23Z | agent-a018cc81 | claude | claude-sonnet-5-5 | low | write | claude | completed | 1.1 | 0 | 0 | split |
| 2026-10-29T13:27Z | agent-a0a5711a | claude | claude-sonnet-5-5 | medium | build | claude | completed | 3 | 0 | 0 | split, gates=14 |
| 2026-10-29T13:30Z | agent-a0d26209 | claude | claude-sonnet-5-5 | low | verify | claude | completed | 1.4 | 0 | 0 | split, lane=agent-a0a5711a343bb9166. |
| 2026-10-29T14:09Z | agent-a7a68d7c | claude | claude-sonnet-5-5 | low | write | claude | completed | 1.2 | 0.1 | 0 | split |
| 2026-10-29T14:12Z | agent-a1533722 | claude | claude-sonnet-5-5 | medium | build | claude | completed | 2.8 | 0 | 0 | split, gates=15 |
| 2026-10-29T14:13Z | agent-ac2ac6e5 | claude | claude-sonnet-5-5 | low | verify | claude | completed | 1 | 0 | 0 | split, lane=agent-a1533722df52ec416. |
<!-- generated from model-register.runs.json; edit with register.py, not by hand -->
