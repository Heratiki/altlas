# External lanes: codex, agy and opencode through the lanes tool

Set up 2026-10-29. The orchestrator drives codex and agy through the lane-panel
`lanes` tool (`mcp__lane-panel__lanes`), not through the `codex:` or `agy:`
plugins, and routes each lane to
the tool that is strongest at that kind of work. General method:
`~/.claude/multi-lane-orchestration.md`.

## Limits at the same time

| Kind | At most | Note |
|---|---|---|
| Claude subagent lanes | 5 | Disjoint owned paths. |
| codex lane | 1 | The `lanes` tool refuses a second one unless `force`. |
| agy lane | 1 | The `lanes` tool refuses a second one unless `force`. |
| opencode lane | 1 | The `lanes` tool refuses a second one unless `force`. |
| Test harness run | 1 | Only `suite-verifier` or the orchestrator. |

No two lanes, of any kind, own the same file at the same time.

## The lanes tool

`action`: `start`, `list`, `tail`, `stop`. `start` takes `runner` (`codex` |
`agy` | `opencode`), `lane`, `label`, `prompt` (`"@<abs>/BRIEF.md"`), and `model`,
`effort`, `task`, `sandbox`, `resume`, `force`. The tool runs in the session's
repo. For a lane in its own worktree, use the CLI with the worktree as the
repo:

```bash
python3 ~/.claude/plugins/marketplaces/heratiki-mods/bin/lane.py start <worktree> codex <id> "<label>" --model <m> --effort <e> -- @<abs>/BRIEF.md
```

The lane's files: `<repo>/.claude/lane-panel/lanes/<id>/` (`prompt.md`,
`out.log`, `result.md` = the final answer, `exit.json` = the end). A codex
fix round: `resume: <id>`. If a start is refused for a pool or the lane limit,
tell the maintainer; use `force` only on their go. If a CLI looks logged out,
ask first.

## tmux windows (alternative, only where tmux is installed)

Only when `command -v tmux` succeeds and `tmux_session` is not none. Session
`none`, windows `codex` and `agy`. Reuse them; make them only if
gone:

```bash
S=none; R=/home/heratiki/Source/altlas
tmux new-window -d -t "$S" -n codex -c "$R"
tmux new-window -d -t "$S" -n agy -c "$R"
```

zsh: write `"${S}:codex"`, never `$S:codex`. Start commands (full
permissions, maintainer's standing ruling):

```bash
none
none
```

The codex header must show full permissions. If a start is refused, ask the
maintainer to start it in the window.

## Routing by strength

| Work | Route |
|---|---|
| Ticket with written acceptance, bounded files | Claude `lane-dev`, Sonnet |
| Ticket with design decisions inside | Claude `lane-dev`, Opus |
| Hard or order-sensitive diagnosis; second implementation; long refactor | codex workhorse, high |
| Adversarial review of a risky lane before merge | codex (a different model from the builder) |
| Simple research, audits, drafts, bounded reading | agy Gemini Flash FIRST |
| Harder research after a measured Flash failure | agy Pro-class |
| Bounded code or research when the Claude, codex or agy pools are high | opencode, a `zai-coding-plan/` model |
| Tickets and ADR prose | Claude `ticket-writer` |
| Test runs | Claude `suite-verifier` |
| Verdicts: design rulings, play/visual pass-fail | the maintainer |

Current model names: `docs/agents/model-register.md`. Always pass the model
explicitly (`-m`, `--model`); config defaults go stale.

## One lane, start to end

1. `scripts/lane-worktree.sh add lane/<id>`. codex works only inside it; agy
   research writes only its output file.
2. Brief from `docs/agents/lane-brief-TEMPLATE.md` to
   `.scratch/lanes/<id>/BRIEF.md` (absolute paths), ledger to
   `.scratch/lanes/<id>/GATES.md`. Read the pools (`~/.claude/lane-usage.sh`).
3. Start it: the CLI above with the worktree as the repo (or the `lanes` tool
   for a lane that needs no worktree), `-- @<abs>/BRIEF.md`, model named.
4. Wait with `~/.claude/skills/watch/watch.sh file <lane dir>/exit.json`; read
   `result.md` and `REPORT.md`. Flash may end a turn early with nothing
   written: a fix round (`--resume`), once or twice.
5. tmux windows instead (only where tmux is installed): codex `/quit` the
   previous lane, wait for the shell prompt, start codex with `-C <worktree>`
   (never `/new` in codex; it moves into its own copy); agy `/new`. Then
   `tmux send-keys -t "${S}:codex" -l "Read <abs>/BRIEF.md and do exactly what it says. Do not stop to ask."`
   and `tmux send-keys -t "${S}:codex" Enter`; confirm with `capture-pane`.
   Wait with `watch.sh tmux codex <abs>/REPORT.md`. Answer a file-permission
   prompt with `1` Enter, a survey with `0` Enter.
6. On return the orchestrator runs every `CHECK:` itself from the worktree.
   A failed gate goes back to the same lane as one fix round. Then
   `lane-verifier`, merge, `scripts/lane-worktree.sh rm lane/<id>`.
7. Read the pools again; record the delta and the outcome in the register.

## Balance and sizing

- A pool above 80% of its 5-hour window, or above 85% of its week with more
  than a day left, takes no new lane.
- Claude's weekly pool is shared with the orchestrator: above 70%, research
  goes to agy and diagnosis to codex first.
- Keep one codex and one agy lane busy when suitable work exists.
- Start at the lowest model and effort that has worked for the class; raise
  one step only after a measured model-fault failure.

## opencode (maintainer ruling 2026-10-29)

- Use only Z.ai models (`zai-coding-plan/<model>`) until 2026-11-21. They
  count on the Z.ai pool. Do not use `opencode-go/` models before that date;
  after it, ask the maintainer.
- opencode has no default model. Name `model` on every start, or the runner
  refuses it.
- opencode is headless only (no tmux window). The command the runner uses:
  `opencode run --standalone --auto --format json --model <provider/model>`.
