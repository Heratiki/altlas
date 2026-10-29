# Lane brief template (codex and agy)

codex and agy do not load the `unlazy` skill. This template copies what it
enforces: gates written before the work, each gate a command that runs with
no setup, evidence pasted from real output, and no gate deleted to pass.
Copy it to `.scratch/lanes/<id>/BRIEF.md` in the main tree and fill every
`<...>`. Write the ledger to `.scratch/lanes/<id>/GATES.md`. Start the lane
with the `lanes` tool, `prompt: "@<abs>/.scratch/lanes/<id>/BRIEF.md"`.

---

# Lane <id>: <ticket title>

You are a lane in /home/heratiki/Source/altlas (AltLAS, a Python and PyTorch code-generation learner with benchmark tasks and model routing). Work only in
`<absolute worktree path>` on branch `lane/<id>`. Do not stop to ask; if a
ruling is missing, write the question in REPORT.md and stop that part.

## Read first

- `<abs>/docs/tickets/<T-NNN-*.md>` — your contract.
- `<abs>/CLAUDE.md` — conventions and the "do not change" rules.
- `<abs>/.scratch/lanes/<id>/GATES.md` — your gates.

## You own

- `<path>` ... (touch nothing else; other lanes run at the same time)

## Never

- Run the test harness (python benchmark_runner.py) or any docker/server command.
- `git add -A`, `git push`, merge to main, `pkill -f`.
- Install dependencies in the worktree.
- Delete or weaken an existing test assertion.
- Change a gate's CHECK or EXPECT. If a gate cannot be met, write
  `ABANDON: G<n> <reason>` at column 1 and say so in REPORT.md.

## Gates (copy of GATES.md; the orchestrator re-runs every CHECK)

The ledger uses the unlazy format, so `gate-check` can run it
(`node ~/.claude/skills/unlazy/scripts/gate-check.mjs <GATES.md>`):

- A gate starts at column 1 with `- [ ] G<n>: <outcome>`.
- `CHECK:`, `EXPECT:` and `EVIDENCE:` are indented two spaces under it.
- A gate passes only when CHECK exits 0 AND its output contains EXPECT
  (plain substring, or `/regex/flags`).

Fixed gates for every code lane:

    - [ ] G1: The commit changes only owned paths.
      CHECK: git diff --name-only main...HEAD | grep -vxE '<owned path regex>' | wc -l
      EXPECT: /^0$/m
      EVIDENCE: pending

    - [ ] G2: Static gates pass.
      CHECK: python -m py_compile runner.py altlas_ui.py benchmark_runner.py core/*.py agent_core/*.py evaluator/*.py guardian/*.py reinforcer/*.py task/*.py memory/*.py && node scripts/check-tickets.mjs && echo STATIC-OK
      EXPECT: STATIC-OK
      EVIDENCE: pending

    - [ ] G3: The ticket Status is landed.
      CHECK: grep -m1 '^Status:' <ticket path>
      EXPECT: Status: landed (lane/<id>, <date>)
      EVIDENCE: pending

Lane gates (one per acceptance criterion and ruling):

    - [ ] G10: <...>
      CHECK: <...>
      EXPECT: <...>
      EVIDENCE: pending

## Finish

1. Commit on `lane/<id>` with explicit paths. Message: subject, blank line,
   body, blank line, then `Co-Authored-By: <your model> <noreply@...>` and
   any session trailer the orchestrator gives.
2. Write `<abs>/.scratch/lanes/<id>/REPORT.md`: branch, commit hash, each gate
   with its EVIDENCE, open questions, and any script or procedure worth
   keeping for reuse.
3. Stop. The orchestrator verifies and merges.
