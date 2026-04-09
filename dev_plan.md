# AltLAS Development Plan (2026 Refresh)

## Current Position

AltLAS has more infrastructure than the original project docs admit. The repository already includes a modular training loop, scoring, execution, duplicate detection, reporting, provider integration, and multi-language scaffolding. The main weakness is no longer “can the system run?” but “can it be measured, routed, and improved reliably?”

The design gap is now larger than the implementation gap. The docs assumed a 2025-era bootstrapping learner. The code and the model ecosystem have moved on.

## Strategic Goal

Turn AltLAS into a measurable hybrid coding-agent platform with a clear research lane for reinforcement learning experiments.

## Immediate Priorities

### 1. Evaluation Foundation

- add an automated benchmark runner over `task/benchmarks/`
- record pass/fail, score, latency, and error-class metrics
- create a regression baseline before changing architecture further
- stop relying on ad hoc long runs as the main signal

### 2. Execution Hardening

- enforce per-task timeout, memory, and CPU limits in `evaluator/executor.py`
- improve error classification so reports stop collapsing into `Unknown Error`
- make language-specific execution paths explicit and testable
- strengthen sandbox assumptions before increasing autonomy

### 3. Architecture Modernization

- formalize a model router instead of a single provider preference list
- support role-based routing: planner, coder, critic, judge, embedding, safety
- distinguish local, hosted, and specialist model tiers in config and docs
- keep provider integration behind stable interfaces

### 4. Research Loop Modernization

- re-baseline the RL generator against the abstract-token vocabulary
- add smaller, measurable benchmark ladders before attempting harder tasks
- use traces and eval failures to drive replay, hints, and future distillation
- keep the RL lane experimental until it can beat deterministic baselines on simple tasks

### 5. Testing and Maintenance

- add a real `tests/` package for loader, executor, scorer, and config parsing
- add backward-compat checks for the modularized runner flow
- treat docs, benchmarks, and config examples as first-class maintenance targets

## Near-Term Sequence

1. build the eval harness
2. harden the executor
3. clean up error taxonomy and reporting
4. implement model routing
5. rerun the easiest benchmarks and establish a fresh baseline

## Out of Scope for Now

- chasing larger model architectures without better evals
- productizing multi-agent behavior before sandboxing is solid
- claiming generalized coding ability from single-task experimental runs
