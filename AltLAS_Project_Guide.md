# AltLAS Project Guide

## Purpose

AltLAS exists to explore autonomous code-solving systems that can improve through execution feedback, memory, evaluation, and model-assisted iteration. It is no longer accurate to describe the project as a purely isolated learner. The codebase already contains a hybrid stack, and the design should acknowledge that directly.

## Two Tracks

AltLAS now has two legitimate tracks:

### 1. Research Track

Use reinforcement learning, replay, reward shaping, and structured tasks to study whether a compact generator can improve through repeated attempts and tool feedback.

### 2. Systems Track

Build a modern coding agent that routes work across local models, hosted models, deterministic tools, and evaluators while remaining measurable, debuggable, and safe.

These tracks should share infrastructure, but they should not be confused with each other.

## Design Principles

- **Capability-based, not model-branded**: architecture should be written around planner, coder, critic, judge, retriever, and safety roles.
- **Provider-agnostic**: local inference, hosted APIs, and future engines should fit behind one routing layer.
- **Eval-first**: benchmark and regression results matter more than intuition about model quality.
- **Local-first where practical**: private and inexpensive loops should prefer local models and deterministic tooling.
- **Safe execution by default**: generated code must run in constrained environments with explicit policy checks.
- **Clear separation of concerns**: orchestration, generation, execution, scoring, memory, and reporting remain modular.

## What Exists Today

- a PyTorch generator and training loop
- benchmark tasks in JSON
- execution, scoring, safety, and duplicate detection
- report generation and run persistence
- an LLM provider abstraction aimed at local engines such as LM Studio and vLLM

## What AltLAS Should Become

The intended 2026 design is a layered system:

1. interface and orchestration
2. model router
3. planning and coding agents
4. deterministic tools and sandboxed execution
5. evaluators and judges
6. memory, traces, and reports
7. learning and distillation loops fed by evals

## Practical Rule

If a feature improves observability, execution safety, or evaluation quality, it is foundational work. If a feature adds model complexity without improving measurement, it is probably premature.
