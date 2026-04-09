# AltLAS Evaluation Strategy

## Goal

Evals are the control system for the project. Without them, AltLAS will keep accumulating infrastructure without proving improvement.

## What Must Be Measured

For each benchmark or task family, record:

- pass/fail
- numeric score
- attempt count to success
- execution latency
- model tier used
- error class
- safety failures

## Benchmark Ladder

Start with small, deterministic tasks and move upward only after the lower rung is stable.

### Level 1: Syntax and Output

- print exact value
- basic arithmetic
- simple conditional
- simple loop

### Level 2: Structured Logic

- function definitions
- variable transformations
- list operations
- small refactors

### Level 3: Repair and Critique

- fix broken code
- explain failures
- improve low-quality solutions

### Level 4: Tool-Using Agent Tasks

- inspect files
- modify code
- rerun checks
- summarize changes

## Harness Requirements

The repository should gain an automated runner that:

- iterates over `task/benchmarks/`
- stores structured results
- compares against the previous baseline
- can be used in local development and CI

## Evaluation Rules

- do not trust a single long run
- do not compare models without fixed prompts and task sets
- do not change routing and scoring at the same time without capturing a baseline first

## Immediate Implementation Targets

- add a benchmark runner script
- create a machine-readable results file under `memory/reports/`
- classify executor failures into stable categories
- expose summary tables in the terminal UI and reports

## Success Definition

A change is meaningful only if it improves benchmark results, reduces cost or latency at equal quality, or improves safety and observability without hurting task performance.
