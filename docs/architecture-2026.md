# AltLAS Architecture (2026)

## Why This Refresh Exists

AltLAS was originally framed around a much narrower model environment. That is outdated. As of April 2026, current model ecosystems include stronger local open models, longer context windows, multimodal inputs, better function calling, cheaper hosted reasoning, and better deployment paths for local inference. Google's Gemma 4 release is a concrete example of that shift: Google describes Gemma 4 as purpose-built for advanced reasoning and agentic workflows, available under Apache 2.0, and supporting text, image, and audio input with context windows up to 256K.

This changes the architecture. AltLAS should not assume one learner, one model, one loop.

## Target Architecture

### 1. Interface Layer

- CLI and terminal UI
- future API or web entrypoints
- task submission, run control, and status display

### 2. Orchestration Layer

- run planning
- task decomposition
- retries, cancellation, and state transitions
- trace capture for every attempt

### 3. Model Routing Layer

- chooses local vs hosted vs specialist models
- routes by capability, latency, privacy, and cost
- exposes stable roles such as planner, coder, critic, judge, and embedding model

### 4. Tooling and Execution Layer

- sandboxed code execution
- file, shell, and retrieval tools
- language-specific adapters
- policy enforcement and guardrails

### 5. Evaluation Layer

- deterministic checks
- benchmark scoring
- safety judgments
- regression comparisons against prior runs

### 6. Learning Layer

- reinforcement learning experiments
- replay and trace reuse
- future distillation and policy improvement
- benchmark-driven tuning rather than anecdotal iteration

### 7. Memory and Observability Layer

- attempt history
- fingerprints
- reports and metrics
- run artifacts and future dataset exports

## Current Implementation Mapping

The repository already contains a partial version of this architecture:

- `runner.py` and `core/`: orchestration and run lifecycle
- `agent_core/`: generator and model logic
- `evaluator/`: execution
- `reinforcer/`: scoring and feedback
- `guardian/`: safety checks
- `memory/`: reports and persisted state
- `llm_provider.py`: early provider abstraction

## Main Gaps

- no real model router yet
- executor resource controls are incomplete
- eval automation is weak
- the research loop and practical agent loop are still mixed together

## Architectural Rule

Every major feature should answer three questions:

1. what role does it serve
2. how is it measured
3. how does it fail safely

## References

- Google Gemma 4 announcement: `https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/`
- Gemma models overview: `https://ai.google.dev/gemma/docs`
- OpenAI API models and agent tooling docs: `https://developers.openai.com/api/docs/models/compare`
