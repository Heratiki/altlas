# AltLAS Model Strategy

## Principle

AltLAS should choose models by role and constraints, not by whichever model is currently fashionable.

## Model Tiers

### Tier 0: Deterministic Tools

Use tools first when they are enough:

- code execution
- file inspection
- syntax checks
- structured scoring
- benchmark comparison

### Tier 1: Local General Models

Use local models for cheap, private, high-iteration loops:

- draft generation
- local repair suggestions
- summarization of recent traces
- benchmark triage

This is where current open models such as Gemma 4-class systems make a meaningful difference compared with the repository's original design assumptions.

### Tier 2: Hosted Frontier Models

Use stronger hosted models selectively for:

- planning
- complex critique
- hard bug repair
- synthesis across long traces
- dataset and rubric generation

These models should be gated by eval value, privacy policy, and cost.

### Tier 3: Specialist Models

Use specialist models or endpoints for:

- embeddings and retrieval
- safety and moderation
- speech or multimodal understanding
- ranking and judging

## Role-Based Routing

The router should expose explicit roles:

- `planner`
- `coder`
- `critic`
- `judge`
- `retriever`
- `embedding`
- `safety`

Each role may map to a different provider or local engine.

## Current Repository Fit

Today the repository has a basic provider abstraction in `llm_provider.py` and configuration support in `config.ini`, but it still assumes a single provider preference path. That should evolve into:

- provider registry
- role-to-model mapping
- fallback policy
- local-first preference rules
- benchmark-based routing evaluation

## Routing Policy

Prefer the cheapest path that can meet the benchmark target:

1. deterministic tools only
2. local model plus tools
3. hosted planner or critic plus local execution
4. hosted end-to-end fallback

## Recommended Near-Term Work

- extend config from provider settings to role settings
- support multiple local engines cleanly
- record which model tier handled each attempt
- compare outcomes by tier in reports

## Anti-Pattern

Do not hard-code AltLAS around one flagship model. The design should survive model churn.
