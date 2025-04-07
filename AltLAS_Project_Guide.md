# 🧠 AltLAS Project Guide

> *AltLAS: Loses A Lot, Succeeds*

## 🧾 Project Purpose

AltLAS is a sandboxed, emergent coding agent that learns how to solve programming tasks via trial, failure, reinforcement, and adaptation. It starts with no knowledge and gradually evolves intelligent behavior. It is *not* a traditional LLM wrapper—AltLAS is a bottom-up learner.

---

## 🧱 System Architecture (High-Level)

### 1. `runner.py`
- Main orchestrator loop
- Executes attempts, scores them, logs them
- Stops on success or loop-detection

### 2. `agent_core/`
- `generator.py`: Generates new code attempts
- Can start with rule-based output and evolve over time

### 3. `task/`
- Contains task definitions and goals
- Example: `hello_world.py` defines success as outputting "hello world"

### 4. `evaluator/`
- `executor.py`: Runs the generated code in a sandbox (Docker container)
- Captures stdout/stderr

### 5. `reinforcer/`
- `scorer.py`: Compares output to task criteria and returns reward (0.0–1.0)

### 6. `guardian/`
- `safety.py`: Filters unsafe or redundant code attempts before execution

### 7. `memory/`
- `logger.py`: Stores useful attempt data, deletes junk
- `fingerprints.py`: Prevents repeating the same failures endlessly

---

## 🧠 Key Philosophies

- Reward-worthy behavior is logged; junk is discarded.
- Fingerprinting prevents infinite loops of same code.
- LLMs (Claude, Gemini, etc.) are used **only if needed** and at AltLAS' request.
- Eventually, AltLAS may design its own language, DSLs, or problem abstractions.

---

## 🧪 Training Workflow

1. Generate code (`agent_core`)
2. Check for safety + novelty (`guardian`)
3. Execute safely (`evaluator`)
4. Score result (`reinforcer`)
5. Log if valuable (`memory`)
6. Repeat or halt

---

## 🧩 Future Modules

- `orchestrator/` – for routing multiple tasks
- `critic/` – for self-analysis and reward hacking prevention
- `vector_store/` – if vector embeddings are needed
- `model_team/` – LLM advisors once AltLAS matures

---

## 💭 Notes for Copilot / Roo Code

- Do not overuse classes unless needed. Start simple and composable.
- Always assume the agent is stupid until it earns abstraction.
- Modularize so future behavior evolution is painless.
- Use minimal dependencies outside of ONNX, numpy, Docker, and Python stdlib.

