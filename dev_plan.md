# AltLAS Development Plan

## Overview

This document outlines the development plan for improving the AltLAS reinforcement learning code generation system. The plan addresses several key issues identified during initial testing while maintaining the core philosophy of allowing the agent to develop its own approach to coding through emergent learning.

## Current Issues

- **Excessive Hinting**: The system provides hints far too frequently (~every 15 attempts)
- **Poor Generated Code Quality**: Best attempts show non-functional patterns (strings of "a" characters)
- **Model Initialization Concerns**: Initial weights may not be properly initialized
- **Training Signal Problems**: The model isn't learning effectively based on scores/rewards

## Implementation Plan

Each task below should be completed while preserving existing functionality. Tasks include a status indicator: 
- [ ] Not started
- [🔄] In progress
- [✓] Completed

### 1. Hint System Optimization

**Goal**: Reduce hint frequency to make it truly a last resort (e.g., 1 in 1000+ attempts)

- [✓] **1.1 Adjust Configuration Parameters**
  - [✓] Increase `stuck_check_window` from 15 to 1000-2000 attempts in `config.ini`
  - [✓] Increase `stuck_threshold` from 0.01 to 0.05-0.1 in `config.ini`
  - [✓] Document parameter meaning and effects in comments
  - [✓] Add `hint_probability_on_stuck` parameter
  - [✓] Add `max_consecutive_stuck_checks` parameter

- [✓] **1.2 Implement Progressive Hinting Logic in `runner.py`**
  - [✓] Add a consecutive non-improvement counter that increases when best score doesn't improve
  - [✓] Add randomization function (e.g., 1% chance of providing a hint when stuck is detected)
  - [ ] Implement exponential backoff for hint frequency (longer wait between hints) *Note: Partially done via consecutive checks, but true backoff not implemented yet.*

- [✓] **1.3 Create Hint Analytics**
  - [✓] Track and log when hints are requested and provided (via counters)
  - [ ] Measure impact of hints on score improvement
  - [✓] Add hint usage statistics to the Rich UI

### 2. Model Initialization and Weight Validation

**Goal**: Ensure proper model initialization to avoid pathological learning patterns

- [✓] **2.1 Implement Initialization Diagnostics**
  - [✓] Add logging of weight statistics in `model.py` (mean, std, min, max)
  - [ ] Create weight histogram visualization function
  - [✓] Add explicit initialization methods (Xavier/Glorot, Kaiming He, Orthogonal) in `model.py`

- [✓] **2.2 Create Weight Validation Functions**
  - [✓] Implement sanity checks for initial weight distribution in `generator.py` (`_validate_initial_weights`)
  - [✓] Add validation against known pathological patterns (near-zero std, large mean, NaN/Inf)
  - [ ] Create recovery mechanism for bad initialization states

- [✓] **2.3 Add Runtime Monitoring**
  - [✓] Track gradient statistics during learning (log gradient norm in `generator.py`)
  - [ ] Monitor weight changes over time to detect vanishing/exploding gradients
  - [ ] Add early warning for training instability

### 3. Training Signal Enhancement

**Goal**: Improve the learning signal to promote more effective exploration and learning

- [✓] **3.1 Enhance Reward Mechanism in `scorer.py`**
  - [✓] Add more nuanced scoring for early attempts (higher base score for errors)
  - [ ] Implement reward normalization to prevent scaling issues
  - [✓] Create difficulty-appropriate reward scales (using `difflib` for partial scores)

- [✓] **3.2 Adjust Learning Mechanisms in `generator.py`**
  - [✓] Add entropy regularization to encourage exploration
  - [✓] Implement gradient clipping to prevent training instability
  - [ ] Create dynamic learning rate based on performance

- [✓] **3.3 Improve REINFORCE Implementation**
  - [✓] Verify correct gradient flow and parameter updates (added logging)
  - [✓] Add baseline for variance reduction (EMA baseline)
  - [ ] Support for experience replay of successful attempts

### 4. Tokenization and Vocabulary Improvements

**Goal**: Ensure the vocabulary and tokenization support effective code generation

- [✓] **4.1 Enhance Vocabulary in `memory/vocab.json`**
  - [✓] Analyze current vocabulary for gaps
  - [✓] Add common coding constructs and Python syntax elements (punctuation, operators, newline)
  - [✓] Ensure proper special tokens (PAD, SOS, EOS, UNK) - *Verified*

- [✓] **4.2 Improve Tokenizer in `tokenizer.py`**
  - [✓] Enhance encoding/decoding functionality (greedy matching, newline handling)
  - [✓] Add better handling for unknown tokens (basic UNK mapping)
  - [ ] Implement normalization of code before tokenization
  - [ ] Add handling for INDENT/DEDENT tokens

- [✓] **4.3 Add Token Usage Analytics**
  - [✓] Track token distribution in generated code (initialized counter and counting logic in `generator.py`)
  - [ ] Monitor token co-occurrence patterns
  - [ ] Identify over/under-utilized tokens

### 5. Fingerprinting and Duplicate Detection

**Goal**: Improve duplicate detection to avoid repetitive unproductive patterns

- [✓] **5.1 Enhance Fingerprinting in `memory/fingerprints.py`**
  - [✓] Improve hash function to better capture code semantics (via improved normalization)
  - [✓] Implement normalized code representation before fingerprinting (preserve case, normalize whitespace)
  - [ ] Add syntax-aware pattern detection

- [✓] **5.2 Improve Duplicate Handling in `runner.py`**
  - [ ] Add penalty for repeatedly generating similar but ineffective patterns
  - [ ] Implement memory of failed approaches
  - [ ] Create "exploration boost" when stuck in repetitive patterns
  - [✓] Add fingerprint hash to duplicate status message for debugging

### 6. Debugging and Monitoring

**Goal**: Improve visibility into learning process

- [✓] **6.1 Enhance Logging Throughout Codebase**
  - [✓] Add detailed logging of model state during training (init stats, loss, grad norm)
  - [ ] Implement visualization for token probabilities
  - [✓] Log gradient norms and parameter updates
  - [✓] Add periodic logging of top token frequencies
  - [✓] Add final token frequency table
  - [✓] Add detailed statistics to final run summary
  - [✓] Centralize logging configuration using `RichHandler` to prevent UI disruption
  - [✓] Implement file logging (`RotatingFileHandler`) for DEBUG+ messages to `altlas_run.log`

- [✓] **6.2 Create Benchmark Tasks**
  - [✓] Implement simple validation tasks with known solutions (created `benchmark_add_two_numbers.json`)
  - [✓] Update TaskLoader to load tasks from JSON files recursively
  - [✓] Add command-line argument (`--task`) to `runner.py` to specify task
  - [ ] Track progress against benchmarks
  - [ ] Add benchmark tests to CI/CD pipeline

### 7. Detection of Improper Model Initialization

**Goal**: Automatically detect and fix improper model initialization

- [✓] **7.1 Implement Statistical Validation**
  - [✓] Add checks for weight distribution statistics
  - [✓] Verify activation patterns on sample inputs (via initial output entropy check)
  - [ ] Compare initial outputs against random baseline

- [🔄] **7.2 Monitor Early Training Signals**
  - [✓] Track initial loss values and trajectories (via existing logging)
  - [✓] Monitor gradient magnitude during early training (via existing logging)
  - [✓] Examine token distribution in early generations (via token frequency logging)
  - *Note: Further analysis/visualization needed.*

- [ ] **7.3 Add Automatic Correction**
  - [ ] Implement warning system for suspicious initialization
  - [ ] Create automatic reinitialization for pathological cases
  - [ ] Add logging for initialization events

### 8. Runtime Control and Stability

**Goal**: Improve user control over runs and ensure stability.

- [ ] **8.1 Implement Graceful Exit (Ctrl+C)**
  - [✓] Add `try...except KeyboardInterrupt` around main loop in `runner.py`.
  - [✓] Implement logic to save state on clean exit (success, max attempts, Ctrl+C) but not on error.
  - [✓] Update final summary message for user interruption.

## Priority Order

1. **First Priority: Hint System Optimization (1.1, 1.2)** - [✓]
   - This will have immediate impact by reducing external guidance
   
2. **Second Priority: Model Initialization (2.1, 2.2, 7.1)** - [✓]
   - Addresses the root cause of poor code generation quality
   
3. **Third Priority: Training Signal Enhancement (3.1, 3.2)** - [✓]
   - Improves the learning dynamics once initialization is fixed
   
4. **Fourth Priority: Fingerprinting Improvements (5.1, 5.2)** - [✓]
   - Reduces wasted learning cycles on repeated failures
   
5. **Fifth Priority: Monitoring and Analytics (6.1, 7.2)** - [✓]
   - Provides visibility into training process for further refinement

6. **Sixth Priority: Vocabulary and Tokenization (4.1, 4.2)** - [✓]
   - Addresses underlying representation for long-term improvement

## Tracking Progress

As tasks are completed, update their status in this document:
- [ ] → [🔄] → [✓]

Additionally, document any unexpected challenges, solutions found, or new insights in a section below:

## Development Notes

*   **2025-04-07:** Completed initial implementation pass for priorities 1-6 and part of 7. Key changes include: reduced hinting frequency, explicit weight initialization & validation, improved REINFORCE (baseline, entropy, clipping), expanded vocabulary, improved tokenizer (greedy matching), enhanced fingerprinting (case-sensitive), extensive logging additions, benchmark task creation, and CLI task selection.
*   **Observation:** Running `benchmark_add_two_numbers` showed successful initialization and functioning training loop. Code quality improved from nonsensical strings but still requires significant training to solve the task. Hinting is now much less frequent.
*   **Next Steps:** Focus on longer training runs, analyzing logs (Task 7.2), refining tokenizer (INDENT/DEDENT), and potentially adding more complex duplicate handling (Task 5.2). Implemented graceful Ctrl+C handling in runner.py. Centralized logging with RichHandler to fix UI disruption.

---

## Emergent Learning Philosophy

This plan aims to address technical issues while preserving the core emergent learning philosophy of AltLAS. The system should continue to develop its own approach to coding rather than being heavily guided. Changes to the hint system, training signal, and model initialization are designed to create a more effective learning environment, not to constrain the agent's exploration.

The goal remains to create an agent that learns to code in its own way, with the capacity to tackle increasingly complex tasks across different programming languages.