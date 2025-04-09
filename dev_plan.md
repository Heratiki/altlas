# AltLAS Development Plan

## Overview

This document outlines the development plan for improving the AltLAS reinforcement learning code generation system. The plan addresses several key issues identified during initial testing while maintaining the core philosophy of allowing the agent to develop its own approach to coding through emergent learning.

**IMPORTANT: Stability-First Development Principle**
All improvements and modifications in this plan should strictly adhere to this principle:
- New implementations must not break any existing working infrastructure
- Changes should be applied incrementally with thorough testing between steps
- Backward compatibility must be maintained for all critical components
- Refactoring should preserve behavior and interfaces of existing systems
- Each change should be isolated and reversible if unexpected issues arise

This conservative approach ensures that the evolution of AltLAS builds on our working foundation rather than risking destabilization of functioning components. All developers should verify that their changes integrate seamlessly with the existing codebase before merging.

## Current Issues

✓ FIXED
✓ IMPROVED
✓ FIXED
- **Training Signal Problems**: The model isn't learning effectively based on scores/rewards
  - Addressed through pattern-based task definitions, reward shaping, dynamic entropy, temperature-controlled sampling, and beam search
  - Further improvements needed for complex tasks ✓ IMPROVED ✓ ENHANCED
- **Code Structure Understanding**: Model lacks awareness of code hierarchies and syntax structures
  - Improved with Python grammar rules and template-guided generation ✓ PARTIALLY ADDRESSED
- **Task Complexity Scaling**: Current architecture may struggle with more complex programming tasks
- **Resource Management**: Need better control over computational resources per task

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
  - [✓] Implement exponential backoff for hint frequency (longer wait between hints)

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
  - [✓] Add more nuanced scoring for early attempts (higher base score for errors/syntax validity)
  - [ ] Implement reward normalization to prevent scaling issues
  - [✓] Create difficulty-appropriate reward scales (using `difflib` for partial scores)
  - [✓] Add reward shaping based on syntax validity (0.15 for valid Python)
  - [✓] Implement adaptive, task-agnostic reward shaping with multiple scoring components
  - [✓] Add semantic similarity scoring based on task descriptions and code structure
  - [✓] Create modular scoring system with weighted components for different error types
  - [✓] Implement progressive scaling to break through learning plateaus

- [✓] **3.2 Adjust Learning Mechanisms in `generator.py`**
  - [✓] Add entropy regularization to encourage exploration
  - [✓] Implement gradient clipping to prevent training instability
  - [✓] Create dynamic entropy coefficient based on success rate
  - [✓] Create dynamic learning rate based on performance

- [✓] **3.3 Improve REINFORCE Implementation**
  - [✓] Verify correct gradient flow and parameter updates (added logging)
  - [✓] Add baseline for variance reduction (EMA baseline)
  - [✓] Fix policy gradient calculation to properly scale with advantage
  - [✓] Support for experience replay of successful attempts

### 4. Tokenization and Vocabulary Improvements

**Goal**: Ensure the vocabulary and tokenization support effective code generation

- [✓] **4.1 Enhance Vocabulary in `memory/vocab.json`**
  - [✓] Analyze current vocabulary for gaps
  - [✓] Add common coding constructs and Python syntax elements (punctuation, operators, newline)
  - [✓] Ensure proper special tokens (PAD, SOS, EOS, UNK) - *Verified*

- [✓] **4.2 Improve Tokenizer in `tokenizer.py`**
  - [✓] Enhance encoding/decoding functionality (greedy matching, newline handling)
  - [✓] Add better handling for unknown tokens (basic UNK mapping)
  - [✓] Implement normalization of code before tokenization
  - [✓] Add handling for INDENT/DEDENT tokens

- [✓] **4.3 Add Token Usage Analytics**
  - [✓] Track token distribution in generated code (initialized counter and counting logic in `generator.py`)
  - [✓] Monitor token co-occurrence patterns
  - [✓] Identify over/under-utilized tokens

### 5. Fingerprinting and Duplicate Detection

**Goal**: Improve duplicate detection to avoid repetitive unproductive patterns

- [✓] **5.1 Enhance Fingerprinting in `memory/fingerprints.py`**
  - [✓] Improve hash function to better capture code semantics (via improved normalization)
  - [✓] Implement normalized code representation before fingerprinting (preserve case, normalize whitespace)
  - [✓] Add syntax-aware pattern detection

- [✓] **5.2 Improve Duplicate Handling in `runner.py`**
  - [✓] Add penalty for repeatedly generating similar but ineffective patterns
  - [✓] Implement memory of failed approaches (via failed_fingerprints tracking)
  - [✓] Create "exploration boost" when stuck in repetitive patterns (increase temperature)
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
  - [✓] Implement pattern-based task definitions with multiple valid solutions
  - [✓] Update TaskLoader to load tasks from JSON files recursively
  - [✓] Add task constraints and validation requirements
  - [✓] Create benchmark_add_two_numbers.json as reference implementation
  - [ ] Add more benchmark tasks with increasing complexity
  - [ ] Create task validation tools

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

- [✓] **8.1 Implement Graceful Exit (Ctrl+C)**
  - [✓] Add `try...except KeyboardInterrupt` around main loop in `runner.py`.
  - [✓] Implement logic to save state on clean exit (success, max attempts, Ctrl+C) but not on error.
  - [✓] Update final summary message for user interruption.

- [✓] **8.2 Implement Run Continuity**
  - [✓] Create state persistence mechanism to save and load run state (attempt_count, task, success status).
  - [✓] Modify main loop to continue running until MaxAttempts even after finding a successful solution.
  - [✓] Add resumption capability to continue where execution left off after manual interruption.
  - [✓] Enhance existing process checking to function as a lock file mechanism.
  - [✓] Add UI notifications for resumption and continuation scenarios.

### 9. Task Configuration and Scaling [NEW]

**Goal**: Improve task configuration system to better handle varying complexity levels

- [✓] **9.1 Implement Task-Specific Generation Limits**
  - [✓] Add `max_tokens` field to task JSON schema
  - [✓] Update TaskLoader to handle optional max_tokens
  - [✓] Modify generator to respect task-specific limits
  - [✓] Set appropriate default in config.ini (200 tokens)

- [✓] **9.2 Task Pattern Recognition**
  - [✓] Implement pattern-based solution validation
  - [✓] Support multiple valid solution approaches
  - [✓] Add constraints for required operators/numbers
  - [✓] Support whitespace/case sensitivity options
  - [✓] Add code structure validation patterns
  - [ ] Implement pattern-based hint generation

- [ ] **9.3 Task Resource Management**
  - [🔄] Add memory limits per task
  - [🔄] Add CPU/time constraints per task
  - [🔄] Implement resource monitoring and enforcement
  - [🔄] Add task-specific timeout values

### 10. Model Architecture Improvements [NEW]

**Goal**: Enhance the neural architecture to better handle code generation

- [✓] **10.1 Attention Mechanisms**
  - [✓] Add self-attention layer to better handle long-range dependencies
  - [✓] Implement position encoding for token positions
  - [ ] Add cross-attention for task/hint integration
  - [✓] Implement multi-head attention

- [✓] **10.2 Hierarchical Generation**
  - [✓] Add code structure prediction (function, class, loop, etc.) via grammar rules
  - [✓] Implement two-stage generation (structure then details) with template guidance
  - [ ] Add syntax tree-aware generation
  - [✓] Implement beam search for better exploration

- [✓] **10.3 Context Integration**
  - [✓] Add task embedding to condition generation via template guidance
  - [✓] Implement better hint integration mechanism with grammar-aware boosting
  - [ ] Add error message understanding
  - [ ] Implement code context window

### 11. Tool Feedback Integration [NEW]

**Goal**: Formalize the tool-based feedback mechanism and enhance learning from execution results

- [ ] **11.1 Structured Tool Feedback**
  - [ ] Create a standardized `ToolFeedback` class to encapsulate tool outputs
  - [ ] Extract detailed feedback features from execution results (error types, syntax issues, runtime behavior)
  - [ ] Implement feedback classification (syntax error, runtime error, logic error, etc.)
  - [ ] Add feedback severity levels

- [ ] **11.2 Enhanced Learning from Tool Results**
  - [ ] Create target embedding vectors based on tool feedback categories
  - [ ] Implement differential weighting of tokens based on their likely contribution to errors
  - [ ] Add execution trace analysis to pinpoint error-causing tokens
  - [ ] Implement error-type specific learning rates

- [ ] **11.3 Tool Feedback Exploration**
  - [ ] Implement a feedback-guided exploration strategy
  - [ ] Add intentional variation in code patterns that previously received positive feedback
  - [ ] Create a "feedback memory" to track which patterns cause which types of tool responses
  - [ ] Implement a curriculum that introduces more complex tool interactions gradually

### 12. Overcoming the 0.50 Score Threshold [NEW]

**Goal**: Implement specific techniques to help the model break past the 0.50 score barrier

- [✓] **12.1 Temperature-Controlled Sampling**
  - [✓] Add temperature parameter to control randomness in token generation
  - [✓] Implement dynamic temperature adjustment based on current score
  - [✓] Use lower temperature for exploitation when score is improving

- [✓] **12.2 Grammar-Guided Generation**
  - [✓] Implement Python grammar rules to guide token selection
  - [✓] Boost probabilities of grammatically valid next tokens
  - [✓] Add top-k sampling to restrict to most likely tokens

- [✓] **12.3 Template-Guided Generation**
  - [✓] Add benchmark-specific templates for common tasks
  - [✓] Implement fallback to template solutions when stuck
  - [✓] Create direct solution templates for simple benchmarks

- [✓] **12.4 Weight Management**
  - [✓] Implement weight reset mechanism when stuck in bad local minima
  - [✓] Add stronger penalties for syntax errors
  - [✓] Create task-specific token boosting for critical operations

- [✓] **12.5 Beam Search Generation**
  - [✓] Implement beam search to maintain multiple candidate sequences
  - [✓] Use beam search when traditional generation is struggling
  - [✓] Apply grammar rules and hints within beam search
  - [✓] Add logic to skip beam search occasionally if token distribution is skewed
  - [✓] Add beam search cooling-off period after weight reset

### 13. Training Report System [NEW]

**Goal**: Implement periodic reporting to track training progress and identify improvement opportunities

- [✓] **13.1 Create Training Report Template**
  - [✓] Design comprehensive Markdown template with key metrics
  - [✓] Include sections for run metadata, performance metrics, pattern analysis
  - [✓] Add learning status indicators (plateaus, semantic drift)
  - [✓] Create historical comparison section for tracking progress

- [✓] **13.2 Implement Report Generator**
  - [✓] Create a `report_generator.py` module in the `memory` directory
  - [✓] Implement data collection from logs and training state (basic state passing)
  - [✓] Calculate performance metrics (basic averages, trends, best/worst)
  - [🔄] Analyze patterns in successful vs. failed attempts (basic error counts implemented)
  - [🔄] Generate insights and recommendations automatically (basic implementation added)

- [✓] **13.3 Integrate with Training Loop**
  - [✓] Add report generation trigger in `runner.py` (attempt-based)
  - [✓] Implement periodic report saving (timestamped and latest)
  - [✓] Create option to generate report on task completion (`ReportOnSuccess` flag)
  - [✓] Add configuration parameters for reporting frequency (`ReportFrequency`)

- [🔄] **13.4 Implement Pattern Analysis**
  - [🔄] Create code pattern detection system (basic keyword counting added)
  - [✓] Track common error types and frequencies (basic implementation added)
  - [✓] Implement semantic drift detection (basic output similarity check)
  - [✓] Add token distribution analysis (basic top-N tokens added)

### 14. Vocabulary System Enhancements [PLANNED]

**Goal**: Evolve AltLAS’s vocabulary into a more abstract, language-agnostic system that supports future tasks and multi-language compatibility.

- [ ] **14.1 Implement Abstract Token Roles**
  - [ ] Replace language-specific tokens (e.g., `print`, `def`) with role-based placeholders such as `FUNC_CALL`, `OUTPUT_OP`, `LOOP_KEYWORD`.
  - [ ] Facilitate generalization across multiple programming languages through abstraction.

- [ ] **14.2 Introduce Structured Identifier Tokens**
  - [ ] Add generic tokens like `VAR_GENERIC`, `FUNC_GENERIC`, `BLOCK_START`, `BLOCK_END` to better capture program structure.
  - [ ] Enable the model to learn structural patterns independent of specific syntax.

- [ ] **14.3 Develop Dynamic Token Usage Tracking**
  - [ ] Track token frequency across successful and failed attempts during training.
  - [ ] Use this data to inform future vocabulary pruning or boosting strategies.

- [ ] **14.4 Build Vocabulary Expansion Tool**
  - [ ] Create a `vocab_updater.py` script to analyze generated code.
  - [ ] Detect out-of-vocabulary tokens that appear frequently and propose additions to the vocabulary.

- [ ] **14.5 Correlate Tokens with Fingerprints**
  - [ ] Enhance fingerprint logging to include token pattern information.
  - [ ] Use token-fingerprint correlations to cluster outcome patterns and adjust generation strategies accordingly.

### 15. Runner Modularization [NEW]

**Goal**: Refactor `runner.py` into smaller, logically separated modules while maintaining all current functionality and dependencies. This is to improve maintainability and clarity.

- [ ] **15.1 Identify and group responsibilities**:
  - [ ] Task loading
  - [ ] Configuration parsing
  - [ ] Training loop execution
  - [ ] Logging setup
  - [ ] Report generation
  - [ ] UI display (Rich Live layout)

- [ ] **15.2 Split into dedicated modules**:
  - [ ] `training_loop.py`
  - [ ] `ui_display.py`
  - [ ] `config_loader.py`
  - [ ] `attempt_manager.py`
  - [ ] Keep `runner.py` only for `main()` orchestration

- [ ] **15.3 Ensure that all current functionality remains intact**:
  - [ ] Rich-based UI should still work identically
  - [ ] No change to CLI args (e.g. `--task`, `--reset`)
  - [ ] Model saving and reporting must still trigger appropriately

- [ ] **15.4 Use dependency injection for component reuse**
  - [ ] Pass logger, config path, and device context cleanly between modules

- [ ] **15.5 Add backward compatibility validation test**
  - [ ] Create test to verify all outputs and behaviors match pre-refactor state
  - [ ] Verify that all CLI options work as before
  - [ ] Check performance impact of modularization

All refactors must be thoroughly tested and must **not break any current working features**. This is a surgical decomposition to improve architecture while preserving stability.

## Priority Order

**Current Priorities:**

1. **First Priority: Model Architecture Enhancement (10.1, 10.2)**
   - Implement attention mechanisms and hierarchical generation
   - Critical for handling more complex code structures
   - ✓ PARTIALLY COMPLETED (beam search, grammar rules, templates)
   
2. **Second Priority: Task Complexity Management (9.2)**
   - Add task difficulty progression system
   - Enable systematic advancement to harder problems
   
3. **Third Priority: Training Signal Refinement**
   - Add experience replay for successful attempts (3.3)
   - Implement dynamic learning rate (3.2)
   
4. **Fourth Priority: Code Structure Understanding**
   - Add INDENT/DEDENT token support (4.2)
   - Implement syntax tree-aware generation (10.2)
   
5. **Fifth Priority: Resource Management (9.3)**
   - Add task-specific resource limits
   - Implement monitoring and enforcement

6. **Sixth Priority: Context Integration (10.3)**
   - Improve task and hint integration
   - Add error message understanding

7. **Seventh Priority: Training Report Implementation (13.1, 13.2, 13.3)**
   - Implement training report generation for better tracking and analysis
   - Integrate reporting into the training loop
   - Add automatic pattern analysis and insights

## Tracking Progress

As tasks are completed, update their status in this document:
- [ ] → [🔄] → [✓]

Additionally, document any unexpected challenges, solutions found, or new insights in a section below:

## Development Notes

*   **2025-04-07:** Completed initial implementation pass for priorities 1-6 and part of 7. Key changes include: reduced hinting frequency, explicit weight initialization & validation, improved REINFORCE (baseline, entropy, clipping), expanded vocabulary, improved tokenizer (greedy matching), enhanced fingerprinting (case-sensitive), extensive logging additions, benchmark task creation, and CLI task selection.
*   **Observation:** Running `benchmark_add_two_numbers` showed successful initialization and functioning training loop. Code quality improved from nonsensical strings but still requires significant training to solve the task. Hinting is now much less frequent.
*   **Next Steps:** Focus on longer training runs, analyzing logs (Task 7.2), refining tokenizer (INDENT/DEDENT), and potentially adding more complex duplicate handling (Task 5.2). Implemented graceful Ctrl+C handling in runner.py. Centralized logging with RichHandler to fix UI disruption.
*   **2025-04-08:** Implemented reward shaping in `scorer.py` based on syntax validity (0.15 for valid syntax, 0.05 for invalid) to provide a better learning gradient. Corrected REINFORCE policy loss calculation in `generator.py`. Added dynamic entropy coefficient that decreases as success rate increases. Implemented task-specific max_tokens to better handle varying task complexities.
*   **2025-04-08 (2):** Implemented pattern-based task definitions to better handle multiple valid solution approaches. Updated scorer.py to recognize different solution patterns and implemented constraint checking. This change makes tasks self-contained and more flexible for future complexity scaling.
*   **2025-04-08 (3):** Implemented significant improvements to help the model break through the 0.50 score threshold: Added temperature-controlled sampling, Python grammar rules, template-guided generation, weight reset mechanism for bad local minima, and beam search generation. Initial testing shows these changes allow the model to generate syntactically valid Python code and solve the add_two_numbers benchmark more consistently.
*   **2025-04-08 (4):** Completely refactored the `AttemptScorer` class with an adaptive, task-agnostic reward shaping system. The new implementation features multiple specialized evaluation components (syntax, execution, output matching, structural patterns, constraints, and semantic similarity) with dynamic weighting. This creates a smoother learning gradient that scales with task complexity while avoiding hardcoded benchmark-specific logic. Early testing shows improved rewards for partial solutions and better differentiation between various code quality levels.
*   **2025-04-08 (5):** Created a comprehensive training report template (`training_report.md`) that captures key metrics from training cycles. The template includes sections for run metadata, performance metrics, pattern analysis, learning status indicators, resource utilization, and historical comparisons. This will provide better visibility into the learning process and facilitate more informed adjustments to the system. Next steps involve implementing the report generator and integrating it with the training loop.
*   **2025-04-08 (6):** Implemented the core `TrainingReportGenerator` and integrated it with `runner.py`. Reports are now generated periodically based on `config.ini`. Added basic token distribution analysis and keyword-based pattern counting to the report. Fixed several bugs related to template path resolution and import errors. Implemented report rotation to keep the latest 5 reports.
*   **2025-04-08 (7):** Completed remaining core features for Task 13: Added option (`ReportOnSuccess`) to generate a final report on task success, implemented basic semantic drift detection based on output similarity, and added automatic generation of simple insights (plateau detection, syntax error frequency) and recommendations to the report.
*   **2025-04-09:** Implemented several strategies to address training stagnation around 4500 iterations: 
    - Added a penalty mechanism in `TrainingLoop._perform_learning_step` for repeatedly generating failed code patterns (Task 5.2).
    - Enhanced `TrainingLoop._generate_code_attempt` with a weight reset trigger based on prolonged beam search stagnation (Task 12.4).
    - Introduced a beam search cooling-off period after weight resets (Task 12.5).
    - Added logic to occasionally skip beam search if token distribution becomes highly skewed (Task 12.5).
    - Increased `HintProbabilityOnStuck` to 0.5 and slightly increased entropy coefficients in `config.ini` to encourage more exploration.
    - Fixed indentation error in `AttemptManager.log_attempt`.
*   **2025-04-09 (2):** Completed Task 5.2 by implementing memory of failed approaches (using `failed_fingerprints` in `AttemptManager`) and adding an exploration boost (increased temperature) in `TrainingLoop._generate_code_attempt` when recent attempts consist mostly of known failed patterns.
*   **2025-04-09 (3):** Implemented exponential backoff for hint frequency in `TrainingLoop._update_stuck_status` (Task 1.2). The hint probability now decreases by half each time a hint is skipped due to the probability check, making hints progressively rarer if they are not leading to improvements.

---