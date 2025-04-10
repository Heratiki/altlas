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
- The rich UI display now shows an estimated total time to 100% completion based on current progress and elapsed time.

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
- **Model Distribution Collapse**: Model occasionally gets stuck in pathological states generating the same tokens repeatedly
  - Emergency distribution recovery helps, but more robust mechanisms needed

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
  - [✓] Measure impact of hints on score improvement (track score before/after hint)
  - [✓] Add hint usage statistics to the Rich UI
  - [✓] Add average hint improvement to training report

### 2. Model Initialization and Weight Validation

**Goal**: Ensure proper model initialization to avoid pathological learning patterns

- [✓] **2.1 Implement Initialization Diagnostics**
  - [✓] Add logging of weight statistics in `model.py` (mean, std, min, max)
  - [✓] Create weight histogram visualization function (*Implemented in report generator, but removed by user*)
  - [✓] Add explicit initialization methods (Xavier/Glorot, Kaiming He, Orthogonal) in `model.py`

- [✓] **2.2 Create Weight Validation Functions**
  - [✓] Implement sanity checks for initial weight distribution in `generator.py` (`_validate_initial_weights`)
  - [✓] Add validation against known pathological patterns (near-zero std, large mean, NaN/Inf)
  - [✓] Create recovery mechanism for bad initialization states

- [✓] **2.3 Add Runtime Monitoring**
  - [✓] Track gradient statistics during learning (log gradient norm in `generator.py`)
  - [✓] Monitor weight changes over time to detect vanishing/exploding gradients (`TrainingLoop._monitor_training_stability`)
  - [✓] Add early warning for training instability (`TrainingLoop._monitor_training_stability`)
- [✓] **2.4 Validate Model Parameter Compatibility**
  - [✓] Check that `hidden_dim % num_attention_heads == 0` if attention enabled
  - [✓] Check that `embedding_dim % num_attention_heads == 0` if attention enabled
  - [✓] Raise clear errors if incompatible

### 3. Training Signal Enhancement

**Goal**: Improve the learning signal to promote more effective exploration and learning

- [✓] **3.1 Enhance Reward Mechanism in `scorer.py`**
  - [✓] Add more nuanced scoring for early attempts (higher base score for errors/syntax validity)
  - [✓] Implement reward normalization to prevent scaling issues (running mean/std in TrainingLoop)
  - [✓] Create difficulty-appropriate reward scales (using `difflib` for partial scores)
  - [✓] Add reward shaping based on syntax validity (0.15 for valid Python)
  - [✓] Implement adaptive, task-agnostic reward shaping with multiple scoring components
  - [✓] Add semantic similarity scoring based on task descriptions and code structure
  - [✓] Create modular scoring system with weighted components for different error types
  - [✓] Implement progressive scaling to break through learning plateaus
  - *Note: Review potential redundancy between `score` and `calculate_reward` methods in `scorer.py`.*

- [✓] **3.2 Adjust Learning Mechanisms in `generator.py`**
  - [✓] Add entropy regularization to encourage exploration
  - [✓] Implement gradient clipping to prevent training instability
  - [✓] Create dynamic entropy coefficient based on success rate
  - [✓] Create dynamic learning rate based on performance
  - [✓] Add adaptive entropy-based early stopping with configurable minimum tokens before stop
  - [✓] Inject optional noise into logits to encourage exploration (configurable)
  - [✓] Dynamically increase temperature if repetitive token overuse is detected across recent generations
  - [✓] Penalize and log empty or invalid generations to improve diagnostics and feedback
  - [✓] Make all exploration and penalty mechanisms configurable, abstract, and token-agnostic

- [✓] **3.3 Improve REINFORCE Implementation**
  - [✓] Verify correct gradient flow and parameter updates (added logging)
  - [✓] Add baseline for variance reduction (EMA baseline)
  - [✓] Fix policy gradient calculation to properly scale with advantage
  - [✓] Support for experience replay of successful attempts

  - [ ] Explore per-token or delayed reward mechanisms.

### 4. Tokenization and Vocabulary Improvements

**Goal**: Ensure the vocabulary and tokenization support effective code generation

- [🔄] **4.1 Enhance Vocabulary in `memory/vocab.json`**
  - [✓] Analyze current vocabulary for gaps
  - [✓] Add common coding constructs and Python syntax elements (punctuation, operators, newline) - *Replaced with abstract tokens*
  - [✓] Ensure proper special tokens (PAD, SOS, EOS, UNK) - *Verified*
  - [✓] Transitioned to language-agnostic abstract tokens (e.g., `OUTPUT_OP`, `FUNC_DEF`). *Requires model retraining.*

- [🔄] **4.2 Improve Tokenizer in `tokenizer.py`**
  - [✓] Enhance encoding/decoding functionality (greedy matching, newline handling)
  - [✓] Add better handling for unknown tokens (basic UNK mapping)
  - [✓] Implement normalization of code before tokenization
  - [✓] Add handling for INDENT/DEDENT tokens - *Removed due to abstract vocabulary.*

- [✓] **4.3 Add Token Usage Analytics**
  - [✓] Track token distribution in generated code (initialized counter and counting logic in `generator.py`)
  - [✓] Monitor token co-occurrence patterns
  - [✓] Identify over/under-utilized tokens
  
- [✓] **4.4 Implement Adaptive Token Imbalance Recovery**
  - [✓] Track token imbalance warnings with cooldown logic (`tokenizer.py`)
  - [✓] Apply entropy boost if imbalance persists > N iterations (*Handled via generator logic/temperature*)
  - [✓] Penalize overused tokens directly in logit sampling (softmax) (`generator.py`)
  - [✓] Add config option: `TokenImbalancePenalty = 0.2` (`config.ini`)

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
  - [✓] Add more benchmark tasks with increasing complexity (hello_variable, simple_if, basic_for_loop added)
  - [ ] Create task validation tools

### 7. Detection of Improper Model Initialization

**Goal**: Automatically detect and fix improper model initialization

- [✓] **7.1 Implement Statistical Validation**
  - [✓] Add checks for weight distribution statistics
  - [✓] Verify activation patterns on sample inputs (via initial output entropy check)
  - [ ] Compare initial outputs against random baseline

- [✓] **7.2 Monitor Early Training Signals**
  - [✓] Track initial loss values and trajectories (via existing logging)
  - [✓] Monitor gradient magnitude during early training (via existing logging)
  - [✓] Examine token distribution in early generations (via token frequency logging)
  - *Note: Further analysis/visualization needed.*

- [✓] **7.3 Add Automatic Correction**
  - [✓] Implement warning system for suspicious initialization
  - [✓] Create automatic reinitialization for pathological cases
  - [✓] Add logging for initialization events

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
  - [🔄] Implement pattern-based hint generation

- [🔄] **9.3 Task Resource Management**
  - [ ] Add memory limits per task (*Executor needs enhancement*)
  - [ ] Add CPU/time constraints per task (*Executor needs enhancement*)
  - [ ] Implement resource monitoring and enforcement (*Executor needs enhancement*)
  - [✓] Add task-specific timeout values (*Implemented in executor via config*)
  - *Note: Current implementation only handles basic timeout. Robust sandboxing and resource limits (CPU, memory) require significant executor enhancements (See Task 17).*

### 10. Model Architecture Improvements [NEW]

**Goal**: Enhance the neural architecture to better handle code generation

- [✓] **10.1 Attention Mechanisms**
  - [✓] Add multi-head self-attention layer
  - [✓] Add residual connection over attention output
  - [✓] Add positional encoding (sinusoidal and learned)
  - [✓] Add optional LayerNorm after LSTM and attention
  - [✓] Implement auto-configuration logic to enable/disable features based on model size
  - [✓] Add runtime override of features via `[ModelFlags]` in config.ini
  - [✓] Log model feature configuration at startup
  - [ ] Add cross-attention for task/hint integration

- [✓] **10.2 Hierarchical Generation**
  - [✓] Add code structure prediction via abstract grammar rules (`ABSTRACT_GRAMMAR_RULES`).
  - [✓] Implement template guidance using agnostic templates (`generate_agnostic_template`).
  - [ ] Add syntax tree-aware generation
  - [✓] Implement beam search for better exploration

  - [✓] Implement early stopping during generation based on low entropy or repetition (`EarlyStop...` configs).
- [✓] **10.3 Context Integration**
  - [✓] Add task embedding to condition generation via template guidance
  - [✓] Implement better hint integration mechanism with agnostic hint parsing (`_parse_hint`).
  - [ ] Add error message understanding
  - [ ] Implement code context window

### 11. Tool Feedback Integration [NEW]

**Goal**: Formalize the tool-based feedback mechanism and enhance learning from execution results

- [✓] **11.1 Structured Tool Feedback**
  - [✓] Create a standardized `ToolFeedback` class to encapsulate tool outputs
  - [✓] Extract detailed feedback features from execution results (error types, syntax issues, runtime behavior)
  - [✓] Implement feedback classification (syntax error, runtime error, logic error, etc.)
  - [✓] Add feedback severity levels

- [✓] **11.2 Enhanced Learning from Tool Results**
  - [✓] Create target embedding vectors based on tool feedback categories (Requires model changes - see Task 10)
  - [✓] Implement differential weighting of tokens based on their likely contribution to errors (via feedback severity and type) (`generator.learn`)
  - [ ] Add execution trace analysis to pinpoint error-causing tokens (Requires executor changes)
  - [✓] Implement error-type specific learning rates (adjust LR reduction based on severity/type) (`generator._update_learning_rate`)

- [✓] **11.3 Tool Feedback Exploration**
  - [✓] Implement a feedback-guided exploration strategy (adjust temperature/penalize tokens based on last feedback)
  - [✓] Add intentional variation in code patterns that previously received positive feedback (boost tokens from successful experiences)
  - [✓] Create a "feedback memory" to track which patterns cause which types of tool responses (store feedback per fingerprint, use history for stronger penalties)
  - [✓] Implement a curriculum that introduces more complex tool interactions gradually (Requires more complex tasks - see Task 6.2/9)

### 12. Overcoming the 0.50 Score Threshold [NEW]

**Goal**: Implement specific techniques to help the model break past the 0.50 score barrier

- [✓] **12.1 Temperature-Controlled Sampling**
  - [✓] Add temperature parameter to control randomness in token generation
  - [✓] Implement dynamic temperature adjustment based on current score
  - [✓] Use lower temperature for exploitation when score is improving

- [✓] **12.2 Grammar-Guided Generation**
  - [✓] Implement abstract grammar rules (`ABSTRACT_GRAMMAR_RULES`) to guide token selection.
  - [✓] Boost probabilities of grammatically valid next tokens
  - [✓] Add top-k sampling to restrict to most likely tokens

  - [✓] Implement decay for grammar boost influence over time (`GrammarBoostDecay` config).
- [✓] **12.3 Template-Guided Generation**
  - [✓] Add agnostic template generation based on task keywords (`generate_agnostic_template`).
  - [✓] Implement fallback to template solutions when stuck (logic remains).
  - [ ] Create direct solution templates for simple benchmarks (agnostic version needed).

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
  - [✓] Analyze patterns in successful vs. failed attempts (AST-based structure analysis implemented)
  - [✓] Generate insights and recommendations automatically (basic implementation added)

- [✓] **13.3 Integrate with Training Loop**
  - [✓] Add report generation trigger in `runner.py` (attempt-based)
  - [✓] Implement periodic report saving (timestamped and latest)
  - [✓] Create option to generate report on task completion (`ReportOnSuccess` flag)
  - [✓] Add configuration parameters for reporting frequency (`ReportFrequency`)

- [✓] **13.4 Implement Pattern Analysis**
  - [✓] Create code pattern detection system (AST-based structure analysis implemented)
  - [✓] Track common error types and frequencies (basic implementation added)
  - [✓] Implement semantic drift detection (basic output similarity check)
  - [✓] Add token distribution analysis (basic top-N tokens added)

### 14. Vocabulary System Enhancements [PLANNED]

**Goal**: Evolve AltLAS's vocabulary into a more abstract, language-agnostic system that supports future tasks and multi-language compatibility.

- [✓] **14.1 Implement Abstract Token Roles**
  - [✓] Replaced language-specific tokens with role-based placeholders (`OUTPUT_OP`, `FUNC_DEF`, etc.) in `memory/vocab.json`.
  - [✓] Facilitates generalization but requires model retraining and tokenizer/generator adaptation.

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

### 20. Configuration System Improvements [NEW]

**Goal:** Improve configuration robustness, clarity, and runtime flexibility.

- [✓] **20.1 Improve Config File Structure**
  - [✓] Clean up and comment all config.ini sections
  - [✓] Add `[ModelFlags]` section for runtime feature toggles
- [✓] **20.2 Fix Inline Comment Parsing**
  - [✓] Update all configparser initializations with `inline_comment_prefixes=(';', '#')`
  - [✓] Prevents parsing errors from inline comments
- [✓] **20.3 Integrate Config Flags into Runtime**
  - [✓] Parse `[ModelFlags]` in generator.py
  - [✓] Pass feature flags to model at init and reset
  - [✓] Override model auto-config with explicit flags

### 15. Runner Modularization [NEW]

**Goal**: Refactor `runner.py` into smaller, logically separated modules while maintaining all current functionality and dependencies. This is to improve maintainability and clarity.

- [✓] **15.1 Identify and group responsibilities**:
  - [✓] Task loading
  - [✓] Configuration parsing
  - [✓] Training loop execution
  - [✓] Logging setup
  - [✓] Report generation
  - [✓] UI display (Rich Live layout)

- [✓] **15.2 Split into dedicated modules**:
  - [✓] `training_loop.py`
  - [✓] `ui_display.py`
  - [✓] `config_loader.py`
  - [✓] `attempt_manager.py`
  - [✓] Keep `runner.py` only for `main()` orchestration

- [ ] **15.3 Ensure that all current functionality remains intact**: (*Requires testing*)
  - [ ] Rich-based UI should still work identically
  - [ ] No change to CLI args (e.g. `--task`, `--reset`)
  - [ ] Model saving and reporting must still trigger appropriately

- [ ] **15.4 Use dependency injection for component reuse**: (*Partially done, review needed*)
  - [ ] Pass logger, config path, and device context cleanly between modules
  
- [ ] **15.5 Add backward compatibility validation test**:
  - [ ] Create test to verify all outputs and behaviors match pre-refactor state
  - [ ] Verify that all CLI options work as before
  - [ ] Check performance impact of modularization
- *Note: `memory/logger.py` and `memory/fingerprints.py` confirmed obsolete; functionality merged into `core/attempt_manager.py`. Files moved to `deprecated/` directory (ignored by git).*
  - [ ] Verify that all CLI options work as before
  - [ ] Check performance impact of modularization

### 16. Refactor External LLM Integration for vLLM [PLANNED]

**Goal**: Replace current OPENAI/LM Studio implementation with vLLM to improve performance, reduce costs, and optimize usage patterns across the codebase.

- [🔄] **16.1 Set Up vLLM Infrastructure**
  - [✓] Create a dedicated `llm_provider` module to abstract LLM API interactions
  - [✓] Implement vLLM server configuration with appropriate model options (via `config.ini`)
  - [✓] Add fallback mechanisms for when vLLM is unavailable (vLLM -> LM Studio)
  - [ ] Set up batching capabilities to optimize throughput (*Stubbed for now*)
  - [✓] Implement configuration options for vLLM-specific parameters (`[LLM]` section in `config.ini`)

- [🔄] **16.2 Implement Request Optimization**
  - [✓] Add request batching to combine multiple similar requests (sequential stub implemented)
  - [✓] Implement caching to avoid redundant LLM calls (simple LRU cache added)
  - [ ] Create a prompt template system to standardize and optimize prompts
  - [ ] Implement token counting to minimize input sizes
  - [ ] Add asynchronous request handling to improve throughput

- [🔄] **16.3 Migrate Existing LLM Calls**
  - [✓] Identify all current LLM integration points (hint generation, report analysis)
  - [✓] Refactor `TrainingReportGenerator._call_local_llm` to use the new vLLM provider
  - [✓] Update hint generation in `TrainingLoop._get_hint_from_advisor`
  - [ ] Create compatibility layer for existing code (*Provider handles fallback*)
  - *Note (2025-04-09): Increased LM Studio API timeout in `TrainingReportGenerator._call_local_llm` to 120s to mitigate premature timeouts.*

- [🔄] **16.4 Implement Performance Monitoring**
  - [✓] Add instrumentation to track LLM request latency
  - [✓] Monitor token usage and costs (basic token counting)
  - [✓] Implement adaptive request throttling based on system load (latency-based delay)
  - [✓] Create detailed logging for LLM interactions (latency, tokens, errors)
  - [ ] Add performance metrics to training reports

- [ ] **16.5 Enhance LLM-Based Features**
  - [ ] Improve hint quality with more specialized prompts
  - [ ] Add progressive hint generation (increasing specificity with successive failures)
  - [ ] Implement code review capabilities for generated solutions
  - [ ] Add automatic prompt tuning based on hint effectiveness
  - [ ] Create an LLM-based code analyzer for deeper report insights

### 17. Execution Safety and Sandboxing [NEW, PLANNED]

**Goal**: Enhance the safety and robustness of code execution by implementing stronger sandboxing and resource controls.

- [ ] **17.1 Implement Robust Sandboxing**
  - [ ] Explore containerization (Docker) or process isolation (nsjail) for execution.
  - [ ] Define strict filesystem access controls (allow only necessary paths).
  - [ ] Disable network access by default during execution.
- [ ] **17.2 Resource Limit Enforcement**
  - [ ] Implement memory limits per execution (complementary to Task 9.3).
  - [ ] Implement CPU usage limits per execution (complementary to Task 9.3).
  - [ ] Refine timeout mechanism for better reliability.
- [ ] **17.3 Static Code Analysis Integration**
  - [ ] Integrate tools like Bandit for pre-execution safety checks.
  - [ ] Develop custom AST analysis rules to detect potentially harmful patterns beyond basic regex.
- [ ] **17.4 Capability Limiting**
  - [ ] Investigate methods to restrict access to dangerous Python modules/built-ins within the execution context.

### 18. Parallel Training Loop Implementation [NEW, PLANNED]

**Goal**: Parallelize the code generation, execution, and scoring process to significantly improve training throughput and efficiency.

- [ ] **18.1 Analyze Bottlenecks**
  - [ ] Profile current `TrainingLoop` to identify major time sinks (generation, execution, scoring, learning).
  - [ ] Determine which components are most suitable for parallelization.
- [ ] **18.2 Design Parallel Architecture**
  - [ ] Explore using `multiprocessing` or `asyncio` for parallel execution of attempts.
  - [ ] Design a worker pool system for code execution and scoring.
  - [ ] Define mechanisms for distributing tasks (code attempts) to workers and collecting results.
- [ ] **18.3 Refactor Core Components for Parallelism**
  - [ ] Modify `TrainingLoop` to manage parallel workers and aggregate results.
  - [ ] Ensure `CodeExecutor` and `AttemptScorer` are thread-safe or suitable for parallel execution (consider instance-per-worker).
  - [ ] Adapt `CodeGenerator` if parallel generation (e.g., batching) is feasible.
  - [ ] Modify `AttemptManager` to handle concurrent logging and state updates safely (e.g., using locks or queues).
- [ ] **18.4 Implement State Synchronization**
  - [ ] Develop strategies for synchronizing shared state (e.g., model weights, optimizer state, best score, attempt count) between the main process and workers.
  - [ ] Address potential race conditions and ensure data consistency during learning updates.
- [ ] **18.5 Configuration and Control**
  - [ ] Add configuration options (`config.ini`) to enable/disable parallelism and control the number of workers.
  - [ ] Ensure UI (`ui_display.py`) can handle updates from multiple sources or aggregates statistics correctly.
- [ ] **18.6 Testing and Benchmarking**
  - [ ] Implement tests to verify correctness under parallel execution.
  - [ ] Benchmark performance gains and overhead of the parallel implementation.
  - [ ] Analyze resource utilization (CPU, memory) under parallel load.

### 19. Multi-Language Support with Language Maps [NEW]

**Goal**: Implement language-agnostic code normalization and multi-language support through a dynamic mapping system.

- [✓] **19.1 Create Language Mapping Directory Structure**
  - [✓] Create `/workspaces/altlas/language_maps/` directory
  - [✓] Implement `python.json` with mappings from abstract tokens to Python syntax
  - [✓] Create `default.json` as a fallback mapping
  - [✓] Add initial support for JavaScript with `javascript.json`
  - [✓] Create `execution_config.json` for language-specific execution settings
  - [✓] Document mapping format and conventions

- [✓] **19.2 Update Task Definition Format**
  - [✓] Add `target_language` field to task JSON schema (e.g., "python", "javascript", "cpp")
  - [✓] Update `TaskLoader` to parse the `target_language` field
  - [✓] Ensure backward compatibility for tasks without the field (default to "python")
  - [✓] Add validation for supported language values
  - [✓] Update existing benchmark tasks with explicit `target_language` field

- [✓] **19.3 Enhance AttemptScorer to Support Multiple Languages**
  - [✓] Add `load_language_map(language_name)` method to dynamically load appropriate mappings
  - [✓] Implement caching to avoid reloading maps repeatedly
  - [✓] Modify `normalize_for_scoring` to use language-specific mappings
  - [✓] Update method signatures to accept task object with language information
  - [✓] Create fallback mechanisms for unsupported languages

- [✓] **19.4 Make Syntax Checking Language-Aware**
  - [✓] Update `_evaluate_syntax` to conditionally use language-specific validation
  - [✓] Keep `ast.parse()` for Python code validation
  - [✓] Implement neutral scoring (e.g., 0.1) for other languages without specific validators
  - [✓] Design pluggable architecture for adding language-specific syntax validators
  - [✓] Document the syntax validation logic and extension points

- [✓] **19.5 Update Generator and Tokenizer**
  - [✓] Ensure `generator.py` properly handles abstract tokens for all languages
  - [✓] Review template generation to support language-agnostic patterns
  - [✓] Update any language-specific logic in the tokenizer
  - [✓] Test token distribution monitoring with multi-language support
  - [✓] Add language-specific grammar rules (where applicable)

- [✓] **19.6 Make Code Executor Language-Aware**
  - [✓] Update `executor.py` to support multiple programming languages
  - [✓] Add configuration-based language-to-command mapping
  - [✓] Add configuration-based language-to-file-extension mapping
  - [✓] Modify `training_loop.py` to pass language information to executor
  - [✓] Test execution with different programming languages
  
- [ ] **19.7 Testing and Validation**
  - [ ] Create multi-language benchmark tasks (starting with Python and JavaScript)
  - [ ] Implement tests to verify correct normalization across languages
  - [ ] Validate training performance on non-Python tasks
  - [ ] Measure and optimize any performance impact from dynamic language loading
  - [ ] Document language support status and limitations

### 21. LLM-Guided Model Health Monitoring [NEW, PLANNED]

**Goal**: Create a robust system for monitoring and correcting pathological model states using LLM guidance

- [ ] **21.1 Design Model Health Metrics**
  - [ ] Implement comprehensive probability distribution analysis (entropy, kurtosis, skewness)
  - [ ] Track distribution collapse events (consecutive zero-entropy outputs)
  - [ ] Monitor gradient statistics over time with anomaly detection
  - [ ] Create token diversity metrics to identify repetitive patterns
  - [ ] Track exploration-exploitation balance through custom metrics

- [ ] **21.2 Build LLM Diagnostic Interface**
  - [ ] Create `model_health_monitor.py` module with LLM integration
  - [ ] Develop standardized diagnostic protocol for collecting metrics
  - [ ] Design prompt templates for different diagnostic scenarios
  - [ ] Implement selective LLM invocation based on severity triggers
  - [ ] Build a "diagnosis history" to track interventions and outcomes

- [ ] **21.3 Implement Partial Weight Reset Strategy**
  - [ ] Design selective weight reset methods (output layer only, specific layers)
  - [ ] Implement gradient re-normalization without full reset
  - [ ] Create "weight checkpointing" to restore to previous healthy states
  - [ ] Develop progressive reset strategy (least invasive to most invasive)
  - [ ] Add entropy injection mechanisms that preserve learned patterns

- [ ] **21.4 Create Multi-Model Guidance System**
  - [ ] Implement cycling through different LLM models for diagnosis
  - [ ] Create consensus mechanism to prevent single-model bias
  - [ ] Add model-switching based on task type or error pattern
  - [ ] Implement confidence scoring for model recommendations
  - [ ] Build automated prompt adjustment based on model performance

- [ ] **21.5 Integrate with Training Loop**
  - [ ] Add health check scheduling in training loop (every N attempts)
  - [ ] Implement intervention triggers based on anomaly detection
  - [ ] Create logging and reporting for health-related interventions
  - [ ] Add configuration options for health monitoring frequency
  - [ ] Implement emergency intervention for critical distribution collapse

- [ ] **21.6 Automate Recovery Actions**
  - [ ] Create an action framework for implementing LLM recommendations
  - [ ] Build pattern-specific intervention strategies
  - [ ] Implement temperature management based on distribution analysis
  - [ ] Add token probability redistribution for targeted recovery
  - [ ] Create automatic task simplification when model is struggling

### 22. Benchmark-Agnostic Recovery Strategies [NEW, PLANNED]

**Goal**: Expand on the current emergency distribution recovery with more sophisticated, language-agnostic approaches

- [ ] **22.1 Implement Distribution Analysis**
  - [ ] Create probability distribution visualization tools
  - [ ] Add entropy profiling across multiple generation attempts
  - [ ] Implement anomaly detection for distribution patterns
  - [ ] Track correlation between distribution shapes and outcomes
  - [ ] Create benchmark-independent distribution norms

- [ ] **22.2 Develop Progressive Recovery Techniques**
  - [ ] Implement staged recovery based on distribution severity
  - [ ] Create token-level intervention strategies
  - [ ] Add layer-specific gradient clamping during recovery
  - [ ] Implement exploration rate boosting for repetitive patterns
  - [ ] Design recovery verification through distribution metrics

- [ ] **22.3 Add Adaptive Learning Rate Management**
  - [ ] Create learning rate cycling based on collapse detection
  - [ ] Implement temporary gradient scaling during recovery
  - [ ] Add parameter-specific learning rate adjustments
  - [ ] Build adaptive momentum management for stability
  - [ ] Implement recovery-specific optimizer configurations

- [ ] **22.4 Create Benchmark-Specific Recovery Profiles**
  - [ ] Analyze collapse patterns across different benchmark types
  - [ ] Build category-based recovery profiles (math, string, loop, etc.)
  - [ ] Implement adaptive token boosting based on task keywords
  - [ ] Create benchmark complexity assessment to guide recovery
  - [ ] Build a recovery strategy database that grows with experience

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

8. **Eighth Priority: vLLM Integration (16.1, 16.2)**
   - Replace OPENAI/LM Studio with vLLM for better performance and control
   - Implement request optimization to reduce costs and improve throughput

9. **Ninth Priority: Multi-Language Support (19.1, 19.2, 19.3)**
   - ✓ COMPLETED: Implement language mapping system for language-agnostic code generation
   - ✓ COMPLETED: Add JavaScript as second supported language
   - ✓ COMPLETED: Make syntax checking language-aware
   - ✓ COMPLETED: Make code executor language-aware

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
*   **2025-04-09 (4):** Implemented hint impact tracking (Task 1.3). Added logic to `TrainingLoop` to record the score before a hint is provided and the score on the subsequent attempt. Passed this history to `TrainingReportGenerator` and added the average hint improvement metric to the report template (`training_report.md`).
*   **2025-04-09 (5):** Reviewed Task 11.1 (Structured Tool Feedback). The existing `ToolFeedback` class in `reinforcer/tool_feedback.py` already implements the core requirements, including classification, severity, and relevant token identification. Marked Task 11.1 as complete.
*   **2025-04-09 (6):** Implemented parts of Task 11.2 (Enhanced Learning from Tool Results): Added differential token weighting in `generator.learn` based on feedback severity and type. Modified `generator._update_learning_rate` to adjust LR reduction based on error severity/type. Noted that implementing target embeddings and execution trace analysis requires more significant architectural changes.
*   **2025-04-09 (7):** Implemented feedback-guided exploration (Task 11.3). Stored last step's feedback in `TrainingLoop`. Modified `generator.generate` and `generator.generate_with_beam_search` to accept `last_feedback` and use it to adjust temperature and penalize problematic tokens.
*   **2025-04-09 (8):** Completed more parts of Task 11.3: Added boosting for tokens from recent high-reward experiences in the generator methods. Implemented feedback memory persistence in `AttemptManager` and used this history in the generator methods to apply stronger penalties to tokens consistently involved in errors for a given code fingerprint.
*   **2025-04-09 (9):** Enhanced pattern analysis in `TrainingReportGenerator` (Task 13.4) by replacing basic keyword counting with AST analysis to identify common code structures (functions, loops, conditionals, etc.) in the highest and lowest scoring attempts.
*   **2025-04-09 (10):** Completed feedback memory implementation (Task 11.3). Passed feedback history for the previous fingerprint to generator methods. Added logic to analyze this history and apply stronger penalties to tokens frequently associated with errors for that fingerprint.
*   **2025-04-09 (11):** Added three new benchmark tasks (`hello_variable`, `simple_if`, `basic_for_loop`) to increase complexity variety (Task 6.2). Reviewed Task 15 (Runner Modularization) and updated status based on current file structure, marking 15.1 and 15.2 as complete.
*   **2025-04-09 (12):** Implemented reward normalization (Task 3.1) in `TrainingLoop._perform_learning_step` using running mean and standard deviation to stabilize the learning signal passed to the generator.
*   **2025-04-09 (13):** Implemented a `reset_weights` method in the `CodeGenerator` class to fix an error when attempting to reset model weights. This enables the system to break out of local minima more effectively by reinitializing the model with fresh weights when training stagnates.
*   **2025-04-09 (14):** Improved the hint generation system in `TrainingLoop` with a more scaffolding-oriented approach. Modified `_get_hint_from_advisor` to provide simpler, more code-focused hints (keywords, function names, operators, or short code snippets) rather than full sentences. Added a new `_post_process_hint` method to ensure brevity and remove explanatory phrases from generated hints.
*   **2025-04-09 (15):** Reduced log file size from 5MB to 2.5MB per file in `runner.py` and configured log cleanup with the `--reset` flag. Modified `AttemptManager.reset_training_state()` to also delete log files and training report files when resetting for a fresh training run.
*   **2025-04-09 (16):** Added LM Studio integration to training reports by implementing `_call_local_llm` and `add_llm_observations` methods in `TrainingReportGenerator`. This feature automatically connects to a locally running LM Studio model to generate AI-powered observations and insights about the training process, adding them to the end of each report.
*   **2025-04-09 (17):** Implemented several generator enhancements: Repetition Penalty (within sequence), Grammar Boost Decay, and Early Stopping (Entropy/Repetition). Configurable via `config.ini`.
*   **2025-04-09 (18):** Transitioned vocabulary (`memory/vocab.json`) to language-agnostic abstract tokens (Task 14.1). Updated `tokenizer.py` (removed INDENT/DEDENT) and `agent_core/generator.py` (removed Python grammar/templates, added abstract grammar/templates/hint parsing). **This is a breaking change requiring model retraining (`--reset` flag).**
*   **2025-04-10:** Implemented multi-language support (Task 19). Updated `executor.py` to support different programming languages based on task language. Created `execution_config.json` with language-specific command and file extension mappings. Modified `training_loop.py` to pass language information to the executor. Verified that the `AttemptScorer` class already had language-specific capabilities through `normalize_for_scoring` and `_evaluate_syntax` methods. Confirmed that `generator.py` was already using a language-agnostic approach for generating code. The system can now train on and execute code in various programming languages, not just Python.
*   **2025-04-10 (2):** Started Task 16 (vLLM Integration). Created `llm_provider.py` module with `LLMProvider` class to abstract LLM calls. Implemented support for vLLM and fallback to LM Studio, configurable via new `[LLM]` section in `config.ini`. Migrated LLM calls in `TrainingReportGenerator` and `TrainingLoop` (for hints) to use the new provider. Removed old LLM call methods from `TrainingLoop`.
*   **2025-04-10 (3):** Implemented request caching and batch interface in `LLMProvider` (Task 16.2). Added basic performance monitoring: latency measurement, token counting, request counters, and detailed logging (Task 16.4). These optimizations reduce redundant LLM calls and provide insights into LLM usage patterns.
*   **2025-04-10 (4):** Implemented adaptive request throttling in `LLMProvider` based on average latency (Task 16.4). If average latency exceeds a configurable threshold, the provider delays new requests to avoid overloading the LLM server.
