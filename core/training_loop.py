#!/usr/bin/env python3
"""
AltLAS TrainingLoop - Module for managing the core training loop.
"""

import os
import sys
import time
import json
import random
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import necessary components (adjust paths as needed)
from agent_core.generator import CodeGenerator
from guardian.safety import SafetyChecker
from evaluator.executor import CodeExecutor
from reinforcer.scorer import AttemptScorer
from memory.report_generator import TrainingReportGenerator
from task.task_loader import Task
from core.attempt_manager import AttemptManager
from core.ui_display import UIDisplay
from core.config_loader import ConfigLoader

log = logging.getLogger(__name__)

class TrainingLoop:
    """
    Manages the core training loop for AltLAS.
    """
    def __init__(self, 
                 config_loader: ConfigLoader,
                 attempt_manager: AttemptManager,
                 ui_display: UIDisplay,
                 generator: CodeGenerator,
                 safety_checker: SafetyChecker,
                 executor: CodeExecutor,
                 scorer: AttemptScorer,
                 report_generator: TrainingReportGenerator,
                 current_task: Task,
                 device: str):
        """
        Initialize the TrainingLoop.
        
        Args:
            config_loader: Instance of ConfigLoader.
            attempt_manager: Instance of AttemptManager.
            ui_display: Instance of UIDisplay.
            generator: Instance of CodeGenerator.
            safety_checker: Instance of SafetyChecker.
            executor: Instance of CodeExecutor.
            scorer: Instance of AttemptScorer.
            report_generator: Instance of TrainingReportGenerator.
            current_task: The current task object.
            device: The PyTorch device to use.
        """
        self.config_loader = config_loader
        self.attempt_manager = attempt_manager
        self.ui_display = ui_display
        self.generator = generator
        self.safety_checker = safety_checker
        self.executor = executor
        self.scorer = scorer
        self.report_generator = report_generator
        self.current_task = current_task
        self.device = device
        
        # Load configurations
        self.runner_config = self.config_loader.get_runner_config()
        self.scorer_config = self.config_loader.get_scorer_config()
        
        # Initialize loop state
        self.attempt_count = 0
        self.success = False
        self.user_interrupted = False # Flag for Ctrl+C
        self.run_exception = None # Store any unexpected exception
        
        # Stuck detection parameters
        self.last_best_score_at_check = 0.0
        self.attempts_since_last_check = 0
        self.consecutive_stuck_count = 0
        self.current_hint = None
        
        # Statistics counters
        self.stats = {
            "Total Attempts": 0,
            "Success Attempts": 0,
            "Error Attempts": 0,
            "Duplicate Attempts": 0,
            "Unsafe Attempts": 0,
            "Hints Requested": 0,
            "Hints Provided": 0,
            "Highest Score": 0.0,
            "Best Attempt": 0,
            "Best Code": None,
            "Current Entropy Coef": self.generator.max_entropy_coefficient # Initial value
        }
        
        # Additional counters
        self.stuck_events = 0
        self.beam_search_uses = 0
        
        # Weight reset tracking
        self.consecutive_beam_search_without_improvement = 0
        self.last_improvement_iteration = 0
        self.weight_reset_count = 0
        self.beam_search_cooling_off_period = 0
        
        # Hint backoff tracking
        self.hint_backoff_level = 0
        
        # Hint impact tracking
        self.hint_just_provided = False
        self.score_before_hint = None
        self.hint_impact_history = [] # List to store (score_before, score_after) tuples
        
        # Feedback from last step
        self.last_tool_feedback = None
        
        # Reward normalization stats
        self.reward_sum = 0.0
        self.reward_sum_sq = 0.0
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_std = 1.0 # Initialize std to 1 to avoid division by zero initially
    
    def set_user_interrupted(self):
        """Set the user interrupted flag."""
        self.user_interrupted = True
        log.warning("🏃 User interrupted the run (Ctrl+C). Exiting gracefully.")
    
    def run(self):

        # Check for task-specific resource limits
        task_limits = getattr(self.current_task, 'resource_limits', {})
        if task_limits:
            mem_limit = task_limits.get('memory_limit')
            cpu_limit = task_limits.get('cpu_limit')
            timeout = task_limits.get('timeout')
            if mem_limit:
                logging.info(f"Task memory limit: {mem_limit} MB")
            if cpu_limit:
                logging.info(f"Task CPU limit: {cpu_limit}%")
            if timeout:
                logging.info(f"Task timeout: {timeout} seconds")
            # TODO: Enforce these limits during execution

        """Execute the main training loop."""
        max_attempts = self.runner_config['max_attempts']
        success_threshold = self.scorer_config['success_threshold']
        
        # Initialize UI
        task_id = self.ui_display.initialize(max_attempts, self.current_task.name)
        
        # Load previous state if available
        self._load_initial_state()
        
        # Start the Live UI display
        live = self.ui_display.start()
        try:
            with live: # Use the context manager
                while self.attempt_count < max_attempts and not self.user_interrupted:
                    try:
                        self.attempt_count += 1
                        self.stats["Total Attempts"] = self.attempt_count
                        self.attempts_since_last_check += 1
                        
                        # --- Periodic Training Report Generation ---
                        self._maybe_generate_report()
                        
                        # --- Stuck Detection & Hinting ---
                        self._update_stuck_status()
                        
                        # --- Generate Code --- 
                        code_attempt, generated_ids = self._generate_code_attempt()
                        if code_attempt is None or generated_ids is None:
                            self.ui_display.add_status_message("⚠️ Generator returned None unexpectedly. Skipping attempt.")
                            continue
                        
                        # --- Safety/Novelty Check --- 
                        if not self.safety_checker.is_safe(code_attempt):
                            self.ui_display.add_status_message("⚠️ Unsafe code attempt detected. Skipping.")
                            log.warning(f"⚠️ Unsafe code attempt detected at attempt {self.attempt_count}. Skipping.")
                            self.stats["Unsafe Attempts"] += 1
                            continue
                        
                        fingerprint = self.attempt_manager.get_fingerprint(code_attempt)
                        if self.attempt_manager.is_duplicate(fingerprint):
                            self.ui_display.add_status_message(f"⚠️ Duplicate code attempt detected (Hash: {fingerprint[:8]}...). Skipping.")
                            log.warning(f"⚠️ Duplicate code attempt detected at attempt {self.attempt_count} (Hash: {fingerprint}). Skipping.")
                            self.stats["Duplicate Attempts"] += 1
                            continue
                        
                        # --- Execute Code --- 
                        result = self.executor.execute(code_attempt)
                        
                        # --- Score Result --- 
                        score = self.scorer.score(code_attempt, result, self.current_task)
                        
                        # --- Update status message --- 
                        self.ui_display.add_status_message(f"📝 Attempt {self.attempt_count} scored {score:.2f}")
                        if score >= success_threshold:
                            self.stats["Success Attempts"] += 1
                        else:
                            self.stats["Error Attempts"] += 1
                        
                        # --- Log Attempt --- 
                        self.attempt_manager.log_attempt(self.attempt_count, code_attempt, result, score, fingerprint)
                        
                        # --- Perform Learning Step --- 
                        self._perform_learning_step(score, generated_ids, code_attempt, result)
                        
                        # --- Update UI --- 
                        self._update_ui_display(code_attempt)
                        
                        # --- Check for Success --- 
                        if score >= success_threshold and not self.success:
                            self.success = True
                            self.ui_display.add_status_message("🎉 Success! Continuing to run until max attempts...")
                            log.info(f"🎉 SUCCESS! Task solved on attempt {self.attempt_count}. Will continue until max attempts.")
                        
                        # --- Save Run State --- 
                        self._save_current_state()
                        
                    except Exception as inner_e:
                        # Log errors using the configured logger
                        log.error(f"❌ Error during attempt {self.attempt_count}: {type(inner_e).__name__} - {inner_e}", exc_info=True)
                        self.ui_display.add_status_message(f"❌ Error in attempt {self.attempt_count}. See logs. Skipping attempt.")
                        continue
                # --- End Main Loop ---
                
        except Exception as e:
            log.exception(f"Unhandled exception during training: {e}")
            self.run_exception = e # Store exception for final summary
        finally:
            # Explicitly stop the live display before printing final summary
            if live and live.is_started:
                live.refresh() # Refresh one last time
                live.stop()
                log.debug("Live display explicitly stopped after final refresh.")
                time.sleep(0.1) # Add a small delay
            
            # Perform final actions (saving, reporting, summarizing)
            self._finalize_run()
    
    def _load_initial_state(self):
        """Load state from a previous run if applicable."""
        previous_state = self.attempt_manager.load_run_state(task_name=self.current_task.name)
        if previous_state:
            self.attempt_count = previous_state["attempt_count"]
            self.success = previous_state["success"]
            log.info(f"🔄 Resuming from previous run: attempt_count={self.attempt_count}, success={self.success}")
            self.ui_display.add_status_message(f"🔄 Resuming from previous run (attempt {self.attempt_count})")
            
            # If already successful but continuing for more attempts
            if self.success:
                self.ui_display.add_status_message("🎉 Already found successful solution, continuing for remaining attempts")
            
            # Update initial UI state
            self.ui_display.update_attempt(self.attempt_count, self.runner_config['max_attempts'])
        else:
            log.info("Starting fresh run (no previous state or reset requested)")
    
    def _save_current_state(self):
        """Save the current run state."""
        best_score_val, best_attempt_num = self.attempt_manager.get_best_score_info()
        self.attempt_manager.save_run_state(
            attempt_count=self.attempt_count,
            task_name=self.current_task.name,
            success=self.success,
            best_score=best_score_val,
            best_attempt=best_attempt_num
        )
    
    def _maybe_generate_report(self):
        """Generate a training report if the frequency condition is met."""
        report_frequency = self.runner_config['report_frequency']
        if report_frequency > 0 and self.attempt_count % report_frequency == 0:
            log.info(f"Attempt {self.attempt_count}: Triggering training report generation.")
            try:
                gen_state = self.generator.get_state()
                scr_state = self.scorer.get_state()
                attempt_history = self.attempt_manager.get_history()

                log.debug(f"Calling generate_and_save_report with history length: {len(attempt_history)}")
                report_path = self.report_generator.generate_and_save_report(
                    attempt_count=self.attempt_count,
                    max_attempts=self.runner_config['max_attempts'],
                    current_task=self.current_task,
                    history=attempt_history,
                    generator_state=gen_state,
                    scorer_state=scr_state,
                    start_time=self.ui_display.start_time, # Get start time from UI display
                    hints_provided=self.stats["Hints Provided"],
                    success_count=self.stats["Success Attempts"],
                    stuck_events=self.stuck_events,
                    beam_search_uses=self.beam_search_uses,
                    hint_impact_history=self.hint_impact_history # Pass hint impact data
                )
                if report_path:
                    log.info(f"📊 Training report generation successful. Report saved to {report_path}")
                    self.ui_display.add_status_message(f"📊 Report generated: {Path(report_path).name}")
                else:
                    log.warning(f"⚠️ Training report generation process indicated failure for attempt {self.attempt_count}.")
            except Exception as e:
                log.error(f"⚠️ UNEXPECTED Error during training report generation trigger for attempt {self.attempt_count}: {e}", exc_info=True)
                self.ui_display.add_status_message(f"❌ Error generating report for attempt {self.attempt_count}")
    
    def _update_stuck_status(self):
        """Check for stuck state and potentially request a hint."""
        stuck_check_window = self.runner_config['stuck_check_window']
        stuck_threshold = self.runner_config['stuck_threshold']
        max_consecutive_stuck_checks = self.runner_config['max_consecutive_stuck_checks']
        base_hint_probability_on_stuck = self.runner_config['hint_probability_on_stuck']
        
        if self.attempts_since_last_check >= stuck_check_window:
            current_best_score = self.attempt_manager.get_best_score()
            if current_best_score - self.last_best_score_at_check < stuck_threshold:
                self.consecutive_stuck_count += 1
                self.stuck_events += 1  # Increment stuck detection counter
                status_msg = f"📉 Stuck detected (Check {self.consecutive_stuck_count}/{max_consecutive_stuck_checks})"
                self.ui_display.add_status_message(status_msg)
                log.info(f"{status_msg}. Score hasn't improved enough.")
                
                if self.consecutive_stuck_count >= max_consecutive_stuck_checks:
                    # Calculate effective hint probability with exponential backoff
                    effective_hint_probability = base_hint_probability_on_stuck / (2 ** self.hint_backoff_level)
                    log.debug(f"Checking hint probability: base={base_hint_probability_on_stuck:.3f}, backoff_level={self.hint_backoff_level}, effective={effective_hint_probability:.3f}")

                    if random.random() < effective_hint_probability:
                        self.stats["Hints Requested"] += 1
                        # Store score *before* getting the hint
                        self.score_before_hint = self.attempt_manager.get_best_score()
                        self.current_hint = self._get_hint_from_advisor()
                        if self.current_hint:
                            self.ui_display.add_status_message(f"🤔 Hint requested: {self.current_hint[:50]}...")
                            log.info(f"🤔 Hint requested (Effective Prob: {effective_hint_probability:.3f}): {self.current_hint}")
                            self.stats["Hints Provided"] += 1
                            self.hint_backoff_level = 0 # Reset backoff after successful hint
                            self.hint_just_provided = True # Flag that a hint was just given for the next attempt
                        else:
                            self.ui_display.add_status_message("🤷 Advisor couldn't generate hint")
                            log.warning(f"🤷 Advisor couldn't generate hint (Effective Prob: {effective_hint_probability:.3f})")
                            # Don't reset backoff if advisor fails
                            self.score_before_hint = None # Reset score_before_hint if hint generation failed
                            self.hint_just_provided = False
                        self.consecutive_stuck_count = 0 # Reset stuck count after hint attempt (success or fail)
                    else:
                        log.info(f"🚫 Hint skipped due to backoff probability (Effective Prob: {effective_hint_probability:.3f}) despite meeting consecutive stuck threshold.")
                        self.hint_backoff_level += 1 # Increase backoff level when hint is skipped
                        self.current_hint = None
                        self.hint_just_provided = False
                        self.score_before_hint = None
                else:
                    self.current_hint = None # No hint if not reached max consecutive checks
                    self.hint_just_provided = False
                    self.score_before_hint = None
            else:
                # Progress detected
                if self.consecutive_stuck_count > 0:
                    log.info("📈 Progress detected. Resetting consecutive stuck counter and hint backoff.")
                    self.ui_display.add_status_message("📈 Progress detected. Resetting stuck counter.")
                self.consecutive_stuck_count = 0
                self.hint_backoff_level = 0 # Reset backoff level on progress
                self.current_hint = None
                self.hint_just_provided = False
                self.score_before_hint = None
            
            # Reset check window
            self.last_best_score_at_check = current_best_score
            self.attempts_since_last_check = 0
        else:
            # If not checking window, ensure hint state is reset if not actively stuck or hinting
            if self.consecutive_stuck_count == 0:
                 self.current_hint = None
                 self.hint_just_provided = False
                 self.score_before_hint = None
    
    def _generate_code_attempt(self) -> Tuple[Optional[str], Optional[List[int]]]:
        """Generate a code attempt using the appropriate method."""
        current_best_score = self.attempt_manager.get_best_score()
        
        # --- Get Feedback History for Previous Fingerprint --- 
        feedback_history_for_fp = None
        if self.attempt_count > 1 and self.attempt_manager.attempts:
            last_attempt = self.attempt_manager.attempts[-1]
            last_fingerprint = last_attempt.get('fingerprint')
            last_score = last_attempt.get('score', 1.0)
            # Only use feedback memory if the last attempt failed and had a fingerprint
            if last_fingerprint and last_score < 0.9:
                feedback_history_for_fp = self.attempt_manager.feedback_memory.get(last_fingerprint)
                if feedback_history_for_fp:
                    log.debug(f"Retrieved feedback history ({len(feedback_history_for_fp)} entries) for previous fingerprint {last_fingerprint[:8]}...")
        # --- End Get Feedback History ---
        
        # Check if we've made progress since the last time we checked
        if current_best_score > self.last_best_score_at_check + 0.01:
            # Progress has been made
            self.last_improvement_iteration = self.attempt_count
            self.consecutive_beam_search_without_improvement = 0
            
            # If progress was made, we'll reset beam search cooling off period
            self.beam_search_cooling_off_period = 0
        
        # Check if we've been stuck in beam search for too long without improvement
        beam_search_stagnation_threshold = 500  # After 500 beam search attempts without improvement, try reset
        if (self.beam_search_uses > 1000 and  # Only activate after significant beam search usage
            self.attempt_count - self.last_improvement_iteration > 1000 and  # No improvement for 1000 attempts
            self.consecutive_beam_search_without_improvement >= beam_search_stagnation_threshold):
            
            # Reset model weights to break out of local minimum
            self.ui_display.add_status_message(f"🔄 Resetting model weights to break out of local minimum")
            log.warning(f"🔄 Beam search has been used {self.consecutive_beam_search_without_improvement} times without improvement. Resetting model weights.")
            
            # Reinitialize model weights
            try:
                self.generator.reset_weights()
                self.weight_reset_count += 1
                self.ui_display.add_status_message(f"🔄 Model weights reset successfully (reset #{self.weight_reset_count})")
                log.info(f"🔄 Model weights reset successfully (reset #{self.weight_reset_count})")
                
                # After reset, we'll use normal generation for a while to stabilize
                self.beam_search_cooling_off_period = 200
                self.consecutive_beam_search_without_improvement = 0
                
                # Use a high temperature immediately after reset to encourage exploration
                temperature = 1.0
                log.info(f"Using increased temperature={temperature} after weight reset to encourage exploration")
                
                return self.generator.generate(
                    self.current_task, 
                    self.attempt_manager.get_history(max_entries=5),  # Fewer history entries to reduce bias from old patterns
                    hint=self.current_hint, 
                    temperature=temperature,
                    last_feedback=self.last_tool_feedback, 
                    feedback_history_for_fp=feedback_history_for_fp # Pass feedback history
                )
            except Exception as reset_error:
                log.error(f"❌ Error resetting model weights: {str(reset_error)}")
                self.ui_display.add_status_message(f"❌ Error resetting model weights")
                # Fall through to normal generation logic
        
        # If we're in the cooling-off period after weight reset, avoid beam search
        if self.beam_search_cooling_off_period > 0:
            self.beam_search_cooling_off_period -= 1
            log.debug(f"In beam search cooling-off period ({self.beam_search_cooling_off_period} attempts left)")
            
            # Dynamic temperature based on recent progress
            if self.attempt_count - self.last_improvement_iteration < 100:
                temperature = 0.7  # Moderate exploration
            else:
                temperature = 0.9  # More exploration
                
            return self.generator.generate(
                self.current_task, 
                self.attempt_manager.get_history(max_entries=10),
                hint=self.current_hint,
                temperature=temperature,
                last_feedback=self.last_tool_feedback, 
                feedback_history_for_fp=feedback_history_for_fp # Pass feedback history
            )
            
        # Token distribution analysis to detect repetitive patterns
        if hasattr(self.generator, 'token_frequency') and self.generator.token_frequency:
            # Check for extreme token imbalance - symptom of getting stuck
            sorted_freq = sorted(self.generator.token_frequency.items(), key=lambda x: x[1], reverse=True)
            if sorted_freq and len(sorted_freq) > 1:
                top_token, top_count = sorted_freq[0]
                second_token, second_count = sorted_freq[1]
                
                # If top token is dramatically overrepresented (10x more than second token)
                if top_count > second_count * 10 and top_count > 10000:
                    self.ui_display.add_status_message(f"⚠️ Token distribution imbalance detected ('{top_token}' overused)")
                    log.warning(f"Token distribution imbalance: '{top_token}' appears {top_count} times (unhealthy ratio)")
                    
                    # If in beam search mode already, this indicates beam search is getting stuck too
                    if self.consecutive_stuck_count >= 2:
                        # Skip beam search occasionally even when stuck to break repetitive patterns
                        if random.random() < 0.4:  # 40% chance to skip beam search even when stuck
                            self.ui_display.add_status_message(f"🔄 Skipping beam search to break repetitive pattern")
                            temperature = 1.0  # High temperature for exploration
                            return self.generator.generate(
                                self.current_task,
                                self.attempt_manager.get_history(max_entries=5),
                                hint=self.current_hint,
                                temperature=temperature,
                                last_feedback=self.last_tool_feedback, 
                                feedback_history_for_fp=feedback_history_for_fp # Pass feedback history
                            )
        
        # Check for repetitive failures to boost exploration
        apply_exploration_boost = False
        if self.attempt_manager.check_recent_repetitive_failures(lookback=30):
            # If recent history is dominated by known failed patterns, boost exploration
            apply_exploration_boost = True
            log.info("Boosting exploration due to recent repetitive failures.")
            self.ui_display.add_status_message("🚀 Boosting exploration (repetitive failures)")

        # Normal generation strategy logic
        if self.consecutive_stuck_count >= 2 and self.beam_search_cooling_off_period <= 0 and not apply_exploration_boost:
            # Only use beam search when stuck, not cooling off, and not boosting exploration
            self.ui_display.add_status_message(f"🔄 Using beam search generation (stuck count: {self.consecutive_stuck_count})")
            log.info(f"🔄 Using beam search generation due to stuck detection (count: {self.consecutive_stuck_count})")
            self.beam_search_uses += 1
            self.consecutive_beam_search_without_improvement += 1
            
            # Vary beam search parameters based on how long we've been stuck
            if self.consecutive_beam_search_without_improvement > 300:
                # If beam search has been used a lot without progress, increase beam width
                beam_width = 5
                log.info(f"Increasing beam width to {beam_width} due to prolonged stagnation")
            else:
                beam_width = 3
                
            return self.generator.generate_with_beam_search(
                self.current_task, 
                self.attempt_manager.get_history(max_entries=10),
                hint=self.current_hint,
                beam_width=beam_width,
                last_feedback=self.last_tool_feedback, 
                feedback_history_for_fp=feedback_history_for_fp # Pass feedback history
            )
        else:
            # Use standard generation with dynamic temperature
            temperature = 0.8 # Default exploration temperature
            if apply_exploration_boost:
                temperature = 1.0 # Max exploration boost
            elif current_best_score > 0.5:
                temperature = 0.5  # More focused when we're getting close
            elif current_best_score > 0.3:
                temperature = 0.7  # Moderate exploration/exploitation balance
            else:
                # For lower scores, check if we're stuck in a pattern
                if self.attempt_count - self.last_improvement_iteration > 500:
                    temperature = 0.9  # High exploration when stuck at low scores
                # else temperature remains 0.8 (default)
            
            log.debug(f"Using standard generation with temperature: {temperature}")
            return self.generator.generate(
                self.current_task, 
                self.attempt_manager.get_history(max_entries=10),
                hint=self.current_hint, 
                temperature=temperature,
                last_feedback=self.last_tool_feedback, 
                feedback_history_for_fp=feedback_history_for_fp # Pass feedback history
            )
    
    def _perform_learning_step(self, score: float, generated_ids: List[int], 
                              code_attempt: str, result: Dict[str, Any]):
        """Perform the learning step using the generator."""
        # --- Reward Normalization (Update Stats) ---
        self.reward_count += 1
        self.reward_sum += score
        self.reward_sum_sq += score**2
        # Update mean and std using Welford's online algorithm or simpler method for stability
        # Simple running mean and std calculation:
        self.reward_mean = self.reward_sum / self.reward_count
        # Calculate variance: E[X^2] - (E[X])^2
        variance = (self.reward_sum_sq / self.reward_count) - (self.reward_mean ** 2)
        # Ensure variance is non-negative due to potential floating point errors
        variance = max(0, variance)
        self.reward_std = variance ** 0.5
        # Avoid division by zero if std dev is very small
        epsilon = 1e-8 
        if self.reward_std < epsilon:
            self.reward_std = epsilon
        # --- End Reward Normalization Stats Update ---
        
        # Calculate dynamic entropy coefficient
        if self.attempt_count > 0:
            success_rate = self.stats["Success Attempts"] / self.attempt_count
            # Anneal entropy: start high, decrease as success rate increases
            annealing_factor = 1.0 - success_rate
            current_entropy_coef = (
                self.generator.min_entropy_coefficient + 
                (self.generator.max_entropy_coefficient - self.generator.min_entropy_coefficient) * annealing_factor
            )
            # Ensure it doesn't go below the minimum
            current_entropy_coef = max(self.generator.min_entropy_coefficient, current_entropy_coef)
        else:
            current_entropy_coef = self.generator.max_entropy_coefficient # Start with max entropy
        
        self.stats["Current Entropy Coef"] = current_entropy_coef
        
        # Create tool feedback from execution result
        tool_feedback = self.scorer.get_tool_feedback(code_attempt, result)
        
        # Calculate more graduated rewards using the specialized reward calculation method
        use_calculate_reward = self.runner_config.get('use_calculated_reward', True)
        
        if use_calculate_reward:
            # ExecutionResult is an object, not a dict, so access attributes directly
            success = hasattr(result, 'status') and result.status == 'success'
            syntax_valid = tool_feedback.feedback_type != 'syntax_error'
            execution_valid = success or tool_feedback.feedback_type not in ['execution_timeout']
            has_output = bool(getattr(result, 'stdout', '').strip())
            
            # Check for structural elements in code
            has_structure = True
            has_correct_ops = False
            has_almost_correct_result = False
            correct_format = False
            
            try:
                import ast
                tree = ast.parse(code_attempt)
                # Simple detection of correct operations
                if hasattr(self.current_task, 'constraints') and 'required_operators' in self.current_task.constraints:
                    required_ops = self.current_task.constraints['required_operators']
                    has_correct_ops = all(op in code_attempt for op in required_ops)
                
                # Check for almost correct results and format
                if has_output:
                    expected_output = getattr(self.current_task, 'success_criteria', {}).get('expected_output', '').strip()
                    actual_output = getattr(result, 'stdout', '').strip()
                    
                    # If expected output exists, check similarity
                    if expected_output:
                        import difflib
                        similarity = difflib.SequenceMatcher(None, actual_output, expected_output).ratio()
                        has_almost_correct_result = similarity > 0.8
                        correct_format = similarity > 0.95
                        
            except Exception:
                pass  # Silently skip if analysis fails

            # Calculate reward using the more graduated system
            calculated_reward = self.scorer.calculate_reward(
                code_output=getattr(result, 'stdout', ''),
                success=success,
                syntax_valid=syntax_valid,
                execution_valid=execution_valid,
                has_output=has_output,
                has_structure=has_structure,
                has_correct_ops=has_correct_ops,
                has_almost_correct_result=has_almost_correct_result,
                correct_format=correct_format,
                config=self.scorer_config
            )
            
            # If we have a valid calculated reward, use it instead of the baseline score
            # This provides more graduated feedback for learning
            if calculated_reward > 0.001:
                log.debug(f"Using calculated_reward={calculated_reward:.4f} instead of baseline score={score:.4f}")
                score = calculated_reward
        
        # Apply penalty for repeatedly generating failed patterns
        fingerprint = self.attempt_manager.get_fingerprint(code_attempt)
        failure_penalty = self.attempt_manager.get_failure_penalty(fingerprint)
        if failure_penalty > 0:
            original_score = score
            score -= failure_penalty
            score = max(0, score) # Ensure score doesn't go below 0
            log.info(f"Applied failure penalty of {failure_penalty:.2f} for repeated failed pattern (fingerprint: {fingerprint[:8]}...). Score reduced from {original_score:.4f} to {score:.4f}")
            self.ui_display.add_status_message(f"📉 Penalty applied for repeated failure ({failure_penalty:.2f})")

        # --- Hint Impact Tracking ---
        if self.hint_just_provided and self.score_before_hint is not None:
            score_after_hint = score # Use the final score for this attempt
            self.hint_impact_history.append((self.score_before_hint, score_after_hint))
            improvement = score_after_hint - self.score_before_hint
            log.info(f"Hint impact recorded: Score before={self.score_before_hint:.4f}, Score after={score_after_hint:.4f}, Improvement={improvement:+.4f}")
            # Reset hint tracking state for the next potential hint
            self.hint_just_provided = False
            self.score_before_hint = None
        # --- End Hint Impact Tracking ---

        # --- Normalize Score for Learning --- 
        normalized_score = (score - self.reward_mean) / self.reward_std
        log.debug(f"Reward Normalization: Raw={score:.4f}, Mean={self.reward_mean:.4f}, Std={self.reward_std:.4f}, Normalized={normalized_score:.4f}")
        # --- End Normalize Score ---

        # Pass the NORMALIZED score with dynamic coefficient and tool feedback to the learn method
        self.generator.learn(normalized_score, generated_ids, current_entropy_coef, tool_feedback=tool_feedback)
        
        # Store the feedback for the next generation step
        self.last_tool_feedback = tool_feedback
    
    def _update_ui_display(self, code_attempt: str):
        """Update the Rich UI display with the latest statistics."""
        best_score_val, best_attempt_num = self.attempt_manager.get_best_score_info()
        self.stats["Highest Score"] = best_score_val
        self.stats["Best Attempt"] = best_attempt_num
        
        # Get best attempt's code
        best_attempt_info = self.attempt_manager.get_attempt(best_attempt_num)
        if best_attempt_info:
            self.stats["Best Code"] = best_attempt_info.get('code', 'N/A')
        else:
             self.stats["Best Code"] = None # Reset if best attempt not found (e.g., after reset)
        
        self.ui_display.update_attempt(
            self.attempt_count, 
            self.runner_config['max_attempts'], 
            stats=self.stats, 
            hint=self.current_hint
        )
        
        # Log top tokens periodically
        log_frequency = self.runner_config['log_frequency']
        top_tokens_to_log = self.runner_config['top_tokens_to_log']
        if log_frequency > 0 and self.attempt_count % log_frequency == 0:
            if hasattr(self.generator, 'token_frequency') and self.generator.token_frequency:
                sorted_freq = sorted(self.generator.token_frequency.items(), key=lambda item: item[1], reverse=True)
                top_n = sorted_freq[:top_tokens_to_log]
                freq_str = ", ".join([f"'{token}': {count}" for token, count in top_n])
                log.info(f"[Attempt {self.attempt_count}] Top {top_tokens_to_log} generated tokens: {freq_str}") 
                log.debug(f"Attempt {self.attempt_count} full token frequencies: {self.generator.token_frequency}")
    
    def _finalize_run(self):
        """Perform final actions at the end of the run."""
        log.info("Finishing run...")
        
        # --- Save Learned Weights (Only on Clean Exit) ---
        if self.run_exception is None and (
               self.success or self.attempt_count >= self.runner_config['max_attempts'] or self.user_interrupted
           ):
            log.info("Attempting to save model and optimizer state...")
            self.generator.save_weights()
        elif self.run_exception is not None:
             log.warning(f"Run terminated due to an error ({type(self.run_exception).__name__}). Model state will NOT be saved.")
        else: 
             log.warning("Unknown run termination state. Model state will NOT be saved.")
        
        # --- Generate Final Report on Success ---
        if self.success and self.runner_config['report_on_success']:
           log.info("Task completed successfully. Generating final report...")
           try:
               gen_state = self.generator.get_state()
               scr_state = self.scorer.get_state()
               attempt_history = self.attempt_manager.get_history()

               log.debug(f"Calling generate_and_save_report (on success) with history length: {len(attempt_history)}")
               report_path = self.report_generator.generate_and_save_report(
                   attempt_count=self.attempt_count,
                   max_attempts=self.runner_config['max_attempts'],
                   current_task=self.current_task,
                   history=attempt_history,
                   generator_state=gen_state,
                   scorer_state=scr_state,
                   start_time=self.ui_display.start_time,
                   hints_provided=self.stats["Hints Provided"],
                   success_count=self.stats["Success Attempts"],
                   stuck_events=self.stuck_events,
                   beam_search_uses=self.beam_search_uses,
                   hint_impact_history=self.hint_impact_history # Pass hint impact data
               )
               if report_path:
                   log.info(f"📊 Final training report saved to {report_path}")
               else:
                   log.warning(f"⚠️ Final training report generation process indicated failure.")
           except Exception as final_report_e:
               log.error(f"⚠️ Error during final report generation: {final_report_e}", exc_info=True)
        
        # --- Log Final Token Frequencies --- 
        if hasattr(self.generator, 'token_frequency'):
            self.ui_display.print_token_frequencies(self.generator.token_frequency)
        
        # --- Print Summary --- 
        best_score_val, best_attempt_num = self.attempt_manager.get_best_score_info()
        final_stats = {
            "Total Attempts": self.stats["Total Attempts"],
            "Success Attempts": self.stats["Success Attempts"],
            "Error Attempts": self.stats["Error Attempts"],
            "Duplicate Attempts": self.stats["Duplicate Attempts"],
            "Unsafe Attempts": self.stats["Unsafe Attempts"],
            "Hints Requested": self.stats["Hints Requested"],
            "Hints Provided": self.stats["Hints Provided"]
        }
        self.ui_display.print_summary(
            success=self.success,
            user_interrupted=self.user_interrupted,
            attempt_count=self.attempt_count,
            max_attempts=self.runner_config['max_attempts'],
            best_score=best_score_val,
            best_attempt=best_attempt_num,
            run_stats=final_stats
        )

    # --- Advisor Hint Generation Logic --- 
    # (Moved from runner.py, slightly adapted to be methods of the class)
    
    def _get_hint_from_advisor(self) -> Optional[str]:
        """
        Get a hint from an external LLM advisor based on the current task and execution history.
        
        Returns:
            str or None: A hint to guide the agent, or None if no hint could be generated
        """
        try:
            history = self.attempt_manager.get_history(max_entries=5)
            
            # Extract the most relevant information from history
            recent_history = history[-5:] if history else []
            
            # Get the most recent errors and scores
            errors = [
                h.get('result', {}).get('error', '') 
                for h in recent_history 
                if h.get('result', {}).get('status') == 'error' and h.get('result', {}).get('error')
            ]
            recent_scores = [h.get('score', 0) for h in recent_history]
            
            # Get code samples from recent attempts
            code_samples = [h.get('code', '') for h in recent_history if h.get('code')]
            
            # Construct a prompt for the LLM
            prompt = f"""You are an AI programming advisor helping a learning agent solve a coding task.

TASK DESCRIPTION:
{self.current_task.name}
{self.current_task.description}

RECENT SCORES (higher is better):
{recent_scores}

RECENT ERRORS:
{errors[:3]}  # Limiting to most recent 3 errors

MOST RECENT CODE ATTEMPT:
```python
{code_samples[-1] if code_samples else 'No attempts yet'}
```

Based on this information, provide ONE specific, concise hint that would help the agent improve. 
Focus on ONE problem at a time. Limit your hint to one or two sentences.
"""
            
            # Call the LLM API
            hint = self._call_llm_api(prompt)
            
            if hint:
                log.info(f"💡 Advisor Hint: {hint}")
                return hint
            else:
                log.info("🤷 Advisor could not generate a hint")
                return None
                
        except Exception as e:
            log.error(f"⚠️ Error getting hint from advisor: {str(e)}")
            # Fallback to a generic hint if the API call fails
            return "Try a different approach."

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        Call an external LLM API to get a response.
        
        Returns:
            str or None: The LLM's response or None if the call failed
        """
        # Check for API key in environment or config
        api_key = os.environ.get('OPENAI_API_KEY')
        
        # If no API key, try using a local LLM via LM Studio
        if not api_key:
            return self._call_local_llm(prompt)
        
        try:
            # OpenAI API call
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful programming advisor giving concise hints."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=10
            )
            
            if response.status_code == 200:
                response_data = response.json()
                hint = response_data['choices'][0]['message']['content'].strip()
                return hint
            else:
                log.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            log.error(f"Error calling LLM API: {str(e)}")
            return None

    def _call_local_llm(self, prompt: str) -> Optional[str]:
        """
        Call a local LLM using LM Studio running on the host machine.
        """
        try:
            base_url = "http://localhost:1234/v1"
            log.info("🌐 Using localhost with network=host to connect to LM Studio")
            
            headers = {"Content-Type": "application/json"}
            chosen_model = None
            
            try: # Inner try for model discovery
                models_response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [model["id"] for model in models_data.get("data", [])]
                    log.info(f"🤖 Available LM Studio models: {available_models}")
                    
                    preferred_models = ["wizardcoder", "codellama", "code-llama", "stable-code", "starcoder"]
                    for preferred in preferred_models:
                        for model in available_models:
                            if preferred.lower() in model.lower():
                                chosen_model = model
                                break
                        if chosen_model: break
                    
                    if not chosen_model and available_models: chosen_model = available_models[0]
                    
                    if not chosen_model:
                        log.warning("⚠️ No models available in LM Studio")
                        return self._generate_static_hint(prompt)
                        
                    log.info(f"🧠 Using LM Studio model: {chosen_model}")
                else:
                    log.warning(f"⚠️ Failed to retrieve models from LM Studio: {models_response.status_code} - {models_response.text}")
                    return self._generate_static_hint(prompt)
            except Exception as model_e: # Added except block for inner try
                log.warning(f"⚠️ Error querying available models: {str(model_e)}")
                return self._generate_static_hint(prompt)
                
            # Proceed with API call only if model discovery didn't fail early
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful programming advisor giving concise hints."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            if chosen_model: data["model"] = chosen_model
            
            response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                hint = response_data['choices'][0]['message']['content'].strip()
                return hint
            else:
                log.warning(f"⚠️ Local LLM API error: {response.status_code} - {response.text}")
                return self._generate_static_hint(prompt)
                
        except Exception as e: # Added except block for outer try
            log.warning(f"⚠️ Error calling local LLM: {str(e)}")
            return self._generate_static_hint(prompt)