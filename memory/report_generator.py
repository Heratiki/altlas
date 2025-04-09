"""
Training Report Generator for AltLAS
Implements periodic reporting to track training progress and identify improvement opportunities.
"""

import os
import logging
import difflib
import json
import re
import statistics
import collections
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Default values for placeholders if data is missing
DEFAULT_PLACEHOLDER = "N/A"

class TrainingReportGenerator:
    def __init__(self, report_dir="memory/reports", template_path="training_report.md"):
        try:
            self.report_dir = Path(report_dir).resolve() # Resolve path immediately
            # Resolve template path relative to project root (parent of this file's parent)
            project_root = Path(__file__).parent.parent.resolve()
            self.template_path = project_root / template_path
            self.metrics_state_file = self.report_dir / "metrics_state.json"
            self.metrics_state = self._load_metrics_state()

            logging.info(f"ReportGenerator initialized. Report dir: {self.report_dir}, Template path: {self.template_path}")

            # Explicitly log directory creation attempt
            logging.info(f"Ensuring report directory exists: {self.report_dir}")
            self.report_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Report directory check/creation complete.")

            if not self.template_path.exists():
                logging.error(f"Report template file not found at resolved path: {self.template_path}")
                raise FileNotFoundError(f"Report template not found at {self.template_path}")
            logging.info(f"Report template found at: {self.template_path}")

        except Exception as e:
            logging.error(f"Error during TrainingReportGenerator initialization: {e}", exc_info=True)
            # Re-raise the exception to prevent using a potentially broken generator
            raise

    def _load_metrics_state(self) -> Dict:
        """Load the persistent metrics state from disk."""
        if self.metrics_state_file.exists():
            try:
                with open(self.metrics_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metrics state: {e}", exc_info=True)
        
        # Return default state if file doesn't exist or has error
        return {
            'all_time_high_score': 0.0,
            'all_time_high_code': DEFAULT_PLACEHOLDER,
            'all_time_high_output': DEFAULT_PLACEHOLDER,
            'all_time_low_score': 1.0,
            'all_time_low_code': DEFAULT_PLACEHOLDER,
            'all_time_low_output': DEFAULT_PLACEHOLDER,
            'cumulative_error_counts': {},
            'historical_patterns': {
                'successful': [],
                'failed': []
            },
            'last_report_metrics': {},
        }

    def _save_metrics_state(self):
        """Save the current metrics state to disk."""
        try:
            with open(self.metrics_state_file, 'w') as f:
                json.dump(self.metrics_state, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metrics state: {e}", exc_info=True)

    def _update_metrics_state(self, metrics: Dict):
        """Update the persistent metrics state with new metrics."""
        # Update all-time bests
        if metrics['high_score'] > self.metrics_state['all_time_high_score']:
            self.metrics_state['all_time_high_score'] = metrics['high_score']
            self.metrics_state['all_time_high_code'] = metrics['high_scoring_code']
            self.metrics_state['all_time_high_output'] = metrics['high_scoring_output']

        # Update all-time worsts (if lower)
        if metrics['low_score'] < self.metrics_state['all_time_low_score']:
            self.metrics_state['all_time_low_score'] = metrics['low_score']
            self.metrics_state['all_time_low_code'] = metrics['low_scoring_code']
            self.metrics_state['all_time_low_output'] = metrics['low_scoring_output']

        # Update cumulative error counts
        for error_type, count in metrics['error_counts'].items():
            if error_type in self.metrics_state['cumulative_error_counts']:
                self.metrics_state['cumulative_error_counts'][error_type] += count
            else:
                self.metrics_state['cumulative_error_counts'][error_type] = count

        # Store current metrics for next comparison
        self.metrics_state['last_report_metrics'] = {
            'success_rate': metrics.get('success_rate', 0.0),
            'moving_avg_score': metrics.get('moving_avg_score', 0.0),
            'high_score': metrics.get('high_score', 0.0),
            'entropy_coefficient': metrics.get('entropy_coefficient', 0.0)
        }

        # Save updated state
        self._save_metrics_state()

    def _load_template(self) -> str:
        """Loads the Markdown report template."""
        try:
            with open(self.template_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading report template: {e}")
            # Fallback to a basic template string if file loading fails
            return "# AltLAS Training Report\n\nError: Could not load template file."

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into a human-readable string (e.g., 1h 2m 3s)."""
        if seconds < 0: return "0s"
        delta = timedelta(seconds=int(seconds))
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []
        if delta.days > 0: parts.append(f"{delta.days}d")
        if hours > 0: parts.append(f"{hours}h")
        if minutes > 0: parts.append(f"{minutes}m")
        if secs > 0 or not parts: parts.append(f"{secs}s") # Show seconds if it's the only unit or non-zero
        return " ".join(parts) if parts else "0s"

    def _calculate_metrics(self, history: List[Dict], window_size=100) -> Dict[str, Any]:
        """Calculates various metrics from the attempt history."""
        metrics = {
            'moving_avg_score': 0.0,
            'score_trend': "Stable",
            'high_score': 0.0,
            'high_scoring_code': DEFAULT_PLACEHOLDER,
            'high_scoring_output': DEFAULT_PLACEHOLDER,
            'low_score': 1.0,
            'low_scoring_code': DEFAULT_PLACEHOLDER,
            'low_scoring_output': DEFAULT_PLACEHOLDER,
            'error_counts': collections.Counter(),
            'avg_gen_time': 0.0, # Placeholder
            'avg_exec_time': 0.0, # Placeholder
            'semantic_drift_status': "Analysis N/A", # Default value
            'top_tokens': [],
            'successful_patterns': [],
            'failed_patterns': [],
        }

        if not history:
            return metrics

        scores = [h.get('score', 0.0) for h in history]
        
        # Moving Average & Trend
        current_window = scores[-window_size:]
        if current_window:
            metrics['moving_avg_score'] = statistics.mean(current_window)

        if len(scores) > window_size * 2:
            previous_window = scores[-(window_size * 2):-window_size]
            prev_avg = statistics.mean(previous_window)
            diff = metrics['moving_avg_score'] - prev_avg
            if diff > 0.02: metrics['score_trend'] = "Increasing"
            elif diff < -0.02: metrics['score_trend'] = "Declining"
            else: metrics['score_trend'] = "Stable"
        elif len(scores) > window_size:
             # Compare current window to overall average if not enough history for two windows
             overall_avg = statistics.mean(scores)
             diff = metrics['moving_avg_score'] - overall_avg
             if diff > 0.02: metrics['score_trend'] = "Increasing"
             elif diff < -0.02: metrics['score_trend'] = "Declining"

        # Best/Worst Attempts
        best_attempt = max(history, key=lambda h: h.get('score', -1))
        worst_attempt = min(history, key=lambda h: h.get('score', 2))

        metrics['high_score'] = best_attempt.get('score', 0.0)
        metrics['high_scoring_code'] = best_attempt.get('code', DEFAULT_PLACEHOLDER)
        metrics['high_scoring_output'] = best_attempt.get('result', {}).get('stdout', DEFAULT_PLACEHOLDER) or DEFAULT_PLACEHOLDER

        metrics['low_score'] = worst_attempt.get('score', 1.0)
        metrics['low_scoring_code'] = worst_attempt.get('code', DEFAULT_PLACEHOLDER)
        metrics['low_scoring_output'] = worst_attempt.get('result', {}).get('stdout', DEFAULT_PLACEHOLDER) or DEFAULT_PLACEHOLDER

        # Error Counts
        for h in history:
            result = h.get('result', {})
            if result.get('status') == 'error':
                error_type = result.get('error_type', 'Unknown Error')
                # Simplify error type (e.g., "NameError: name 'x' is not defined" -> "NameError")
                simple_error_type = error_type.split(':')[0].strip()
                metrics['error_counts'][simple_error_type] += 1

        # TODO: Calculate Avg Gen/Exec Time if available in history

        # TODO: Calculate Avg Gen/Exec Time if available in history

        # Basic Pattern Analysis (Keyword Counting)
        keywords = ['def', 'class', 'for', 'while', 'if', 'return', 'print', '+', '=', 'import']
        if metrics['high_scoring_code'] != DEFAULT_PLACEHOLDER:
             metrics['successful_patterns'] = sorted([f"{kw}: {metrics['high_scoring_code'].count(kw)}" for kw in keywords if metrics['high_scoring_code'].count(kw) > 0], key=lambda x: int(x.split(': ')[1]), reverse=True)[:3]
        if metrics['low_scoring_code'] != DEFAULT_PLACEHOLDER:
             metrics['failed_patterns'] = sorted([f"{kw}: {metrics['low_scoring_code'].count(kw)}" for kw in keywords if metrics['low_scoring_code'].count(kw) > 0], key=lambda x: int(x.split(': ')[1]), reverse=True)[:3]


        # --- Semantic Drift Detection (Basic) ---
        drift_window_size = 20 # How many recent attempts to check for output similarity
        if len(history) >= drift_window_size:
            recent_outputs = [h.get('result', {}).get('stdout', '').strip() for h in history[-drift_window_size:]]
            similarities = []
            for i in range(len(recent_outputs) - 1):
                # Avoid comparing empty strings which gives high similarity
                if recent_outputs[i] and recent_outputs[i+1]:
                    ratio = difflib.SequenceMatcher(None, recent_outputs[i], recent_outputs[i+1]).ratio()
                    similarities.append(ratio)

            if similarities: # Check if we have any similarity scores
                avg_similarity = statistics.mean(similarities)
                logging.debug(f"Semantic Drift Check: Avg output similarity (last {drift_window_size}) = {avg_similarity:.2f}")
                # If outputs are dissimilar (low avg similarity) AND score is plateaued
                if avg_similarity < 0.3 and metrics['score_trend'] == "Stable":
                    metrics['semantic_drift_status'] = f"Potential drift detected (Avg similarity {avg_similarity:.2f} < 0.3 while score is stable)"
                elif avg_similarity < 0.5:
                     metrics['semantic_drift_status'] = f"Outputs show high variability (Avg similarity {avg_similarity:.2f})"
                else:
                     metrics['semantic_drift_status'] = f"Outputs relatively consistent (Avg similarity {avg_similarity:.2f})"
            else:
                 metrics['semantic_drift_status'] = "Not enough comparable outputs in window"
        else:
             metrics['semantic_drift_status'] = "Not enough history for drift analysis"
        # --- End Semantic Drift Detection ---

        # --- Token Co-occurrence and Utilization ---
        from collections import defaultdict
        co_occurrence = defaultdict(lambda: defaultdict(int))
        token_freq = collections.Counter()
        recent_attempts = history[-window_size:] if len(history) > window_size else history
        for h in recent_attempts:
            tokens = h.get('tokens', [])  # Assumes token IDs or strings stored
            for i, t1 in enumerate(tokens):
                token_freq[t1] += 1
                for t2 in tokens[i+1:]:
                    co_occurrence[t1][t2] += 1
                    co_occurrence[t2][t1] += 1
        metrics['token_co_occurrence'] = {k: dict(v) for k, v in co_occurrence.items()}
        metrics['token_frequencies'] = dict(token_freq)
        # Identify over/under-utilized tokens
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        metrics['overused_tokens'] = sorted_tokens[:5]
        metrics['underused_tokens'] = [t for t, c in token_freq.items() if c <= 2][:5]


        return metrics

    def _populate_template(self, template: str, data: Dict[str, Any]) -> str:
        """Replaces placeholders in the template string with data."""
        populated = template
        for key, value in data.items():
            placeholder = f"{{{key.upper()}}}" # Placeholders are like {PLACEHOLDER_NAME}
            # Format values appropriately
            if isinstance(value, float):
                formatted_value = f"{value:.4f}" # Format floats nicely
            elif isinstance(value, dict) or isinstance(value, list):
                 # Basic formatting for dicts/lists, might need refinement
                formatted_value = json.dumps(value, indent=2)
            else:
                formatted_value = str(value)

            # Correctly reassign the result of replace back to populated
            populated = populated.replace(placeholder, formatted_value)

        # Clean up any remaining placeholders
        populated = re.sub(r'\{[A-Z0-9_]+\}', DEFAULT_PLACEHOLDER, populated)
        return populated

    def _generate_insights(self, metrics: Dict, history: List[Dict], hints_provided: int) -> Dict[str, str]:
        """Generates simple insights and recommendations based on metrics."""
        insights = {
            'observations': [],
            'recommendation_1': "N/A",
            'recommendation_2': "N/A",
            'recommendation_3': "N/A",
            'plateau_status': "Analysis N/A", # Default
        }
        num_attempts = len(history)
        if num_attempts < 50: # Need some history for meaningful insights
             insights['observations'].append("Not enough history for detailed insights.")
             return insights

        # --- Plateau Detection ---
        # Simple check: score trend stable and score below a certain threshold?
        plateau_threshold = 0.6 # Example threshold
        if metrics.get('score_trend') == "Stable" and metrics.get('moving_avg_score', 0) < plateau_threshold:
            insights['plateau_status'] = f"Potential plateau detected (Score stable around {metrics.get('moving_avg_score', 0):.2f})"
            insights['observations'].append(insights['plateau_status'])
            insights['recommendation_1'] = "Consider adjusting hyperparameters (LR, entropy) or exploring different generation strategies (e.g., beam search if not already used)."
        else:
             insights['plateau_status'] = "No significant plateau detected."

        # --- Syntax Error Analysis ---
        syntax_error_count = metrics.get('error_counts', {}).get('SyntaxError', 0)
        total_errors = sum(metrics.get('error_counts', {}).values())
        # Add check for total_errors > 0 to prevent division by zero
        if total_errors > 0 and (syntax_error_count / total_errors) > 0.3: # If > 30% of errors are syntax
            insights['observations'].append("Syntax errors appear common. Review grammar rules or generation constraints.")
            insights['recommendation_2'] = "Focus on improving syntax generation (e.g., strengthen grammar guidance, check tokenization)."

        # --- Hint Usage Analysis ---
        if insights['plateau_status'].startswith("Potential plateau") and hints_provided < (num_attempts / 500): # Example: low hint usage despite plateau
             insights['observations'].append(f"Low hint usage ({hints_provided}) observed despite potential plateau.")
             insights['recommendation_3'] = "Review hint trigger logic/probability if progress is stalled."

        # --- Final Formatting ---
        if not insights['observations']:
            insights['observations'] = ["No specific issues automatically detected."]
        insights['observations'] = "\n".join(f"- {obs}" for obs in insights['observations'])

        # Fill remaining recommendations if slots are empty
        rec_list = [insights['recommendation_1'], insights['recommendation_2'], insights['recommendation_3']]
        default_recs = ["Analyze score trends.", "Review common errors.", "Monitor resource usage."]
        filled_recs = [rec for rec in rec_list if rec != "N/A"]
        next_default = 0
        while len(filled_recs) < 3 and next_default < len(default_recs):
            if default_recs[next_default] not in filled_recs:
                 filled_recs.append(default_recs[next_default])
            next_default += 1
        insights['recommendation_1'], insights['recommendation_2'], insights['recommendation_3'] = filled_recs[:3]


        return insights


    def generate_and_save_report(self,
                                 attempt_count: int,
                                 max_attempts: int,
                                 current_task: Any, # Task object
                                 history: List[Dict],
                                 generator_state: Dict,
                                 scorer_state: Dict,
                                 start_time: float,
                                 hints_provided: int,
                                 success_count: int,
                                 stuck_events: int = 0,
                                 beam_search_uses: int = 0):
        """
        Generates a training report based on the current state and saves it.

        Args:
            attempt_count (int): Current attempt number.
            max_attempts (int): Total attempts configured for the run.
            current_task (Any): The task object being worked on.
            history (List[Dict]): List of dictionaries representing past attempts.
            generator_state (Dict): State dictionary from CodeGenerator.get_state().
            scorer_state (Dict): State dictionary from AttemptScorer.get_state().
            start_time (float): Timestamp when the training run started.
            hints_provided (int): Number of hints provided during the run.
            success_count (int): Number of successful attempts.
            stuck_events (int): Number of stuck events detected during the run.
            beam_search_uses (int): Number of times beam search was used during the run.
        """
        try:
            # Initialize status messages list
            status_messages = []
            
            template = self._load_template()
            now = datetime.utcnow()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            session_id = now.strftime("%Y%m%d_%H%M%S") # Simple session ID based on time

            # --- Calculate Metrics ---
            metrics = self._calculate_metrics(history)
            elapsed_time_sec = datetime.utcnow().timestamp() - start_time
            success_rate = (success_count / attempt_count * 100) if attempt_count > 0 else 0

            # --- Token Frequency Analysis ---
            token_freq = generator_state.get('token_frequency', {})
            top_tokens = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)[:5]
            token_distribution_status = f"Top 5: {', '.join([f'{t[0]}({t[1]})' for t in top_tokens])}" if top_tokens else "No data"

            # --- Generate Insights ---
            generated_insights = self._generate_insights(metrics, history, hints_provided)

            # --- Calculate Changes from Previous Report ---
            previous_metrics = self.metrics_state.get('last_report_metrics', {})
            success_rate_change = success_rate - previous_metrics.get('success_rate', 0.0)
            score_change = metrics['moving_avg_score'] - previous_metrics.get('moving_avg_score', 0.0)
            high_score_change = metrics['high_score'] - previous_metrics.get('high_score', 0.0)
            entropy_change = generator_state.get('max_entropy_coefficient', 0.0) - previous_metrics.get('entropy_coefficient', 0.0)

            # --- Check All-Time Bests ---
            all_time_high_score = self.metrics_state.get('all_time_high_score', 0.0)
            if metrics['high_score'] > all_time_high_score:
                status_messages.insert(0, f"🏆 New all-time best score: {metrics['high_score']:.4f}")
                logging.info(f"New all-time best score achieved: {metrics['high_score']:.4f}")

            # --- Prepare Data Dictionary for Template ---
            report_data = {
                'timestamp': timestamp_str,
                'session_id': session_id,
                'task_name': getattr(current_task, 'name', DEFAULT_PLACEHOLDER),
                'total_attempts': attempt_count,
                'successful_attempts': success_count,
                'success_rate': f"{success_rate:.2f}",
                'training_time': self._format_time(elapsed_time_sec),

                'moving_avg_score': metrics['moving_avg_score'],
                'score_trend': metrics['score_trend'],
                'learning_rate': generator_state.get('current_learning_rate', DEFAULT_PLACEHOLDER),
                'entropy_coefficient': generator_state.get('max_entropy_coefficient', DEFAULT_PLACEHOLDER),

                # Use all-time bests if better than current
                'high_score': max(metrics['high_score'], self.metrics_state.get('all_time_high_score', 0.0)),
                'highest_scoring_code': (metrics['high_scoring_code'] 
                    if metrics['high_score'] >= self.metrics_state.get('all_time_high_score', 0.0)
                    else self.metrics_state.get('all_time_high_code', DEFAULT_PLACEHOLDER)),
                'highest_scoring_output': (metrics['high_scoring_output']
                    if metrics['high_score'] >= self.metrics_state.get('all_time_high_score', 0.0)
                    else self.metrics_state.get('all_time_high_output', DEFAULT_PLACEHOLDER)),
                
                # Use all-time worsts for comparison
                'low_score': min(metrics['low_score'], self.metrics_state.get('all_time_low_score', 1.0)),
                'lowest_scoring_code': (metrics['low_scoring_code']
                    if metrics['low_score'] <= self.metrics_state.get('all_time_low_score', 1.0)
                    else self.metrics_state.get('all_time_low_code', DEFAULT_PLACEHOLDER)),
                'lowest_scoring_output': (metrics['low_scoring_output']
                    if metrics['low_score'] <= self.metrics_state.get('all_time_low_score', 1.0)
                    else self.metrics_state.get('all_time_low_output', DEFAULT_PLACEHOLDER)),

                # Common Patterns (Basic Keyword Counts) - Top 3
                'pattern_1': metrics['successful_patterns'][0] if len(metrics['successful_patterns']) > 0 else DEFAULT_PLACEHOLDER,
                'pattern_2': metrics['successful_patterns'][1] if len(metrics['successful_patterns']) > 1 else DEFAULT_PLACEHOLDER,
                'pattern_3': metrics['successful_patterns'][2] if len(metrics['successful_patterns']) > 2 else DEFAULT_PLACEHOLDER,
                'failed_pattern_1': metrics['failed_patterns'][0] if len(metrics['failed_patterns']) > 0 else DEFAULT_PLACEHOLDER,
                'failed_pattern_2': metrics['failed_patterns'][1] if len(metrics['failed_patterns']) > 1 else DEFAULT_PLACEHOLDER,
                'failed_pattern_3': metrics['failed_patterns'][2] if len(metrics['failed_patterns']) > 2 else DEFAULT_PLACEHOLDER,

                # Common Errors Table (Cumulative + Current)
                'error_type_1': DEFAULT_PLACEHOLDER, 'count_1': DEFAULT_PLACEHOLDER, 'error_example_1': DEFAULT_PLACEHOLDER,
                'error_type_2': DEFAULT_PLACEHOLDER, 'count_2': DEFAULT_PLACEHOLDER, 'error_example_2': DEFAULT_PLACEHOLDER,
                'error_type_3': DEFAULT_PLACEHOLDER, 'count_3': DEFAULT_PLACEHOLDER, 'error_example_3': DEFAULT_PLACEHOLDER,

                # Learning Status Placeholders
                'plateau_status': generated_insights['plateau_status'],
                'semantic_drift_status': metrics['semantic_drift_status'],
                'token_distribution_status': token_distribution_status,
                'hint_usage': hints_provided,
                'hint_utilization': DEFAULT_PLACEHOLDER,

                # Score Component Weights and Values
                'syntax_weight': scorer_state.get('component_weights', {}).get('syntax', DEFAULT_PLACEHOLDER),
                'execution_weight': scorer_state.get('component_weights', {}).get('execution', DEFAULT_PLACEHOLDER),
                'output_weight': scorer_state.get('component_weights', {}).get('output', DEFAULT_PLACEHOLDER),
                'structural_weight': scorer_state.get('component_weights', {}).get('structural', DEFAULT_PLACEHOLDER),
                'constraints_weight': scorer_state.get('component_weights', {}).get('constraints', DEFAULT_PLACEHOLDER),
                'semantic_weight': scorer_state.get('component_weights', {}).get('semantic', DEFAULT_PLACEHOLDER),

                # Resource Utilization
                'memory_usage': DEFAULT_PLACEHOLDER,
                'avg_gen_time': metrics['avg_gen_time'],
                'avg_exec_time': metrics['avg_exec_time'],

                # Observations/Recommendations
                'observations': generated_insights['observations'],
                'recommendation_1': generated_insights['recommendation_1'],
                'recommendation_2': generated_insights['recommendation_2'],
                'recommendation_3': generated_insights['recommendation_3'],

                # Historical Comparison (from previous report)
                'prev_success_rate': f"{previous_metrics.get('success_rate', 0.0):.2f}",
                'success_rate_change': f"{success_rate_change:+.2f}",
                'prev_moving_avg': f"{previous_metrics.get('moving_avg_score', 0.0):.4f}",
                'score_change': f"{score_change:+.4f}",
                'prev_high_score': f"{previous_metrics.get('high_score', 0.0):.4f}",
                'high_score_change': f"{high_score_change:+.4f}",
                'prev_entropy': f"{previous_metrics.get('entropy_coefficient', 0.0):.4f}",
                'entropy_change': f"{entropy_change:+.4f}",
                'stuck_events': stuck_events,
                'beam_search_uses': beam_search_uses
            }

            # Insert stuck/beam info into Score Progression section
            report_data['score_trend'] += f" (Stuck events: {stuck_events}, Beam search used: {beam_search_uses})"

            # Populate Error Table Data (Combine current and cumulative)
            combined_errors = metrics['error_counts'] + collections.Counter(self.metrics_state.get('cumulative_error_counts', {}))
            top_errors = combined_errors.most_common(3)
            for i, (err_type, count) in enumerate(top_errors):
                report_data[f'error_type_{i+1}'] = err_type
                report_data[f'count_{i+1}'] = count
                # Find an example error message
                example_msg = DEFAULT_PLACEHOLDER
                for h in reversed(history):
                    result = h.get('result', {})
                    if result.get('status') == 'error':
                         error_type_full = result.get('error_type', '')
                         if error_type_full.startswith(err_type):
                             example_msg = error_type_full.split('\n')[0]
                             break
                report_data[f'error_example_{i+1}'] = f"`{example_msg}`"

            # --- Populate Template ---
            report_content = self._populate_template(template, report_data)

            # --- Save Report ---
            timestamp_filename = f"training_report_{session_id}.md"
            report_path = self.report_dir / timestamp_filename
            latest_report_path = self.report_dir / "latest_report.md"

            logging.info(f"Attempting to save timestamped report to: {report_path}")
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"Timestamped report saved successfully to {report_path}")

            # Overwrite/create the latest report file
            logging.info(f"Attempting to save latest report to: {latest_report_path}")
            with open(latest_report_path, "w", encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"Latest report updated successfully at {latest_report_path}")

            # --- Report Rotation ---
            try:
                self._rotate_reports(keep=5)
            except Exception as rot_e:
                logging.error(f"Error during report rotation: {rot_e}", exc_info=True)

            # Update and save metrics state
            self._update_metrics_state(metrics)

            return str(report_path)

        except Exception as e:
            # Use logging for better error tracking
            logging.error(f"Error generating or saving training report: {e}", exc_info=True)
            return None

    def _rotate_reports(self, keep: int):
        """Keeps only the specified number of most recent timestamped reports."""
        logging.info(f"Starting report rotation, keeping latest {keep} reports.")
        # Find all timestamped report files, excluding latest_report.md
        report_files = list(self.report_dir.glob("training_report_*.md"))
        report_files = [f for f in report_files if f.name != "latest_report.md"]

        if len(report_files) <= keep:
            logging.info(f"Found {len(report_files)} reports, which is not more than {keep}. No rotation needed.")
            return

        # Sort files by modification time (oldest first)
        # Need to import 'os' for getmtime
        import os
        report_files.sort(key=os.path.getmtime)

        # Calculate how many files to delete
        files_to_delete_count = len(report_files) - keep
        logging.info(f"Found {len(report_files)} reports. Deleting {files_to_delete_count} oldest reports.")

        # Delete the oldest files
        for i in range(files_to_delete_count):
            file_to_delete = report_files[i]
            try:
                file_to_delete.unlink()
                logging.info(f"Deleted old report: {file_to_delete.name}")
            except OSError as del_e:
                logging.error(f"Error deleting report file {file_to_delete}: {del_e}")
