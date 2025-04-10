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
import ast
import matplotlib.pyplot as plt
import numpy as np
import torch # Needed for tensor operations
import threading
import queue
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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
            
            # Initialize the background report generation queue and thread
            self.report_queue = queue.Queue()
            self.worker_thread = None
            self.worker_running = False
            
            # Initialize a list to track report generation status
            self.pending_reports = []
            self.completed_reports = []

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

        # --- AST-based Pattern Analysis --- 
        def analyze_code_structure(code_str: str) -> Dict[str, int]:
            """Analyzes code structure using AST and returns node counts."""
            counts = collections.Counter()
            if not code_str or code_str == DEFAULT_PLACEHOLDER:
                return dict(counts)
            try:
                tree = ast.parse(code_str)
                for node in ast.walk(tree):
                    node_type = type(node).__name__
                    # Count common structural nodes
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        counts['FunctionDefs'] += 1
                    elif isinstance(node, ast.ClassDef):
                        counts['ClassDefs'] += 1
                    elif isinstance(node, (ast.For, ast.AsyncFor)):
                        counts['ForLoops'] += 1
                    elif isinstance(node, ast.While):
                        counts['WhileLoops'] += 1
                    elif isinstance(node, ast.If):
                        counts['IfStatements'] += 1
                    elif isinstance(node, ast.Assign):
                        counts['Assignments'] += 1
                    elif isinstance(node, ast.Call):
                        counts['FunctionCalls'] += 1
                    elif isinstance(node, ast.Return):
                        counts['ReturnStatements'] += 1
                    # Add more node types as needed
            except SyntaxError:
                counts['SyntaxError'] = 1 # Mark that parsing failed
            except Exception as e:
                logging.warning(f"AST analysis failed: {e}")
                counts['ASTAnalysisFailed'] = 1
            # Return sorted by count, descending
            return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

        successful_structure = analyze_code_structure(metrics['high_scoring_code'])
        failed_structure = analyze_code_structure(metrics['low_scoring_code'])
        
        # Format for report (Top 3-5 structures)
        metrics['successful_patterns'] = [f"{k}: {v}" for k, v in list(successful_structure.items())[:5]]
        metrics['failed_patterns'] = [f"{k}: {v}" for k, v in list(failed_structure.items())[:5]]
        # --- End AST-based Pattern Analysis ---

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

    def _call_local_llm(self, prompt: str) -> str:
        """
        Call a local LLM using LM Studio running on localhost to generate insights.
        
        Args:
            prompt (str): The prompt to send to the LM Studio API
            
        Returns:
            str: The generated text or a default message if the call fails
        """
        import requests
        import json
        import logging
        
        try:
            base_url = "http://localhost:1234/v1"
            logging.info("🌐 Attempting to connect to LM Studio on localhost:1234")
            
            headers = {"Content-Type": "application/json"}
            chosen_model = None
            
            # First check if LM Studio is available and get available models
            try:
                models_response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [model["id"] for model in models_data.get("data", [])]
                    logging.info(f"🤖 Available LM Studio models: {available_models}")
                    
                    # Prefer coding-specific models
                    preferred_models = ["wizardcoder", "codellama", "code-llama", "stable-code", "starcoder", "llama"]
                    for preferred in preferred_models:
                        for model in available_models:
                            if preferred.lower() in model.lower():
                                chosen_model = model
                                break
                        if chosen_model: break
                    
                    if not chosen_model and available_models:
                        chosen_model = available_models[0]
                    
                    if chosen_model:
                        logging.info(f"🧠 Using LM Studio model: {chosen_model}")
                    else:
                        logging.warning("⚠️ No models available in LM Studio")
                        return "No LM Studio models available to generate observations."
                else:
                    logging.warning(f"⚠️ Failed to retrieve models from LM Studio: {models_response.status_code}")
                    return "Could not connect to LM Studio API."
            except Exception as model_e:
                logging.warning(f"⚠️ Error querying available models: {str(model_e)}")
                return "Error connecting to LM Studio: " + str(model_e)
                
            # Proceed with API call
            data = {
                "messages": [
                    {"role": "system", "content": "You are an AI assistant specializing in code generation and reinforcement learning analysis. Provide insightful observations about training reports."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            if chosen_model:
                data["model"] = chosen_model
            
            logging.info(f"Sending prompt to LM Studio model {chosen_model}")
            response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(data), timeout=120) # Increased timeout to 45s
            
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data['choices'][0]['message']['content'].strip()
                return generated_text
            else:
                logging.warning(f"⚠️ LM Studio API error: {response.status_code} - {response.text}")
                return f"LM Studio API error (HTTP {response.status_code})"
                
        except Exception as e:
            logging.warning(f"⚠️ Error calling local LLM: {str(e)}")
            return f"Error generating LLM observations: {str(e)}"

    def add_llm_observations(self, report_data: dict, history: list, current_task) -> str:
        """
        Generate AI-powered observations about the training run using a local LLM.
        
        Args:
            report_data (dict): The report data dictionary
            history (list): The attempt history
            current_task: The current task object
            
        Returns:
            str: Generated observations text
        """
        # Build a prompt for the LLM with relevant training data
        prompt = f"""
Please analyze this training report data and provide insightful observations about the model's learning process.

TASK: {getattr(current_task, 'name', 'Unknown')}
DESCRIPTION: {getattr(current_task, 'description', 'No description available')}

KEY METRICS:
- Total Attempts: {report_data.get('total_attempts', 'N/A')}
- Success Rate: {report_data.get('success_rate', 'N/A')}%
- Best Score: {report_data.get('high_score', 'N/A')}
- Score Trend: {report_data.get('score_trend', 'N/A')}
- Training Time: {report_data.get('training_time', 'N/A')}

COMMON ERRORS:
- Error 1: {report_data.get('error_type_1', 'N/A')} (Count: {report_data.get('count_1', 'N/A')})
- Error 2: {report_data.get('error_type_2', 'N/A')} (Count: {report_data.get('count_2', 'N/A')})
- Error 3: {report_data.get('error_type_3', 'N/A')} (Count: {report_data.get('count_3', 'N/A')})

CURRENT STATUS:
- Plateau Status: {report_data.get('plateau_status', 'N/A')}
- Semantic Drift: {report_data.get('semantic_drift_status', 'N/A')}
- Hints Provided: {report_data.get('hint_usage', 'N/A')}
- Stuck Events: {report_data.get('stuck_events', 'N/A')}
- Beam Search Uses: {report_data.get('beam_search_uses', 'N/A')}

HIGH SCORING EXAMPLE:
```python
{report_data.get('highest_scoring_code', 'No code available')}
```

Based on this data, please provide:
1. 3-5 key observations about the model's learning process
2. Potential strategies to improve performance
3. Analysis of any learning bottlenecks or issues
4. Suggestions for hyperparameter tuning if applicable

Format your response with clear section headers and bullet points where appropriate.
"""
        # Call the local LLM for analysis
        observations = self._call_local_llm(prompt)
        
        # If observations were generated successfully
        if observations and not observations.startswith("Error"):
            return f"""## AI Observations and Analysis

{observations}

*Generated using a local LM Studio model*
"""
        else:
            # Return a placeholder with instructions if LLM call failed
            return """## AI Observations and Analysis

*LM Studio was not available to generate observations. To enable this feature:*
1. Download and install LM Studio from https://lmstudio.ai/
2. Load a code-specialized model (WizardCoder, CodeLlama, etc.)
3. Start the local server on port 1234
4. Run the training again to get AI-powered observations

*Once set up, this section will contain AI-generated insights about your training run.*
"""

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
                                 beam_search_uses: int = 0,
                                 hint_impact_history: List[Tuple[float, float]] = None):
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
            hint_impact_history (List[Tuple[float, float]], optional): 
                List of (score_before_hint, score_after_hint) tuples.
        """
        try:
            # Initialize status messages list
            status_messages = []
            
            template = self._load_template()
            now = datetime.utcnow()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
            session_id = now.strftime("%Y%m%d_%H%M%S") # Simple session ID based on time

            # --- Generate Weight Histograms ---
            # Pass the generator_state which should contain the model state dict
            histogram_plot_paths = self._generate_weight_histograms(generator_state, session_id)

            # --- Calculate Metrics ---
            metrics = self._calculate_metrics(history)
            elapsed_time_sec = datetime.utcnow().timestamp() - start_time
            success_rate = (success_count / attempt_count * 100) if attempt_count > 0 else 0

            # --- Token Frequency Analysis ---
            token_freq = generator_state.get('token_frequency', {})
            top_tokens = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)[:5]
            token_distribution_status = f"Top 5: {', '.join([f'{t[0]}({t[1]})' for t in top_tokens])}" if top_tokens else "No data"

            # --- Calculate Hint Impact --- 
            avg_hint_improvement = DEFAULT_PLACEHOLDER
            if hint_impact_history:
                improvements = [after - before for before, after in hint_impact_history]
                if improvements:
                    avg_improvement = statistics.mean(improvements)
                    avg_hint_improvement = f"{avg_improvement:+.4f}"
                    logging.info(f"Calculated average hint impact: {avg_hint_improvement} over {len(improvements)} hints.")
                else:
                    logging.info("Hint impact history present but no improvements recorded yet.")
            else:
                hint_impact_history = [] # Ensure it's a list for calculations
                logging.debug("No hint impact history provided for report.")

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
                'avg_hint_improvement': avg_hint_improvement,

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
                'beam_search_uses': beam_search_uses,

                # Add relative paths to histograms for embedding in the report
                'embedding_hist_path': str(histogram_plot_paths.get('embedding.weight', DEFAULT_PLACEHOLDER)) if histogram_plot_paths else DEFAULT_PLACEHOLDER,
                'lstm_ih_hist_path': str(histogram_plot_paths.get('lstm.weight_ih_l0', DEFAULT_PLACEHOLDER)) if histogram_plot_paths else DEFAULT_PLACEHOLDER,
                'lstm_hh_hist_path': str(histogram_plot_paths.get('lstm.weight_hh_l0', DEFAULT_PLACEHOLDER)) if histogram_plot_paths else DEFAULT_PLACEHOLDER,
                'fc_weight_hist_path': str(histogram_plot_paths.get('fc.weight', DEFAULT_PLACEHOLDER)) if histogram_plot_paths else DEFAULT_PLACEHOLDER,
                'fc_bias_hist_path': str(histogram_plot_paths.get('fc.bias', DEFAULT_PLACEHOLDER)) if histogram_plot_paths else DEFAULT_PLACEHOLDER,
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

            # Add LLM-powered observations
            llm_observations = self.add_llm_observations(report_data, history, current_task)
            report_content += f"\n\n{llm_observations}"

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

    def _generate_weight_histograms(self, generator_state: Dict, session_id: str):
        """Generates histograms for key model weights and saves them."""
        try:
            # Check if matplotlib is available
            import matplotlib
            matplotlib.use('Agg') # Use non-interactive backend suitable for saving files
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib not found. Skipping weight histogram generation.")
            return None

        # --- Get Model State Dictionary ---
        # Prioritize getting it directly from generator_state if passed
        model_state_dict = generator_state.get('model_state_dict')
        
        # Fallback: Try loading from file if generator didn't pass it directly
        if not model_state_dict:
            model_state_path = Path(__file__).parent.parent / "memory" / "model_state.pth"
            if model_state_path.exists():
                 try:
                     # Load to CPU to avoid potential CUDA issues in report generation context
                     model_state_dict = torch.load(model_state_path, map_location='cpu')
                     logging.info(f"Loaded model state from {model_state_path} for histograms.")
                 except Exception as e:
                     logging.error(f"Failed to load model state from {model_state_path} for histograms: {e}")
                     return None # Cannot proceed without model state
            else:
                logging.warning("Model state dictionary not available in generator_state or file. Cannot generate weight histograms.")
                return None
        # --- End Get Model State Dictionary ---

        histogram_paths = {}
        # Ensure plots directory exists within the main report directory
        plot_dir = self.report_dir / "plots" / session_id
        plot_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating weight histograms in {plot_dir}...")

        # Define key layers to plot histograms for
        key_layers = {
            'embedding.weight': 'Embedding Weights',
            'lstm.weight_ih_l0': 'LSTM Input Weights (Layer 0)',
            'lstm.weight_hh_l0': 'LSTM Hidden Weights (Layer 0)',
            'fc.weight': 'Output FC Weights',
            'fc.bias': 'Output FC Bias'
        }

        for param_name, plot_title in key_layers.items():
            if param_name in model_state_dict:
                try:
                    # Ensure data is on CPU and converted to numpy
                    param_data = model_state_dict[param_name].cpu().numpy().flatten()
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(param_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                    plt.title(f'{plot_title} Distribution\n(Session: {session_id})', fontsize=14)
                    plt.xlabel("Weight Value", fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    
                    # Add basic stats text to the plot
                    mean_val = np.mean(param_data)
                    std_val = np.std(param_data)
                    min_val = np.min(param_data)
                    max_val = np.max(param_data)
                    stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
                    # Position the text box in the upper right corner
                    plt.text(0.95, 0.95, stats_text,
                             transform=plt.gca().transAxes, # Use axes coordinates
                             fontsize=10, verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

                    # Define plot filename and save path
                    plot_filename = f"hist_{param_name.replace('.', '_')}.png"
                    plot_path = plot_dir / plot_filename
                    plt.savefig(plot_path, bbox_inches='tight') # Save the figure
                    plt.close() # Close the figure explicitly to free memory
                    
                    # Store relative path for linking in the Markdown report
                    # Path should be relative to the report file itself (which is in self.report_dir)
                    relative_plot_path = Path("plots") / session_id / plot_filename
                    histogram_paths[param_name] = relative_plot_path
                    logging.debug(f"Generated histogram for {param_name} at {plot_path}")

                except Exception as e:
                    logging.error(f"Failed to generate histogram for {param_name}: {e}", exc_info=True)
            else:
                 logging.warning(f"Parameter '{param_name}' not found in model state dict for histogram generation.")

        return histogram_paths # Return dictionary of parameter names to relative plot paths

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

    def _worker_thread_function(self):
        """Background worker thread function for processing report generation requests."""
        self.worker_running = True
        logging.info("Background report generation worker thread started")
        
        while self.worker_running:
            try:
                # Get report request from queue with timeout to allow checking worker_running flag
                try:
                    report_request = self.report_queue.get(timeout=1.0)
                except queue.Empty:
                    # No request in queue, continue the loop
                    continue
                
                # Process the report request
                report_id = report_request.get('report_id', f"report_{int(time.time())}")
                logging.info(f"Processing report request {report_id} in background thread")
                
                # Extract parameters from the request
                attempt_count = report_request.get('attempt_count')
                max_attempts = report_request.get('max_attempts')
                current_task = report_request.get('current_task')
                history = report_request.get('history')
                generator_state = report_request.get('generator_state')
                scorer_state = report_request.get('scorer_state')
                start_time = report_request.get('start_time')
                hints_provided = report_request.get('hints_provided')
                success_count = report_request.get('success_count')
                stuck_events = report_request.get('stuck_events', 0)
                beam_search_uses = report_request.get('beam_search_uses', 0)
                hint_impact_history = report_request.get('hint_impact_history')
                
                # Generate and save the report
                try:
                    report_path = self.generate_and_save_report(
                        attempt_count=attempt_count,
                        max_attempts=max_attempts,
                        current_task=current_task,
                        history=history,
                        generator_state=generator_state,
                        scorer_state=scorer_state,
                        start_time=start_time,
                        hints_provided=hints_provided,
                        success_count=success_count,
                        stuck_events=stuck_events,
                        beam_search_uses=beam_search_uses,
                        hint_impact_history=hint_impact_history
                    )
                    
                    # Record that the report was completed successfully
                    self.completed_reports.append({
                        'report_id': report_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'report_path': str(report_path),
                        'status': 'completed'
                    })
                    
                    # Remove from pending list
                    self.pending_reports = [r for r in self.pending_reports if r.get('report_id') != report_id]
                    
                    logging.info(f"Background report generation completed for {report_id}, saved to {report_path}")
                
                except Exception as e:
                    logging.error(f"Error generating report {report_id} in background thread: {e}", exc_info=True)
                    
                    # Record the error
                    self.completed_reports.append({
                        'report_id': report_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'status': 'error',
                        'error_message': str(e)
                    })
                    
                    # Remove from pending list
                    self.pending_reports = [r for r in self.pending_reports if r.get('report_id') != report_id]
                
                # Mark task as done in the queue
                self.report_queue.task_done()
                
            except Exception as e:
                logging.error(f"Unexpected error in report generator worker thread: {e}", exc_info=True)
                # Don't terminate the thread on unexpected errors
        
        logging.info("Background report generation worker thread terminated")

    def start_worker_thread(self):
        """Start the background worker thread if it's not already running."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_running = True
            self.worker_thread = threading.Thread(
                target=self._worker_thread_function,
                daemon=True  # Make thread terminate when main program exits
            )
            self.worker_thread.start()
            logging.info("Started background report generation worker thread")
            return True
        return False

    def stop_worker_thread(self):
        """Stop the background worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_running = False
            # Give the thread a chance to terminate gracefully
            self.worker_thread.join(timeout=2.0)
            if self.worker_thread.is_alive():
                logging.warning("Background report generation worker thread did not terminate gracefully")
            else:
                logging.info("Background report generation worker thread stopped successfully")
            return True
        return False

    def generate_report_async(self,
                             attempt_count: int,
                             max_attempts: int,
                             current_task: Any,
                             history: List[Dict],
                             generator_state: Dict,
                             scorer_state: Dict,
                             start_time: float,
                             hints_provided: int,
                             success_count: int,
                             stuck_events: int = 0,
                             beam_search_uses: int = 0,
                             hint_impact_history: List[Tuple[float, float]] = None) -> str:
        """
        Queue a training report to be generated asynchronously in a background thread.
        
        This method returns immediately and doesn't block the caller. The report will be
        generated by a background worker thread.
        
        Args:
            (Same parameters as generate_and_save_report)
            
        Returns:
            str: A report ID that can be used to check the status of the report generation
        """
        # Start worker thread if not already running
        self.start_worker_thread()
        
        # Generate a unique report ID
        report_id = f"report_{int(time.time())}_{len(self.pending_reports)}"
        
        # Create report request
        report_request = {
            'report_id': report_id,
            'timestamp': datetime.utcnow().isoformat(),
            'attempt_count': attempt_count,
            'max_attempts': max_attempts,
            'current_task': current_task,
            'history': history.copy() if history else [],  # Make a copy to prevent changes while processing
            'generator_state': generator_state,
            'scorer_state': scorer_state,
            'start_time': start_time,
            'hints_provided': hints_provided,
            'success_count': success_count,
            'stuck_events': stuck_events,
            'beam_search_uses': beam_search_uses,
            'hint_impact_history': hint_impact_history,
            'status': 'pending'
        }
        
        # Add to pending list
        self.pending_reports.append({
            'report_id': report_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'pending'
        })
        
        # Add request to queue for processing
        self.report_queue.put(report_request)
        
        logging.info(f"Queued report generation request {report_id} for background processing")
        return report_id

    def get_report_status(self, report_id: str) -> Dict:
        """Get the status of a queued or completed report."""
        # Check pending reports
        for report in self.pending_reports:
            if report.get('report_id') == report_id:
                return report
        
        # Check completed reports
        for report in self.completed_reports:
            if report.get('report_id') == report_id:
                return report
        
        # Not found
        return {'report_id': report_id, 'status': 'not_found'}

    def cleanup(self):
        """Clean up resources before shutdown."""
        self.stop_worker_thread()
        # Wait for any pending reports to finish
        if not self.report_queue.empty():
            try:
                self.report_queue.join(timeout=5.0)
                logging.info("Waited for pending reports to complete during cleanup")
            except Exception as e:
                logging.warning(f"Error waiting for report queue to complete: {e}")
