#!/usr/bin/env python3
"""
Run lightweight benchmark evaluations across task/benchmarks.

This is an eval harness, not a training loop. It measures the current
generator/executor/scorer stack over a fixed number of attempts per task and
stores structured JSON results under memory/reports/.
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from agent_core.generator import CodeGenerator
from core.config_loader import ConfigLoader
from evaluator.executor import CodeExecutor
from guardian.safety import SafetyChecker
from reinforcer.scorer import AttemptScorer
from task.task_loader import TaskLoader
from utils import get_pytorch_device


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def list_benchmark_task_names(task_loader: TaskLoader, selected_task: str = None):
    if selected_task:
        return [selected_task]
    return task_loader.list_task_names(subdirectory="benchmarks")


def evaluate_task(generator, safety_checker, executor, scorer, task, attempts, success_threshold, use_beam_search):
    task_limits = getattr(task, "resource_limits", {}) or {}
    attempt_records = []
    best_attempt = None

    for attempt_index in range(1, attempts + 1):
        generation_start = time.time()
        if use_beam_search:
            code_attempt, generated_ids = generator.generate_with_beam_search(task=task, history=None, hint=None)
        else:
            code_attempt, generated_ids = generator.generate(task=task, history=None, hint=None)
        generation_latency_ms = round((time.time() - generation_start) * 1000, 2)

        if code_attempt is None or generated_ids is None:
            result = {
                "attempt": attempt_index,
                "success": False,
                "score": -1.0,
                "status": "generation_error",
                "error_type": "GenerationError",
                "generation_latency_ms": generation_latency_ms,
                "execution_latency_ms": 0.0,
                "unsafe": False,
                "output": "",
                "code": "",
                "model_role": "generator",
                "model_tier": "research",
            }
            attempt_records.append(result)
            continue

        if not safety_checker.is_safe(code_attempt):
            result = {
                "attempt": attempt_index,
                "success": False,
                "score": -1.0,
                "status": "unsafe",
                "error_type": "UnsafeCode",
                "generation_latency_ms": generation_latency_ms,
                "execution_latency_ms": 0.0,
                "unsafe": True,
                "output": "",
                "code": code_attempt,
                "model_role": "generator",
                "model_tier": "research",
            }
            attempt_records.append(result)
            continue

        execution_result = executor.execute(
            code_attempt,
            language=task.target_language,
            timeout=task_limits.get("timeout"),
            resource_limits=task_limits,
        )
        score = scorer.score(code_attempt, execution_result, task)
        tool_feedback = scorer.get_tool_feedback(code_attempt, execution_result)

        result = {
            "attempt": attempt_index,
            "success": score >= success_threshold,
            "score": round(score, 6),
            "status": execution_result.status,
            "error_type": execution_result.error_type or tool_feedback.feedback_type,
            "generation_latency_ms": generation_latency_ms,
            "execution_latency_ms": round(execution_result.runtime * 1000, 2),
            "unsafe": False,
            "output": execution_result.stdout,
            "code": code_attempt,
            "model_role": "generator",
            "model_tier": "research",
        }
        attempt_records.append(result)

        if best_attempt is None or result["score"] > best_attempt["score"]:
            best_attempt = result

    if best_attempt is None:
        best_attempt = {
            "score": -1.0,
            "status": "not_run",
            "error_type": "NoAttempts",
            "code": "",
            "output": "",
        }

    return {
        "task_name": task.name,
        "target_language": task.target_language,
        "difficulty": getattr(task, "difficulty", "unknown"),
        "attempts_run": attempts,
        "passed": any(record["success"] for record in attempt_records),
        "best_score": best_attempt["score"],
        "best_status": best_attempt["status"],
        "best_error_type": best_attempt["error_type"],
        "best_code": best_attempt["code"],
        "best_output": best_attempt["output"],
        "attempts": attempt_records,
    }


def save_results(report_dir: Path, results_document: dict, output_path: str = None):
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if output_path:
        primary_path = Path(output_path)
        if not primary_path.is_absolute():
            primary_path = Path.cwd() / primary_path
    else:
        primary_path = report_dir / f"benchmark_results_{timestamp}.json"

    latest_path = report_dir / "benchmark_results_latest.json"

    with open(primary_path, "w", encoding="utf-8") as handle:
        json.dump(results_document, handle, indent=2)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(results_document, handle, indent=2)

    return primary_path, latest_path


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluations for AltLAS.")
    parser.add_argument("--task", type=str, help="Run a single benchmark task by name.")
    parser.add_argument("--attempts", type=int, default=3, help="Attempts per benchmark task.")
    parser.add_argument("--beam-search", action="store_true", help="Use beam search generation.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible runs.")
    parser.add_argument("--output", type=str, help="Optional explicit output path for the JSON results.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    configure_logging(args.debug)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_loader = ConfigLoader()
    config_path_str = str(config_loader.get_config_path())
    success_threshold = config_loader.get_scorer_config()["success_threshold"]

    task_loader = TaskLoader()
    task_names = list_benchmark_task_names(task_loader, selected_task=args.task)
    if not task_names:
        raise SystemExit("No benchmark tasks found.")

    device = get_pytorch_device()
    generator = CodeGenerator(config_path=config_path_str, device=device)
    safety_checker = SafetyChecker()
    executor = CodeExecutor(config_path=config_path_str)
    scorer = AttemptScorer(config_path=config_path_str)

    results = []
    started_at = datetime.now(timezone.utc)
    for task_name in task_names:
        logging.info(f"Evaluating benchmark task: {task_name}")
        task = task_loader.load_task(task_name)
        results.append(
            evaluate_task(
                generator=generator,
                safety_checker=safety_checker,
                executor=executor,
                scorer=scorer,
                task=task,
                attempts=args.attempts,
                success_threshold=success_threshold,
                use_beam_search=args.beam_search,
            )
        )

    finished_at = datetime.now(timezone.utc)
    passed_count = sum(1 for item in results if item["passed"])

    results_document = {
        "generated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "attempts_per_task": args.attempts,
        "beam_search": args.beam_search,
        "seed": args.seed,
        "success_threshold": success_threshold,
        "tasks_run": len(results),
        "tasks_passed": passed_count,
        "pass_rate": round((passed_count / len(results)) * 100, 2) if results else 0.0,
        "results": results,
    }

    primary_path, latest_path = save_results(Path("memory/reports"), results_document, output_path=args.output)
    logging.info(f"Saved benchmark results to {primary_path}")
    logging.info(f"Updated latest benchmark results at {latest_path}")


if __name__ == "__main__":
    main()
