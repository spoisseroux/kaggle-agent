"""Summarizer Agent - Document phase execution.

Responsibilities:
- Generate comprehensive phase summaries
- Document key findings and decisions
- Create Telegram notifications
- Log to MLflow
- Recommend next phase actions

Cost: 100% Ollama (zero API cost)
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Dict, Any, List
from pathlib import Path

from core.llm_interface import ask_ollama

log = logging.getLogger(__name__)


def summarize_phase(
    phase: str,
    tasks: List[Dict[str, Any]],
    state: Dict[str, Any],
    notify_telegram: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive phase summary.

    Args:
        phase: Phase name
        tasks: List of completed tasks with results
        state: Current state
        notify_telegram: Send notification via Telegram

    Returns:
        Summary dict with:
        - phase: phase name
        - key_findings: [bullet points]
        - decisions_made: [{decision, reasoning}, ...]
        - metrics: {cv_score, feature_count, etc.}
        - artifacts: [file paths generated]
        - next_phase_recommendations: [suggestions]
        - telegram_summary: concise version for notification
        - full_report: detailed markdown report
    """
    log.info(f"Summarizer agent documenting {phase}")

    # Build summary prompt
    prompt = _build_summary_prompt(phase, tasks, state)

    system = "You are a data science documentation specialist. Summarize experiments clearly and concisely."

    response = ask_ollama(prompt, system=system, think=False)

    # Parse response
    try:
        summary = _parse_summary_response(response, phase)

        # Add metadata
        summary["phase"] = phase
        summary["task_count"] = len(tasks)

        # Generate full markdown report
        summary["full_report"] = _generate_markdown_report(phase, tasks, summary)

        # Save to file
        _save_summary(phase, summary, state)

        # Log to MLflow if applicable
        _log_to_mlflow(phase, summary, state)

        # Notify via Telegram
        if notify_telegram:
            _notify_telegram(summary)

        log.info(f"Phase summary complete: {len(summary['key_findings'])} findings")

        return summary

    except Exception as e:
        log.error(f"Summarizer failed: {e}")
        # Fallback summary
        return _fallback_summary(phase, tasks, state)


def _build_summary_prompt(
    phase: str,
    tasks: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> str:
    """Build prompt for summarizer."""
    # Format task results
    tasks_text = []
    for i, task_result in enumerate(tasks, 1):
        task = task_result.get("task", {})
        result = task_result.get("result", {})
        review = task_result.get("review", {})

        tasks_text.append(f"""
Task {i}: {task.get('name', 'Unknown')}
- Methodology: {task.get('methodology', 'N/A')[:200]}
- Output: {result.get('output', 'N/A')[:200]}
- Review: {review.get('summary', 'N/A')}
- Issues: {len(review.get('issues', []))} issues
""")

    tasks_summary = "\n".join(tasks_text)

    # State summary
    state_text = json.dumps(
        {k: v for k, v in state.items() if k not in ["code", "raw_data"]},
        indent=2,
        default=str,
    )[:800]

    prompt = f"""
Summarize the "{phase}" phase execution for a Kaggle competition.

=== TASKS COMPLETED ===
{tasks_summary}

=== CURRENT STATE ===
{state_text}

---

Generate a comprehensive summary JSON:
{{
  "key_findings": [
    "Finding 1 (concrete, data-driven)",
    "Finding 2",
    "Finding 3"
  ],
  "decisions_made": [
    {{
      "decision": "What was decided",
      "reasoning": "Why this choice was made"
    }}
  ],
  "metrics": {{
    "cv_score": 0.0,
    "feature_count": 0,
    "train_time_seconds": 0,
    "other_metrics": "..."
  }},
  "artifacts": [
    "path/to/file1.csv",
    "path/to/file2.pkl"
  ],
  "next_phase_recommendations": [
    "Recommendation 1 for next phase",
    "Recommendation 2"
  ],
  "telegram_summary": "2-3 sentence summary for Telegram notification (concise, no markdown)"
}}

Guidelines:
- Key findings should be specific and data-driven
- Decisions should explain the reasoning (not just what was done)
- Include actual metric values if available
- List all generated artifacts
- Recommendations should be actionable

Return ONLY the JSON object.
"""

    return prompt


def _parse_summary_response(response: str, phase: str) -> Dict[str, Any]:
    """Parse summary JSON from response."""
    response_clean = response.strip()

    if response_clean.startswith("```json"):
        response_clean = response_clean.split("```json")[1].split("```")[0].strip()
    elif response_clean.startswith("```"):
        response_clean = response_clean.split("```")[1].split("```")[0].strip()

    return json.loads(response_clean)


def _generate_markdown_report(
    phase: str,
    tasks: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    """Generate detailed markdown report."""
    report = [
        f"# {phase} - Phase Summary\n",
        "## Key Findings\n",
    ]

    for finding in summary.get("key_findings", []):
        report.append(f"- {finding}")

    report.append("\n## Decisions Made\n")
    for decision in summary.get("decisions_made", []):
        report.append(f"**{decision.get('decision', '')}**")
        report.append(f"- Reasoning: {decision.get('reasoning', '')}\n")

    report.append("## Metrics\n")
    metrics = summary.get("metrics", {})
    for key, value in metrics.items():
        report.append(f"- {key}: {value}")

    report.append("\n## Artifacts\n")
    for artifact in summary.get("artifacts", []):
        report.append(f"- `{artifact}`")

    report.append("\n## Next Phase Recommendations\n")
    for rec in summary.get("next_phase_recommendations", []):
        report.append(f"- {rec}")

    report.append("\n## Task Details\n")
    for i, task_result in enumerate(tasks, 1):
        task = task_result.get("task", {})
        review = task_result.get("review", {})

        report.append(f"\n### Task {i}: {task.get('name', 'Unknown')}")
        report.append(f"- Methodology: {task.get('methodology', 'N/A')}")
        report.append(f"- Review Score: {review.get('score', 0)}/100")
        report.append(f"- Issues: {len(review.get('issues', []))}")

    return "\n".join(report)


def _save_summary(phase: str, summary: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Save summary to .claude directory."""
    slug = state.get("competition_slug", "unknown")
    output_dir = Path(f".claude/summaries/{slug}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"{phase.lower().replace(' ', '_')}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save markdown
    md_path = output_dir / f"{phase.lower().replace(' ', '_')}.md"
    md_path.write_text(summary.get("full_report", ""))

    log.info(f"Saved summary to {json_path} and {md_path}")


def _log_to_mlflow(phase: str, summary: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Log phase summary to MLflow."""
    try:
        import mlflow

        if mlflow.active_run():
            # Log metrics
            metrics = summary.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{phase}_{key}", value)

            # Log summary as artifact
            mlflow.log_dict(summary, f"summaries/{phase}.json")

            log.info(f"Logged {phase} summary to MLflow")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


def _notify_telegram(summary: Dict[str, Any]) -> None:
    """Send Telegram notification."""
    try:
        phase = summary.get("phase", "Unknown")
        telegram_msg = summary.get("telegram_summary", "Phase complete")

        # Format for Telegram
        message = f"✅ {phase} complete\n\n{telegram_msg}"

        # Send via notify.py
        subprocess.run(
            ["python", "core/notify.py", message],
            capture_output=True,
            timeout=5,
        )

        log.info("Sent Telegram notification")

    except Exception as e:
        log.warning(f"Telegram notification failed: {e}")


def _fallback_summary(
    phase: str,
    tasks: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate fallback summary if parsing fails."""
    return {
        "phase": phase,
        "key_findings": [f"{len(tasks)} tasks completed"],
        "decisions_made": [],
        "metrics": {},
        "artifacts": [],
        "next_phase_recommendations": ["Continue to next phase"],
        "telegram_summary": f"{phase} phase completed with {len(tasks)} tasks",
        "full_report": f"# {phase}\n\nCompleted {len(tasks)} tasks.",
        "error": "Fallback summary - LLM parsing failed",
    }


if __name__ == "__main__":
    # Test summary
    tasks = [
        {
            "task": {
                "name": "Create lag features",
                "methodology": "Lags 1,7,14",
            },
            "result": {
                "output": "Created 3 lag features",
            },
            "review": {
                "summary": "Good implementation",
                "score": 85,
                "issues": [],
            },
        }
    ]

    state = {
        "competition_slug": "store-sales",
        "cv_score": 0.35,
    }

    summary = summarize_phase("Feature Engineering", tasks, state, notify_telegram=False)
    print(json.dumps(summary, indent=2, default=str))
