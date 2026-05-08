"""Ollama planner — designs the next batch of experiments as a priority-ordered JSON plan.

Reads current competition state, notebook insights, and the last reflection,
then uses Ollama (free, local) to produce a concrete experiment plan.
The main Claude agent reads this plan instead of deciding from scratch,
which cuts expensive Claude tokens significantly.

Usage:
    python scripts/plan_experiments.py
    python scripts/plan_experiments.py --competition titanic
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env, notify  # noqa: E402

_PLAN_SCHEMA = """[
  {
    "name": "experiment_name",
    "description": "one-line what this tries",
    "approach": "specific implementation details — model, features, params",
    "expected_cv_delta": 0.005,
    "priority": 1,
    "estimated_runtime_min": 15
  }
]"""


def _active_competition() -> str:
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        return reg.get("active", "")
    except Exception:
        return ""


def _competition_metric(slug: str) -> str:
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        comps = reg.get("competitions", {})
        if slug in comps:
            return comps[slug].get("metric", "unknown")
    except Exception:
        pass
    try:
        md = (REPO_ROOT / "competitions" / "active" / slug / "CLAUDE.md").read_text()
        for line in md.splitlines():
            if "metric" in line.lower():
                return line.strip()[:80]
    except Exception:
        pass
    return "accuracy"


def _get_best_cv(slug: str) -> float | None:
    try:
        import mlflow  # type: ignore
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(slug)
        if not exp:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.cv_score DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].data.metrics.get("cv_score", runs[0].data.metrics.get("cv"))
        return None
    except Exception:
        return None


def _get_recent_experiments(slug: str, n: int = 5) -> str:
    try:
        import mlflow  # type: ignore
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(slug)
        if not exp:
            return "None yet."
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )
        lines = []
        for r in runs:
            cv = r.data.metrics.get("cv_score", r.data.metrics.get("cv", "?"))
            lines.append(f"- {r.info.run_name}: CV={cv}")
        return "\n".join(lines) if lines else "None yet."
    except Exception:
        return "MLflow unavailable."


def _read_file_safe(path: Path, max_chars: int = 2000) -> str:
    try:
        content = path.read_text()
        return content[:max_chars] + ("..." if len(content) > max_chars else "")
    except Exception:
        return ""


def _extract_json(text: str) -> list:
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _plan_with_ollama(
    slug: str,
    metric: str,
    best_cv: float | None,
    recent_experiments: str,
    reflection: str,
    notebook_insights: str,
) -> list[dict]:
    try:
        from core.ollama_client import generate  # noqa: E402
        prompt = (
            f"You are a Kaggle ML planning agent for the '{slug}' competition.\n"
            f"Metric: {metric} | Best CV so far: {best_cv}\n\n"
            f"Recent experiments:\n{recent_experiments}\n\n"
            f"Last reflection / analysis:\n{reflection[:1200]}\n\n"
            f"Insights from top public notebooks:\n{notebook_insights[:1200]}\n\n"
            f"Design the next 5 experiments to maximise CV improvement.\n"
            f"Focus on highest-impact, lowest-risk changes first.\n"
            f"Output ONLY a valid JSON array matching this schema exactly:\n{_PLAN_SCHEMA}\n\n"
            f"No explanation outside the JSON. No markdown. Just the array."
        )
        raw = generate(prompt, think=True, max_tokens=2000)
        return _extract_json(raw)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return []


def _plan_to_markdown(plan: list[dict], slug: str, ts: str) -> str:
    lines = [f"# Experiment Plan — {slug}", f"Generated: {ts}", ""]
    for i, exp in enumerate(plan, 1):
        lines += [
            f"## {i}. {exp.get('name', 'Unnamed')} (priority {exp.get('priority', '?')})",
            f"**Description:** {exp.get('description', '')}",
            f"**Approach:** {exp.get('approach', '')}",
            f"**Expected CV delta:** +{exp.get('expected_cv_delta', '?')}",
            f"**Est. runtime:** {exp.get('estimated_runtime_min', '?')} min",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate experiment plan with Ollama planner")
    parser.add_argument("--competition", "-c", default="")
    args = parser.parse_args()

    _load_env()
    slug = args.competition or _active_competition()
    if not slug:
        print("No active competition. Use --competition <slug>", file=sys.stderr)
        return 1

    print(f"Planning experiments for: {slug}")

    claude_dir = REPO_ROOT / ".claude"
    claude_dir.mkdir(exist_ok=True)

    metric = _competition_metric(slug)
    best_cv = _get_best_cv(slug)
    recent = _get_recent_experiments(slug)
    reflection = _read_file_safe(claude_dir / f"last_reflection_{slug}.md")
    notebook_insights = _read_file_safe(claude_dir / f"notebook_insights_{slug}.md")

    print(f"Best CV: {best_cv} | Metric: {metric}")
    print("Calling Ollama planner (think=True)...")

    plan = _plan_with_ollama(slug, metric, best_cv, recent, reflection, notebook_insights)

    if not plan:
        print("Failed to generate plan.", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).isoformat()

    # Save JSON plan
    json_path = claude_dir / f"experiment_plan_{slug}.json"
    json_path.write_text(json.dumps(plan, indent=2))

    # Save markdown plan
    md_path = claude_dir / f"experiment_plan_{slug}.md"
    md_path.write_text(_plan_to_markdown(plan, slug, ts))

    print(f"\nPlan saved to:\n  {json_path}\n  {md_path}")
    print(f"\nTop experiment: {plan[0].get('name')} — {plan[0].get('description')}")

    top = plan[0]
    notify(
        f"🗺 Experiment plan updated for {slug}: {len(plan)} experiments queued.\n\n"
        f"Top priority: {top.get('name')} — {top.get('description')}\n"
        f"Expected CV delta: +{top.get('expected_cv_delta')} | ETA: {top.get('estimated_runtime_min')}min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
