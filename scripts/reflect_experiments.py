"""Reflect on experiment history to find patterns and guide next steps.

Reads the last N MLflow experiments, asks Ollama to analyse what's working
and why, then stores the reflection in Postgres memory and a local markdown
file for the agent to read at the start of each work cycle.

Usage:
    python scripts/reflect_experiments.py
    python scripts/reflect_experiments.py --competition titanic --last 15
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env, notify  # noqa: E402


def _active_competition() -> str:
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        return reg.get("active", "")
    except Exception:
        return ""


def _get_experiments(slug: str, n: int) -> list[dict]:
    try:
        import mlflow  # type: ignore
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(slug)
        if not exp:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.cv_score DESC"],
            max_results=n,
        )
        result = []
        for r in runs:
            cv = r.data.metrics.get("cv_score", r.data.metrics.get("cv"))
            lb = r.data.metrics.get("lb_score", r.data.metrics.get("lb"))
            result.append({
                "name": r.info.run_name or r.info.run_id[:8],
                "cv": round(cv, 6) if cv is not None else None,
                "lb": round(lb, 6) if lb is not None else None,
                "params": {k: v for k, v in list(r.data.params.items())[:8]},
                "duration_s": (
                    (r.info.end_time - r.info.start_time) // 1000
                    if r.info.end_time else None
                ),
            })
        return result
    except Exception as e:
        print(f"MLflow unavailable: {e}", file=sys.stderr)
        return []


def _format_experiments(experiments: list[dict]) -> str:
    if not experiments:
        return "No experiments logged yet."
    lines = []
    for i, e in enumerate(experiments, 1):
        cv_str = f"CV={e['cv']}" if e["cv"] is not None else "CV=?"
        lb_str = f" LB={e['lb']}" if e["lb"] is not None else ""
        dur_str = f" ({e['duration_s']}s)" if e["duration_s"] else ""
        params_str = ", ".join(f"{k}={v}" for k, v in list(e["params"].items())[:4])
        lines.append(f"{i}. {e['name']}: {cv_str}{lb_str}{dur_str} | {params_str}")
    return "\n".join(lines)


def _reflect_with_ollama(slug: str, experiments: list[dict]) -> str:
    try:
        from core.ollama_client import generate  # noqa: E402
        best_cv = max((e["cv"] for e in experiments if e["cv"] is not None), default=None)
        exp_text = _format_experiments(experiments)
        prompt = (
            f"You are a Kaggle ML expert analysing experiment results for the '{slug}' competition.\n"
            f"Best CV so far: {best_cv}\n\n"
            f"Experiments (best CV first):\n{exp_text}\n\n"
            f"Provide a structured analysis:\n"
            f"1. TOP FACTORS: What 2-3 factors most explain the best-performing experiments?\n"
            f"2. PATTERN: What separates good experiments (top 25%) from bad ones (bottom 25%)?\n"
            f"3. DEAD ENDS: What approaches clearly didn't help and should be dropped?\n"
            f"4. NEXT 3 EXPERIMENTS: Most promising next steps with specific rationale.\n"
            f"   Be specific about model type, features, hyperparameters to try.\n\n"
            f"Be concise and actionable. No generic advice."
        )
        return generate(prompt, think=True, max_tokens=1500)
    except Exception as e:
        return f"[Ollama unavailable: {e}]\n\nExperiments:\n{_format_experiments(experiments)}"


def _pg_save(slug: str, title: str, body: str) -> bool:
    try:
        import os
        import psycopg2  # type: ignore
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO memories (type, title, body, tags, created_at, updated_at)
               VALUES (%s, %s, %s, %s, NOW(), NOW())""",
            ("experiment_reflection", title, body, [slug, "reflection"]),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"pg_save error: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Reflect on experiment history")
    parser.add_argument("--competition", "-c", default="")
    parser.add_argument("--last", type=int, default=10, help="Number of recent experiments to analyse")
    args = parser.parse_args()

    _load_env()
    slug = args.competition or _active_competition()
    if not slug:
        print("No active competition. Use --competition <slug>", file=sys.stderr)
        return 1

    print(f"Loading last {args.last} experiments for: {slug}")
    experiments = _get_experiments(slug, args.last)

    if not experiments:
        print("No experiments found in MLflow.", file=sys.stderr)
        return 1

    print(f"Found {len(experiments)} experiments. Reflecting with Ollama (think=True)...")
    reflection = _reflect_with_ollama(slug, experiments)

    ts = datetime.now(timezone.utc).isoformat()
    title = f"reflection_{slug}_{ts}"

    # Save to Postgres
    _pg_save(slug, title, reflection)

    # Save to local markdown
    claude_dir = REPO_ROOT / ".claude"
    claude_dir.mkdir(exist_ok=True)
    md_path = claude_dir / f"last_reflection_{slug}.md"
    md_path.write_text(
        f"# Experiment reflection — {slug}\n"
        f"Generated: {ts} | Analysed: {len(experiments)} experiments\n\n"
        f"## Experiments analysed\n```\n{_format_experiments(experiments)}\n```\n\n"
        f"## Analysis\n\n{reflection}"
    )
    print(f"\nReflection saved to: {md_path}")
    print(f"\n{reflection}")

    # Short Telegram notify
    short = reflection[:400].rsplit(" ", 1)[0] + "..."
    notify(f"🧠 Experiment reflection for {slug}:\n\n{short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
