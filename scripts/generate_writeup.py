"""Generate a public Kaggle notebook writeup after a competition.

Reads experiment history from MLflow, submission history, and competition
metadata, then uses Ollama (qwen3:14b) to draft a structured writeup
saved as a Jupyter notebook (.ipynb) ready to post to Kaggle.

Usage:
    python scripts/generate_writeup.py --competition titanic
    python scripts/generate_writeup.py  # uses active competition
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _active_competition() -> str:
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        return reg.get("active", "")
    except Exception:
        return ""


def _get_experiments(slug: str, top_n: int = 10) -> list[dict]:
    try:
        import mlflow  # type: ignore
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(slug)
        if not exp:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.cv_score DESC"],
            max_results=top_n,
        )
        result = []
        for r in runs:
            result.append({
                "name": r.info.run_name,
                "cv": r.data.metrics.get("cv_score", r.data.metrics.get("cv")),
                "lb": r.data.metrics.get("lb_score"),
                "params": dict(r.data.params),
                "tags": dict(r.data.tags),
            })
        return result
    except Exception as e:
        print(f"MLflow unavailable: {e}", file=sys.stderr)
        return []


def _get_submissions(slug: str) -> list[dict]:
    sub_dir = REPO_ROOT / "submissions" / slug
    if not sub_dir.exists():
        return []
    subs = []
    for f in sorted(sub_dir.glob("*.json")):
        try:
            subs.append(json.loads(f.read_text()))
        except Exception:
            pass
    return subs


def _ollama_generate(prompt: str) -> str:
    try:
        from core.ollama_client import generate  # type: ignore
        return generate(prompt, think=False)
    except Exception:
        # Fallback: return the prompt as a placeholder
        return f"[Ollama unavailable — fill in manually]\n\nPrompt was:\n{prompt}"


def _build_notebook(sections: dict[str, str], slug: str) -> dict:
    """Build a minimal Jupyter notebook structure."""
    def md_cell(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {},
                "source": text.strip().splitlines(keepends=True)}

    def code_cell(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": text.strip().splitlines(keepends=True)}

    cells = [
        md_cell(f"# {sections['title']}\n\n{sections['intro']}"),
        md_cell(f"## Approach\n\n{sections['approach']}"),
        code_cell("# Key imports used throughout\nimport pandas as pd\nimport numpy as np"),
        md_cell(f"## Feature Engineering\n\n{sections['features']}"),
        md_cell(f"## Model & Results\n\n{sections['results']}"),
        md_cell(f"## What Didn't Work\n\n{sections['negatives']}"),
        md_cell(f"## Takeaways\n\n{sections['takeaways']}"),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def generate_writeup(slug: str) -> Path:
    print(f"Generating writeup for: {slug}")

    experiments = _get_experiments(slug)
    submissions = _get_submissions(slug)

    comp_md_path = REPO_ROOT / "competitions" / "active" / slug / "CLAUDE.md"
    comp_context = comp_md_path.read_text() if comp_md_path.exists() else ""

    best_cv = max((e["cv"] for e in experiments if e["cv"] is not None), default=None)
    best_lb = max((e["lb"] for e in experiments if e["lb"] is not None), default=None)
    top_models = [e["name"] for e in experiments[:3]]

    exp_summary = "\n".join(
        f"- {e['name']}: CV={e['cv']}, LB={e['lb']}, params={e['params']}"
        for e in experiments[:8]
    ) or "No experiments logged."

    context = f"""Competition: {slug}
Best CV: {best_cv}
Best LB: {best_lb}
Top models: {top_models}
Experiments:\n{exp_summary}

Competition notes:\n{comp_context[:1500]}"""

    print("Drafting sections with Ollama...")

    sections = {}

    sections["title"] = _ollama_generate(
        f"Write a concise, engaging Kaggle notebook title for my {slug} solution. "
        f"Best CV: {best_cv}, Best LB: {best_lb}. Just the title, no quotes."
    )

    sections["intro"] = _ollama_generate(
        f"Write a 2-3 paragraph intro for a Kaggle notebook about the {slug} competition. "
        f"Mention the task, evaluation metric, and that this is an AI-assisted solution. "
        f"Context: {context[:800]}"
    )

    sections["approach"] = _ollama_generate(
        f"Describe the ML approach taken for {slug} in 3-5 bullet points. "
        f"Focus on: data preprocessing, feature engineering strategy, model choice, ensembling. "
        f"Context: {context}"
    )

    sections["features"] = _ollama_generate(
        f"Describe the key features engineered for {slug} in a Kaggle writeup style. "
        f"Be specific about what worked. Context: {context}"
    )

    sections["results"] = _ollama_generate(
        f"Summarise the model results for {slug}: best CV={best_cv}, best LB={best_lb}. "
        f"Top experiments: {exp_summary[:600]}. "
        f"Write in Kaggle notebook style, 2-3 paragraphs."
    )

    sections["negatives"] = _ollama_generate(
        f"For the {slug} competition, list 3-5 things that were tried but didn't improve CV. "
        f"Context: {context[:600]}. Be honest and specific — this helps other competitors learn."
    )

    sections["takeaways"] = _ollama_generate(
        f"Write 3-5 key takeaways from the {slug} competition for a Kaggle writeup. "
        f"Include: what you'd do differently, what was most impactful, lessons learned."
    )

    notebook = _build_notebook(sections, slug)

    out_dir = REPO_ROOT / "writeups"
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"{slug}_writeup_{date_str}.ipynb"
    out_path.write_text(json.dumps(notebook, indent=2))

    print(f"\nWriteup saved: {out_path}")
    print("Next steps:")
    print("  1. Review and edit the notebook")
    print(f"  2. Upload to Kaggle: kaggle kernels push -p {out_path.parent}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kaggle competition writeup")
    parser.add_argument("--competition", "-c", default="", help="Competition slug")
    args = parser.parse_args()

    slug = args.competition or _active_competition()
    if not slug:
        print("No active competition found. Use --competition <slug>", file=sys.stderr)
        return 1

    generate_writeup(slug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
