"""Reader Agent - Parse competition docs into structured summaries.

Responsibilities:
- Read competition overview.txt
- Sample training/test data
- Parse notebook insights from past winners
- Generate structured competition_info.txt

Cost: 100% Ollama (zero API cost)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from core.llm_interface import ask_ollama

log = logging.getLogger(__name__)


def read_competition(competition_slug: str, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Read and structure competition information.

    Args:
        competition_slug: Kaggle competition slug
        data_dir: Override data directory (default: data/{slug})

    Returns:
        Structured competition info dict with:
        - problem_type: classification, regression, time series, etc.
        - eval_metric: AUC, RMSE, RMSLE, etc.
        - data_shape: {"train_rows": X, "train_cols": Y, "test_rows": Z}
        - features: [{name, type, missing_pct}, ...]
        - target_info: {name, type, distribution}
        - known_issues: [list of data leaks, issues from discussions]
        - winning_approaches: [approaches from top notebooks]
        - summary: one-paragraph overview
    """
    log.info(f"Reader agent analyzing {competition_slug}")

    # Paths
    comp_dir = Path(f"competitions/active/{competition_slug}")
    data_dir = data_dir or Path(f"data/{competition_slug}")

    overview_path = comp_dir / "overview.txt"
    insights_path = Path(f".claude/notebook_insights_{competition_slug}.md")

    # 1. Load competition overview
    overview = ""
    if overview_path.exists():
        overview = overview_path.read_text()
    else:
        log.warning(f"No overview.txt found at {overview_path}")

    # 2. Sample data
    train_sample = _sample_data(data_dir / "train.csv", rows=10)
    test_sample = _sample_data(data_dir / "test.csv", rows=5)

    # 3. Load notebook insights
    insights = ""
    if insights_path.exists():
        insights = insights_path.read_text()
    else:
        log.warning(f"No notebook insights at {insights_path}")

    # 4. Ask Ollama to structure the information
    prompt = f"""
Analyze this Kaggle competition and extract structured information.

=== COMPETITION OVERVIEW ===
{overview}

=== TRAINING DATA SAMPLE ===
Shape: {train_sample.get('shape', 'unknown')}
Columns: {train_sample.get('columns', [])}
Sample rows:
{train_sample.get('preview', 'N/A')}

=== TEST DATA SAMPLE ===
Shape: {test_sample.get('shape', 'unknown')}
Columns: {test_sample.get('columns', [])}

=== TOP NOTEBOOK INSIGHTS ===
{insights[:2000] if insights else 'No insights available yet'}

---

Generate a JSON object with:
{{
  "problem_type": "classification|regression|time_series|other",
  "eval_metric": "the primary evaluation metric",
  "data_shape": {{"train_rows": X, "train_cols": Y, "test_rows": Z}},
  "features": [
    {{"name": "feature_name", "type": "numeric|categorical|datetime|text", "missing_pct": 0.0}}
  ],
  "target_info": {{
    "name": "target_column_name",
    "type": "binary|multiclass|continuous|other",
    "distribution": "balanced|imbalanced|normal|skewed|other"
  }},
  "known_issues": [
    "any data leaks, train/test differences, known bugs from discussions"
  ],
  "winning_approaches": [
    "common patterns from top notebooks (feature engineering, models, ensembles)"
  ],
  "summary": "one paragraph overview of the competition and key challenges"
}}

Return ONLY the JSON object, no other text.
"""

    system = "You are a data science competition analyst. Extract structured information from competition docs."

    response = ask_ollama(prompt, system=system, think=False)

    # Parse JSON response
    try:
        # Clean up response (sometimes Ollama adds markdown)
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean.split("```json")[1].split("```")[0].strip()
        elif response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1].split("```")[0].strip()

        comp_info = json.loads(response_clean)

        # Add metadata
        comp_info["competition_slug"] = competition_slug
        comp_info["data_dir"] = str(data_dir)

        log.info(f"Reader complete: {comp_info['problem_type']}, metric: {comp_info['eval_metric']}")

        return comp_info

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Reader response as JSON: {e}")
        log.error(f"Response: {response}")

        # Fallback: return minimal info
        return {
            "competition_slug": competition_slug,
            "problem_type": "unknown",
            "eval_metric": "unknown",
            "data_shape": train_sample.get("shape", {}),
            "features": [],
            "target_info": {},
            "known_issues": [],
            "winning_approaches": [],
            "summary": "Failed to parse competition info - manual review needed",
            "data_dir": str(data_dir),
            "error": str(e),
            "raw_response": response[:500],
        }


def _sample_data(path: Path, rows: int = 10) -> Dict[str, Any]:
    """Load sample of CSV data."""
    if not path.exists():
        log.warning(f"Data file not found: {path}")
        return {"error": "file_not_found"}

    try:
        df = pd.read_csv(path, nrows=rows)

        return {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview": df.head(5).to_string(),
            "missing": df.isnull().sum().to_dict(),
        }

    except Exception as e:
        log.error(f"Failed to load data from {path}: {e}")
        return {"error": str(e)}


def save_competition_info(comp_info: Dict[str, Any], output_path: Path) -> None:
    """Save structured competition info to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(comp_info, f, indent=2)

    log.info(f"Saved competition info to {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.agents.reader <competition_slug>")
        sys.exit(1)

    slug = sys.argv[1]
    comp_info = read_competition(slug)

    # Save to .claude directory
    output_path = Path(f".claude/competition_info_{slug}.json")
    save_competition_info(comp_info, output_path)

    print(json.dumps(comp_info, indent=2))
