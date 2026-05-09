"""Analyst Agent - Understand current state and identify opportunities.

Analyzes all past submissions and experiments to:
- Identify what's working vs not working
- Calculate CV vs LB gaps
- Find feature importance trends
- Compare model performance
- Identify bottlenecks
"""
import sys
from pathlib import Path
from typing import Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.langfuse_tracker import track_decision
from core.ollama_client import generate as ask_ollama
from core import memory
import mlflow


@track_decision("analyze_state")
def analyze_state(competition: str) -> dict:
    """
    Analyze current competition state and identify opportunities.

    Returns:
        {
            "current_best": {...},
            "insights": [...],
            "bottlenecks": [...],
            "model_comparison": {...},
            "feature_trends": {...}
        }
    """
    print(f"\n{'='*70}")
    print("ANALYST AGENT - Analyzing competition state")
    print(f"{'='*70}\n")

    # 1. Get all submissions
    submissions = memory.list_submissions(competition, limit=100)
    if not submissions:
        return {
            "current_best": None,
            "insights": ["No submissions found yet - this is a fresh start"],
            "bottlenecks": [],
            "model_comparison": {},
            "feature_trends": {}
        }

    # 2. Find current best
    best_cv = min(submissions, key=lambda s: s.get("cv_score") or float("inf"))
    best_lb = min(submissions, key=lambda s: s.get("lb_score") or float("inf"))

    current_best = {
        "cv": {
            "model": best_cv.get("filename"),
            "score": best_cv.get("cv_score"),
            "lb_score": best_cv.get("lb_score"),
            "gap": abs((best_cv.get("lb_score") or 0) - (best_cv.get("cv_score") or 0)) if best_cv.get("cv_score") and best_cv.get("lb_score") else None
        },
        "lb": {
            "model": best_lb.get("filename"),
            "score": best_lb.get("lb_score"),
            "cv_score": best_lb.get("cv_score"),
            "gap": abs((best_lb.get("lb_score") or 0) - (best_lb.get("cv_score") or 0)) if best_lb.get("cv_score") and best_lb.get("lb_score") else None
        }
    }

    # 3. Calculate CV-LB gaps for all submissions
    gaps = []
    for sub in submissions:
        if sub.get("cv_score") and sub.get("lb_score"):
            gap = abs(sub["lb_score"] - sub["cv_score"])
            gaps.append({
                "model": sub["filename"],
                "cv": sub["cv_score"],
                "lb": sub["lb_score"],
                "gap": gap
            })

    # 4. Model comparison
    model_types = {}
    for sub in submissions:
        filename = sub.get("filename", "")
        model_type = None
        if "xgb" in filename or "xgboost" in filename:
            model_type = "xgboost"
        elif "lgbm" in filename or "lightgbm" in filename:
            model_type = "lightgbm"
        elif "catboost" in filename or "cat" in filename:
            model_type = "catboost"
        elif "ensemble" in filename:
            model_type = "ensemble"

        if model_type:
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append({
                "cv": sub.get("cv_score"),
                "lb": sub.get("lb_score")
            })

    model_comparison = {}
    for model, scores in model_types.items():
        cv_scores = [s["cv"] for s in scores if s["cv"]]
        lb_scores = [s["lb"] for s in scores if s["lb"]]
        model_comparison[model] = {
            "count": len(scores),
            "avg_cv": np.mean(cv_scores) if cv_scores else None,
            "best_cv": min(cv_scores) if cv_scores else None,
            "avg_lb": np.mean(lb_scores) if lb_scores else None,
            "best_lb": min(lb_scores) if lb_scores else None
        }

    # 5. Query MLflow for feature importance trends (if available)
    feature_trends = _analyze_feature_importance(competition)

    # 6. Use Ollama to synthesize insights
    print("Synthesizing insights with Ollama...")

    # Format scores for prompt
    cv_lb_score = f"{current_best['cv']['lb_score']:.5f}" if current_best['cv']['lb_score'] else 'N/A'
    lb_score = f"{current_best['lb']['score']:.5f}" if current_best['lb']['score'] else 'N/A'
    lb_cv_score = f"{current_best['lb']['cv_score']:.4f}" if current_best['lb']['cv_score'] else 'N/A'

    prompt = f"""Analyze this Kaggle competition data and provide insights:

CURRENT BEST:
- Best CV: {current_best['cv']['model']} (CV: {current_best['cv']['score']:.4f}, LB: {cv_lb_score})
- Best LB: {current_best['lb']['model']} (LB: {lb_score}, CV: {lb_cv_score})

CV-LB GAPS:
{_format_gaps(gaps[:5])}

MODEL COMPARISON:
{_format_model_comparison(model_comparison)}

FEATURE IMPORTANCE TRENDS:
{feature_trends.get('summary', 'No feature importance data available yet')}

Provide:
1. 3-5 key insights about what's working
2. 2-3 bottlenecks or issues to address
3. Patterns you notice in the data

Format as JSON:
{{
  "insights": ["insight 1", "insight 2", ...],
  "bottlenecks": ["bottleneck 1", "bottleneck 2", ...],
  "patterns": ["pattern 1", "pattern 2", ...]
}}
"""

    analysis = ask_ollama(prompt, model="qwen3:14b", think=True)

    # Parse JSON response
    import json
    try:
        ollama_insights = json.loads(analysis)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        ollama_insights = {
            "insights": ["Analysis in progress - Ollama response needs JSON formatting"],
            "bottlenecks": [],
            "patterns": []
        }

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Best CV: {current_best['cv']['score']:.4f} ({current_best['cv']['model']})")
    print(f"Best LB: {current_best['lb']['score']:.5f} ({current_best['lb']['model']})")
    print(f"\nInsights: {len(ollama_insights['insights'])}")
    print(f"Bottlenecks: {len(ollama_insights['bottlenecks'])}")
    print(f"{'='*70}\n")

    return {
        "current_best": current_best,
        "insights": ollama_insights.get("insights", []),
        "bottlenecks": ollama_insights.get("bottlenecks", []),
        "patterns": ollama_insights.get("patterns", []),
        "model_comparison": model_comparison,
        "feature_trends": feature_trends,
        "cv_lb_gaps": gaps[:10],  # Top 10 by gap
        "total_submissions": len(submissions)
    }


def _analyze_feature_importance(competition: str) -> dict:
    """Analyze feature importance trends from MLflow experiments."""
    try:
        # This would query MLflow for feature importance data
        # For now, return placeholder
        return {
            "summary": "Feature importance analysis requires MLflow integration",
            "top_features": [],
            "trends": []
        }
    except Exception as e:
        return {
            "summary": f"Error analyzing features: {e}",
            "top_features": [],
            "trends": []
        }


def _format_gaps(gaps: list) -> str:
    """Format CV-LB gaps for display."""
    if not gaps:
        return "No gaps data available"
    lines = []
    for g in gaps:
        lines.append(f"  {g['model']}: CV {g['cv']:.4f}, LB {g['lb']:.5f}, Gap {g['gap']:.4f}")
    return "\n".join(lines)


def _format_model_comparison(comparison: dict) -> str:
    """Format model comparison for display."""
    if not comparison:
        return "No model comparison data available"
    lines = []
    for model, stats in comparison.items():
        best_cv = f"{stats['best_cv']:.4f}" if stats['best_cv'] else "N/A"
        best_lb = f"{stats['best_lb']:.5f}" if stats['best_lb'] else "N/A"
        lines.append(f"  {model.upper()}: {stats['count']} submissions, Best CV: {best_cv}, Best LB: {best_lb}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m agents.analyst <competition_slug>")
        sys.exit(1)

    competition = sys.argv[1]
    result = analyze_state(competition)

    print("\n" + "="*70)
    print("ANALYST REPORT")
    print("="*70)
    print(f"\nCurrent Best:")
    print(f"  CV: {result['current_best']['cv']['score']:.4f} - {result['current_best']['cv']['model']}")
    print(f"  LB: {result['current_best']['lb']['score']:.5f} - {result['current_best']['lb']['model']}")

    print(f"\nInsights:")
    for i, insight in enumerate(result['insights'], 1):
        print(f"  {i}. {insight}")

    print(f"\nBottlenecks:")
    for i, bottleneck in enumerate(result['bottlenecks'], 1):
        print(f"  {i}. {bottleneck}")

    if result['patterns']:
        print(f"\nPatterns:")
        for i, pattern in enumerate(result['patterns'], 1):
            print(f"  {i}. {pattern}")

    print(f"\nModel Comparison:")
    for model, stats in result['model_comparison'].items():
        best_lb = f"{stats['best_lb']:.5f}" if stats['best_lb'] else "N/A"
        print(f"  {model.upper()}: Best LB {best_lb}")
