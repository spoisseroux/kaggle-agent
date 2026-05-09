"""Format structured data for readable Telegram messages.

Telegram doesn't support tables or Markdown well, so we format data
as plain text with clear structure.
"""
from typing import Any, Dict, List


def format_table(rows: List[Dict[str, Any]], title: str = "") -> str:
    """Format a list of dicts as readable text for Telegram.

    Example input:
    [
        {"name": "xgb_v1", "cv": 0.321, "lb": 0.526},
        {"name": "lgbm_v1", "cv": 0.401, "lb": 0.572}
    ]

    Output:
    Title (if provided)

    1. xgb_v1
       CV: 0.321 | LB: 0.526

    2. lgbm_v1
       CV: 0.401 | LB: 0.572
    """
    if not rows:
        return "No data to display"

    lines = []
    if title:
        lines.append(title)
        lines.append("")

    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {row.get('name', 'Unnamed')}")

        # Format key-value pairs
        details = []
        for key, value in row.items():
            if key == 'name':
                continue

            # Format based on type
            if isinstance(value, float):
                formatted = f"{value:.3f}"
            else:
                formatted = str(value)

            details.append(f"{key.upper()}: {formatted}")

        if details:
            lines.append("   " + " | ".join(details))
        lines.append("")

    return "\n".join(lines)


def format_kaggle_submissions(submissions: List[Dict[str, Any]]) -> str:
    """Format Kaggle submissions for Telegram.

    Expects submissions with: fileName, date, publicScore, status
    """
    if not submissions:
        return "No submissions found"

    lines = ["📊 Latest Submissions:", ""]

    for i, sub in enumerate(submissions, 1):
        filename = sub.get('fileName', 'Unknown')
        score = sub.get('publicScore', 'N/A')
        status = sub.get('status', 'Unknown')
        date = sub.get('date', '')[:10] if sub.get('date') else ''

        # Extract model name from filename
        model_name = filename.replace('_submission.csv', '').replace('.csv', '')

        lines.append(f"{i}. {model_name}")

        details = []
        if date:
            details.append(f"Date: {date}")
        if score != 'N/A':
            try:
                details.append(f"LB: {float(score):.3f}")
            except:
                details.append(f"LB: {score}")

        # Simplify status
        if 'COMPLETE' in str(status):
            status_simple = "✅"
        elif 'PENDING' in str(status):
            status_simple = "⏳"
        elif 'ERROR' in str(status):
            status_simple = "❌"
        else:
            status_simple = str(status)

        details.append(status_simple)

        lines.append("   " + " | ".join(details))
        lines.append("")

    return "\n".join(lines)


def format_experiment_results(results: Dict[str, Any]) -> str:
    """Format experiment results for Telegram.

    Example input:
    {
        "name": "XGBoost v5 TimeSeriesSplit",
        "cv_mean": 0.350,
        "cv_std": 0.016,
        "baseline": 0.367,
        "improvement": 0.017,
        "params": {"max_depth": 12, "learning_rate": 0.03}
    }
    """
    lines = [f"🧪 {results.get('name', 'Experiment Results')}", ""]

    # Main metrics
    if 'cv_mean' in results:
        cv = results['cv_mean']
        lines.append(f"CV Score: {cv:.4f}")

        if 'cv_std' in results:
            lines.append(f"Std Dev: ±{results['cv_std']:.4f}")

    # Comparison
    if 'baseline' in results and 'improvement' in results:
        baseline = results['baseline']
        improvement = results['improvement']

        lines.append("")
        lines.append(f"Baseline: {baseline:.4f}")

        if improvement > 0:
            lines.append(f"Improvement: -{improvement:.4f} ✅")
        else:
            lines.append(f"Change: +{abs(improvement):.4f}")

    # Parameters
    if 'params' in results and results['params']:
        lines.append("")
        lines.append("Best Parameters:")
        for key, value in results['params'].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)
