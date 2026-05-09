"""Strategist Agent - Decide what to try next based on analysis.

Creates prioritized experiment queue based on:
- Analyst insights and bottlenecks
- Past experiment history (what worked, what didn't)
- Expected impact vs effort tradeoffs
- Available compute budget
"""
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.langfuse_tracker import track_decision
from core.ollama_client import generate as ask_ollama
from core import memory


@track_decision("strategize_experiments")
def strategize_experiments(
    analysis: dict,
    competition: str,
    max_experiments: int = 5,
    compute_budget_hours: float = 6.0
) -> list[dict]:
    """
    Generate prioritized list of experiments based on analysis.

    Args:
        analysis: Output from Analyst agent
        competition: Competition slug
        max_experiments: Maximum number of experiments to propose
        compute_budget_hours: Available compute time

    Returns:
        List of experiment specifications, ordered by priority
    """
    print(f"\n{'='*70}")
    print("STRATEGIST AGENT - Planning experiments")
    print(f"{'='*70}\n")

    current_best = analysis.get("current_best", {})
    insights = analysis.get("insights", [])
    bottlenecks = analysis.get("bottlenecks", [])
    patterns = analysis.get("patterns", [])
    model_comparison = analysis.get("model_comparison", {})

    # Query past experiments from Qdrant for similar approaches
    past_learnings = _query_past_learnings(competition)

    # Build prompt for Ollama to generate experiment ideas
    prompt = f"""You are a Kaggle competition strategist. Based on the analysis below, propose {max_experiments} experiments to improve performance.

CURRENT STATE:
Best CV: {current_best.get('cv', {}).get('score', 'N/A')}
Best LB: {current_best.get('lb', {}).get('score', 'N/A')}

INSIGHTS:
{_format_list(insights)}

BOTTLENECKS:
{_format_list(bottlenecks)}

PATTERNS:
{_format_list(patterns)}

MODEL COMPARISON:
{json.dumps(model_comparison, indent=2)}

PAST LEARNINGS (similar competitions):
{past_learnings}

CONSTRAINTS:
- Compute budget: {compute_budget_hours} hours
- Must address at least one bottleneck
- Prioritize high-impact, low-effort experiments first

For each experiment, provide:
1. Name (short, descriptive)
2. Hypothesis (what you expect to happen and why)
3. Approach (specific implementation steps)
4. Expected impact (estimated CV/LB improvement range)
5. Effort (estimated time in minutes)
6. Priority (1=highest)
7. Success criteria (how to know if it worked)

Respond ONLY with valid JSON array (no markdown, no explanations):
[
  {{
    "id": "exp_001",
    "name": "...",
    "hypothesis": "...",
    "approach": "...",
    "expected_impact": "+0.01 to +0.03 LB improvement",
    "effort_minutes": 30,
    "priority": 1,
    "success_criteria": "..."
  }}
]
"""

    print("Generating experiment ideas with Ollama (thinking mode)...")
    response = ask_ollama(prompt, model="qwen3:14b", think=True)

    # Parse JSON response
    try:
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        experiments = json.loads(response)

        # Validate and enrich experiments
        for i, exp in enumerate(experiments):
            if "id" not in exp:
                exp["id"] = f"exp_{i+1:03d}"
            if "priority" not in exp:
                exp["priority"] = i + 1

        # Sort by priority
        experiments.sort(key=lambda x: x.get("priority", 999))

        print(f"\n{'='*70}")
        print(f"STRATEGY COMPLETE - {len(experiments)} experiments proposed")
        print(f"{'='*70}")
        for exp in experiments[:3]:
            print(f"\n  {exp['id']}: {exp['name']}")
            print(f"    Priority: {exp['priority']}, Effort: {exp.get('effort_minutes', 'N/A')} min")
            print(f"    Impact: {exp.get('expected_impact', 'N/A')}")
        if len(experiments) > 3:
            print(f"\n  ... and {len(experiments) - 3} more")
        print(f"{'='*70}\n")

        return experiments

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse Ollama response as JSON: {e}")
        print(f"Raw response:\n{response[:500]}")

        # Fallback: return a basic experiment based on bottlenecks
        fallback_experiments = _generate_fallback_experiments(bottlenecks, model_comparison)
        return fallback_experiments


def _query_past_learnings(competition: str) -> str:
    """Query Qdrant for similar successful approaches from past competitions."""
    try:
        # This would query Qdrant for embedded successful experiments
        # For now, return placeholder
        return "No past learnings available yet (Qdrant integration pending)"
    except Exception as e:
        return f"Error querying past learnings: {e}"


def _format_list(items: list) -> str:
    """Format a list of strings for display."""
    if not items:
        return "  (none)"
    return "\n".join([f"  - {item}" for item in items])


def _generate_fallback_experiments(bottlenecks: list, model_comparison: dict) -> list[dict]:
    """Generate basic experiments when Ollama response fails."""
    experiments = []

    # If there are bottlenecks, address the first one
    if bottlenecks:
        experiments.append({
            "id": "exp_001",
            "name": "Address primary bottleneck",
            "hypothesis": f"Resolving '{bottlenecks[0]}' will improve performance",
            "approach": "Investigate and implement fix for identified bottleneck",
            "expected_impact": "+0.01 to +0.05 improvement",
            "effort_minutes": 45,
            "priority": 1,
            "success_criteria": "Bottleneck metric improves or CV-LB gap reduces"
        })

    # Try best-performing model with different params
    if model_comparison:
        best_model = min(model_comparison.items(), key=lambda x: x[1].get("best_lb", float("inf")))
        experiments.append({
            "id": "exp_002",
            "name": f"Tune {best_model[0]} hyperparameters",
            "hypothesis": f"{best_model[0]} is performing best, tuning could improve further",
            "approach": "Run Optuna hyperparameter optimization",
            "expected_impact": "+0.01 to +0.03 improvement",
            "effort_minutes": 60,
            "priority": 2,
            "success_criteria": "CV improves by at least 0.01"
        })

    return experiments


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m agents.strategist <competition_slug>")
        print("  Optional: First run Analyst agent to get analysis JSON")
        sys.exit(1)

    competition = sys.argv[1]

    # Run Analyst first to get current state
    from agents.analyst import analyze_state
    print("Running Analyst first...")
    analysis = analyze_state(competition)

    # Now strategize
    experiments = strategize_experiments(analysis, competition, max_experiments=5)

    print("\n" + "="*70)
    print("EXPERIMENT QUEUE")
    print("="*70)
    for exp in experiments:
        print(f"\n{exp['id']}: {exp['name']}")
        print(f"  Priority: {exp['priority']}")
        print(f"  Hypothesis: {exp['hypothesis']}")
        print(f"  Expected Impact: {exp.get('expected_impact', 'N/A')}")
        print(f"  Effort: {exp.get('effort_minutes', 'N/A')} minutes")
        print(f"  Success Criteria: {exp.get('success_criteria', 'N/A')}")
