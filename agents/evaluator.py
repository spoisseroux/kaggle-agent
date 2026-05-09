"""Evaluator Agent - Assess if experiment met success criteria.

Evaluates experiment results by:
- Comparing to success criteria from Strategist
- Performing root cause analysis if failed
- Generating actionable next steps
- Recording what was learned
"""
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.langfuse_tracker import track_decision
from core.ollama_client import generate as ask_ollama


@track_decision("evaluate_experiment")
def evaluate_experiment(
    experiment: dict,
    implementation_result: dict,
    competition: str
) -> dict:
    """
    Evaluate experiment results against success criteria.

    Args:
        experiment: Original experiment spec from Strategist
        implementation_result: Results from Engineer agent
        competition: Competition slug

    Returns:
        {
            "experiment_id": str,
            "success": bool,
            "assessment": str,
            "root_cause": str | None,
            "next_steps": list[str],
            "learned": str
        }
    """
    print(f"\n{'='*70}")
    print(f"EVALUATOR AGENT - Assessing {experiment['id']}: {experiment['name']}")
    print(f"{'='*70}\n")

    # Check if implementation succeeded
    if not implementation_result.get("success"):
        return _evaluate_failure(experiment, implementation_result)

    # Implementation succeeded - check if it met success criteria
    execution_results = implementation_result.get("execution_results", {})
    cv_score = execution_results.get("cv_score")
    lb_score = execution_results.get("lb_score")

    success_criteria = experiment.get("success_criteria", "")
    expected_impact = experiment.get("expected_impact", "")

    # Use Ollama to analyze results
    prompt = f"""Evaluate this Kaggle experiment result:

EXPERIMENT:
ID: {experiment['id']}
Name: {experiment['name']}
Hypothesis: {experiment['hypothesis']}
Success Criteria: {success_criteria}
Expected Impact: {expected_impact}

RESULTS:
CV Score: {cv_score if cv_score else 'N/A'}
LB Score: {lb_score if lb_score else 'N/A (not submitted)'}
Implementation Attempts: {implementation_result.get('attempts', 1)}

Based on the results:
1. Did the experiment succeed (meet success criteria)?
2. If yes: What worked and why?
3. If no/partial: What went wrong (root cause)?
4. What should we try next (2-3 actionable next steps)?
5. What did we learn (one key takeaway)?

Respond with JSON:
{{
  "success": true/false,
  "assessment": "brief summary of outcome",
  "root_cause": "explanation if failed, otherwise null",
  "next_steps": ["step 1", "step 2", ...],
  "learned": "key takeaway"
}}
"""

    print("Analyzing results with Ollama...")
    response = ask_ollama(prompt, model="qwen3:14b", think=True)

    # Parse JSON
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        evaluation = json.loads(response)

        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Success: {evaluation.get('success', False)}")
        print(f"Assessment: {evaluation.get('assessment', 'N/A')}")
        if evaluation.get('root_cause'):
            print(f"Root Cause: {evaluation.get('root_cause')}")
        print(f"Learned: {evaluation.get('learned', 'N/A')}")
        print(f"{'='*70}\n")

        return {
            "experiment_id": experiment['id'],
            "success": evaluation.get("success", False),
            "assessment": evaluation.get("assessment", ""),
            "root_cause": evaluation.get("root_cause"),
            "next_steps": evaluation.get("next_steps", []),
            "learned": evaluation.get("learned", "")
        }

    except json.JSONDecodeError:
        print(f"ERROR: Failed to parse Ollama response as JSON")
        print(f"Raw response: {response[:500]}")

        # Fallback evaluation
        return {
            "experiment_id": experiment['id'],
            "success": False,
            "assessment": "Evaluation failed - Ollama response parsing error",
            "root_cause": "Could not parse evaluation results",
            "next_steps": ["Retry evaluation", "Manual review"],
            "learned": "Need better JSON response handling"
        }


def _evaluate_failure(experiment: dict, implementation_result: dict) -> dict:
    """Evaluate a failed implementation."""
    print("Implementation failed - analyzing failure...")

    errors = implementation_result.get("validation_errors", [])
    attempts = implementation_result.get("attempts", 0)

    # Simple failure analysis
    root_cause = "Unknown"
    if errors:
        if any("syntax" in str(e).lower() for e in errors):
            root_cause = "Code generation produced syntax errors"
        elif any("import" in str(e).lower() for e in errors):
            root_cause = "Missing or incorrect imports"
        elif any("runtime" in str(e).lower() for e in errors):
            root_cause = "Runtime execution error"
        else:
            root_cause = errors[0] if errors else "Unknown error"

    next_steps = [
        "Simplify experiment approach",
        "Add more context to code generation prompt",
        "Manually review and fix generated code"
    ]

    learned = f"Implementation failed after {attempts} attempts - may need human intervention"

    print(f"\n{'='*70}")
    print(f"EVALUATION: IMPLEMENTATION FAILURE")
    print(f"{'='*70}")
    print(f"Root Cause: {root_cause}")
    print(f"Attempts: {attempts}")
    print(f"{'='*70}\n")

    return {
        "experiment_id": experiment['id'],
        "success": False,
        "assessment": f"Implementation failed after {attempts} attempts",
        "root_cause": root_cause,
        "next_steps": next_steps,
        "learned": learned
    }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m agents.evaluator <competition_slug> <experiment_json> <implementation_result_json>")
        sys.exit(1)

    competition = sys.argv[1]
    experiment_json = sys.argv[2]
    implementation_json = sys.argv[3]

    try:
        experiment = json.loads(experiment_json)
        implementation_result = json.loads(implementation_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        sys.exit(1)

    result = evaluate_experiment(experiment, implementation_result, competition)

    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"Experiment: {result['experiment_id']}")
    print(f"Success: {result['success']}")
    print(f"Assessment: {result['assessment']}")
    if result['root_cause']:
        print(f"Root Cause: {result['root_cause']}")
    print(f"\nNext Steps:")
    for i, step in enumerate(result['next_steps'], 1):
        print(f"  {i}. {step}")
    print(f"\nLearned: {result['learned']}")
