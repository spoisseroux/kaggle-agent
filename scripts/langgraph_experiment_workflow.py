#!/usr/bin/env python3
"""LangGraph workflow for systematic Kaggle experimentation.

Implements StateGraph pattern for Store Sales with validation gates:
- Correlation check (detect alignment bugs)
- Distribution check (detect scaling issues)
- CV threshold (stop bad experiments early)
- LB validation (track actual performance)

Pattern from multi-agent LangGraph notebook adapted for ML workflows.
"""
from typing import TypedDict, Annotated, Literal, List
from operator import add
from pathlib import Path
import subprocess
import json
import numpy as np
import pandas as pd
from datetime import datetime

# State definition
class ExperimentState(TypedDict):
    """Shared state across all workflow nodes."""
    competition: str
    experiment_id: str
    hypothesis: str
    config: dict

    # Execution tracking
    actions_performed: Annotated[List[str], add]  # Accumulator
    current_result: dict

    # Validation gates
    correlation_with_baseline: float
    distribution_check: str  # 'pass', 'fail', 'pending'
    cv_score: float
    lb_score: float

    # Decision
    next_action: str  # 'continue', 'stop', 'submit', 'fix'
    stop_reason: str

# Validation thresholds
CORRELATION_THRESHOLD = 0.95  # Must be >0.95 correlation with working model
CV_IMPROVEMENT_THRESHOLD = 0.01  # Must be at least 1% better
MIN_ACCEPTABLE_CV = 0.60  # Stop if CV > 0.60 (worse than v50)

# Baseline paths
BASELINE_PREDICTIONS = Path("competitions/active/store-sales-time-series-forecasting/predictions/v50_ensemble_test.npy")
BASELINE_CV = 0.3925  # v50 CV
BASELINE_LB = 0.455  # v50 LB


def generate_experiment(state: ExperimentState) -> dict:
    """Plan next experiment based on hypothesis."""
    hypothesis = state["hypothesis"]
    config = state["config"]

    # Generate experiment config
    exp_config = {
        "model_type": config.get("model_type", "lightgbm"),
        "features": config.get("features", "v1"),
        "approach": config.get("approach", "batch"),
        "params": config.get("params", {})
    }

    action = f"Generated experiment config: {exp_config['approach']} {exp_config['model_type']}"

    return {
        "config": exp_config,
        "actions_performed": [action],
        "distribution_check": "pending"
    }


def run_experiment(state: ExperimentState) -> dict:
    """Execute the experiment and get predictions."""
    exp_id = state["experiment_id"]
    config = state["config"]

    # Build script path
    script_name = f"model_{config['model_type']}_{ exp_id}.py"
    script_path = Path(f"competitions/active/store-sales-time-series-forecasting/src/{script_name}")

    if not script_path.exists():
        return {
            "next_action": "stop",
            "stop_reason": f"Script not found: {script_path}",
            "actions_performed": ["Script not found"]
        }

    # Run experiment
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )

        # Parse output for CV score
        cv_score = None
        for line in result.stdout.split('\n'):
            if 'CV:' in line or 'Holdout CV:' in line:
                try:
                    cv_score = float(line.split(':')[-1].strip())
                    break
                except:
                    pass

        if cv_score is None:
            return {
                "next_action": "stop",
                "stop_reason": "Could not parse CV score from output",
                "actions_performed": ["Experiment failed to return CV"]
            }

        action = f"Ran experiment {exp_id}, CV: {cv_score:.4f}"

        return {
            "cv_score": cv_score,
            "current_result": {"cv": cv_score, "output": result.stdout},
            "actions_performed": [action]
        }

    except subprocess.TimeoutExpired:
        return {
            "next_action": "stop",
            "stop_reason": "Experiment timeout (>30 min)",
            "actions_performed": ["Experiment timeout"]
        }
    except Exception as e:
        return {
            "next_action": "stop",
            "stop_reason": f"Experiment error: {e}",
            "actions_performed": [f"Error: {e}"]
        }


def validate_correlation(state: ExperimentState) -> dict:
    """Check correlation with baseline to detect alignment bugs."""
    exp_id = state["experiment_id"]

    # Load experiment predictions
    pred_file = Path(f"competitions/active/store-sales-time-series-forecasting/predictions/{exp_id}_test.npy")

    if not pred_file.exists():
        return {
            "correlation_with_baseline": 0.0,
            "next_action": "stop",
            "stop_reason": "Predictions file not found",
            "actions_performed": ["Predictions not saved"]
        }

    try:
        exp_preds = np.load(pred_file)
        baseline_preds = np.load(BASELINE_PREDICTIONS)

        # Calculate correlation
        correlation = np.corrcoef(exp_preds, baseline_preds)[0, 1]

        action = f"Correlation with baseline: {correlation:.4f}"

        # Check if correlation is acceptable
        if correlation < 0:
            # NEGATIVE correlation = alignment bug (like v32/v33/v63)
            return {
                "correlation_with_baseline": correlation,
                "next_action": "fix",
                "stop_reason": f"ALIGNMENT BUG: Negative correlation ({correlation:.4f})",
                "actions_performed": [action, "🚨 ALIGNMENT BUG DETECTED"]
            }
        elif correlation < CORRELATION_THRESHOLD:
            # Low correlation = something fundamentally different
            return {
                "correlation_with_baseline": correlation,
                "next_action": "stop",
                "stop_reason": f"Low correlation ({correlation:.4f}) - approach too different",
                "actions_performed": [action, "Low correlation - stopping"]
            }
        else:
            # Good correlation - continue
            return {
                "correlation_with_baseline": correlation,
                "actions_performed": [action, "✓ Correlation check passed"]
            }

    except Exception as e:
        return {
            "correlation_with_baseline": 0.0,
            "next_action": "stop",
            "stop_reason": f"Correlation check error: {e}",
            "actions_performed": [f"Error: {e}"]
        }


def validate_distribution(state: ExperimentState) -> dict:
    """Check prediction distribution vs baseline."""
    exp_id = state["experiment_id"]

    pred_file = Path(f"competitions/active/store-sales-time-series-forecasting/predictions/{exp_id}_test.npy")

    try:
        exp_preds = np.load(pred_file)
        baseline_preds = np.load(BASELINE_PREDICTIONS)

        exp_mean = np.mean(exp_preds)
        baseline_mean = np.mean(baseline_preds)
        diff_pct = (exp_mean - baseline_mean) / baseline_mean * 100

        action = f"Prediction mean: {exp_mean:.1f} (baseline: {baseline_mean:.1f}, {diff_pct:+.1f}%)"

        # Check if distribution is reasonable
        if abs(diff_pct) > 20:
            # More than 20% different = likely scaling issue (like v66)
            return {
                "distribution_check": "fail",
                "next_action": "fix",
                "stop_reason": f"Distribution mismatch: {diff_pct:+.1f}% vs baseline",
                "actions_performed": [action, "🚨 SCALING ISSUE DETECTED"]
            }
        elif abs(diff_pct) > 10:
            # 10-20% different = warning but continue
            return {
                "distribution_check": "warning",
                "actions_performed": [action, f"⚠️ Distribution {diff_pct:+.1f}% off baseline"]
            }
        else:
            # Within 10% = good
            return {
                "distribution_check": "pass",
                "actions_performed": [action, "✓ Distribution check passed"]
            }

    except Exception as e:
        return {
            "distribution_check": "fail",
            "next_action": "stop",
            "stop_reason": f"Distribution check error: {e}",
            "actions_performed": [f"Error: {e}"]
        }


def validate_cv(state: ExperimentState) -> dict:
    """Check if CV is acceptable."""
    cv = state["cv_score"]

    improvement = (BASELINE_CV - cv) / BASELINE_CV * 100

    action = f"CV {cv:.4f} vs baseline {BASELINE_CV:.4f} ({improvement:+.1f}%)"

    if cv > MIN_ACCEPTABLE_CV:
        # Too bad, stop
        return {
            "next_action": "stop",
            "stop_reason": f"CV {cv:.4f} too bad (threshold: {MIN_ACCEPTABLE_CV})",
            "actions_performed": [action, "CV too bad - stopping"]
        }
    elif cv > BASELINE_CV:
        # Worse than baseline but not catastrophic
        return {
            "next_action": "stop",
            "stop_reason": f"CV {cv:.4f} worse than baseline {BASELINE_CV:.4f}",
            "actions_performed": [action, "No improvement - stopping"]
        }
    else:
        # Better than baseline!
        if improvement >= CV_IMPROVEMENT_THRESHOLD * 100:
            # Significant improvement - submit
            return {
                "next_action": "submit",
                "actions_performed": [action, f"✅ {improvement:.1f}% improvement - ready to submit"]
            }
        else:
            # Minor improvement
            return {
                "next_action": "continue",
                "actions_performed": [action, f"✓ {improvement:.1f}% improvement"]
            }


def decide_next_action(state: ExperimentState) -> Literal["submit", "stop", "fix", "continue"]:
    """Router function - decides next node based on validation results."""
    next_action = state.get("next_action")

    if next_action:
        return next_action

    # Default: continue if no explicit decision
    return "continue"


def submit_experiment(state: ExperimentState) -> dict:
    """Prepare submission and ask for approval."""
    exp_id = state["experiment_id"]
    cv = state["cv_score"]
    correlation = state["correlation_with_baseline"]

    message = f"""
🎯 EXPERIMENT READY FOR SUBMISSION

ID: {exp_id}
Hypothesis: {state['hypothesis']}

Results:
  CV: {cv:.4f} ({(BASELINE_CV - cv) / BASELINE_CV * 100:+.1f}% vs baseline)
  Correlation: {correlation:.4f}
  Distribution: {state['distribution_check']}

All validation gates passed ✓

Submit to leaderboard?
1) Yes, submit now
2) No, keep for later
"""

    # Ask human for approval
    try:
        result = subprocess.run(
            ["python", "core/ask_human.py", message],
            capture_output=True,
            text=True
        )
        choice = result.stdout.strip()

        if choice == "1":
            return {
                "next_action": "end",
                "actions_performed": ["Approved for submission"],
                "stop_reason": "Ready to submit"
            }
        else:
            return {
                "next_action": "end",
                "actions_performed": ["Saved for later"],
                "stop_reason": "User chose not to submit"
            }
    except Exception as e:
        return {
            "next_action": "end",
            "actions_performed": [f"Error asking for approval: {e}"],
            "stop_reason": "Error in submission approval"
        }


def fix_bug(state: ExperimentState) -> dict:
    """Log bug and stop - human intervention needed."""
    stop_reason = state["stop_reason"]

    message = f"""
🐛 BUG DETECTED IN EXPERIMENT

{stop_reason}

Experiment: {state['experiment_id']}
Hypothesis: {state['hypothesis']}

Actions performed:
{'  '.join('- ' + a for a in state['actions_performed'])}

Human intervention required.
"""

    # Notify via Telegram
    subprocess.run(["python", "core/notify.py", message])

    return {
        "next_action": "end",
        "actions_performed": ["Notified human of bug"],
        "stop_reason": stop_reason
    }


def stop_experiment(state: ExperimentState) -> dict:
    """Stop experiment and log reason."""
    stop_reason = state.get("stop_reason", "Unknown")

    return {
        "actions_performed": [f"Stopped: {stop_reason}"],
        "next_action": "end"
    }


def build_workflow():
    """Build the LangGraph StateGraph for experiments."""
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError:
        print("langgraph not installed - this is a reference implementation")
        print("Install with: pip install langgraph")
        return None

    workflow = StateGraph(ExperimentState)

    # Add nodes
    workflow.add_node("generate", generate_experiment)
    workflow.add_node("run", run_experiment)
    workflow.add_node("check_correlation", validate_correlation)
    workflow.add_node("check_distribution", validate_distribution)
    workflow.add_node("check_cv", validate_cv)
    workflow.add_node("submit", submit_experiment)
    workflow.add_node("fix", fix_bug)
    workflow.add_node("stop", stop_experiment)

    # Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "run")
    workflow.add_edge("run", "check_correlation")

    # Conditional routing after correlation check
    workflow.add_conditional_edges(
        "check_correlation",
        decide_next_action,
        {
            "continue": "check_distribution",
            "fix": "fix",
            "stop": "stop",
            "submit": "submit"
        }
    )

    workflow.add_conditional_edges(
        "check_distribution",
        decide_next_action,
        {
            "continue": "check_cv",
            "fix": "fix",
            "stop": "stop"
        }
    )

    workflow.add_conditional_edges(
        "check_cv",
        decide_next_action,
        {
            "continue": END,  # Passed all checks, experiment complete
            "submit": "submit",
            "stop": "stop"
        }
    )

    workflow.add_edge("submit", END)
    workflow.add_edge("fix", END)
    workflow.add_edge("stop", END)

    return workflow.compile()


def main():
    """Demo usage."""
    print("="*80)
    print("LANGGRAPH EXPERIMENT WORKFLOW")
    print("="*80)
    print()
    print("This workflow provides systematic validation gates:")
    print("  1. Correlation check - detects alignment bugs (v32/v33/v63)")
    print("  2. Distribution check - detects scaling issues (v66)")
    print("  3. CV threshold - stops bad experiments early")
    print("  4. Human approval - before submission")
    print()
    print("To use:")
    print("  1. Install langgraph: pip install langgraph")
    print("  2. Create experiment script")
    print("  3. Run workflow with experiment config")
    print()
    print("Validation thresholds:")
    print(f"  - Correlation > {CORRELATION_THRESHOLD}")
    print(f"  - CV improvement > {CV_IMPROVEMENT_THRESHOLD * 100}%")
    print(f"  - CV < {MIN_ACCEPTABLE_CV}")
    print()
    print("Workflow saved to: scripts/langgraph_experiment_workflow.py")

if __name__ == "__main__":
    main()
