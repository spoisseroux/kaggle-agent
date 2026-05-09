"""DeepEval test suite for Kaggle experiment quality.

Tests experiment configurations before training to catch issues early.
Uses LLM-based evaluation with Ollama (RTX 5070).
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval, AnswerRelevancyMetric
from typing import Dict, Any
from .ollama_model import OllamaModel


# Sample experiment configurations for testing
GOOD_EXPERIMENT = {
    "competition_type": "time-series",
    "validation_strategy": "TimeSeriesSplit",
    "features": ["Lag_1", "Lag_7", "Lag_14", "Roll_mean_7", "Roll_std_7", "day_of_week", "month"],
    "hyperparameters": {
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    "model_type": "xgboost",
    "cv_score": 0.350,
    "baseline_score": 0.367
}

BAD_EXPERIMENT_WRONG_CV = {
    "competition_type": "time-series",
    "validation_strategy": "train_test_split",  # WRONG for time-series
    "features": ["feature_1", "feature_2"],
    "hyperparameters": {"learning_rate": 0.1},
    "model_type": "xgboost",
    "cv_score": 0.450,
    "baseline_score": 0.367
}

BAD_EXPERIMENT_DATA_LEAKAGE = {
    "competition_type": "time-series",
    "validation_strategy": "TimeSeriesSplit",
    "features": ["future_sales", "next_week_demand", "Lag_1"],  # Future leakage!
    "hyperparameters": {"learning_rate": 0.05},
    "model_type": "xgboost",
    "cv_score": 0.250,  # Suspiciously good
    "baseline_score": 0.367
}


def format_experiment_for_eval(exp: Dict[str, Any]) -> str:
    """Format experiment config into a readable prompt for LLM evaluation."""
    return f"""
Competition Type: {exp['competition_type']}
Validation Strategy: {exp['validation_strategy']}
Model: {exp.get('model_type', 'unknown')}

Features ({len(exp.get('features', []))}):
{', '.join(exp.get('features', []))}

Hyperparameters:
{', '.join(f"{k}={v}" for k, v in exp.get('hyperparameters', {}).items())}

Performance:
- CV Score: {exp.get('cv_score', 'N/A')}
- Baseline: {exp.get('baseline_score', 'N/A')}
- Delta: {((exp.get('cv_score', 1) / exp.get('baseline_score', 1) - 1) * 100):.1f}%
"""


def create_validation_metric():
    """Create a custom metric for validating experiment quality."""
    ollama = OllamaModel(model="qwen3:14b")

    return GEval(
        name="Experiment Validation Quality",
        criteria="You are evaluating a Kaggle competition experiment configuration. Check for: "
                 "1) Appropriate validation strategy for the competition type (e.g., TimeSeriesSplit for time-series), "
                 "2) No data leakage in features (no 'future', 'next', 'forward' features), "
                 "3) Reasonable hyperparameters (learning rate 0.001-0.5, max_depth 3-15), "
                 "4) Performance not suspiciously better than baseline (>25% improvement suggests leakage).",
        evaluation_params=["input", "actual_output"],
        threshold=0.7,
        model=ollama,
        evaluation_steps=[
            "Check if validation strategy matches competition type",
            "Scan features for future-leaking keywords",
            "Verify hyperparameters are in reasonable ranges",
            "Check if performance improvement is realistic"
        ]
    )


@pytest.mark.parametrize("experiment,expected_pass", [
    (GOOD_EXPERIMENT, True),
    (BAD_EXPERIMENT_WRONG_CV, False),
    (BAD_EXPERIMENT_DATA_LEAKAGE, False),
])
def test_experiment_validation(experiment, expected_pass):
    """Test experiment configurations for quality issues."""

    # Format experiment for evaluation
    exp_text = format_experiment_for_eval(experiment)

    # Create test case
    test_case = LLMTestCase(
        input=f"Validate this Kaggle experiment configuration:\n{exp_text}",
        actual_output="Valid experiment" if expected_pass else "Invalid experiment",
        context=[exp_text]
    )

    # Run validation metric
    metric = create_validation_metric()

    # Assert with appropriate expectation
    if expected_pass:
        assert_test(test_case, [metric])
    else:
        # For experiments we expect to fail, we actually expect the metric to catch issues
        # So we invert the logic here - the metric should score low for bad experiments
        try:
            assert_test(test_case, [metric])
            # If it passed but we expected failure, that's a problem
            pytest.fail(f"Metric did not catch issues in bad experiment: {experiment['validation_strategy']}")
        except AssertionError:
            # Expected - metric caught the issues
            pass


def test_good_experiment_passes():
    """Test that a properly configured experiment passes validation."""
    exp_text = format_experiment_for_eval(GOOD_EXPERIMENT)

    test_case = LLMTestCase(
        input=f"Evaluate this Kaggle experiment:\n{exp_text}",
        actual_output="This experiment uses proper TimeSeriesSplit validation for time-series data, "
                     "includes appropriate lag and rolling window features, has reasonable hyperparameters, "
                     "and shows realistic 4.6% improvement over baseline.",
        context=[exp_text]
    )

    metric = create_validation_metric()
    assert_test(test_case, [metric])


def test_detect_wrong_validation_strategy():
    """Test that wrong validation strategy is detected."""
    exp_text = format_experiment_for_eval(BAD_EXPERIMENT_WRONG_CV)

    test_case = LLMTestCase(
        input=f"Evaluate this Kaggle experiment:\n{exp_text}",
        actual_output="This experiment has a critical error: using train_test_split for time-series data "
                     "instead of TimeSeriesSplit. This will cause data leakage and overestimate performance.",
        context=[exp_text]
    )

    # This should fail validation due to wrong CV strategy
    metric = create_validation_metric()

    # We expect this to score low (bad experiment)
    metric.measure(test_case)
    assert metric.score < 0.7, f"Metric should catch wrong validation strategy, got score: {metric.score}"
