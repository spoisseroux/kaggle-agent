"""Experiment evaluation using DeepEval or similar frameworks.

Validates experiments before submission to catch issues like:
- Invalid CV strategy for competition type
- Nonsensical feature engineering
- Unexpected performance degradation
- Data leakage risks
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

# Will support both local and remote DeepEval
DEEPEVAL_MODE = os.getenv("DEEPEVAL_MODE", "service")  # or "builtin"
DEEPEVAL_URL = os.getenv("DEEPEVAL_URL", "http://docker:8001")  # Tailscale hostname


class ExperimentEval:
    """Evaluate experiment quality before submission."""

    def __init__(self, mode: str = DEEPEVAL_MODE, url: str = DEEPEVAL_URL):
        self.mode = mode
        self.url = url if mode == "service" else None

    def evaluate_experiment(
        self,
        experiment_config: Dict[str, Any],
        cv_score: float,
        baseline_score: float,
        competition_type: str = "time-series"
    ) -> Dict[str, Any]:
        """
        Evaluate an experiment for quality and correctness.

        Args:
            experiment_config: Dict with model, features, validation_strategy, etc.
            cv_score: Cross-validation score
            baseline_score: Baseline to compare against
            competition_type: Type of competition (time-series, classification, etc.)

        Returns:
            Dict with:
                - passed: bool
                - issues: List[str] - Problems found
                - warnings: List[str] - Potential concerns
                - score: float - Overall quality score 0-1
        """
        # If using service mode, call the API
        if self.mode == "service" and self.url:
            try:
                import httpx
                response = httpx.post(
                    f"{self.url}/evaluate",
                    json={
                        "experiment_config": experiment_config,
                        "cv_score": cv_score,
                        "baseline_score": baseline_score,
                        "competition_type": competition_type
                    },
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"DeepEval service error: {response.status_code}, falling back to builtin")
            except Exception as e:
                print(f"Failed to connect to DeepEval service: {e}, falling back to builtin")

        # Fallback to built-in evaluation
        issues = []
        warnings = []

        # Check 1: Validation strategy
        val_strategy = experiment_config.get("validation_strategy", "unknown")
        if competition_type == "time-series":
            if val_strategy not in ["TimeSeriesSplit", "time_series_split", "temporal"]:
                issues.append(
                    f"Invalid validation for time-series: '{val_strategy}'. "
                    "Should use TimeSeriesSplit for temporal data."
                )

        # Check 2: Performance regression
        if cv_score > baseline_score * 1.1:  # 10% worse
            issues.append(
                f"Significant regression: CV {cv_score:.4f} vs baseline {baseline_score:.4f}. "
                f"This is {((cv_score/baseline_score - 1) * 100):.1f}% worse."
            )
        elif cv_score > baseline_score * 1.05:  # 5-10% worse
            warnings.append(
                f"Minor regression: CV {cv_score:.4f} vs baseline {baseline_score:.4f}"
            )

        # Check 3: Unrealistic improvement (possible data leakage)
        if cv_score < baseline_score * 0.8:  # 20%+ improvement
            warnings.append(
                f"Suspiciously large improvement: {((1 - cv_score/baseline_score) * 100):.1f}%. "
                "Check for data leakage or validation errors."
            )

        # Check 4: Feature count sanity
        features = experiment_config.get("features", [])
        if isinstance(features, list) and len(features) > 100:
            warnings.append(
                f"Very high feature count ({len(features)}). Risk of overfitting."
            )

        # Check 5: Hyperparameters sanity
        params = experiment_config.get("hyperparameters", {})
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if lr > 0.5:
                warnings.append(f"High learning rate ({lr}). May cause instability.")
            elif lr < 0.001:
                warnings.append(f"Very low learning rate ({lr}). Training may be slow.")

        # Calculate overall score
        score = 1.0
        score -= len(issues) * 0.3  # Each issue -0.3
        score -= len(warnings) * 0.1  # Each warning -0.1
        score = max(0.0, min(1.0, score))

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "score": score,
            "recommendation": self._get_recommendation(issues, warnings, score)
        }

    def _get_recommendation(
        self,
        issues: List[str],
        warnings: List[str],
        score: float
    ) -> str:
        """Generate actionable recommendation."""
        if score >= 0.9:
            return "✅ Experiment looks good. Safe to submit."
        elif score >= 0.7:
            return "⚠️ Minor concerns. Review warnings before submitting."
        elif score >= 0.5:
            return "❌ Significant issues found. Fix before submitting."
        else:
            return "🚨 Critical problems. Do not submit - experiment likely invalid."

    def evaluate_features(
        self,
        features: List[str],
        competition_type: str = "time-series"
    ) -> Dict[str, Any]:
        """Evaluate feature engineering logic."""
        issues = []
        warnings = []

        # Time-series specific checks
        if competition_type == "time-series":
            has_lags = any("lag" in f.lower() for f in features)
            has_rolling = any("roll" in f.lower() or "moving" in f.lower() for f in features)

            if not has_lags:
                warnings.append("No lag features found. Consider adding lagged values.")

            if not has_rolling:
                warnings.append("No rolling/moving average features. May miss trends.")

            # Check for future leakage
            suspicious = [f for f in features if "future" in f.lower() or "next" in f.lower()]
            if suspicious:
                issues.append(
                    f"Suspicious features that may leak future data: {suspicious}"
                )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


def quick_eval(cv: float, baseline: float, val_strategy: str = "unknown") -> str:
    """Quick one-line evaluation."""
    evaluator = ExperimentEval()
    result = evaluator.evaluate_experiment(
        {"validation_strategy": val_strategy},
        cv,
        baseline
    )

    if result["passed"]:
        return f"✅ {result['recommendation']}"
    else:
        return f"❌ {result['recommendation']}\nIssues: {', '.join(result['issues'])}"


# Example usage
if __name__ == "__main__":
    evaluator = ExperimentEval()

    # Test case: Good experiment
    result = evaluator.evaluate_experiment(
        {
            "validation_strategy": "TimeSeriesSplit",
            "features": ["Lag_1", "Lag_7", "Roll_mean_7"],
            "hyperparameters": {"learning_rate": 0.05, "max_depth": 8}
        },
        cv_score=0.350,
        baseline_score=0.367,
        competition_type="time-series"
    )
    print("Good experiment:", json.dumps(result, indent=2))

    # Test case: Bad experiment
    result = evaluator.evaluate_experiment(
        {
            "validation_strategy": "train_test_split",  # WRONG for time-series
            "features": ["feature1", "feature2"],
            "hyperparameters": {"learning_rate": 0.05}
        },
        cv_score=0.450,  # Much worse
        baseline_score=0.367,
        competition_type="time-series"
    )
    print("\nBad experiment:", json.dumps(result, indent=2))
