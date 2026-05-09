"""DeepEval API Service - FastAPI wrapper for experiment evaluation."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

app = FastAPI(title="DeepEval Service", version="1.0.0")

# Note: Full DeepEval integration would use actual metrics
# For now, we'll use our custom validation logic and extend with DeepEval later


class ExperimentConfig(BaseModel):
    """Experiment configuration for evaluation."""
    validation_strategy: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    model_type: Optional[str] = "xgboost"


class EvaluationRequest(BaseModel):
    """Request for experiment evaluation."""
    experiment_config: ExperimentConfig
    cv_score: float
    baseline_score: float
    competition_type: str = "time-series"


class EvaluationResponse(BaseModel):
    """Evaluation results."""
    passed: bool
    issues: List[str]
    warnings: List[str]
    score: float
    recommendation: str


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "deepeval"}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_experiment(request: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluate an experiment for quality and correctness.

    Checks:
    - Validation strategy appropriateness
    - Performance regressions
    - Feature engineering sanity
    - Hyperparameter bounds
    - Data leakage risks
    """
    issues = []
    warnings = []

    config = request.experiment_config
    cv = request.cv_score
    baseline = request.baseline_score
    comp_type = request.competition_type

    # Check 1: Validation strategy
    val_strategy = config.validation_strategy.lower()
    if comp_type == "time-series":
        valid_strategies = ["timeseriesplit", "time_series_split", "temporal", "timeseries"]
        if not any(vs in val_strategy for vs in valid_strategies):
            issues.append(
                f"Invalid validation for time-series: '{config.validation_strategy}'. "
                "Should use TimeSeriesSplit for temporal data."
            )

    # Check 2: Performance regression
    regression_pct = ((cv / baseline) - 1) * 100
    if cv > baseline * 1.15:  # 15% worse
        issues.append(
            f"Significant regression: CV {cv:.4f} vs baseline {baseline:.4f} "
            f"({regression_pct:+.1f}%)"
        )
    elif cv > baseline * 1.05:  # 5-15% worse
        warnings.append(
            f"Minor regression: CV {cv:.4f} vs baseline {baseline:.4f} "
            f"({regression_pct:+.1f}%)"
        )

    # Check 3: Unrealistic improvement (possible data leakage)
    improvement_pct = ((baseline / cv) - 1) * 100
    if cv < baseline * 0.75:  # 25%+ improvement
        warnings.append(
            f"Suspiciously large improvement ({improvement_pct:.1f}%). "
            "Check for data leakage or validation errors."
        )

    # Check 4: Feature count sanity
    feature_count = len(config.features)
    if feature_count > 100:
        warnings.append(f"Very high feature count ({feature_count}). Risk of overfitting.")
    elif feature_count < 3:
        warnings.append(f"Very low feature count ({feature_count}). May underfit.")

    # Check 5: Time-series specific feature checks
    if comp_type == "time-series":
        has_lags = any("lag" in f.lower() for f in config.features)
        has_rolling = any("roll" in f.lower() or "moving" in f.lower() for f in config.features)

        if not has_lags:
            warnings.append("No lag features found. Time-series usually benefit from lags.")

        if not has_rolling:
            warnings.append("No rolling/moving average features found.")

        # Check for future leakage
        suspicious_features = [
            f for f in config.features
            if "future" in f.lower() or "next" in f.lower() or "forward" in f.lower()
        ]
        if suspicious_features:
            issues.append(
                f"Features may leak future data: {suspicious_features}"
            )

    # Check 6: Hyperparameter sanity
    params = config.hyperparameters
    if "learning_rate" in params:
        lr = params["learning_rate"]
        if lr > 0.5:
            warnings.append(f"High learning rate ({lr}). May cause instability.")
        elif lr < 0.0001:
            warnings.append(f"Very low learning rate ({lr}). Training may be very slow.")

    if "max_depth" in params:
        depth = params["max_depth"]
        if depth > 15:
            warnings.append(f"Very deep trees ({depth}). High overfitting risk.")

    # Calculate overall quality score
    score = 1.0
    score -= len(issues) * 0.25  # Each issue -0.25
    score -= len(warnings) * 0.08  # Each warning -0.08
    score = max(0.0, min(1.0, score))

    # Generate recommendation
    if len(issues) > 0:
        if score < 0.3:
            recommendation = "🚨 CRITICAL: Multiple serious problems. Do NOT submit."
        elif score < 0.6:
            recommendation = "❌ FAILED: Fix critical issues before submission."
        else:
            recommendation = "⚠️ ISSUES FOUND: Address problems before submitting."
    elif len(warnings) > 2:
        recommendation = "⚠️ CAUTION: Multiple warnings. Review carefully before submission."
    elif len(warnings) > 0:
        recommendation = "✅ PASSED with warnings. Review minor concerns."
    else:
        recommendation = "✅ EXCELLENT: Experiment looks good. Safe to submit."

    return EvaluationResponse(
        passed=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        score=score,
        recommendation=recommendation
    )


@app.post("/evaluate/features")
def evaluate_features(
    features: List[str],
    competition_type: str = "time-series"
) -> Dict[str, Any]:
    """Evaluate feature engineering quality."""
    issues = []
    warnings = []

    if competition_type == "time-series":
        has_temporal = any(
            term in " ".join(features).lower()
            for term in ["lag", "roll", "moving", "shift", "diff"]
        )
        if not has_temporal:
            warnings.append("No temporal features (lag/rolling) for time-series data.")

        # Check for common time features
        has_time_features = any(
            term in " ".join(features).lower()
            for term in ["day", "month", "week", "year", "season", "holiday"]
        )
        if not has_time_features:
            warnings.append("No date/time features found.")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "feature_count": len(features)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
