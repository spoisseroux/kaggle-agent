"""LLM-based experiment evaluation using Ollama on RTX 5070.

This replaces the basic rule-based validation in core/evals.py with
intelligent LLM-based evaluation that can catch subtle issues.
"""
from typing import Dict, Any, List
import json
from . import ollama_client


class ExperimentEvaluator:
    """Evaluate Kaggle experiments using LLM-based analysis."""

    def __init__(self, model_name: str = "qwen3:14b"):
        """Initialize with Ollama model."""
        self.model_name = model_name

    def evaluate_experiment(
        self,
        experiment_config: Dict[str, Any],
        cv_score: float,
        baseline_score: float,
        competition_type: str = "time-series"
    ) -> Dict[str, Any]:
        """
        Evaluate experiment quality using LLM.

        Returns:
            {
                "passed": bool,
                "score": float (0-1),
                "issues": List[str],
                "warnings": List[str],
                "recommendation": str,
                "llm_analysis": str
            }
        """
        # Format experiment for LLM
        exp_text = self._format_experiment(
            experiment_config, cv_score, baseline_score, competition_type
        )

        # Get LLM evaluation
        prompt = f"""You are evaluating a Kaggle competition experiment configuration.

{exp_text}

Analyze this experiment and respond in JSON format with:
{{
  "passed": true/false,
  "critical_issues": ["list of critical problems that must be fixed"],
  "warnings": ["list of minor concerns"],
  "score": 0.0-1.0,
  "explanation": "brief explanation of the evaluation"
}}

Critical issues to check:
1. Validation strategy must match competition type (TimeSeriesSplit for time-series)
2. No data leakage in features (no 'future', 'next', 'forward' keywords)
3. Hyperparameters in reasonable ranges (lr: 0.001-0.5, depth: 3-15)
4. Performance improvement realistic (<25% better than baseline suggests leakage)
5. Regression from baseline (<15% worse is critical, 5-15% is warning)

Respond ONLY with valid JSON, no other text."""

        llm_response = ollama_client.generate(
            prompt,
            model=self.model_name,
            temperature=0.3,
            think=True  # Use thinking mode for better evaluation
        )

        # Parse LLM response
        try:
            result = self._parse_llm_response(llm_response)
            result["llm_analysis"] = llm_response
            return result
        except Exception as e:
            # Fallback if JSON parsing fails
            return {
                "passed": False,
                "score": 0.0,
                "issues": [f"LLM evaluation failed: {e}"],
                "warnings": [],
                "recommendation": "Unable to evaluate - LLM response parsing error",
                "llm_analysis": llm_response
            }

    def _format_experiment(
        self,
        config: Dict[str, Any],
        cv: float,
        baseline: float,
        comp_type: str
    ) -> str:
        """Format experiment details for LLM."""
        delta_pct = ((cv / baseline) - 1) * 100 if baseline > 0 else 0

        features = config.get("features", [])
        feature_list = ", ".join(features[:10])
        if len(features) > 10:
            feature_list += f"... ({len(features)} total)"

        return f"""
**Competition Type:** {comp_type}
**Validation Strategy:** {config.get('validation_strategy', 'unknown')}
**Model:** {config.get('model_type', 'unknown')}

**Features ({len(features)}):**
{feature_list}

**Hyperparameters:**
{json.dumps(config.get('hyperparameters', {}), indent=2)}

**Performance:**
- CV Score: {cv:.4f}
- Baseline: {baseline:.4f}
- Delta: {delta_pct:+.1f}%
"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        response = response.strip()

        # Sometimes LLM adds markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        # Parse JSON
        data = json.loads(response)

        # Standardize format
        return {
            "passed": data.get("passed", False),
            "score": float(data.get("score", 0.0)),
            "issues": data.get("critical_issues", []),
            "warnings": data.get("warnings", []),
            "recommendation": data.get("explanation", "No explanation provided")
        }

    def quick_eval(self, cv: float, baseline: float, val_strategy: str) -> str:
        """Quick one-liner evaluation."""
        config = {"validation_strategy": val_strategy}
        result = self.evaluate_experiment(config, cv, baseline)

        if result["passed"]:
            return f"✅ {result['recommendation']}"
        else:
            issues = "; ".join(result["issues"])
            return f"❌ {result['recommendation']} Issues: {issues}"


# Example usage
if __name__ == "__main__":
    evaluator = ExperimentEvaluator()

    # Test good experiment
    good = {
        "validation_strategy": "TimeSeriesSplit",
        "features": ["Lag_1", "Lag_7", "Roll_mean_7"],
        "hyperparameters": {"learning_rate": 0.05, "max_depth": 8},
        "model_type": "xgboost"
    }
    result = evaluator.evaluate_experiment(good, cv_score=0.350, baseline_score=0.367, competition_type="time-series")
    print("Good experiment:", json.dumps(result, indent=2))

    # Test bad experiment
    bad = {
        "validation_strategy": "train_test_split",
        "features": ["future_sales", "next_week"],
        "hyperparameters": {"learning_rate": 0.05},
        "model_type": "xgboost"
    }
    result = evaluator.evaluate_experiment(bad, cv_score=0.250, baseline_score=0.367, competition_type="time-series")
    print("\nBad experiment:", json.dumps(result, indent=2))
