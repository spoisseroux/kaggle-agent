"""Test Ollama integration with DeepEval."""
import pytest
from .ollama_model import OllamaModel


def test_ollama_model_basic():
    """Test that Ollama model can generate responses."""
    model = OllamaModel(model="qwen3:14b")

    prompt = "Explain in one sentence what TimeSeriesSplit validation is."
    response = model.generate(prompt)

    print(f"\n\nPrompt: {prompt}")
    print(f"Response: {response}\n")

    assert response, "Model should return a response"
    assert len(response) > 20, "Response should be substantive"
    assert "time" in response.lower() or "temporal" in response.lower() or "series" in response.lower(), \
        "Response should mention time/temporal/series"


def test_ollama_experiment_evaluation():
    """Test Ollama can evaluate experiment quality."""
    model = OllamaModel(model="qwen3:14b")

    experiment_desc = """
Evaluate this Kaggle experiment configuration:

Competition Type: time-series
Validation Strategy: train_test_split
Model: xgboost
Features: feature_1, feature_2, next_week_sales
Hyperparameters: learning_rate=0.05
CV Score: 0.250
Baseline: 0.367

Is this a good experiment configuration? Answer with YES or NO and explain why in 1-2 sentences.
"""

    response = model.generate(experiment_desc)

    print(f"\n\nExperiment Evaluation:")
    print(f"Response: {response}\n")

    assert response, "Model should return a response"
    # This experiment is bad (wrong CV strategy for time-series, has future-leaking feature)
    # The model should ideally flag this
    assert len(response) > 30, "Response should explain the evaluation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
