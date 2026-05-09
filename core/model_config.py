"""Dynamic model configuration for agents.

Allows switching Ollama models per agent, task type, or competition.
Can be configured via JSON file or environment variables.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

log = logging.getLogger(__name__)


# Default model assignments
DEFAULT_MODELS = {
    "reader": "qwen3:14b",          # Fast, good at structured extraction
    "planner": "qwen3:14b",         # Strategic reasoning
    "developer": "qwen3:14b",       # Code generation
    "reviewer": "qwen3:14b",        # Code analysis
    "summarizer": "qwen3:14b",      # Documentation
}

# Task-specific overrides (if certain tasks need different models)
TASK_MODELS = {
    "debugging": "qwen3:14b",       # Could use deepseek-coder for debugging
    "feature_engineering": "qwen3:14b",
    "model_training": "qwen3:14b",
}

# Competition-specific overrides (if certain competitions benefit from specific models)
COMPETITION_MODELS = {
    # "nlp-competition": {"developer": "codellama:13b"},
    # "cv-competition": {"developer": "deepseek-coder:33b"},
}


class ModelConfig:
    """Manage model assignments for agents."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize model config.

        Args:
            config_path: Path to JSON config file (optional)
        """
        self.config_path = config_path or Path(".claude/model_config.json")
        self.models = DEFAULT_MODELS.copy()
        self.task_models = TASK_MODELS.copy()
        self.competition_models = COMPETITION_MODELS.copy()

        # Load from file if exists
        if self.config_path.exists():
            self.load()

        # Override from environment variables
        self._load_from_env()

    def get_model(
        self,
        agent: str,
        task_type: Optional[str] = None,
        competition: Optional[str] = None,
    ) -> str:
        """
        Get model for agent with task/competition overrides.

        Priority:
        1. Competition-specific agent override
        2. Task-specific override
        3. Agent default
        4. Global default (qwen3:14b)

        Args:
            agent: Agent name (reader, planner, developer, reviewer, summarizer)
            task_type: Task type (debugging, feature_engineering, etc.)
            competition: Competition slug

        Returns:
            Model name (e.g., "qwen3:14b")
        """
        # Competition-specific agent override
        if competition and competition in self.competition_models:
            comp_models = self.competition_models[competition]
            if agent in comp_models:
                log.info(f"Using competition-specific model for {agent}: {comp_models[agent]}")
                return comp_models[agent]

        # Task-specific override
        if task_type and task_type in self.task_models:
            log.info(f"Using task-specific model for {task_type}: {self.task_models[task_type]}")
            return self.task_models[task_type]

        # Agent default
        if agent in self.models:
            return self.models[agent]

        # Global default
        log.warning(f"No model configured for agent {agent}, using default")
        return "qwen3:14b"

    def set_model(self, agent: str, model: str) -> None:
        """Set model for an agent."""
        self.models[agent] = model
        log.info(f"Set {agent} model to {model}")

    def set_task_model(self, task_type: str, model: str) -> None:
        """Set model for a task type."""
        self.task_models[task_type] = model
        log.info(f"Set {task_type} task model to {model}")

    def set_competition_model(self, competition: str, agent: str, model: str) -> None:
        """Set model for specific agent in specific competition."""
        if competition not in self.competition_models:
            self.competition_models[competition] = {}
        self.competition_models[competition][agent] = model
        log.info(f"Set {competition}/{agent} model to {model}")

    def save(self) -> None:
        """Save config to JSON file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "models": self.models,
            "task_models": self.task_models,
            "competition_models": self.competition_models,
        }

        with self.config_path.open("w") as f:
            json.dump(config, f, indent=2)

        log.info(f"Saved model config to {self.config_path}")

    def load(self) -> None:
        """Load config from JSON file."""
        if not self.config_path.exists():
            return

        with self.config_path.open() as f:
            config = json.load(f)

        self.models.update(config.get("models", {}))
        self.task_models.update(config.get("task_models", {}))
        self.competition_models.update(config.get("competition_models", {}))

        log.info(f"Loaded model config from {self.config_path}")

    def _load_from_env(self) -> None:
        """Load overrides from environment variables."""
        # Format: AGENT_MODEL_READER=qwen3:14b
        for agent in ["reader", "planner", "developer", "reviewer", "summarizer"]:
            env_var = f"AGENT_MODEL_{agent.upper()}"
            if env_var in os.environ:
                model = os.environ[env_var]
                self.models[agent] = model
                log.info(f"Loaded {agent} model from env: {model}")

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available Ollama models.

        Returns:
            Dict with model info from Ollama
        """
        try:
            from core.ollama_client import health

            health_check = health()
            if health_check.get("ok"):
                models = health_check.get("models", [])
                return {
                    "available": True,
                    "models": [m.get("name") for m in models],
                    "details": models,
                }
            else:
                return {
                    "available": False,
                    "error": health_check.get("error"),
                }
        except Exception as e:
            log.error(f"Failed to get available models: {e}")
            return {
                "available": False,
                "error": str(e),
            }

    def recommend_model(
        self,
        task_description: str,
        competition_type: Optional[str] = None,
    ) -> str:
        """
        Recommend best model for a task (future: use LLM to decide).

        Args:
            task_description: Description of the task
            competition_type: Type of competition (nlp, cv, tabular, time_series)

        Returns:
            Recommended model name
        """
        # Simple rule-based for now
        # Future: query LLM or use benchmark data

        desc_lower = task_description.lower()

        # Code-heavy tasks
        if any(word in desc_lower for word in ["debug", "implement", "code", "function"]):
            return "qwen3:14b"  # Could use deepseek-coder if available

        # Planning/strategy tasks
        if any(word in desc_lower for word in ["plan", "strategy", "approach", "design"]):
            return "qwen3:14b"

        # Analysis tasks
        if any(word in desc_lower for word in ["analyze", "review", "check", "validate"]):
            return "qwen3:14b"

        # Default
        return "qwen3:14b"


# Global instance
_model_config: Optional[ModelConfig] = None


def get_model_config() -> ModelConfig:
    """Get global model config instance."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config


def get_agent_model(
    agent: str,
    task_type: Optional[str] = None,
    competition: Optional[str] = None,
) -> str:
    """Quick helper to get model for agent."""
    config = get_model_config()
    return config.get_model(agent, task_type, competition)


if __name__ == "__main__":
    # Test model config
    config = ModelConfig()

    print("Available models:")
    available = config.get_available_models()
    if available["available"]:
        for model in available["models"]:
            print(f"  - {model}")

    print("\nCurrent agent models:")
    for agent in ["reader", "planner", "developer", "reviewer", "summarizer"]:
        model = config.get_model(agent)
        print(f"  {agent}: {model}")

    print("\nRecommendations:")
    print(f"  Debugging: {config.recommend_model('debug complex error')}")
    print(f"  Planning: {config.recommend_model('plan feature engineering strategy')}")
    print(f"  Code gen: {config.recommend_model('implement lag features')}")
