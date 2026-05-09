"""Ollama model integration for DeepEval.

Uses local Ollama instance with qwen3:14b on RTX 5070.
"""
from deepeval.models import DeepEvalBaseLLM
import requests
import json
from typing import Optional


class OllamaModel(DeepEvalBaseLLM):
    """Custom Ollama model for DeepEval evaluations."""

    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        super().__init__(model)

    def load_model(self):
        """Ollama doesn't need to load - always available via API."""
        return self.model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from Ollama."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Async generate - just call sync version for now."""
        return self.generate(prompt, **kwargs)

    def get_model_name(self) -> str:
        """Return model name for logging."""
        return f"ollama:{self.model}"
