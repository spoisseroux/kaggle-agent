"""Wrapper around the OpenRouter API for code generation.

Uses OpenRouter's programming models for reliable code generation where
Ollama struggles with syntax correctness.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Load .env file into environment."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        key = k.strip()
        val = v.strip()
        os.environ.setdefault(key, val)


_load_env()  # Load on import

DEFAULT_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro-20260423")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def health() -> dict:
    """Check if OpenRouter API is accessible."""
    if not API_KEY:
        return {"ok": False, "error": "OPENROUTER_API_KEY not set"}
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://github.com/spoisseroux/kaggle-agent",
            "X-Title": "Kaggle Agent",
        }
        r = httpx.get(f"{DEFAULT_URL}/models", headers=headers, timeout=5)
        return {"ok": r.status_code == 200, "models_available": r.status_code == 200}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout_s: float = 180.0,
) -> str:
    """One-shot completion. Returns the assistant message text."""
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/spoisseroux/kaggle-agent",
        "X-Title": "Kaggle Agent",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = httpx.post(f"{DEFAULT_URL}/chat/completions", json=payload, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    try:
        data = r.json()
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response (status {r.status_code}): {r.text[:500]}") from e
    return data["choices"][0]["message"]["content"]


def stream(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout_s: float = 600.0,
) -> Iterator[str]:
    """Streaming completion. Yields chunks of the response."""
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://github.com/spoisseroux/kaggle-agent",
        "X-Title": "Kaggle Agent",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    with httpx.stream("POST", f"{DEFAULT_URL}/chat/completions", json=payload, headers=headers, timeout=timeout_s) as r:
        r.raise_for_status()
        import json
        for line in r.iter_lines():
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                obj = json.loads(line)
            except Exception:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content")
            if chunk:
                yield chunk


if __name__ == "__main__":
    print(health())
