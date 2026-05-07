"""Wrapper around the Ollama HTTP API for the kaggle agent.

The agent uses the local model (qwen3:14b) for boilerplate tasks; the
think=True flag toggles the Qwen3 thinking-mode prefix so the model
returns deeper reasoning for complex feature pipelines or debugging.
"""
from __future__ import annotations

import os
from typing import Iterator

import httpx

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")


def health() -> dict:
    try:
        r = httpx.get(f"{DEFAULT_URL}/api/tags", timeout=3)
        return {"ok": r.status_code == 200, "models": r.json().get("models", [])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    think: bool = False,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout_s: float = 120.0,
) -> str:
    """One-shot completion. Returns the assistant message text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user = prompt if not think else f"/think\n\n{prompt}"
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = httpx.post(f"{DEFAULT_URL}/api/chat", json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def stream(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    think: bool = False,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout_s: float = 600.0,
) -> Iterator[str]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user = prompt if not think else f"/think\n\n{prompt}"
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with httpx.stream("POST", f"{DEFAULT_URL}/api/chat", json=payload, timeout=timeout_s) as r:
        r.raise_for_status()
        import json
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            chunk = obj.get("message", {}).get("content")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


if __name__ == "__main__":
    print(health())
