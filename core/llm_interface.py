"""LLM interface abstraction - switch between Claude Code and Claude API.

This module provides a unified interface for LLM calls that can route to:
1. Claude Code (current) - via ask_human.py for escalation
2. Claude API (future) - via Anthropic SDK
3. Ollama (local) - via ollama_client.py

Usage:
    from core.llm_interface import get_llm_response, LLMMode

    # Local Ollama (default for agents)
    response = get_llm_response(prompt, mode=LLMMode.LOCAL)

    # Escalate to Claude (current: Claude Code, future: API)
    response = get_llm_response(prompt, mode=LLMMode.CLAUDE, reason="Complex debugging needed")
"""
from __future__ import annotations

import os
import json
import subprocess
from enum import Enum
from typing import Optional, Dict, Any
import logging

from core.ollama_client import generate as ollama_generate
from core.ask_human import ask_human

log = logging.getLogger(__name__)


class LLMMode(Enum):
    """LLM routing modes."""
    LOCAL = "local"      # Ollama (qwen3:14b)
    CLAUDE = "claude"    # Claude Code or API (configurable)


class LLMBackend(Enum):
    """Claude backend implementation."""
    CLAUDE_CODE = "claude_code"  # Current: escalate via ask_human.py
    CLAUDE_API = "claude_api"    # Future: direct API calls


# Configuration
CLAUDE_BACKEND = os.environ.get("CLAUDE_BACKEND", "claude_code")
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")


def get_llm_response(
    prompt: str,
    *,
    mode: LLMMode = LLMMode.LOCAL,
    system: Optional[str] = None,
    think: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    reason: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """
    Get LLM response with automatic routing.

    Args:
        prompt: The prompt text
        mode: LLMMode.LOCAL (Ollama) or LLMMode.CLAUDE (escalate)
        system: System message (optional)
        think: Enable thinking mode (Ollama only)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        reason: Why escalating to Claude (logged for debugging)
        timeout: Timeout in seconds

    Returns:
        LLM response text
    """
    if mode == LLMMode.LOCAL:
        return _call_ollama(prompt, system, think, temperature, max_tokens, timeout)
    elif mode == LLMMode.CLAUDE:
        return _call_claude(prompt, system, temperature, max_tokens, reason)
    else:
        raise ValueError(f"Unknown LLM mode: {mode}")


def _call_ollama(
    prompt: str,
    system: Optional[str],
    think: bool,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    """Call local Ollama (qwen3:14b)."""
    try:
        return ollama_generate(
            prompt,
            system=system,
            think=think,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout,
        )
    except Exception as e:
        log.error(f"Ollama call failed: {e}")
        raise


def _call_claude(
    prompt: str,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
    reason: Optional[str],
) -> str:
    """
    Call Claude (backend configured via CLAUDE_BACKEND env var).

    Backends:
    - claude_code: Escalate to Claude Code via ask_human.py (current)
    - claude_api: Direct API call via Anthropic SDK (future)
    """
    backend = LLMBackend(CLAUDE_BACKEND)

    if backend == LLMBackend.CLAUDE_CODE:
        return _escalate_to_claude_code(prompt, system, reason)
    elif backend == LLMBackend.CLAUDE_API:
        return _call_claude_api(prompt, system, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown Claude backend: {backend}")


def _escalate_to_claude_code(
    prompt: str,
    system: Optional[str],
    reason: Optional[str],
) -> str:
    """
    Escalate to Claude Code via ask_human.py.

    The agent asks the user (Claude Code top-level orchestrator) for help.
    Claude Code sees the request and provides the response.
    """
    escalation_message = f"""
🧠 Agent needs Claude reasoning:

Reason: {reason or "Complex task requiring strategic thinking"}

System context: {system or "None"}

Task:
{prompt}

Please provide your response below.
"""

    log.info(f"Escalating to Claude Code: {reason}")

    # ask_human sends to Telegram and waits for reply
    # In this case, the "human" is actually Claude Code orchestrator
    response = ask_human(escalation_message)

    return response


def _call_claude_api(
    prompt: str,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Call Claude API directly (future implementation).

    Requires ANTHROPIC_API_KEY env var.
    """
    if not CLAUDE_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set - cannot use claude_api backend")

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed - run: pip install anthropic")

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "",
        messages=messages,
    )

    return response.content[0].text


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return len(text) // 4


def should_escalate(
    agent_name: str,
    context: str,
    attempt: int = 0,
    error: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Decide if task should escalate from Ollama to Claude.

    Returns:
        (should_escalate, reason)
    """
    # Developer: escalate after 3 debugging attempts
    if agent_name == "Developer" and attempt >= 3:
        return True, f"Debugging failed after {attempt} attempts"

    # Planner: escalate if context mentions contradictions
    if agent_name == "Planner":
        if any(word in context.lower() for word in ["contradiction", "unclear", "ambiguous", "conflicting"]):
            return True, "Plan has contradictions or ambiguity"

    # Any agent: explicit escalation request
    if "REQUEST_CLAUDE_REASONING" in context:
        return True, "Explicit escalation requested"

    # Any agent: complex error that mentions "unclear root cause"
    if error and "unclear" in error.lower():
        return True, "Unclear error - needs strategic debugging"

    return False, None


# Quick helpers
def ask_ollama(prompt: str, *, system: str | None = None, think: bool = False) -> str:
    """Quick helper for Ollama calls."""
    return get_llm_response(prompt, mode=LLMMode.LOCAL, system=system, think=think)


def ask_claude(prompt: str, *, system: str | None = None, reason: str | None = None) -> str:
    """Quick helper for Claude escalation."""
    return get_llm_response(prompt, mode=LLMMode.CLAUDE, system=system, reason=reason)


if __name__ == "__main__":
    # Test local mode
    print("Testing Ollama (local):")
    response = ask_ollama("What is 2+2?", system="You are a helpful math assistant.")
    print(f"Response: {response}\n")

    # Test escalation detection
    print("Testing escalation logic:")
    tests = [
        ("Developer", "Simple syntax error", 1, None),
        ("Developer", "Complex bug", 4, None),
        ("Planner", "This plan has contradictions", 0, None),
        ("Planner", "Clear and straightforward", 0, None),
    ]

    for agent, context, attempt, error in tests:
        should, reason = should_escalate(agent, context, attempt, error)
        print(f"{agent} (attempt {attempt}): {should} - {reason}")
