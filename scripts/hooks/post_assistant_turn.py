#!/usr/bin/env python3
"""Post-assistant turn hook - Log to Langfuse for observability.

Logs each assistant response to Langfuse with:
- Estimated token usage & cost
- Response time/latency
- Tool calls made
- Model used
- Conversation context

This enables cost tracking and usage monitoring for Claude Code.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Load environment
env_path = REPO_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# Import after env is loaded
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://docker:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")

# Claude pricing (as of 2025)
# https://www.anthropic.com/pricing
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},  # per 1M tokens
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5": {"input": 0.25, "output": 1.25},
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def extract_model(event: Dict[str, Any]) -> str:
    """Extract model name from event metadata."""
    # Default to sonnet if not specified
    return event.get("model", "claude-sonnet-4-5")


def extract_tool_calls(content: list) -> list[Dict[str, Any]]:
    """Extract tool calls from assistant message content."""
    tools = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tools.append({
                "name": block.get("name"),
                "id": block.get("id"),
                "input_preview": str(block.get("input", {}))[:100]
            })
    return tools


def extract_text(content: list) -> str:
    """Extract text content from message."""
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts)


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost in USD."""
    pricing = PRICING.get(model, PRICING["claude-sonnet-4-5"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def log_to_langfuse(event: Dict[str, Any]) -> bool:
    """Log assistant turn to Langfuse."""
    if not HAS_REQUESTS or not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return False

    try:
        content = event.get("content", [])
        if isinstance(content, str):
            text = content
            tools = []
        else:
            text = extract_text(content)
            tools = extract_tool_calls(content)

        model = extract_model(event)

        # Estimate tokens (very rough - actual may differ)
        output_tokens = estimate_tokens(text)
        # Assume context size based on conversation length
        input_tokens = event.get("estimated_input_tokens", output_tokens * 2)

        cost = calculate_cost(input_tokens, output_tokens, model)

        trace_id = f"claude-{event.get('turn_id', int(datetime.now().timestamp()))}"

        # Build trace
        batch = [{
            "id": trace_id,
            "type": "trace-create",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "body": {
                "id": trace_id,
                "name": f"Assistant Turn - {model}",
                "userId": "kaggle-agent",
                "sessionId": event.get("conversation_id", "unknown"),
                "metadata": {
                    "model": model,
                    "tool_count": len(tools),
                    "tools_used": [t["name"] for t in tools],
                    "response_length": len(text),
                },
                "input": {
                    "estimated_input_tokens": input_tokens,
                    "context_size": "estimated"
                },
                "output": {
                    "text_preview": text[:500] if text else "[tool calls only]",
                    "output_tokens": output_tokens,
                    "tools": tools,
                },
                "usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                    "unit": "TOKENS"
                },
                "metrics": {
                    "cost_usd": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            }
        }]

        response = requests.post(
            f"{LANGFUSE_HOST}/api/public/ingestion",
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            json={"batch": batch},
            timeout=5
        )

        return response.status_code == 207

    except Exception as e:
        # Silently fail - don't break the agent
        return False


def main() -> int:
    """Hook entry point."""
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0

    # Log to Langfuse
    log_to_langfuse(event)

    return 0


if __name__ == "__main__":
    sys.exit(main())
