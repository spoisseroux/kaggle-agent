#!/usr/bin/env python3
"""Telegram Delivery Enforcer + Langfuse Logger

Monitors the conversation and ensures:
1. All assistant responses are sent to Telegram (if missing notify.py call)
2. All assistant turns are logged to Langfuse for cost/usage tracking

How it works:
1. Watches the conversation JSONL file
2. For each assistant turn:
   - Checks if notify.py was called, auto-sends to Telegram if not
   - Logs to Langfuse with token estimates and cost tracking
3. Maintains a state file to avoid duplicate processing

Usage:
    python scripts/telegram_enforcer.py --watch

Or as a one-time check:
    python scripts/telegram_enforcer.py --check-last
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

STATE_FILE = REPO_ROOT / ".telegram_enforcer_state.json"
CONVERSATION_DIR = Path.home() / ".claude" / "projects" / "-home-keehar-kaggle-agent"

# Langfuse config
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://docker:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")

# Claude pricing (per 1M tokens)
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5": {"input": 0.25, "output": 1.25},
}


def load_state() -> dict[str, Any]:
    """Load the last processed message index."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_checked_index": 0, "last_checked_conversation": None}


def save_state(state: dict[str, Any]) -> None:
    """Save the last processed message index."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_latest_conversation() -> Path | None:
    """Find the most recent conversation JSONL file."""
    if not CONVERSATION_DIR.exists():
        return None

    # Conversation files are UUIDs.jsonl directly in the project dir
    conversations = list(CONVERSATION_DIR.glob("*.jsonl"))
    if not conversations:
        return None

    # Sort by modification time, get most recent
    return max(conversations, key=lambda p: p.stat().st_mtime)


def extract_text_from_message(msg: dict[str, Any]) -> str:
    """Extract text content from an assistant message."""
    content = msg.get("content", [])
    if isinstance(content, str):
        return content

    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))

    return "\n\n".join(texts).strip()


def has_notify_call(msg: dict[str, Any]) -> bool:
    """Check if this turn includes a notify.py call."""
    content = msg.get("content", [])
    if isinstance(content, str):
        return False

    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_input = block.get("input", {})
            command = tool_input.get("command", "")
            if "notify.py" in command:
                return True

    return False


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def extract_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from message."""
    tools = []
    content = msg.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append({
                    "name": block.get("name"),
                    "id": block.get("id"),
                })
    return tools


def log_to_langfuse(msg: dict[str, Any], text: str, conv_id: str) -> bool:
    """Log assistant turn to Langfuse."""
    if not HAS_REQUESTS or not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return False

    try:
        tools = extract_tool_calls(msg)
        output_tokens = estimate_tokens(text)
        input_tokens = output_tokens * 2  # Rough estimate

        # Assume Sonnet 4.5
        model = "claude-sonnet-4-5"
        pricing = PRICING[model]
        cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]

        trace_id = f"assistant-{conv_id}-{int(time.time() * 1000)}"

        batch = [{
            "id": trace_id,
            "type": "trace-create",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "body": {
                "id": trace_id,
                "name": f"Assistant Turn",
                "userId": "kaggle-agent",
                "sessionId": conv_id,
                "metadata": {
                    "model": model,
                    "tool_count": len(tools),
                    "tools_used": [t["name"] for t in tools],
                },
                "output": {
                    "text_preview": text[:500] if text else "[tool calls only]",
                    "full_length": len(text),
                },
                "usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                    "unit": "TOKENS"
                },
            }
        }]

        # Add generation with cost
        generation_id = f"gen-{trace_id}"
        batch.append({
            "id": generation_id,
            "type": "generation-create",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "body": {
                "id": generation_id,
                "traceId": trace_id,
                "name": model,
                "model": model,
                "modelParameters": {},
                "input": {"estimated": True},
                "output": text[:1000],
                "usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                },
                "metadata": {
                    "cost_usd": round(cost, 6),
                    "cost_breakdown": {
                        "input": round((input_tokens / 1_000_000) * pricing["input"], 6),
                        "output": round((output_tokens / 1_000_000) * pricing["output"], 6),
                    }
                }
            }
        })

        response = requests.post(
            f"{LANGFUSE_HOST}/api/public/ingestion",
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            json={"batch": batch},
            timeout=5
        )

        return response.status_code == 207

    except Exception as e:
        print(f"Langfuse logging failed: {e}", file=sys.stderr)
        return False


def send_to_telegram(text: str) -> bool:
    """Send text to Telegram via notify.py."""
    if not text or len(text) < 10:  # Don't send very short messages
        return False

    try:
        # Prepend warning that this is an auto-send
        full_text = f"🔔 Auto-forwarded (no notify.py call):\n\n{text}"
        result = subprocess.run(
            ["python", str(REPO_ROOT / "core" / "notify.py"), full_text],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to send: {e}", file=sys.stderr)
        return False


def check_conversation(conv_file: Path, state: dict[str, Any]) -> int:
    """Check a conversation file for missing notify.py calls and log to Langfuse.

    Returns the number of messages auto-sent to Telegram.
    """
    if not conv_file.exists():
        return 0

    lines = conv_file.read_text().strip().split("\n")
    messages = [json.loads(line) for line in lines if line.strip()]

    last_checked = state.get("last_checked_index", 0)
    auto_sent = 0

    # Extract conversation ID from filename
    conv_id = conv_file.stem

    # Process only new messages since last check
    processed_count = 0
    assistant_count = 0
    for i, msg in enumerate(messages[last_checked:], start=last_checked):
        processed_count += 1
        if msg.get("role") != "assistant":
            continue

        assistant_count += 1
        # Extract text content
        text = extract_text_from_message(msg)

        # Log ALL assistant turns to Langfuse (even tool-use-only for complete tracking)
        log_result = log_to_langfuse(msg, text, conv_id)
        if log_result:
            if text:
                print(f"[{i}] ✓ Logged to Langfuse ({len(text)} chars text)", flush=True)
            else:
                print(f"[{i}] ✓ Logged to Langfuse (tool-use only)", flush=True)
        else:
            print(f"[{i}] ✗ Langfuse logging failed", flush=True)

        # Skip Telegram delivery if no text (tool-use-only messages)
        if not text:
            continue

        # Check if notify.py was called in this turn
        if has_notify_call(msg):
            continue

        # No notify.py call found - auto-send to Telegram
        print(f"[{i}] Missing notify.py call, auto-sending to Telegram", flush=True)
        if send_to_telegram(text):
            auto_sent += 1
            print(f"  ✓ Sent {len(text)} chars", flush=True)
        else:
            print(f"  ✗ Failed to send", flush=True)

    if processed_count > 0:
        print(f"Processed {processed_count} messages, {assistant_count} assistant turns", flush=True)

    # Update state
    state["last_checked_index"] = len(messages)
    state["last_checked_conversation"] = str(conv_file)
    save_state(state)

    return auto_sent


def watch_mode() -> None:
    """Continuously watch for new messages and enforce Telegram delivery."""
    print("Telegram Enforcer: watching for missing notify.py calls...")
    print("Press Ctrl+C to stop")

    state = load_state()

    try:
        while True:
            conv_file = get_latest_conversation()
            if not conv_file:
                time.sleep(5)
                continue

            # Check if we're on a new conversation
            if state.get("last_checked_conversation") != str(conv_file):
                print(f"New conversation detected: {conv_file.parent.name}")
                state["last_checked_index"] = 0
                state["last_checked_conversation"] = str(conv_file)

            auto_sent = check_conversation(conv_file, state)
            if auto_sent > 0:
                print(f"Auto-sent {auto_sent} message(s) to Telegram")

            time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        print("\nStopping telegram enforcer")


def check_last() -> None:
    """One-time check of the most recent assistant message."""
    conv_file = get_latest_conversation()
    if not conv_file:
        print("No conversation found", file=sys.stderr)
        return

    state = load_state()
    auto_sent = check_conversation(conv_file, state)

    if auto_sent > 0:
        print(f"Auto-sent {auto_sent} message(s) to Telegram")
    else:
        print("All recent messages were properly sent to Telegram ✓")


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram Delivery Enforcer")
    parser.add_argument("--watch", action="store_true", help="Watch mode (continuous)")
    parser.add_argument("--check-last", action="store_true", help="Check last message only")

    args = parser.parse_args()

    if args.watch:
        watch_mode()
    elif args.check_last:
        check_last()
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
