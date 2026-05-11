#!/usr/bin/env python3
"""Telegram Delivery Enforcer

Monitors the conversation and ensures all assistant responses are sent to Telegram.

How it works:
1. Watches the conversation JSONL file
2. For each assistant turn, checks if notify.py was called in that turn
3. If not, extracts text output and auto-sends via notify.py
4. Maintains a state file to avoid duplicate sends

Usage:
    python scripts/telegram_enforcer.py --watch

Or as a one-time check:
    python scripts/telegram_enforcer.py --check-last
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STATE_FILE = REPO_ROOT / ".telegram_enforcer_state.json"
CONVERSATION_DIR = Path.home() / ".claude" / "projects" / "-home-keehar-kaggle-agent"


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
    """Check a conversation file for missing notify.py calls.

    Returns the number of messages auto-sent.
    """
    if not conv_file.exists():
        return 0

    lines = conv_file.read_text().strip().split("\n")
    messages = [json.loads(line) for line in lines if line.strip()]

    last_checked = state.get("last_checked_index", 0)
    auto_sent = 0

    # Process only new messages since last check
    for i, msg in enumerate(messages[last_checked:], start=last_checked):
        if msg.get("role") != "assistant":
            continue

        # Extract text content
        text = extract_text_from_message(msg)
        if not text:
            continue

        # Check if notify.py was called in this turn
        if has_notify_call(msg):
            continue

        # No notify.py call found - auto-send
        print(f"[{i}] Missing notify.py call, auto-sending to Telegram")
        if send_to_telegram(text):
            auto_sent += 1
            print(f"  ✓ Sent {len(text)} chars")
        else:
            print(f"  ✗ Failed to send")

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
