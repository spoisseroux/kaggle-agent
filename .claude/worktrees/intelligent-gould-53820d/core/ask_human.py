"""Blocking ask-human prompt.

Sends a question to Telegram (and the chat bus), then waits for a human
reply via the message bus. The human can reply from either Telegram or the
web UI; both write to the same SQLite-backed bus.

Usage:
    answer=$(python core/ask_human.py "Submit lgbm_v7? CV: 0.8821")
    echo "human said: $answer"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.message_bus import create_ask, post_agent_message, wait_for_reply  # noqa: E402
from core.notify import _load_env, send_telegram  # noqa: E402


def ask(question: str, timeout_s: int | None = None) -> str | None:
    _load_env()
    ask_id = create_ask(question)
    prompt = f"❓ {question}\n\nReply to this message to answer."
    sent = send_telegram(prompt)
    post_agent_message(question, source="agent", ask_id=ask_id,
                       status="delivered" if sent else "pending")
    return wait_for_reply(ask_id, timeout_s=timeout_s)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python core/ask_human.py <question> [--timeout SECONDS]", file=sys.stderr)
        return 2
    args = sys.argv[1:]
    timeout: int | None = None
    if "--timeout" in args:
        i = args.index("--timeout")
        timeout = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    question = " ".join(args)
    reply = ask(question, timeout_s=timeout)
    if reply is None:
        print("", end="")
        return 1
    print(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
