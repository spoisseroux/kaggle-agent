"""Fire-and-forget notification.

Sends a Telegram message and records it on the local message bus so it
shows up in the web UI chat. Errors are swallowed — never block the agent.

Usage:
    python core/notify.py "Training complete. CV: 0.8721"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


def notify(text: str) -> None:
    _load_env()
    sent = send_telegram(text)
    try:
        from core.message_bus import post_agent_message
        post_agent_message(text, source="agent",
                           status="delivered" if sent else "pending")
    except Exception:
        pass


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python core/notify.py <message>", file=sys.stderr)
        return 2
    notify(" ".join(sys.argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
