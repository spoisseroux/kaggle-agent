"""Fire-and-forget notification.

Sends a Telegram message and records it on the local message bus so it
shows up in the web UI chat. Errors are swallowed — never block the agent.

Long messages are automatically split at paragraph boundaries so they
stay within Telegram's 4096-character limit.

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

TELEGRAM_MAX_CHARS = 4000


def _load_env() -> None:
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
        # Force override for TELEGRAM_CHAT_ID to use .env value
        if key == "TELEGRAM_CHAT_ID":
            os.environ[key] = val
        else:
            os.environ.setdefault(key, val)


def _split_message(text: str) -> list[str]:
    """Split text into ≤TELEGRAM_MAX_CHARS chunks, breaking at paragraph boundaries."""
    if len(text) <= TELEGRAM_MAX_CHARS:
        return [text]
    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).lstrip() if current else para
        if len(candidate) <= TELEGRAM_MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Paragraph itself too long — hard-split by line
            if len(para) <= TELEGRAM_MAX_CHARS:
                current = para
            else:
                for i in range(0, len(para), TELEGRAM_MAX_CHARS):
                    chunks.append(para[i : i + TELEGRAM_MAX_CHARS])
                current = ""
    if current:
        chunks.append(current)
    return chunks or [text[:TELEGRAM_MAX_CHARS]]


def send_telegram(text: str) -> bool:
    _load_env()  # Ensure .env is loaded with correct TELEGRAM_CHAT_ID
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    all_ok = True
    for chunk in _split_message(text):
        try:
            r = httpx.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk,
                      "disable_web_page_preview": True},
                timeout=8,
            )
            if r.status_code != 200:
                all_ok = False
        except Exception:
            all_ok = False
    return all_ok


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
