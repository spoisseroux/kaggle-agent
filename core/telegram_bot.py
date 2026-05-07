"""Telegram bot — long polls Telegram and forwards messages onto the bus.

Runs as a systemd service. Reverse direction (agent->user) is handled by
notify.py / ask_human.py; this file only ingests inbound human messages.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env  # noqa: E402

_load_env()

from telegram import Update  # noqa: E402
from telegram.ext import Application, MessageHandler, filters, ContextTypes  # noqa: E402

from core.message_bus import init_db, post_human_message  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s telegram_bot %(levelname)s %(message)s",
)
log = logging.getLogger("telegram_bot")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = str(update.message.chat_id)
    expected = os.environ.get("TELEGRAM_CHAT_ID", "")
    if expected and chat_id != expected:
        log.warning("ignoring message from unauthorised chat_id=%s", chat_id)
        return
    text = update.message.text.strip()
    if not text:
        return
    msg_id = post_human_message(text, source="telegram")
    log.info("ingested telegram message %s (%d chars)", msg_id, len(text))


def main() -> int:
    init_db()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set")
        return 2
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("starting Telegram polling")
    app.run_polling(allowed_updates=["message"], drop_pending_updates=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
