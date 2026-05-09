"""Enhanced Telegram bot with group chat support.

Supports:
- Private 1-on-1 chats (original behavior)
- Group chats with multiple authorized users
- @mention requirement in groups
- User attribution (who asked what)
- Commands: /status, /help, /queue

Configuration via environment variables:
- TELEGRAM_BOT_TOKEN: Bot token
- TELEGRAM_CHAT_ID: Chat ID (private or group)
- TELEGRAM_AUTHORIZED_USERS: Comma-separated user IDs (optional)
- TELEGRAM_REQUIRE_MENTION: "true" to require @mention in groups (default: false)
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
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes  # noqa: E402

from core.message_bus import init_db, post_human_message, wake_agent  # noqa: E402
from core.notify import send_telegram  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s telegram_bot %(levelname)s %(message)s",
)
log = logging.getLogger("telegram_bot")


def is_authorized(update: Update) -> tuple[bool, str | None]:
    """
    Check if user and chat are authorized.

    Returns:
        (authorized: bool, reason: str | None)
    """
    if not update.message:
        return False, "No message"

    chat_id = str(update.message.chat_id)
    user_id = str(update.message.from_user.id) if update.message.from_user else None
    username = update.message.from_user.username if update.message.from_user else "Unknown"

    # Check chat ID
    expected_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if expected_chat and chat_id != expected_chat:
        return False, f"Unauthorized chat {chat_id} (expected {expected_chat})"

    # Check user whitelist (if configured)
    authorized_users = os.environ.get("TELEGRAM_AUTHORIZED_USERS", "")
    if authorized_users:
        authorized_list = [uid.strip() for uid in authorized_users.split(",")]
        if user_id not in authorized_list:
            return False, f"Unauthorized user {username} ({user_id})"

    return True, None


def should_respond(update: Update) -> tuple[bool, str | None]:
    """
    Check if bot should respond to this message.

    In groups, optionally requires @mention.

    Returns:
        (should_respond: bool, reason: str | None)
    """
    if not update.message:
        return False, "No message"

    chat_type = update.message.chat.type
    text = update.message.text or ""

    # In private chats, always respond
    if chat_type == "private":
        return True, None

    # In groups, check if @mention is required
    require_mention = os.environ.get("TELEGRAM_REQUIRE_MENTION", "false").lower() == "true"

    if require_mention:
        # Check if bot was mentioned
        bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "")
        if bot_username and f"@{bot_username}" not in text:
            return False, "Not mentioned (group requires @mention)"

    return True, None


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    if not update.message or not update.message.text:
        return

    # Log the chat_id for debugging
    chat_id = str(update.message.chat_id)
    chat_type = update.message.chat.type
    username = update.message.from_user.username if update.message.from_user else "Unknown"
    log.warning(f"MESSAGE RECEIVED: chat_id={chat_id}, type={chat_type}, from={username}, text={update.message.text[:50]}")

    # Check authorization
    authorized, auth_reason = is_authorized(update)
    if not authorized:
        log.warning(f"REJECTED: {auth_reason}")
        return

    # Check if we should respond
    should_reply, reply_reason = should_respond(update)
    if not should_reply:
        log.debug(f"Skipping message: {reply_reason}")
        return

    text = update.message.text.strip()
    if not text:
        return

    # Get user info for attribution
    user_info = ""
    if update.message.from_user:
        username = update.message.from_user.username or update.message.from_user.first_name or "Unknown"
        user_info = f"[{username}] "

    # Remove @mentions from text
    bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "")
    if bot_username:
        text = text.replace(f"@{bot_username}", "").strip()

    # Post to message bus with user attribution
    attributed_text = f"{user_info}{text}" if user_info else text
    msg_id = post_human_message(attributed_text, source="telegram")
    log.info("ingested telegram message %s from %s (%d chars)",
             msg_id, update.message.from_user.username if update.message.from_user else "unknown", len(text))

    # Immediate ack
    try:
        send_telegram("Got it, thinking...")
    except Exception:
        pass

    # Wake agent
    wake_agent(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    authorized, _ = is_authorized(update)
    if not authorized:
        return

    try:
        # Get current status
        from core import gpu_monitor
        from core import system_control

        gpu_status = gpu_monitor.get_gpu_status()
        system_status = system_control.get_status()

        status_msg = "📊 Kaggle Agent Status\n\n"
        status_msg += f"System: {system_status.get('state', 'unknown')}\n"
        status_msg += f"GPU: {gpu_status.get('utilization', 'N/A')}% | {gpu_status.get('memory_used', 'N/A')}/{gpu_status.get('memory_total', 'N/A')} MB\n"

        from core.message_bus import get_pending_instructions
        pending = len(get_pending_instructions(claim=False))
        if pending > 0:
            status_msg += f"\nPending messages: {pending}"

        await update.message.reply_text(status_msg)
    except Exception as e:
        log.error(f"Error in /status: {e}")
        await update.message.reply_text("Error fetching status")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    authorized, _ = is_authorized(update)
    if not authorized:
        return

    help_msg = """🤖 Kaggle Agent Commands

/status - Show current system status
/help - Show this help message
/queue - Show pending message queue

To interact:
- In private chat: just type your message
- In group chat: @mention me or send message (depends on config)

Examples:
- "What's the current best CV?"
- "Run feature engineering experiments"
- "Submit the XGBoost v10 model"
"""
    await update.message.reply_text(help_msg)


async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /queue command."""
    authorized, _ = is_authorized(update)
    if not authorized:
        return

    try:
        from core.message_bus import get_pending_instructions
        pending = len(get_pending_instructions(claim=False))
        msg = f"📬 Pending instructions: {pending}"
        await update.message.reply_text(msg)
    except Exception as e:
        log.error(f"Error in /queue: {e}")
        await update.message.reply_text("Error fetching queue")


def main() -> int:
    init_db()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set")
        return 2

    # Log configuration
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    authorized_users = os.environ.get("TELEGRAM_AUTHORIZED_USERS", "")
    require_mention = os.environ.get("TELEGRAM_REQUIRE_MENTION", "false")
    bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "")

    log.info("Starting Telegram bot")
    log.info(f"  Chat ID: {chat_id}")
    log.info(f"  Authorized users: {authorized_users if authorized_users else 'all'}")
    log.info(f"  Require @mention: {require_mention}")
    log.info(f"  Bot username: @{bot_username}")

    app = Application.builder().token(token).build()

    # Add handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("queue", cmd_queue))

    log.info("Starting polling")
    app.run_polling(allowed_updates=["message"], drop_pending_updates=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
