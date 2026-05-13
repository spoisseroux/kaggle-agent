"""Telegram bot — long polls Telegram and forwards messages onto the bus.

Runs as a systemd service. Reverse direction (agent->user) is handled by
notify.py / ask_human.py; this file only ingests inbound human messages.

Slash commands (handled here directly, not forwarded to the agent):
    /help          list commands
    /status        agent state, active competition
    /config        full system diagnostic — model, services, account
    /model         show current model or change it (e.g. /model opus)
    /restart       restart the agent (picks up settings changes)
    /pause /resume /stop  agent state controls
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env  # noqa: E402

_load_env()

import httpx  # noqa: E402
from telegram import Update  # noqa: E402
from telegram.ext import (  # noqa: E402
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
)

from core.message_bus import init_db, post_human_message, wake_agent  # noqa: E402
from core.notify import send_telegram  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s telegram_bot %(levelname)s %(message)s",
)
log = logging.getLogger("telegram_bot")

API_BASE = f"http://localhost:{os.environ.get('API_PORT', '8765')}"

# Short aliases → full model IDs for /model command
MODEL_ALIASES = {
    "opus": "claude-opus-4-7",
    "opus-4-7": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-5",
    "sonnet-4-5": "claude-sonnet-4-5",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


# ── auth gate ───────────────────────────────────────────────────────────────

def _is_authorized(update: Update) -> bool:
    if not update.message:
        return False
    chat_id = str(update.message.chat_id)
    expected = os.environ.get("TELEGRAM_CHAT_ID", "")
    return not expected or chat_id == expected


# ── service health checks ───────────────────────────────────────────────────

def _check_postgres() -> tuple[bool, str]:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        return False, "no DSN configured"
    try:
        import psycopg2
        with psycopg2.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                ver = cur.fetchone()[0].split(",")[0]
        # Extract host from DSN for the status line
        host = dsn.split("@")[1].split("/")[0] if "@" in dsn else "?"
        return True, f"{host} — {ver}"
    except Exception as e:
        return False, f"FAIL: {str(e)[:80]}"


def _check_qdrant() -> tuple[bool, str]:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        r = httpx.get(f"{url}/collections", timeout=3)
        if r.status_code == 200:
            data = r.json().get("result", {}).get("collections", [])
            names = [c["name"] for c in data]
            return True, f"{url} — {len(names)} collections ({', '.join(names[:5])})"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"FAIL: {str(e)[:80]}"


def _check_mlflow() -> tuple[bool, str]:
    url = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        r = httpx.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            return True, url
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"FAIL: {str(e)[:80]}"


def _check_ollama() -> tuple[bool, str]:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        r = httpx.get(f"{url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, f"{url} — {len(models)} models"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"FAIL: {str(e)[:80]}"


def _check_api() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{API_BASE}/system/state", timeout=3)
        if r.status_code == 200:
            data = r.json()
            return True, f"state={data.get('state', '?')} active={data.get('active_competition', '?')}"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"FAIL: {str(e)[:80]}"


def _get_current_model() -> str:
    try:
        r = httpx.get(f"{API_BASE}/system/model", timeout=3)
        if r.status_code == 200:
            return r.json().get("model", "unknown")
    except Exception:
        pass
    # Fallback: read the settings file directly
    try:
        p = REPO_ROOT / ".claude" / "kaggle_settings.json"
        return json.loads(p.read_text()).get("model", "unknown")
    except Exception:
        return "unknown"


# ── command handlers ────────────────────────────────────────────────────────

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    text = (
        "Available commands:\n\n"
        "/status   — agent state and active competition\n"
        "/config   — full system diagnostic (model, services, account)\n"
        "/model    — show current model, or /model opus|sonnet|haiku to change\n"
        "/restart  — restart the agent (picks up settings changes)\n"
        "/pause    — pause the agent\n"
        "/resume   — resume after pause\n"
        "/stop     — stop the agent\n\n"
        "Any other message goes straight to the agent."
    )
    await update.message.reply_text(text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    ok, detail = _check_api()
    emoji = "🟢" if ok else "🔴"
    await update.message.reply_text(f"{emoji} {detail}")


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    lines = ["📊 System diagnostic\n"]

    # Model
    model = _get_current_model()
    lines.append(f"Model: {model}\n")

    # Services
    lines.append("Services:")
    for name, fn in [
        ("API", _check_api),
        ("Postgres", _check_postgres),
        ("Qdrant", _check_qdrant),
        ("MLflow", _check_mlflow),
        ("Ollama", _check_ollama),
    ]:
        ok, detail = fn()
        mark = "✅" if ok else "❌"
        lines.append(f"  {mark} {name}: {detail}")

    # Kaggle account
    lines.append("")
    user = os.environ.get("KAGGLE_USERNAME", "(unset)")
    key = os.environ.get("KAGGLE_KEY", "")
    key_prefix = key[:8] + "..." if key else "(unset)"
    lines.append(f"Kaggle: {user} (key: {key_prefix})")

    # Active competition (from registry)
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        lines.append(f"Active competition: {reg.get('active', 'none')}")
    except Exception:
        pass

    await update.message.reply_text("\n".join(lines))


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    args = context.args or []

    if not args:
        current = _get_current_model()
        text = (
            f"Current model: {current}\n\n"
            "To change:\n"
            "  /model opus     — claude-opus-4-7\n"
            "  /model sonnet   — claude-sonnet-4-5\n"
            "  /model haiku    — claude-haiku-4-5\n\n"
            "Changing the model auto-restarts the agent."
        )
        await update.message.reply_text(text)
        return

    alias = args[0].lower().strip()
    model_id = MODEL_ALIASES.get(alias, alias)
    try:
        r = httpx.post(f"{API_BASE}/system/model",
                       json={"model": model_id}, timeout=8)
        if r.status_code != 200:
            await update.message.reply_text(
                f"❌ Set failed: HTTP {r.status_code} — {r.text[:150]}"
            )
            return
        data = r.json()
        if data.get("restarted"):
            await update.message.reply_text(
                f"✅ Model set to {data['model']}\n"
                f"Agent is restarting now — back online in ~10s."
            )
        else:
            await update.message.reply_text(
                f"✅ Model already set to {data['model']} (no change)"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    try:
        r = httpx.post(f"{API_BASE}/system/restart", timeout=8)
        if r.status_code == 200:
            await update.message.reply_text(
                "🔄 Restart triggered — agent back online in ~10s."
            )
        else:
            await update.message.reply_text(
                f"❌ Restart failed: HTTP {r.status_code}"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


def make_simple_action(action: str):
    """Factory for /pause /resume /stop handlers."""
    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update):
            return
        try:
            r = httpx.post(f"{API_BASE}/system/{action}", timeout=8)
            if r.status_code == 200:
                emoji = {"pause": "⏸", "resume": "▶️", "stop": "⏹"}.get(action, "✅")
                await update.message.reply_text(f"{emoji} {action.capitalize()}d")
            else:
                await update.message.reply_text(
                    f"❌ {action} failed: HTTP {r.status_code}"
                )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    return handler


# ── regular text handler (forwards to agent) ────────────────────────────────

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not _is_authorized(update):
        log.warning("ignoring message from unauthorised chat_id=%s",
                    update.message.chat_id)
        return
    text = update.message.text.strip()
    if not text:
        return
    msg_id = post_human_message(text, source="telegram")
    log.info("ingested telegram message %s (%d chars)", msg_id, len(text))
    try:
        send_telegram("Got it, thinking...")
    except Exception:
        pass
    wake_agent(text)


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> int:
    init_db()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set")
        return 2
    app = Application.builder().token(token).build()

    # Slash commands handled locally — never forwarded to the agent
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("config", cmd_config))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("restart", cmd_restart))

    app.add_handler(CommandHandler("pause", make_simple_action("pause")))
    app.add_handler(CommandHandler("resume", make_simple_action("resume")))
    app.add_handler(CommandHandler("stop", make_simple_action("stop")))

    # Anything else: forward to the agent
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    log.info("starting Telegram polling with slash commands enabled")
    app.run_polling(allowed_updates=["message"], drop_pending_updates=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
