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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env  # noqa: E402

_load_env()

import httpx  # noqa: E402
from telegram import (  # noqa: E402
    BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update,
)
from telegram.ext import (  # noqa: E402
    Application, CallbackQueryHandler, CommandHandler, MessageHandler,
    filters, ContextTypes,
)

from core.message_bus import init_db, post_human_message, wake_agent  # noqa: E402
from core.models_catalog import (  # noqa: E402
    alert_if_new_models, get_available_models, resolve_alias,
)
from core.notify import send_telegram  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s telegram_bot %(levelname)s %(message)s",
)
log = logging.getLogger("telegram_bot")

API_BASE = f"http://localhost:{os.environ.get('API_PORT', '8765')}"
AGENT_SESSION = "kaggle-agent"
LOGIN_SESSION = "kaggle-login"
LOGIN_STATE_FILE = Path("/tmp/kaggle_agent_login_state.json")
LOGIN_TIMEOUT_S = 600  # 10 min for user to come back with the code


# ── agent state helpers ─────────────────────────────────────────────────────

def _tmux_session_exists(name: str) -> bool:
    try:
        r = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True, timeout=2,
        )
        return r.returncode == 0
    except Exception:
        return False


def _agent_is_running() -> bool:
    return _tmux_session_exists(AGENT_SESSION)


# ── login-flow state (file-backed, survives bot restart) ────────────────────

def _set_login_pending() -> None:
    LOGIN_STATE_FILE.write_text(json.dumps({
        "awaiting_code": True,
        "started_at": time.time(),
    }))


def _get_login_state() -> dict | None:
    if not LOGIN_STATE_FILE.exists():
        return None
    try:
        data = json.loads(LOGIN_STATE_FILE.read_text())
    except Exception:
        return None
    if time.time() - data.get("started_at", 0) > LOGIN_TIMEOUT_S:
        _clear_login_state()
        return None
    return data


def _clear_login_state() -> None:
    try:
        LOGIN_STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


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


def _check_claude_backend() -> tuple[str, str]:
    """Identify which Claude backend the agent uses.

    Returns (backend_label, detail).
    """
    creds_path = Path.home() / ".claude" / ".credentials.json"
    has_oauth = False
    oauth_detail = ""
    if creds_path.exists():
        try:
            data = json.loads(creds_path.read_text())
            oauth = data.get("claudeAiOauth")
            if oauth and oauth.get("accessToken"):
                has_oauth = True
                exp_ms = oauth.get("expiresAt", 0)
                if exp_ms:
                    remaining_min = int((exp_ms / 1000 - time.time()) / 60)
                    if remaining_min > 0:
                        oauth_detail = f"token expires in {remaining_min}m"
                    else:
                        oauth_detail = f"token expired {-remaining_min}m ago"
        except Exception:
            pass

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    backend_env = os.environ.get("CLAUDE_BACKEND", "").lower()

    # CLI backend (Claude Code subscription) is the default
    if backend_env == "cli" or (has_oauth and not has_api_key):
        return ("Claude Max subscription (OAuth)",
                oauth_detail or "no token expiry info")
    if backend_env == "api" or has_api_key:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        prefix = key[:10] + "..." if key else "(unset)"
        return ("Anthropic API (metered)", f"key: {prefix}")
    if backend_env == "openrouter" and has_openrouter:
        return ("OpenRouter", "via OPENROUTER_API_KEY")
    if has_openrouter and has_oauth:
        # OpenRouter key present but not configured as backend
        return ("Claude Max (OAuth) — OpenRouter unused for agent",
                oauth_detail or "")
    return ("unknown", "no credentials detected")


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
        "/model    — show current model, or pick a new one via buttons\n"
        "/login    — re-authenticate Claude (URL sent here, paste code back)\n"
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

    # Model + backend (where the inference bills against)
    model = _get_current_model()
    backend, backend_detail = _check_claude_backend()
    lines.append(f"Model:   {model}")
    lines.append(f"Backend: {backend}")
    if backend_detail:
        lines.append(f"         ({backend_detail})")
    lines.append("")

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


def _apply_model_change(model_id: str) -> str:
    """Hit the API to set the model. Returns user-facing result string."""
    try:
        r = httpx.post(f"{API_BASE}/system/model",
                       json={"model": model_id}, timeout=8)
        if r.status_code != 200:
            return f"❌ Set failed: HTTP {r.status_code} — {r.text[:150]}"
        data = r.json()
        if data.get("restarted"):
            return (f"✅ Model set to {data['model']}\n"
                    f"Agent is restarting — back online in ~10s.")
        return f"✅ Already on {data['model']} (no change)"
    except Exception as e:
        return f"❌ Error: {e}"


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return
    args = context.args or []
    available = get_available_models()

    if args:
        # Old-style: /model opus
        resolved = resolve_alias(args[0])
        if not resolved:
            avail = ", ".join(m["tier"] for m in available)
            await update.message.reply_text(
                f"Unknown model '{args[0]}'. Try: {avail}"
            )
            return
        await update.message.reply_text(_apply_model_change(resolved))
        return

    # No args: show current + inline buttons for each available model
    current = _get_current_model()
    text = (
        f"Current model: {current}\n\n"
        "Tap to change (auto-restarts the agent):"
    )
    rows = []
    for m in available:
        check = " ✓" if m["id"] == current else ""
        rows.append([InlineKeyboardButton(
            f"{m['label']}{check}",
            callback_data=f"model:{m['id']}",
        )])
    await update.message.reply_text(
        text, reply_markup=InlineKeyboardMarkup(rows),
    )


async def on_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline-button taps from /model."""
    query = update.callback_query
    if not query or not query.data:
        return
    # Auth check on the underlying message
    chat_id = str(query.message.chat_id) if query.message else ""
    expected = os.environ.get("TELEGRAM_CHAT_ID", "")
    if expected and chat_id != expected:
        await query.answer("Unauthorised", show_alert=True)
        return
    await query.answer()  # dismiss the "loading" spinner

    if not query.data.startswith("model:"):
        return
    model_id = query.data.split(":", 1)[1]
    result = _apply_model_change(model_id)
    # Edit the original message to show the result
    try:
        await query.edit_message_text(result)
    except Exception:
        # Fallback: send a new message
        if query.message:
            await query.message.reply_text(result)


_URL_CONTINUATION = __import__("re").compile(r"^[A-Za-z0-9%._~:/?#\[\]@!$&'()*+,;=\-]+$")


def _extract_wrapped_oauth_url(pane_text: str) -> str | None:
    """Find a Claude OAuth URL in tmux pane output, joining hard-wrapped lines.

    The Claude TUI renders the URL inside a fixed-width box, hard-wrapping
    every ~80 chars. Strategy: find the line containing 'https://...claude...',
    then greedily concatenate subsequent lines that look like URL continuations
    (no whitespace, only URL-safe characters).
    """
    lines = [ln.strip() for ln in pane_text.splitlines()]
    for i, ln in enumerate(lines):
        if "://" not in ln:
            continue
        # Must look like a Claude/Anthropic OAuth URL
        if not any(d in ln for d in ("claude.com", "claude.ai", "anthropic.com")):
            continue
        idx = ln.find("http")
        if idx < 0:
            continue
        url = ln[idx:]
        # Append continuation lines
        j = i + 1
        while j < len(lines) and lines[j] and _URL_CONTINUATION.match(lines[j]):
            url += lines[j]
            j += 1
        # Final sanity check — must contain the expected oauth path
        if "oauth" in url and ("state=" in url or "client_id=" in url):
            return url
    return None


async def cmd_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a fresh `claude /login` flow and pipe the auth URL to Telegram.

    The user opens the URL in a browser, logs in, gets a code, then sends
    the code back here as their next message — we feed it to the CLI and
    restart the agent on success.
    """
    if not _is_authorized(update):
        return

    await update.message.reply_text("🔑 Starting Claude login flow…")

    # Recreate the login tmux session cleanly
    subprocess.run(["tmux", "kill-session", "-t", LOGIN_SESSION],
                   capture_output=True)
    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", LOGIN_SESSION,
         "-c", "/home/keehar/kaggle-agent"],
        capture_output=True,
    )
    if r.returncode != 0:
        await update.message.reply_text(
            f"❌ Couldn't create tmux session: {r.stderr.decode()[:200]}"
        )
        return

    # Fire `claude /login` in the new session
    subprocess.run([
        "tmux", "send-keys", "-t", LOGIN_SESSION,
        "source /home/keehar/kaggle-venv/bin/activate && claude /login",
        "Enter",
    ])

    # First screen is a 3-option menu (subscription/API/3rd-party). Option 1
    # is highlighted by default — wait for the menu to render, then press
    # Enter to select Claude subscription, which generates the OAuth URL.
    import asyncio
    import re
    await asyncio.sleep(3)
    subprocess.run([
        "tmux", "send-keys", "-t", LOGIN_SESSION, "Enter",
    ])

    # Poll the pane up to ~15s for the auth URL to appear.
    # The TUI hard-wraps the URL across multiple lines (fixed-width box),
    # so we scan line-by-line and concatenate URL-shape continuations.
    url = None
    for _ in range(15):
        await asyncio.sleep(1)
        pane = subprocess.run(
            ["tmux", "capture-pane", "-t", LOGIN_SESSION, "-p",
             "-S", "-200"],
            capture_output=True, text=True,
        )
        url = _extract_wrapped_oauth_url(pane.stdout)
        if url:
            break

    if not url:
        await update.message.reply_text(
            "❌ Couldn't extract the login URL after 15s.\n"
            "Run `claude /login` manually in tmux."
        )
        return

    _set_login_pending()
    await update.message.reply_text(
        f"Open this in your browser, log in, and copy the code back here:\n\n"
        f"{url}\n\n"
        f"⏱ I'll wait 10 minutes for your reply. Send the code as your next "
        f"message (it usually starts with 'sk-ant-' or is a long random string)."
    )


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

async def _handle_login_code(update: Update, code: str) -> None:
    """Pipe the user-provided login code into the kaggle-login tmux session
    and check whether credentials.json was updated. Restart agent on success.
    """
    import asyncio
    creds_path = Path.home() / ".claude" / ".credentials.json"
    mtime_before = creds_path.stat().st_mtime if creds_path.exists() else 0

    subprocess.run(["tmux", "send-keys", "-t", LOGIN_SESSION, code, "Enter"])

    # Poll for credentials file to update (means login succeeded)
    success = False
    for _ in range(12):
        await asyncio.sleep(1)
        if creds_path.exists() and creds_path.stat().st_mtime > mtime_before:
            success = True
            break

    if not success:
        # Capture the pane so the user can see what went wrong
        pane = subprocess.run(
            ["tmux", "capture-pane", "-t", LOGIN_SESSION, "-p", "-S", "-15"],
            capture_output=True, text=True,
        )
        tail = pane.stdout.strip().split("\n")[-10:]
        await update.message.reply_text(
            "❌ Login didn't complete in 12s. Last output:\n\n"
            + "\n".join(tail)[-1500:]
            + "\n\nSend another /login to retry, or open tmux manually."
        )
        return

    _clear_login_state()
    # Kill the login session — it's done
    subprocess.run(["tmux", "kill-session", "-t", LOGIN_SESSION],
                   capture_output=True)
    # Restart the agent so it picks up the fresh token
    try:
        httpx.post(f"{API_BASE}/system/restart", timeout=5)
    except Exception:
        # API might be down too — fall back to direct script
        subprocess.Popen(
            ["bash", "/home/keehar/kaggle-agent/scripts/restart_agent.sh"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    await update.message.reply_text(
        "✅ Login successful. Agent restarting — back online in ~10s."
    )


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

    # 1. Are we waiting on a login code from a previous /login?
    if _get_login_state():
        log.info("received text while awaiting login code — piping to CLI")
        await _handle_login_code(update, text)
        return

    # 2. Is the agent actually running? Don't fake "thinking..." if it isn't.
    if not _agent_is_running():
        await update.message.reply_text(
            "⚠️ Agent is offline — no tmux session running.\n\n"
            "If your Claude token expired:  /login  (does it via Telegram)\n"
            "If services need to come back:  /restart\n"
            "Check what broke:               /config"
        )
        return

    # 3. Normal path — forward to the agent
    msg_id = post_human_message(text, source="telegram")
    log.info("ingested telegram message %s (%d chars)", msg_id, len(text))
    try:
        send_telegram("Got it, thinking...")
    except Exception:
        pass
    wake_agent(text)


# ── entry point ─────────────────────────────────────────────────────────────

# ── command catalog (also fed to setMyCommands for autocomplete) ────────────

BOT_COMMANDS = [
    ("help",    "List available commands"),
    ("status",  "Agent state and active competition"),
    ("config",  "Full diagnostic — model, services, account"),
    ("model",   "Show current model or change it"),
    ("login",   "Re-authenticate Claude via Telegram"),
    ("restart", "Restart the agent"),
    ("pause",   "Pause the agent"),
    ("resume",  "Resume the agent"),
    ("stop",    "Stop the agent"),
]


async def _on_startup(app: Application) -> None:
    """Register commands with Telegram so they autocomplete in the chat UI."""
    try:
        await app.bot.set_my_commands(
            [BotCommand(name, desc) for name, desc in BOT_COMMANDS]
        )
        log.info("registered %d bot commands with Telegram", len(BOT_COMMANDS))
    except Exception as e:
        log.warning("set_my_commands failed: %s", e)

    # Check for new Anthropic models (24h cached, so safe to run on every boot)
    try:
        alert_if_new_models()
    except Exception as e:
        log.warning("new-models check failed: %s", e)


def main() -> int:
    init_db()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("TELEGRAM_BOT_TOKEN not set")
        return 2

    app = (
        Application.builder()
        .token(token)
        .post_init(_on_startup)
        .build()
    )

    # Slash commands handled locally — never forwarded to the agent
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("config", cmd_config))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("restart", cmd_restart))
    app.add_handler(CommandHandler("pause", make_simple_action("pause")))
    app.add_handler(CommandHandler("resume", make_simple_action("resume")))
    app.add_handler(CommandHandler("stop", make_simple_action("stop")))

    # Inline-keyboard button taps from /model
    app.add_handler(CallbackQueryHandler(on_model_callback, pattern=r"^model:"))

    # Anything else: forward to the agent
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    log.info("starting Telegram polling with slash commands enabled")
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
