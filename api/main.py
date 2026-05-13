"""FastAPI bridge that the agency-platform web UI calls into.

Runs on :8765 inside WSL2; the Cloudflare Tunnel exposes it at
${CLOUDFLARE_TUNNEL_URL}. All endpoints documented in PRD §10/§17.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env  # noqa: E402

_load_env()

from core import competition_manager as cm  # noqa: E402
from core import gpu_monitor  # noqa: E402
from core import memory  # noqa: E402
from core import message_bus  # noqa: E402
from core import ollama_client  # noqa: E402
from core import system_control  # noqa: E402
from core.notify import send_telegram  # noqa: E402

# DeepEval integration
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_DSN = os.environ.get("POSTGRES_DSN")
    DEEPEVAL_ENABLED = bool(POSTGRES_DSN)
except ImportError:
    DEEPEVAL_ENABLED = False
    POSTGRES_DSN = None

API_VERSION = "0.1.0"
START_TIME = dt.datetime.utcnow()

# Settings file for agent-specific config (model selection etc.)
KAGGLE_SETTINGS_PATH = REPO_ROOT / ".claude" / "kaggle_settings.json"

from core.models_catalog import get_available_models, get_model_ids  # noqa: E402

app = FastAPI(title="Kaggle Agent API", version=API_VERSION)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not origins:
    origins = ["*"]

# Always allow Vercel deployments and production domain
origins.extend([
    "https://*.vercel.app",
    "https://kaggle-ui.nnaq.net",
    "https://app.nnaq.net",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_PATH = REPO_ROOT / "logs"


# ---------- helpers ----------

def _active_slug() -> str | None:
    reg_path = REPO_ROOT / "competitions" / "registry.json"
    if not reg_path.exists():
        return None
    try:
        return json.loads(reg_path.read_text()).get("active")
    except Exception:
        return None


def _service_active(name: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", name],
                           capture_output=True, text=True, timeout=4)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def _langfuse_online() -> bool:
    """Check if Langfuse is responding (runs in Docker, not systemd)."""
    try:
        import requests
        r = requests.get("http://docker:3000/", timeout=2)
        # Langfuse returns 200 for root page
        return r.status_code == 200
    except Exception:
        return False


def _trend_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    n = len(values)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(values) / n
    num = sum((xs[i] - mx) * (values[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-9
    return num / den


def _read_kaggle_settings() -> dict:
    if not KAGGLE_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(KAGGLE_SETTINGS_PATH.read_text())
    except Exception:
        return {}


def _write_kaggle_settings(data: dict) -> None:
    KAGGLE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    KAGGLE_SETTINGS_PATH.write_text(json.dumps(data, indent=2))


def _post_ack() -> None:
    """Post an immediate acknowledgment before the agent has time to respond."""
    message_bus.post_agent_message("Got it, thinking...", source="system", status="delivered")
    try:
        send_telegram("Got it, thinking...")
    except Exception:
        pass


# ---------- health ----------

@app.get("/health")
def health() -> dict:
    pg = memory.postgres_status()
    qd = memory.qdrant_status()
    mc = memory.mcp_status()
    oll = ollama_client.health()
    services = {
        "postgres": pg.get("status") == "online",
        "qdrant": qd.get("status") == "online",
        "mcp": mc.get("status") == "online",
        "ollama": bool(oll.get("ok")),
        "telegram": _service_active("telegram-bot"),
        "mlflow": _service_active("mlflow"),
        "langfuse": {
            "url": "http://docker:3000/project/cmoxvzksu0006rttfvmavgrc9",
            "type": "external_link",
            "status": _langfuse_online(),
        },
    }
    try:
        gpu = gpu_monitor.snapshot()
    except Exception as e:
        gpu = {"error": str(e)}
    overall = "online" if all([services["postgres"], services["ollama"], services["telegram"]]) else (
        "degraded" if any(services.values()) else "offline"
    )
    return {
        "status": overall,
        "services": services,
        "active_competition": _active_slug(),
        "gpu": gpu,
        "uptime_s": int((dt.datetime.utcnow() - START_TIME).total_seconds()),
        "last_updated": dt.datetime.utcnow().isoformat() + "Z",
    }


# ---------- usage stats ----------

@app.get("/usage/stats")
def usage_stats() -> dict:
    """Get GPU vs Claude Code usage statistics."""
    import sys
    sys.path.append("/home/keehar/kaggle-agent")

    try:
        from core.usage_tracker import get_tracker
        tracker = get_tracker()
        usage = tracker.get_current_usage()

        # Calculate GPU usage (rough estimate based on multi-agent mode history)
        # Multi-agent mode = mostly GPU/Ollama
        # Single-agent mode = mostly Claude Code

        # Get mode history from orchestrator if available
        try:
            from core.hybrid_orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            total_tasks = len(orchestrator.mode_history)

            if total_tasks > 0:
                multi_count = sum(1 for mode, _, _ in orchestrator.mode_history if mode == "multi")
                single_count = total_tasks - multi_count

                # Rough estimate: multi-agent is 67% GPU, single is 100% Claude
                gpu_pct = (multi_count * 0.67 + single_count * 0.0) / total_tasks * 100
                claude_pct = 100 - gpu_pct
            else:
                gpu_pct = 0
                claude_pct = 100
        except Exception:
            gpu_pct = 0
            claude_pct = 100

        return {
            "gpu_usage_pct": round(gpu_pct, 1),
            "claude_usage_pct": round(claude_pct, 1),
            "api_usage": {
                "minute_tokens": usage["minute_tokens"],
                "minute_requests": usage["minute_requests"],
                "day_tokens": usage["day_tokens"],
                "day_requests": usage["day_requests"],
                "minute_token_pct": round(usage["minute_token_pct"] * 100, 1),
                "minute_request_pct": round(usage["minute_request_pct"] * 100, 1),
                "day_token_pct": round(usage["day_token_pct"] * 100, 1),
            },
            "mode_history": {
                "total_tasks": total_tasks if 'total_tasks' in locals() else 0,
                "multi_agent": multi_count if 'multi_count' in locals() else 0,
                "single_agent": single_count if 'single_count' in locals() else 0,
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "gpu_usage_pct": 0,
            "claude_usage_pct": 100
        }


# ---------- system state machine ----------

@app.get("/system/state")
def system_state() -> dict:
    s = system_control.get_state()
    run_stage = system_control.get_run_stage()
    return {**s, **run_stage, "active_competition": _active_slug()}


@app.post("/system/pause")
def system_pause() -> dict:
    result = system_control.pause()
    try:
        send_telegram("⏸ Agent paused.")
    except Exception:
        pass
    return {**result, "active_competition": _active_slug()}


@app.post("/system/resume")
def system_resume() -> dict:
    result = system_control.resume()
    try:
        send_telegram("▶ Agent resumed.")
    except Exception:
        pass
    return {**result, "active_competition": _active_slug()}


@app.post("/system/stop")
def system_stop() -> dict:
    result = system_control.stop()
    try:
        send_telegram("■ Agent stopped.")
    except Exception:
        pass
    return {**result, "active_competition": _active_slug()}


@app.post("/system/checkpoint")
def system_checkpoint() -> dict:
    """Signal the agent to save mid-run state before stopping."""
    result = system_control.checkpoint()
    # Also wake the agent so it notices the checkpoint request quickly
    message_bus.wake_agent("checkpoint requested — please save state now")
    return {**result, "active_competition": _active_slug()}


@app.get("/system/ssh-info")
def ssh_info() -> dict:
    host = os.environ.get("TAILSCALE_HOSTNAME", "kaggle")
    user = os.environ.get("USER", "keehar")
    connected = False
    try:
        r = subprocess.run(["tailscale", "status", "--json"],
                           capture_output=True, text=True, timeout=4)
        if r.returncode == 0:
            data = json.loads(r.stdout)
            connected = bool(data.get("BackendState") == "Running")
    except Exception:
        connected = False
    return {
        "tailscale_hostname": host,
        "ssh_command": f"ssh {user}@{host}",
        "tmux_command": f"ssh {user}@{host} -t tmux attach -t kaggle-agent",
        "tailscale_connected": connected,
    }


@app.get("/system/hooks")
def system_hooks() -> dict:
    """Return installed Claude Code hooks from .claude/settings.json."""
    settings_path = REPO_ROOT / ".claude" / "settings.json"
    if not settings_path.exists():
        return {"hooks": []}

    try:
        settings = json.loads(settings_path.read_text())
        hooks_config = settings.get("hooks", {})

        # Flatten hooks structure: { event: [{ matcher, hooks: [...] }] }
        # into: [{ event, matcher, command }]
        result = []
        for event, matchers in hooks_config.items():
            for matcher_obj in matchers:
                matcher = matcher_obj.get("matcher", "")
                for hook in matcher_obj.get("hooks", []):
                    result.append({
                        "event": event,
                        "matcher": matcher,
                        "command": hook.get("command", ""),
                    })

        return {"hooks": result}
    except Exception as e:
        return {"hooks": [], "error": str(e)}


# ---------- model selection ----------

@app.get("/system/model")
def get_model() -> dict:
    settings = _read_kaggle_settings()
    current = settings.get("model", "claude-sonnet-4-6")
    return {"model": current, "available": get_available_models()}


class ModelUpdate(BaseModel):
    model: str


@app.post("/system/model")
def set_model(body: ModelUpdate) -> dict:
    valid_ids = get_model_ids()
    if body.model not in valid_ids:
        raise HTTPException(400, f"unknown model; available: {valid_ids}")
    settings = _read_kaggle_settings()
    previous_model = settings.get("model")
    settings["model"] = body.model
    _write_kaggle_settings(settings)

    # If the model actually changed, auto-restart the agent so it picks up
    # the new model without requiring the user to manually restart.
    restarted = False
    if previous_model != body.model:
        try:
            import subprocess
            subprocess.Popen(
                ["bash", str(REPO_ROOT / "scripts" / "restart_agent.sh")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            restarted = True
        except Exception:
            pass

    return {
        "model": body.model,
        "previous": previous_model,
        "restarted": restarted,
        "note": "Model change triggers an agent restart" if restarted else "No change",
    }


@app.post("/system/restart")
def system_restart() -> dict:
    """Restart the Claude Code agent in the kaggle-agent tmux session."""
    try:
        import subprocess
        subprocess.Popen(
            ["bash", str(REPO_ROOT / "scripts" / "restart_agent.sh")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"restart failed: {e}")


# ---------- competitions ----------

class CompetitionNew(BaseModel):
    slug: str
    metric: str
    deadline: str
    name: str | None = None
    higher_better: bool = True
    submissions_max: int | None = None


class CompetitionSwitch(BaseModel):
    slug: str


@app.get("/competitions")
def competitions() -> list[dict]:
    return memory.list_competitions() or list(json.loads((REPO_ROOT / "competitions" / "registry.json").read_text()).get("competitions", {}).values())


@app.get("/competitions/active")
def competitions_active() -> dict:
    slug = _active_slug()
    if not slug:
        return {"active": None}
    info = memory.get_competition(slug) or {"slug": slug}
    return {"active": slug, **info}


@app.get("/competitions/active/trajectory")
def competitions_trajectory() -> dict:
    slug = _active_slug()
    if not slug:
        raise HTTPException(404, "no active competition")
    comp = memory.get_competition(slug) or {"slug": slug}
    exps = memory.list_experiments(slug, limit=50)
    subs = memory.list_submissions(slug, limit=50)
    cv_history = [e.get("cv_mean") for e in reversed(exps) if e.get("cv_mean") is not None]
    lb_history = [s.get("lb_score") for s in reversed(subs) if s.get("lb_score") is not None]
    deadline = comp.get("deadline")
    deadline_days = None
    if deadline:
        try:
            d = deadline if isinstance(deadline, dt.datetime) else dt.datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            deadline_days = max(0, (d.date() - dt.date.today()).days)
        except Exception:
            deadline_days = None
    pct = None
    if comp.get("lb_rank") and comp.get("total_teams"):
        pct = 100.0 * (1.0 - (comp["lb_rank"] - 1) / max(comp["total_teams"], 1))
    return {
        "slug": slug,
        "metric": comp.get("metric"),
        "deadline_days": deadline_days,
        "best_cv": comp.get("best_cv"),
        "best_lb": comp.get("best_lb"),
        "lb_rank": comp.get("lb_rank"),
        "total_teams": comp.get("total_teams"),
        "percentile": pct,
        "cv_history": cv_history,
        "lb_history": lb_history,
        "cv_trend_slope": _trend_slope(cv_history),
        "projected_percentile": pct,
        "submissions_used": comp.get("submissions_used"),
        "submissions_max": comp.get("submissions_max"),
    }


@app.post("/competitions/new")
def competitions_new(body: CompetitionNew) -> dict:
    cm.cmd_new(body.slug, body.metric, body.deadline, name=body.name,
               higher_better=body.higher_better, submissions_max=body.submissions_max)
    return {"created": body.slug}


@app.post("/competitions/switch")
def competitions_switch(body: CompetitionSwitch) -> dict:
    cm.cmd_switch(body.slug)
    return {"active": body.slug}


# ---------- experiments / submissions / leaderboard ----------

def _fmt_experiment(r: dict) -> dict:
    """Map kaggle_experiments Postgres row → frontend shape."""
    config = r.get("config") or {}
    if isinstance(config, str):
        try:
            import json as _json
            config = _json.loads(config)
        except Exception:
            config = {}
    tags: dict = {}
    if r.get("model_type"):
        tags["model"] = r["model_type"]
    if r.get("feature_set"):
        tags["feature_set"] = r["feature_set"]
    if r.get("data_version"):
        tags["data_version"] = r["data_version"]
    name = "_".join(filter(None, [r.get("model_type"), r.get("feature_set")])) or str(r.get("id", ""))
    ts = r.get("created_at")
    if hasattr(ts, "timestamp"):
        ts = ts.timestamp()
    return {
        "id":          str(r.get("id", "")),
        "name":        name,
        "competition": r.get("competition_slug"),
        "mlflow_run":  r.get("mlflow_run_id"),
        "cv_score":    r.get("cv_mean"),
        "cv_std":      r.get("cv_std"),
        "lb_score":    r.get("lb_score"),
        "params":      config,
        "tags":        tags,
        "notes":       r.get("notes"),
        "duration_s":  r.get("training_time_s"),
        "ts":          ts,
    }


def _fmt_submission(r: dict) -> dict:
    """Map kaggle_submissions Postgres row → frontend shape."""
    ts = r.get("submitted_at")
    if hasattr(ts, "timestamp"):
        ts = ts.timestamp()
    return {
        "id":           str(r.get("id", "")),
        "competition":  r.get("competition_slug"),
        "filename":     r.get("filename"),
        "cv_score":     r.get("cv_score"),
        "lb_score":     r.get("lb_score"),
        "lb_rank":      r.get("lb_rank"),
        "total_teams":  r.get("total_teams"),
        "percentile":   r.get("percentile"),
        "description":  r.get("notes"),
        "submitted_at": ts,
    }


@app.get("/experiments")
def experiments(competition: str | None = None, limit: int = 50) -> dict:
    # competition=all or omitted → return everything
    slug = None if (not competition or competition == "all") else competition
    rows = memory.list_experiments(slug, limit=limit)
    return {"experiments": [_fmt_experiment(r) for r in rows]}


@app.get("/submissions")
def submissions(competition: str | None = None, limit: int = 50) -> dict:
    # competition=all or omitted → return everything
    slug = None if (not competition or competition == "all") else competition
    rows = memory.list_submissions(slug, limit=limit)
    return {"submissions": [_fmt_submission(r) for r in rows]}


@app.get("/leaderboard/{slug}")
def leaderboard(slug: str) -> dict:
    """Best-effort leaderboard using the Kaggle CLI."""
    try:
        r = subprocess.run(
            ["kaggle", "competitions", "leaderboard", "-c", slug, "-s", "-v"],
            capture_output=True, text=True, timeout=20,
        )
        return {"raw": r.stdout, "stderr": r.stderr, "rc": r.returncode}
    except FileNotFoundError:
        raise HTTPException(503, "kaggle CLI not installed")
    except Exception as e:
        raise HTTPException(500, f"leaderboard fetch failed: {e}")


# ---------- logs ----------

@app.get("/logs")
def logs(lines: int = 200) -> dict:
    out: dict[str, list[str]] = {}
    if not LOG_PATH.exists():
        return {"files": {}}
    for p in sorted(LOG_PATH.glob("*.log")):
        try:
            tail = p.read_text(errors="replace").splitlines()[-lines:]
            out[p.name] = tail
        except Exception:
            out[p.name] = []
    return {"files": out}


# ---------- GPU SSE ----------

@app.get("/gpu/stream")
async def gpu_stream():
    async def gen():
        while True:
            try:
                snap = gpu_monitor.snapshot()
                yield f"data: {json.dumps(snap, default=str)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- chat ----------

class ChatMessage(BaseModel):
    text: str


class MarkReadBody(BaseModel):
    up_to_id: str


def _fmt_message(r: dict) -> dict:
    """Normalise a message_bus row to the frontend wire format.

    DB schema:  id(hex), role('human'|'agent'), source, text, status, created_at
    Frontend:   id,      role('user'|'agent'),  text,   ts,   read
    """
    return {
        "id":   r["id"],
        "role": "user" if r["role"] == "human" else "agent",
        "text": r["text"],
        "ts":   r["created_at"],
        # human messages are always "read" (user sent them);
        # agent messages are read once delivered / not still pending
        "read": r["role"] == "human" or r.get("status") != "pending",
    }


@app.get("/chat/messages")
def chat_messages(limit: int = 50, before: str | None = None) -> dict:
    """Return paginated messages, oldest-first.

    ``before`` is a message id; only messages older than that id are returned.
    """
    rows = message_bus.list_recent(limit + 1, before_id=before)  # fetch +1 to detect has_more
    has_more = len(rows) > limit
    rows = rows[:limit]
    # list_recent returns newest-first; reverse to oldest-first for the frontend
    rows = list(reversed(rows))
    return {
        "messages": [_fmt_message(r) for r in rows],
        "has_more": has_more,
    }


@app.get("/chat/unread_count")
def chat_unread_count() -> dict:
    return {"count": message_bus.count_unread()}


@app.post("/chat/mark_read")
def chat_mark_read(body: MarkReadBody) -> dict:
    message_bus.mark_read_up_to(body.up_to_id)
    return {"ok": True}


@app.post("/chat/send")
def chat_send(body: ChatMessage) -> dict:
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    msg_id = message_bus.post_human_message(text, source="web")
    # Immediate ack so the user knows it landed before the agent responds
    _post_ack()
    # Wake Claude Code's stdin so it processes the message now
    message_bus.wake_agent(text)
    # Forward to telegram so phone stays in sync
    try:
        send_telegram(f"[web] {text}")
    except Exception:
        pass
    return {"ok": True, "id": msg_id}


@app.websocket("/chat/ws")
async def chat_ws(ws: WebSocket) -> None:
    await ws.accept()
    last_seen = 0.0
    try:
        # initial dump
        recent = message_bus.list_recent(50)
        for m in reversed(recent):
            await ws.send_text(json.dumps(m, default=str))
            last_seen = max(last_seen, float(m["created_at"]))
        while True:
            try:
                payload = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                data = json.loads(payload)
                text = (data.get("text") or "").strip()
                if text:
                    msg_id = message_bus.post_human_message(text, source="web")
                    # Immediate ack + wake
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _post_ack)
                    await loop.run_in_executor(None, message_bus.wake_agent, text)
                    try:
                        send_telegram(f"[web] {text}")
                    except Exception:
                        pass
                    await ws.send_text(json.dumps({"ack": msg_id}))
            except asyncio.TimeoutError:
                pass
            recent = message_bus.list_recent(50)
            for m in sorted(recent, key=lambda r: r["created_at"]):
                if float(m["created_at"]) > last_seen:
                    await ws.send_text(json.dumps(m, default=str))
                    last_seen = float(m["created_at"])
    except WebSocketDisconnect:
        return


def _set_winsize(fd: int, cols: int, rows: int) -> None:
    """Push a TIOCSWINSZ ioctl so the PTY (and tmux) know the terminal size."""
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _read_fd(fd: int) -> bytes:
    """Non-blocking read from PTY master.

    Returns b"" when nothing is ready within 0.5 s.
    Returns b"" and sets errno to EIO when the slave side closes — the
    caller must treat an empty return with EIO as EOF, not an error.
    Raises only for genuine unexpected errors.
    """
    import errno as _errno
    try:
        r, _, _ = select.select([fd], [], [], 0.5)
        if not r:
            return b""
        return os.read(fd, 4096)
    except OSError as e:
        if e.errno == _errno.EIO:
            # Slave PTY closed (tmux exited / detached) — signal EOF
            return b"\x00"   # sentinel: caller checks for this
        raise


@app.websocket("/terminal/ws")
async def terminal_ws(ws: WebSocket) -> None:
    """Full PTY terminal over WebSocket — attaches to the kaggle-agent tmux session.

    Protocol (client → server):
      {"type": "input",  "data": "<keystrokes>"}
      {"type": "resize", "cols": 220, "rows": 30}

    Protocol (server → client):
      {"type": "output", "data": "<utf-8 string>"}
      {"type": "disconnect"}   ← sent when the tmux session ends
    """
    await ws.accept()

    try:
        master_fd, slave_fd = pty.openpty()
        _set_winsize(master_fd, 220, 30)  # sensible default before first resize

        # attach-session is more reliable than new-session -A when the session
        # already exists; TERM must be set or tmux exits immediately.
        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        proc = subprocess.Popen(
            ["tmux", "attach-session", "-t", "kaggle-agent"],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            close_fds=True, preexec_fn=os.setsid,
            env=env,
        )
        os.close(slave_fd)
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).error("Terminal PTY setup failed: %s", e, exc_info=True)
        await ws.close(4000)
        return

    loop = asyncio.get_running_loop()   # get_event_loop() is deprecated in 3.10+
    stop = asyncio.Event()

    async def pty_to_ws() -> None:
        """Forward PTY output → WebSocket. Stops on EIO (tmux exited)."""
        while not stop.is_set():
            data = await loop.run_in_executor(None, _read_fd, master_fd)
            if data == b"\x00":          # EIO sentinel — slave closed
                await ws.send_text(json.dumps({"type": "disconnect"}))
                stop.set()
                break
            if data:
                await ws.send_text(json.dumps({
                    "type": "output",
                    "data": data.decode("utf-8", errors="replace"),
                }))

    async def ws_to_pty() -> None:
        """Forward WebSocket input → PTY (keystrokes + resize events)."""
        try:
            while not stop.is_set():
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "input":
                    os.write(master_fd, msg["data"].encode())
                elif msg.get("type") == "resize":
                    _set_winsize(master_fd, int(msg["cols"]), int(msg["rows"]))
        except WebSocketDisconnect:
            pass
        except OSError:
            pass
        finally:
            stop.set()

    try:
        await asyncio.gather(pty_to_ws(), ws_to_pty())
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            proc.terminate()
        except Exception:
            pass


# ---------- ollama / memory ----------

@app.get("/ollama/status")
def ollama_status() -> dict:
    return ollama_client.health()


@app.get("/memory/status")
def memory_status() -> dict:
    pg = memory.postgres_status()
    qd = memory.qdrant_status()
    mc = memory.mcp_status()
    try:
        ts = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=4,
        )
        ts_data = json.loads(ts.stdout) if ts.returncode == 0 else {}
        connected = ts_data.get("BackendState") == "Running"
        peers = ts_data.get("Peer") or {}
        latency_ms: int | None = None
        for p in peers.values():
            if p.get("HostName") == "docker":
                connected = bool(p.get("Online"))
                break
    except Exception:
        connected = False
        latency_ms = None
    return {
        "postgres": pg,
        "qdrant": qd,
        "mcp": mc,
        "tailscale": {"connected": connected, "latency_ms": pg.get("latency_ms")},
    }


# ---------- DeepEval endpoints ----------

class DeepEvalSubmission(BaseModel):
    experiment_name: str
    passed: bool
    score: float
    issues: list[str]
    warnings: list[str]
    recommendation: str
    cv_score: float
    baseline_score: float
    competition: str | None = None


@app.post("/deepeval/submit")
def submit_deepeval(submission: DeepEvalSubmission):
    """Submit a DeepEval result."""
    if not DEEPEVAL_ENABLED:
        raise HTTPException(503, "DeepEval not configured (missing POSTGRES_DSN)")

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor()

        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS deepeval_results (
                id SERIAL PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                passed BOOLEAN NOT NULL,
                score FLOAT NOT NULL,
                issues JSONB NOT NULL,
                warnings JSONB NOT NULL,
                recommendation TEXT NOT NULL,
                cv_score FLOAT NOT NULL,
                baseline_score FLOAT NOT NULL,
                competition TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        # Insert result
        import time
        timestamp = int(time.time())

        cur.execute("""
            INSERT INTO deepeval_results
            (experiment_name, timestamp, passed, score, issues, warnings, recommendation, cv_score, baseline_score, competition)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            submission.experiment_name,
            timestamp,
            submission.passed,
            submission.score,
            json.dumps(submission.issues),
            json.dumps(submission.warnings),
            submission.recommendation,
            submission.cv_score,
            submission.baseline_score,
            submission.competition or _active_slug(),
        ))

        result_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return {"success": True, "id": result_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to store DeepEval result: {e}")


@app.get("/deepeval/results")
def get_deepeval_results(competition: str | None = None, limit: int = 50):
    """Get DeepEval results, optionally filtered by competition."""
    if not DEEPEVAL_ENABLED:
        return []  # Return empty list if not configured

    try:
        conn = psycopg2.connect(POSTGRES_DSN)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        if competition:
            cur.execute("""
                SELECT id, experiment_name, timestamp, passed, score, issues, warnings,
                       recommendation, cv_score, baseline_score, competition
                FROM deepeval_results
                WHERE competition = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (competition, limit))
        else:
            cur.execute("""
                SELECT id, experiment_name, timestamp, passed, score, issues, warnings,
                       recommendation, cv_score, baseline_score, competition
                FROM deepeval_results
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

        results = []
        for row in cur.fetchall():
            results.append({
                "id": str(row["id"]),
                "experiment_name": row["experiment_name"],
                "timestamp": row["timestamp"],
                "passed": row["passed"],
                "score": row["score"],
                "issues": row["issues"] if isinstance(row["issues"], list) else json.loads(row["issues"]),
                "warnings": row["warnings"] if isinstance(row["warnings"], list) else json.loads(row["warnings"]),
                "recommendation": row["recommendation"],
                "cv_score": row["cv_score"],
                "baseline_score": row["baseline_score"],
            })

        cur.close()
        conn.close()

        return results
    except Exception as e:
        # If table doesn't exist yet, return empty list
        if "does not exist" in str(e):
            return []
        raise HTTPException(500, f"Failed to fetch DeepEval results: {e}")


# ---------- entrypoint ----------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
