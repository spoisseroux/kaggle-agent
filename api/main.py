"""FastAPI bridge that the agency-platform web UI calls into.

Runs on :8765 inside WSL2; the Cloudflare Tunnel exposes it at
${CLOUDFLARE_TUNNEL_URL}. All endpoints documented in PRD §10/§17.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
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
from core.notify import send_telegram  # noqa: E402

API_VERSION = "0.1.0"
START_TIME = dt.datetime.utcnow()

app = FastAPI(title="Kaggle Agent API", version=API_VERSION)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not origins:
    origins = ["*"]
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


def _trend_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    n = len(values)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(values) / n
    num = sum((xs[i] - mx) * (values[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-9
    return num / den


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

@app.get("/experiments")
def experiments(competition: str | None = None, limit: int = 50) -> list[dict]:
    return memory.list_experiments(competition, limit=limit)


@app.get("/submissions")
def submissions(competition: str | None = None, limit: int = 50) -> list[dict]:
    return memory.list_submissions(competition, limit=limit)


@app.get("/leaderboard/{slug}")
def leaderboard(slug: str) -> dict:
    """Best-effort leaderboard using the Kaggle CLI."""
    try:
        env = os.environ.copy()
        # Kaggle CLI accepts KAGGLE_KEY too, but we standardise on KAGGLE_TOKEN.
        if "KAGGLE_KEY" not in env and env.get("KAGGLE_TOKEN"):
            env["KAGGLE_KEY"] = env["KAGGLE_TOKEN"]
        r = subprocess.run(
            ["kaggle", "competitions", "leaderboard", "-c", slug, "-s", "-v"],
            capture_output=True, text=True, env=env, timeout=20,
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


@app.get("/chat/messages")
def chat_messages(limit: int = 50) -> list[dict]:
    return list(reversed(message_bus.list_recent(limit)))


@app.post("/chat/send")
def chat_send(body: ChatMessage) -> dict:
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    msg_id = message_bus.post_human_message(text, source="web")
    # forward to telegram so phone stays in sync
    try:
        send_telegram(f"[web] {text}")
    except Exception:
        pass
    return {"id": msg_id}


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


# ---------- entrypoint ----------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", "8765"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
