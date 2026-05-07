"""Pause/resume/stop state machine for the Claude Code agent.

States persisted to `state.json` in the repo root:
    running   — agent loop alive, Ollama running
    paused    — agent processes SIGSTOP'd, Ollama stopped (FastAPI/Telegram
                bot intentionally stay up to receive commands)
    stopped   — agent tmux session killed, Ollama stopped

The FastAPI service (`kaggle-api.service`) and Telegram bot
(`telegram-bot.service`) are NEVER touched by these operations: they must
stay alive so the user can resume from anywhere.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "state.json"
SESSION = "kaggle-agent"

log = logging.getLogger(__name__)


# ---------- state file ----------

def _read_state_file() -> dict:
    if not STATE_PATH.exists():
        return {"state": "running", "since": time.time()}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"state": "running", "since": time.time()}


def _write_state_file(state: str) -> dict:
    payload = {"state": state, "since": time.time()}
    STATE_PATH.write_text(json.dumps(payload))
    return payload


# ---------- tmux helpers ----------

def session_exists() -> bool:
    try:
        r = subprocess.run(
            ["tmux", "has-session", "-t", SESSION],
            capture_output=True, timeout=4,
        )
        return r.returncode == 0
    except Exception:
        return False


def _list_descendants(pid: int) -> list[int]:
    """Return PID and all descendants by walking pgrep -P."""
    out: list[int] = []
    try:
        r = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True, timeout=3,
        )
        children = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
        for c in children:
            out.append(c)
            out.extend(_list_descendants(c))
    except Exception:
        pass
    return out


def list_session_pids() -> list[int]:
    """Return all PIDs running inside the kaggle-agent tmux session."""
    if not session_exists():
        return []
    try:
        r = subprocess.run(
            ["tmux", "list-panes", "-t", SESSION, "-F", "#{pane_pid}"],
            capture_output=True, text=True, timeout=4,
        )
        if r.returncode != 0:
            return []
        pane_pids = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
    except Exception:
        return []
    pids = list(pane_pids)
    for p in pane_pids:
        pids.extend(_list_descendants(p))
    return sorted(set(pids))


def _signal_pids(pids: list[int], sig: int) -> int:
    sent = 0
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            log.warning("permission denied sending signal to pid %s", pid)
        except Exception as e:
            log.warning("kill(%s, %s) failed: %s", pid, sig, e)
    return sent


# ---------- cgroup freeze (WSL2-friendly) ----------
#
# On WSL2 with systemd, SIGSTOP/SIGCONT delivered cross-cgroup don't reliably
# pause the target — we observed a CPU-bound python in the tmux scope keep
# running at 100% after kill -STOP succeeded with no error. cgroupv2's
# `cgroup.freeze` is the modern, supported way to pause everything in a
# cgroup atomically. The tmux pane lives in
# /sys/fs/cgroup/user.slice/.../app-tmux.slice/tmux-spawn-*.scope/ and the
# `cgroup.freeze` file there is writable by the user who owns the scope.

def _scope_path(pid: int) -> Path | None:
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text().strip()
    except Exception:
        return None
    # cgroupv2 line format: "0::/user.slice/...scope"
    if "::" not in line:
        return None
    rel = line.split("::", 1)[1].lstrip("/")
    return Path("/sys/fs/cgroup") / rel


def _set_freeze(pids: list[int], frozen: bool) -> int:
    """Write to cgroup.freeze for each unique cgroup these pids belong to."""
    scopes: set[Path] = set()
    for pid in pids:
        sp = _scope_path(pid)
        if sp:
            scopes.add(sp)
    written = 0
    for sp in scopes:
        f = sp / "cgroup.freeze"
        if not f.exists():
            continue
        try:
            f.write_text("1" if frozen else "0")
            written += 1
        except PermissionError:
            log.warning("no permission to write %s", f)
        except Exception as e:
            log.warning("freeze write failed for %s: %s", f, e)
    return written


# ---------- ollama ----------

def _ollama_stop() -> None:
    subprocess.run(
        ["sudo", "-n", "/bin/systemctl", "stop", "ollama"],
        check=False, timeout=15,
    )


def _ollama_start() -> None:
    subprocess.run(
        ["sudo", "-n", "/bin/systemctl", "start", "ollama"],
        check=False, timeout=15,
    )


# ---------- public API ----------

def get_state() -> dict:
    """Return persisted state, reconciling with reality.

    If state.json says running but no tmux session exists, the agent must
    have crashed or been stopped externally — surface that as 'stopped'
    without rewriting the file (we don't know if a 'pause' should still
    be remembered).
    """
    s = _read_state_file()
    declared = s.get("state", "running")
    has_session = session_exists()
    if declared == "running" and not has_session:
        return {"state": "stopped", "since": s.get("since", time.time()),
                "note": "session_missing"}
    if declared == "stopped" and has_session:
        # Someone manually started the session — promote to running.
        return _write_state_file("running") | {"note": "session_reappeared"}
    return s


def pause() -> dict:
    pids = list_session_pids()
    frozen_scopes = _set_freeze(pids, True)
    _ollama_stop()
    log.info("pause: froze %d cgroup scope(s) covering %d pids, ollama stopped",
             frozen_scopes, len(pids))
    return _write_state_file("paused") | {
        "paused_pids": len(pids),
        "frozen_scopes": frozen_scopes,
    }


def resume() -> dict:
    if not session_exists():
        # Recreate tmux session running the agent loop.
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", SESSION,
             "-c", str(REPO_ROOT),
             f"bash {REPO_ROOT}/scripts/start_agent.sh"],
            check=False, timeout=10,
        )
        log.info("resume: created new tmux session")
    pids = list_session_pids()
    thawed = _set_freeze(pids, False)
    # Belt-and-braces: if anything got SIGSTOP'd elsewhere, wake it up.
    _signal_pids(pids, signal.SIGCONT)
    _ollama_start()
    log.info("resume: thawed %d cgroup scope(s) covering %d pids, ollama started",
             thawed, len(pids))
    return _write_state_file("running") | {
        "resumed_pids": len(pids),
        "thawed_scopes": thawed,
    }


def stop() -> dict:
    pids = list_session_pids()
    # If currently frozen, thaw first so signals can be received.
    _set_freeze(pids, False)
    _signal_pids(pids, signal.SIGCONT)
    _signal_pids(pids, signal.SIGTERM)
    time.sleep(1.0)
    still = [p for p in pids if _pid_alive(p)]
    if still:
        _signal_pids(still, signal.SIGKILL)
    if session_exists():
        subprocess.run(
            ["tmux", "kill-session", "-t", SESSION],
            check=False, timeout=10,
        )
    _ollama_stop()
    log.info("stop: terminated %d pids, killed tmux session, ollama stopped",
             len(pids))
    return _write_state_file("stopped") | {"terminated_pids": len(pids)}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "state"
    fn = {
        "state": get_state, "pause": pause, "resume": resume, "stop": stop,
    }.get(cmd)
    if not fn:
        print(f"usage: python -m core.system_control [state|pause|resume|stop]",
              file=sys.stderr)
        sys.exit(2)
    print(json.dumps(fn(), indent=2, default=str))
