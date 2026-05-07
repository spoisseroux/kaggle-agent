"""SQLite-backed message bus for chat between human and agent.

Roles
- agent: messages produced by the kaggle agent (notifications, questions)
- human: messages produced by the human (Telegram or web UI)

Sources
- telegram, web, agent, system

Statuses
- new        : just arrived (used for unsolicited human messages)
- claimed    : agent picked it up and is acting on it
- consumed   : matched to a specific ask_human() call
- delivered  : sent (for agent->human messages already delivered to telegram)
- pending    : agent->human, not yet delivered
"""
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get(
    "KAGGLE_MESSAGE_BUS_DB",
    str(Path(__file__).resolve().parent.parent / "message_bus.sqlite3"),
))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                role        TEXT NOT NULL,
                source      TEXT NOT NULL,
                text        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'new',
                ask_id      TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS ix_messages_role_status ON messages(role, status)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_messages_ask_id ON messages(ask_id)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_messages_created ON messages(created_at)")


def _new_id() -> str:
    return uuid.uuid4().hex


def post_human_message(text: str, source: str = "telegram", ask_id: str | None = None) -> str:
    """Insert a human-originated message. If ask_id is set, it answers a pending ask."""
    init_db()
    msg_id = _new_id()
    now = time.time()
    status = "consumed" if ask_id else "new"
    with _connect() as c:
        c.execute(
            "INSERT INTO messages(id, role, source, text, status, ask_id, created_at, updated_at) "
            "VALUES (?, 'human', ?, ?, ?, ?, ?, ?)",
            (msg_id, source, text, status, ask_id, now, now),
        )
    return msg_id


def post_agent_message(text: str, source: str = "agent", ask_id: str | None = None,
                       status: str = "delivered") -> str:
    """Insert an agent-originated message."""
    init_db()
    msg_id = _new_id()
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT INTO messages(id, role, source, text, status, ask_id, created_at, updated_at) "
            "VALUES (?, 'agent', ?, ?, ?, ?, ?, ?)",
            (msg_id, source, text, status, ask_id, now, now),
        )
    return msg_id


def create_ask(text: str) -> str:
    """Record an ask_human prompt. Returns the ask_id used to match the reply."""
    init_db()
    ask_id = _new_id()
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT INTO messages(id, role, source, text, status, ask_id, created_at, updated_at) "
            "VALUES (?, 'agent', 'ask', ?, 'pending', ?, ?, ?)",
            (_new_id(), text, ask_id, now, now),
        )
    return ask_id


def wait_for_reply(ask_id: str, timeout_s: int | None = None, poll_s: float = 1.0) -> str | None:
    """Block until a human message arrives unmatched after the ask was created.

    Strategy: any 'new' human message that arrives after the ask was created is
    treated as the answer (one ask at a time). We mark it 'consumed' and tag it
    with the ask_id for traceability.
    """
    init_db()
    with _connect() as c:
        row = c.execute(
            "SELECT created_at FROM messages WHERE ask_id = ? AND source = 'ask'",
            (ask_id,),
        ).fetchone()
        if not row:
            return None
        ask_created = row["created_at"]
    deadline = None if timeout_s is None else (time.time() + timeout_s)
    while True:
        with _connect() as c:
            row = c.execute(
                "SELECT id, text FROM messages "
                "WHERE role='human' AND status='new' AND created_at >= ? "
                "ORDER BY created_at ASC LIMIT 1",
                (ask_created,),
            ).fetchone()
            if row:
                c.execute(
                    "UPDATE messages SET status='consumed', ask_id=?, updated_at=? WHERE id=?",
                    (ask_id, time.time(), row["id"]),
                )
                return row["text"]
        if deadline is not None and time.time() >= deadline:
            return None
        time.sleep(poll_s)


def get_pending_instructions(claim: bool = True) -> list[dict]:
    """Return unsolicited human messages and (by default) mark them claimed.

    Used by the agent at the top of each loop iteration to receive redirects
    from Telegram or the web chat without an explicit ask.
    """
    init_db()
    with _connect() as c:
        rows = c.execute(
            "SELECT id, text, source, created_at FROM messages "
            "WHERE role='human' AND status='new' ORDER BY created_at ASC",
        ).fetchall()
        result = [dict(r) for r in rows]
        if claim and result:
            ids = [r["id"] for r in result]
            qmarks = ",".join(["?"] * len(ids))
            c.execute(
                f"UPDATE messages SET status='claimed', updated_at=? WHERE id IN ({qmarks})",
                (time.time(), *ids),
            )
    return result


def list_recent(limit: int = 50) -> list[dict]:
    init_db()
    with _connect() as c:
        rows = c.execute(
            "SELECT id, role, source, text, status, ask_id, created_at "
            "FROM messages ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"message bus initialised at {DB_PATH}")
