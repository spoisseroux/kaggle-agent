"""Kaggle leaderboard rank monitor.

Runs hourly via systemd timer. Silently updates the rank state file;
sends a Telegram notification only when rank changes (improvement or drop).

Usage:
    python scripts/monitor_leaderboard.py
    python scripts/monitor_leaderboard.py --competition titanic
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.notify import _load_env, notify  # noqa: E402


def _active_competition() -> str:
    try:
        reg = json.loads((REPO_ROOT / "competitions" / "registry.json").read_text())
        return reg.get("active", "")
    except Exception:
        return ""


def _get_leaderboard_rank(slug: str, username: str) -> dict | None:
    """Try kaggle competitions leaderboard --show --csv and find our entry."""
    try:
        r = subprocess.run(
            ["kaggle", "competitions", "leaderboard", slug, "--show", "--csv"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"leaderboard fetch failed: {r.stderr[:200]}", file=sys.stderr)
            return None

        reader = csv.DictReader(io.StringIO(r.stdout.strip()))
        rows = list(reader)
        total = len(rows)

        for row in rows:
            team = row.get("teamName", row.get("TeamName", ""))
            if username.lower() in team.lower():
                rank_raw = row.get("rank", row.get("Rank", "0"))
                score_raw = row.get("score", row.get("Score", "0"))
                return {
                    "rank": int(rank_raw),
                    "score": float(score_raw),
                    "total_teams": total,
                }

        # Not on leaderboard yet
        return {"rank": None, "score": None, "total_teams": total}
    except Exception as e:
        print(f"_get_leaderboard_rank error: {e}", file=sys.stderr)
        return None


def _load_state(slug: str) -> dict:
    path = REPO_ROOT / ".claude" / f"leaderboard_state_{slug}.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_state(slug: str, state: dict) -> None:
    path = REPO_ROOT / ".claude" / f"leaderboard_state_{slug}.json"
    path.parent.mkdir(exist_ok=True)
    state["checked_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Kaggle leaderboard rank")
    parser.add_argument("--competition", "-c", default="")
    args = parser.parse_args()

    _load_env()
    slug = args.competition or _active_competition()
    if not slug:
        print("No active competition. Use --competition <slug>", file=sys.stderr)
        return 0  # Non-fatal — timer fires even without active competition

    username = os.environ.get("KAGGLE_USERNAME", "")
    if not username:
        print("KAGGLE_USERNAME not set in .env", file=sys.stderr)
        return 1

    print(f"Checking leaderboard: {slug} (user: {username})")
    current = _get_leaderboard_rank(slug, username)

    if current is None:
        print("Could not fetch leaderboard.", file=sys.stderr)
        return 1

    prev = _load_state(slug)
    new_rank = current.get("rank")
    old_rank = prev.get("rank")
    total = current.get("total_teams", "?")
    score = current.get("score")

    rank_str = f"#{new_rank}" if new_rank else "unranked"
    print(f"Current: {rank_str} / {total} teams | Score: {score}")

    if new_rank is not None and old_rank is not None:
        if new_rank < old_rank:
            notify(
                f"🚀 Rank improved! {slug}: #{old_rank} → #{new_rank} / {total} teams\n"
                f"Score: {score}"
            )
        elif new_rank > old_rank:
            notify(
                f"⚠️ Rank dropped: {slug}: #{old_rank} → #{new_rank} / {total} teams\n"
                f"Score: {score} — others caught up."
            )
        else:
            print("Rank unchanged — no notification sent.")
    elif new_rank is not None and old_rank is None:
        # First time we appear on the leaderboard
        notify(
            f"🏁 First leaderboard entry! {slug}: #{new_rank} / {total} teams\n"
            f"Score: {score}"
        )
    else:
        print("Not yet on leaderboard — no notification sent.")

    _save_state(slug, {
        "rank": new_rank,
        "score": score,
        "total_teams": total,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
