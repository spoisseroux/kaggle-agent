"""Competition lifecycle CLI: new / switch / archive / list / status.

Maintains:
  - competitions/registry.json   — local index, single source of truth for `active`
  - competitions/active/<slug>/  — per-competition working directory
  - kaggle_competitions table    — long-term shared state on docker postgres
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "competitions" / "registry.json"
ACTIVE_DIR = REPO_ROOT / "competitions" / "active"
ARCHIVED_DIR = REPO_ROOT / "competitions" / "archived"
TEMPLATE_DIR = REPO_ROOT / "competitions" / "template"

sys.path.insert(0, str(REPO_ROOT))


def _load() -> dict:
    if not REGISTRY.exists():
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps({"active": None, "competitions": {}}, indent=2))
    return json.loads(REGISTRY.read_text())


def _save(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2, default=str))


def cmd_new(slug: str, metric: str, deadline: str, *, name: str | None = None,
            higher_better: bool = True, submissions_max: int | None = None) -> None:
    """Create a new competition workspace and register it."""
    target = ACTIVE_DIR / slug
    if target.exists():
        raise SystemExit(f"competition '{slug}' already exists at {target}")
    if not TEMPLATE_DIR.exists():
        raise SystemExit(f"template dir missing at {TEMPLATE_DIR}")
    shutil.copytree(TEMPLATE_DIR, target)
    (target / "data").mkdir(exist_ok=True)
    claude_md = target / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(
            f"# {name or slug}\n\n"
            f"- slug: `{slug}`\n"
            f"- metric: `{metric}` (higher_better={higher_better})\n"
            f"- deadline: `{deadline}`\n\n"
            f"See repo-root CLAUDE.md for the master workflow.\n"
        )
    reg = _load()
    reg["competitions"][slug] = {
        "slug": slug, "name": name or slug, "metric": metric,
        "higher_better": higher_better, "deadline": deadline,
        "status": "active", "submissions_max": submissions_max,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    if reg.get("active") is None:
        reg["active"] = slug
    _save(reg)
    try:
        from core.memory import upsert_competition
        upsert_competition(slug, name=name or slug, metric=metric,
                           higher_better=higher_better,
                           deadline=deadline, status="active",
                           submissions_max=submissions_max)
    except Exception as e:
        print(f"[warn] could not record competition in postgres: {e}", file=sys.stderr)
    print(json.dumps({"created": slug, "active": reg["active"]}, indent=2))


def cmd_switch(slug: str) -> None:
    reg = _load()
    if slug not in reg["competitions"]:
        raise SystemExit(f"unknown competition: {slug}")
    reg["active"] = slug
    _save(reg)
    print(json.dumps({"active": slug}, indent=2))


def cmd_archive(slug: str) -> None:
    reg = _load()
    if slug not in reg["competitions"]:
        raise SystemExit(f"unknown competition: {slug}")
    src = ACTIVE_DIR / slug
    dst = ARCHIVED_DIR / slug
    ARCHIVED_DIR.mkdir(parents=True, exist_ok=True)
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    reg["competitions"][slug]["status"] = "archived"
    if reg.get("active") == slug:
        reg["active"] = None
    _save(reg)
    try:
        from core.memory import upsert_competition
        upsert_competition(slug, status="archived")
    except Exception:
        pass
    print(json.dumps({"archived": slug}, indent=2))


def cmd_list() -> None:
    reg = _load()
    print(json.dumps(reg, indent=2, default=str))


def cmd_status() -> None:
    reg = _load()
    active = reg.get("active")
    if not active:
        print(json.dumps({"active": None}, indent=2))
        return
    info = reg["competitions"].get(active, {})
    print(json.dumps({"active": active, **info}, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(prog="competition_manager")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_new = sub.add_parser("new")
    p_new.add_argument("--slug", required=True)
    p_new.add_argument("--metric", required=True)
    p_new.add_argument("--deadline", required=True)
    p_new.add_argument("--name")
    p_new.add_argument("--higher-better", default="true",
                       choices=["true", "false"])
    p_new.add_argument("--submissions-max", type=int)
    p_sw = sub.add_parser("switch")
    p_sw.add_argument("slug")
    p_ar = sub.add_parser("archive")
    p_ar.add_argument("slug")
    sub.add_parser("list")
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "new":
        cmd_new(args.slug, args.metric, args.deadline,
                name=args.name,
                higher_better=(args.higher_better == "true"),
                submissions_max=args.submissions_max)
    elif args.cmd == "switch":
        cmd_switch(args.slug)
    elif args.cmd == "archive":
        cmd_archive(args.slug)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "status":
        cmd_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
