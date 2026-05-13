"""Single source of truth for the available Claude model list.

Design
------
- One curated list (FALLBACK_MODELS) that both the FastAPI service and the
  Telegram bot read from. The Windows tray fetches via the API.
- An optional background check (``check_for_new_models``) hits OpenRouter
  and reports any tier where Anthropic has shipped something newer than
  what we have curated — sent as a Telegram alert rather than auto-applied
  so we never feed Claude Code a malformed model ID.

When Anthropic ships a new model:
1. The next OpenRouter check (daily) sends you a Telegram alert
2. You update FALLBACK_MODELS in this file (one line per tier)
3. Bot + tray pick it up on next API restart — no other code changes needed
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / ".claude" / "models_check_cache.json"
CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h between OpenRouter pings

# Canonical list. Format is Claude Code-compatible (dashes, no dots).
# Update when Anthropic ships new models — alert script reminds you.
FALLBACK_MODELS: list[dict[str, str]] = [
    {"id": "claude-haiku-4-5-20251001",
     "label": "Haiku 4.5 · fast/cheap",
     "tier": "haiku"},
    {"id": "claude-sonnet-4-6",
     "label": "Sonnet 4.6 · balanced",
     "tier": "sonnet"},
    {"id": "claude-opus-4-7",
     "label": "Opus 4.7 · powerful",
     "tier": "opus"},
]


def get_available_models() -> list[dict[str, str]]:
    """Return the curated list — used by API, tray, and bot."""
    return FALLBACK_MODELS


def get_model_ids() -> list[str]:
    return [m["id"] for m in FALLBACK_MODELS]


def resolve_alias(alias: str) -> str | None:
    """Map 'opus' / 'sonnet' / 'haiku' or partial IDs → canonical model ID."""
    alias = alias.lower().strip()
    for m in FALLBACK_MODELS:
        if m["id"] == alias or m["tier"] == alias:
            return m["id"]
    for m in FALLBACK_MODELS:
        if alias in m["id"]:
            return m["id"]
    return None


# ── Background "new model available?" check ─────────────────────────────────

def _classify_tier(model_id: str) -> str | None:
    lid = model_id.lower()
    if "haiku" in lid:
        return "haiku"
    if "sonnet" in lid:
        return "sonnet"
    if "opus" in lid:
        return "opus"
    return None


def _version_tuple(model_id: str) -> tuple[int, ...]:
    """Extract version numbers for comparison: 'opus-4-7' or 'opus-4.7-fast' → (4, 7)."""
    nums = re.findall(r"\d+", model_id)
    # Take the first 2-3 numbers, ignore date suffixes (8-digit numbers)
    out = tuple(int(n) for n in nums if len(n) < 4)
    return out or (0,)


def _fetch_openrouter_anthropic_models() -> list[str] | None:
    """Pull Anthropic models from OpenRouter as a flat list of IDs."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=6,
        )
        if r.status_code != 200:
            return None
        return [m["id"].split("/", 1)[1]
                for m in r.json().get("data", [])
                if m.get("id", "").startswith("anthropic/claude")]
    except Exception as e:
        log.warning("OpenRouter fetch failed: %s", e)
        return None


def check_for_new_models() -> list[dict[str, str]]:
    """Compare OpenRouter's catalog to FALLBACK_MODELS by tier version.

    Returns a list of {tier, latest_openrouter, current_local} dicts for
    each tier where OpenRouter has something newer. Empty list = up to date.
    """
    remote = _fetch_openrouter_anthropic_models()
    if not remote:
        return []

    # Find the newest per tier on OpenRouter
    remote_latest: dict[str, str] = {}
    for mid in remote:
        # Skip variant suffixes — only consider clean tier-version IDs
        if mid.endswith(("-fast", "-thinking", "-bedrock", "-vertex")):
            continue
        tier = _classify_tier(mid)
        if not tier:
            continue
        v = _version_tuple(mid)
        existing = remote_latest.get(tier)
        if existing is None or v > _version_tuple(existing):
            remote_latest[tier] = mid

    # Compare against local
    local_by_tier = {m["tier"]: m["id"] for m in FALLBACK_MODELS}
    newer: list[dict[str, str]] = []
    for tier, remote_id in remote_latest.items():
        local_id = local_by_tier.get(tier, "")
        if _version_tuple(remote_id) > _version_tuple(local_id):
            newer.append({
                "tier": tier,
                "latest_openrouter": remote_id,
                "current_local": local_id or "(none)",
            })
    return newer


def alert_if_new_models() -> bool:
    """Run the check; if there's a new model, send Telegram + update cache.

    Designed to be called once a day (e.g. from the Telegram bot on startup
    or a systemd timer). Honours the 24h cache to avoid spam.

    Returns True if an alert was sent.
    """
    # Cache check — don't run more than once per TTL
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
            if time.time() - cache.get("checked_at", 0) < CACHE_TTL_SECONDS:
                return False
        except Exception:
            pass

    newer = check_for_new_models()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "checked_at": time.time(),
        "newer": newer,
    }, indent=2))

    if not newer:
        log.info("models catalog up to date")
        return False

    lines = ["🆕 New Claude model(s) available\n"]
    for n in newer:
        lines.append(
            f"  {n['tier']}: {n['latest_openrouter']}\n"
            f"     (currently using {n['current_local']})"
        )
    lines.append(
        "\nUpdate FALLBACK_MODELS in core/models_catalog.py to enable them, "
        "then restart the API."
    )
    try:
        from core.notify import send_telegram, _load_env
        _load_env()
        send_telegram("\n".join(lines))
    except Exception as e:
        log.warning("Telegram alert failed: %s", e)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Current curated list:")
    for m in get_available_models():
        print(f"  {m['id']:35s} {m['label']}")
    print("\nChecking OpenRouter for newer models…")
    newer = check_for_new_models()
    if newer:
        print("New models found:")
        for n in newer:
            print(f"  {n}")
    else:
        print("Up to date.")
