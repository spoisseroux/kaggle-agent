"""download_guard.py — size-aware download safety check.

WSL2 disk reality
-----------------
WSL2 uses a dynamically-allocated VHDX file that lives on the Windows C: drive.
``df /`` inside WSL shows up to 1 TB "free" because that is the VHDX *ceiling*,
but the file can only physically grow by consuming free space on C:.

So the real free space available to any download is:
    min(WSL reported free, Windows C: free)

This module always checks *both* and uses the lower figure.

Call ``check_before_download()`` before any large download. It will:
  1. Determine the download size (Kaggle API, HuggingFace Hub, HTTP HEAD, or caller-supplied)
  2. Read free space on both the WSL filesystem and Windows C:
  3. Warn via Telegram if C: is running low (< LOW_WINDOWS_C_GB, default 80 GB)
  4. Refuse if effective free space < 2× download or < MIN_FREE_GB
  5. Ask via Telegram for any download > CONFIRM_THRESHOLD_GB (default 10 GB)
     or any download of unknown size

Usage
-----
    from core.download_guard import check_before_download

    ok = check_before_download(
        kind="kaggle_dataset",
        identifier="store-sales-time-series-forecasting",
        dest_dir="data/store-sales-time-series-forecasting",
    )
    if not ok:
        raise SystemExit("Download cancelled")

    ok = check_before_download(kind="hf_model", identifier="microsoft/phi-2", dest_dir="models/")
    ok = check_before_download(kind="url", identifier="https://example.com/big.tar.gz", dest_dir="/tmp")
"""
from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── configurable thresholds (override via .env) ────────────────────────────
# Ask for human approval if download exceeds this size
CONFIRM_THRESHOLD_GB: float = float(os.environ.get("DOWNLOAD_CONFIRM_GB", "10.0"))

# Refuse outright if effective free space < this after accounting for download
MIN_FREE_GB: float = float(os.environ.get("DOWNLOAD_MIN_FREE_GB", "20.0"))

# Warn (but don't block) if Windows C: free drops below this
LOW_WINDOWS_C_GB: float = float(os.environ.get("DOWNLOAD_LOW_C_GB", "80.0"))

# Require this many times the download size to be free (extraction headroom)
DISK_HEADROOM_FACTOR: float = 2.0

# Mount point where Windows C: appears inside WSL2
WINDOWS_C_MOUNT: str = "/mnt/c"
# ──────────────────────────────────────────────────────────────────────────


def _fmt_size(n_bytes: Optional[int]) -> str:
    if n_bytes is None:
        return "unknown size"
    for unit, thr in [("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n_bytes >= thr:
            return f"{n_bytes / thr:.1f} {unit}"
    return f"{n_bytes} B"


def _disk_status(dest_dir: str | Path) -> dict:
    """Return a dict with free-space info for both WSL and Windows C:.

    Keys:
        wsl_free_bytes   – free bytes on the WSL filesystem
        win_free_bytes   – free bytes on Windows C: (None if unmounted)
        effective_bytes  – min(wsl, win) — the real constraint
        wsl_free_gb      – float GB
        win_free_gb      – float GB or None
        effective_gb     – float GB
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    wsl_free = shutil.disk_usage(str(dest)).free
    wsl_gb = wsl_free / (1 << 30)

    win_free: Optional[int] = None
    win_gb: Optional[float] = None
    try:
        win_free = shutil.disk_usage(WINDOWS_C_MOUNT).free
        win_gb = win_free / (1 << 30)
    except Exception:
        log.debug("Could not read Windows C: free space at %s", WINDOWS_C_MOUNT)

    if win_free is not None:
        effective = min(wsl_free, win_free)
    else:
        effective = wsl_free
    effective_gb = effective / (1 << 30)

    return {
        "wsl_free_bytes": wsl_free,
        "win_free_bytes": win_free,
        "effective_bytes": effective,
        "wsl_free_gb": wsl_gb,
        "win_free_gb": win_gb,
        "effective_gb": effective_gb,
    }


def _disk_summary(status: dict) -> str:
    """One-line human-readable disk summary."""
    if status["win_free_gb"] is not None:
        return (
            f"WSL free: {status['wsl_free_gb']:.1f} GB  |  "
            f"Windows C: free: {status['win_free_gb']:.1f} GB  |  "
            f"Effective: {status['effective_gb']:.1f} GB"
        )
    return f"WSL free: {status['wsl_free_gb']:.1f} GB"


# ── size lookups ───────────────────────────────────────────────────────────

def _kaggle_dataset_size(slug: str) -> Optional[int]:
    """Return total bytes for a Kaggle competition or dataset slug."""
    for cmd in [
        ["kaggle", "competitions", "files", "-c", slug, "-v", "--csv"],
        ["kaggle", "datasets", "files", slug, "-v", "--csv"],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                reader = csv.DictReader(io.StringIO(r.stdout))
                total = 0
                for row in reader:
                    val = (row.get("size") or row.get("totalBytes") or "0").strip()
                    try:
                        total += int(val)
                    except ValueError:
                        pass
                if total > 0:
                    return total
        except Exception as e:
            log.debug("kaggle size check failed (%s): %s", cmd[1], e)
    return None


def _hf_model_size(model_id: str) -> Optional[int]:
    """Estimate HuggingFace model size via Hub API."""
    try:
        import json as _json
        import urllib.request
        url = f"https://huggingface.co/api/models/{model_id}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read())
        total = sum(
            s.get("size", 0) for s in data.get("siblings", [])
            if isinstance(s.get("size"), int)
        )
        return total if total > 0 else None
    except Exception as e:
        log.debug("HF size check failed: %s", e)
        return None


def _url_size(url: str) -> Optional[int]:
    """Try HTTP HEAD to get Content-Length."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as e:
        log.debug("URL HEAD failed: %s", e)
        return None


# ── main API ───────────────────────────────────────────────────────────────

def check_before_download(
    kind: str,
    identifier: str,
    dest_dir: str | Path = ".",
    size_bytes: Optional[int] = None,
    context: str = "",
) -> bool:
    """Gate a download with a size + disk-space check.

    Parameters
    ----------
    kind        : "kaggle_dataset" | "hf_model" | "url" | "generic"
    identifier  : competition slug, model id, or URL
    dest_dir    : destination directory (used for WSL disk-space check)
    size_bytes  : pre-known size in bytes; skips the remote lookup if set
    context     : human-readable label shown in Telegram messages

    Returns
    -------
    True  → safe to proceed
    False → user declined, disk too full, or unrecoverable error
    """
    from core.notify import send_telegram

    label = context or f"{kind}:{identifier}"

    # ── 1. determine download size ────────────────────────────────────────
    if size_bytes is None:
        log.info("Checking size of %s …", label)
        if kind == "kaggle_dataset":
            size_bytes = _kaggle_dataset_size(identifier)
        elif kind == "hf_model":
            size_bytes = _hf_model_size(identifier)
        elif kind == "url":
            size_bytes = _url_size(identifier)
        # "generic" or failed lookup → size_bytes stays None → triggers confirm

    size_str = _fmt_size(size_bytes)
    log.info("Download size for %s: %s", label, size_str)

    # ── 2. disk space check ───────────────────────────────────────────────
    status = _disk_status(dest_dir)
    effective_free = status["effective_bytes"]
    effective_gb   = status["effective_gb"]
    win_gb         = status["win_free_gb"]

    # Warn if Windows C: is running low (regardless of this download)
    if win_gb is not None and win_gb < LOW_WINDOWS_C_GB:
        send_telegram(
            f"⚠️ Windows C: drive is low on space: {win_gb:.1f} GB free\n"
            f"The WSL virtual disk lives on C:, so WSL cannot grow further.\n"
            f"Consider freeing space on Windows before running more experiments."
        )

    # Hard block: not enough room for the download + headroom
    needed_bytes = int((size_bytes or 0) * DISK_HEADROOM_FACTOR)
    needed_gb    = needed_bytes / (1 << 30)

    if size_bytes and effective_free < needed_bytes:
        send_telegram(
            f"❌ Not enough disk space for {label}\n\n"
            f"Download: {size_str}\n"
            f"Need (×2 headroom): {_fmt_size(needed_bytes)}\n"
            f"{_disk_summary(status)}\n\n"
            f"Free up space on Windows C: or WSL before retrying."
        )
        log.error("Disk full for %s: need %.1f GB effective, have %.1f GB", label, needed_gb, effective_gb)
        return False

    # Hard block: effective free below absolute minimum
    if effective_gb < MIN_FREE_GB:
        send_telegram(
            f"🚨 Disk critically low — refusing download of {label}\n\n"
            f"{_disk_summary(status)}\n"
            f"Minimum required: {MIN_FREE_GB} GB\n\n"
            f"Free space before retrying."
        )
        log.error("Disk critically low: %.1f GB effective free", effective_gb)
        return False

    # ── 3. large-download confirmation ────────────────────────────────────
    threshold_bytes = int(CONFIRM_THRESHOLD_GB * (1 << 30))
    size_known = size_bytes is not None
    needs_confirm = (not size_known) or (size_bytes >= threshold_bytes)

    if needs_confirm:
        from core.ask_human import ask

        if size_known:
            prompt = (
                f"📦 Large download: {label}\n\n"
                f"Size: {size_str}\n"
                f"{_disk_summary(status)}\n\n"
                f"Proceed? Reply yes / no"
            )
        else:
            prompt = (
                f"📦 Download size unknown: {label}\n\n"
                f"{_disk_summary(status)}\n"
                f"Could not fetch size before downloading.\n\n"
                f"Proceed? Reply yes / no"
            )

        reply = (ask(prompt) or "").strip().lower()
        if reply not in ("yes", "y", "ok", "sure", "go", "proceed", "yep", "1"):
            log.info("Download declined by user: %s", label)
            send_telegram(f"⏭️ Download skipped: {label}")
            return False
        log.info("Download approved by user: %s", label)

    # ── 4. approved — notify and proceed ─────────────────────────────────
    send_telegram(
        f"⬇️ Downloading: {label}  ({size_str})\n"
        f"{_disk_summary(status)}"
    )
    return True


def disk_health_check() -> None:
    """Standalone check — call periodically to warn about low disk space.

    Does not gate any download; just sends a Telegram alert if either
    Windows C: or the WSL effective free space is below warning thresholds.
    """
    from core.notify import send_telegram

    status = _disk_status(".")
    win_gb = status["win_free_gb"]
    eff_gb = status["effective_gb"]

    warnings = []
    if win_gb is not None and win_gb < LOW_WINDOWS_C_GB:
        warnings.append(f"⚠️ Windows C: {win_gb:.1f} GB free (threshold: {LOW_WINDOWS_C_GB} GB)")
    if eff_gb < MIN_FREE_GB * 2:
        warnings.append(f"⚠️ Effective WSL free: {eff_gb:.1f} GB (min for downloads: {MIN_FREE_GB} GB)")

    if warnings:
        send_telegram(
            "💾 Disk space alert:\n\n"
            + "\n".join(warnings)
            + f"\n\n{_disk_summary(status)}"
        )
