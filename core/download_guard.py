"""download_guard.py — size-aware download safety check.

Call ``check_before_download()`` before any large download. It will:
  1. Try to determine the download size (Kaggle API, HTTP HEAD, or caller-supplied)
  2. Check that the destination disk has enough free space (2× the download)
  3. If the download is above the confirmation threshold, ask the human via
     Telegram before proceeding

Usage
-----
    from core.download_guard import check_before_download

    # Kaggle competition dataset
    ok = check_before_download(
        kind="kaggle_dataset",
        identifier="store-sales-time-series-forecasting",
        dest_dir="data/store-sales-time-series-forecasting",
    )
    if not ok:
        raise SystemExit("Download cancelled by user or disk full")

    # Hugging Face model
    ok = check_before_download(
        kind="hf_model",
        identifier="microsoft/phi-2",
        dest_dir="models/phi-2",
    )

    # Generic URL
    ok = check_before_download(
        kind="url",
        identifier="https://example.com/bigfile.tar.gz",
        dest_dir="/tmp",
    )
"""
from __future__ import annotations

import os
import shutil
import subprocess
import csv
import io
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── configurable thresholds ────────────────────────────────────────────────
# Ask for human approval if the download exceeds this size
CONFIRM_THRESHOLD_GB: float = float(os.environ.get("DOWNLOAD_CONFIRM_GB", "1.0"))

# Refuse outright if free disk < this many GB after the download
MIN_FREE_GB: float = float(os.environ.get("DOWNLOAD_MIN_FREE_GB", "10.0"))

# How many times the download size must be available free (headroom for
# extraction, temp files, model shards, etc.)
DISK_HEADROOM_FACTOR: float = 2.0
# ──────────────────────────────────────────────────────────────────────────


def _fmt_size(n_bytes: Optional[int]) -> str:
    if n_bytes is None:
        return "unknown size"
    for unit, thr in [("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n_bytes >= thr:
            return f"{n_bytes / thr:.1f} {unit}"
    return f"{n_bytes} B"


def _free_bytes(path: str | Path) -> int:
    """Return free bytes on the filesystem containing *path*."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(str(p)).free


def _kaggle_dataset_size(slug: str) -> Optional[int]:
    """Return total size in bytes of a Kaggle competition dataset."""
    try:
        # Try competition dataset first
        r = subprocess.run(
            ["kaggle", "competitions", "files", "-c", slug, "-v", "--csv"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            reader = csv.DictReader(io.StringIO(r.stdout))
            total = 0
            for row in reader:
                size_str = (row.get("size") or row.get("totalBytes") or "0").strip()
                try:
                    total += int(size_str)
                except ValueError:
                    pass
            if total > 0:
                return total
        # Try dataset slug (user/dataset format)
        r = subprocess.run(
            ["kaggle", "datasets", "files", slug, "-v", "--csv"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout.strip():
            reader = csv.DictReader(io.StringIO(r.stdout))
            total = sum(
                int((row.get("size") or row.get("totalBytes") or "0").strip())
                for row in reader
                if (row.get("size") or row.get("totalBytes") or "").strip().isdigit()
            )
            if total > 0:
                return total
    except Exception as e:
        log.debug("kaggle size check failed: %s", e)

    # Fall back to metadata endpoint
    try:
        r = subprocess.run(
            ["kaggle", "competitions", "list", "--csv", "-s", slug],
            capture_output=True, text=True, timeout=15,
        )
        # Can't get size this way reliably; return None to trigger confirm
    except Exception:
        pass
    return None


def _hf_model_size(model_id: str) -> Optional[int]:
    """Estimate HuggingFace model size via Hub API."""
    try:
        import urllib.request, json as _json
        url = f"https://huggingface.co/api/models/{model_id}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read())
        siblings = data.get("siblings", [])
        total = sum(s.get("size", 0) for s in siblings if isinstance(s.get("size"), int))
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
    kind : "kaggle_dataset" | "hf_model" | "url" | "generic"
    identifier : slug, model id, or URL
    dest_dir : where files will land (used for disk-space check)
    size_bytes : pre-known size; skips the remote lookup if provided
    context : human-readable description shown in the Telegram message

    Returns
    -------
    True  → safe to proceed
    False → user declined, disk too full, or error
    """
    from core.notify import send_telegram  # local import to avoid circular

    label = context or f"{kind}:{identifier}"

    # ── 1. determine size ────────────────────────────────────────────────
    if size_bytes is None:
        log.info("Checking size of %s …", label)
        if kind == "kaggle_dataset":
            size_bytes = _kaggle_dataset_size(identifier)
        elif kind == "hf_model":
            size_bytes = _hf_model_size(identifier)
        elif kind in ("url", "generic"):
            size_bytes = _url_size(identifier) if kind == "url" else None
        # Unknown size: log and treat as if large (will trigger confirm)

    size_str = _fmt_size(size_bytes)
    log.info("Download size for %s: %s", label, size_str)

    # ── 2. disk space check ──────────────────────────────────────────────
    free = _free_bytes(dest_dir)
    free_gb = free / (1 << 30)
    needed = (size_bytes or 0) * DISK_HEADROOM_FACTOR
    needed_gb = needed / (1 << 30)

    if size_bytes and free < needed:
        msg = (
            f"❌ Not enough disk space for {label}\n\n"
            f"Download: {size_str}\n"
            f"Need (2×): {_fmt_size(int(needed))}\n"
            f"Available: {_fmt_size(free)}\n\n"
            f"Free up space on the WSL disk before retrying."
        )
        log.error("Disk too full for %s: need %.1f GB, have %.1f GB", label, needed_gb, free_gb)
        send_telegram(msg)
        return False

    if free_gb < MIN_FREE_GB:
        msg = (
            f"⚠️ Disk critically low — aborting download of {label}\n\n"
            f"Free: {free_gb:.1f} GB (minimum: {MIN_FREE_GB} GB)\n"
            f"Free up space and retry."
        )
        log.error("Disk critically low: %.1f GB free", free_gb)
        send_telegram(msg)
        return False

    # ── 3. confirmation for large downloads ──────────────────────────────
    threshold_bytes = int(CONFIRM_THRESHOLD_GB * (1 << 30))
    size_known = size_bytes is not None
    is_large = (not size_known) or (size_bytes >= threshold_bytes)

    if is_large:
        from core.ask_human import ask  # local import
        if size_known:
            prompt = (
                f"📦 Large download: {label}\n\n"
                f"Size: {size_str}\n"
                f"Disk free: {free_gb:.1f} GB\n\n"
                f"Proceed? Reply yes/no"
            )
        else:
            prompt = (
                f"📦 Download size unknown: {label}\n\n"
                f"Disk free: {free_gb:.1f} GB\n"
                f"Could not determine size in advance.\n\n"
                f"Proceed? Reply yes/no"
            )
        reply = (ask(prompt) or "").strip().lower()
        if reply not in ("yes", "y", "ok", "sure", "go", "proceed", "1"):
            log.info("Download declined by user: %s", label)
            send_telegram(f"⏭️ Download skipped: {label}")
            return False
        log.info("Download approved by user: %s", label)

    send_telegram(
        f"⬇️ Starting download: {label}  ({size_str})\n"
        f"Disk free: {free_gb:.1f} GB"
    )
    return True
