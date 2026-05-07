"""VRAM allocation manager for the RTX 5070 (12GB total).

Budget (from PRD §7):
    Total:               12,227 MB
    Windows reserve:      1,500 MB
    Safe WSL2 budget:    10,700 MB
    Ollama (Qwen3-14B):   9,200 MB
    Training minimum:     9,500 MB free before starting

Use `request_training_vram()` before any GPU training; it stops the Ollama
service so the runner releases its share, then waits for VRAM to free.
After training, `release_training_vram()` restarts Ollama.
"""
from __future__ import annotations

import logging
import subprocess
import time

import httpx

"""
PRD §7 budget assumes Windows holds ~1.5 GB. On this WSL2 install Windows
reserves closer to 3.0 GB (Tailscale, Docker Desktop, browser GPU procs),
so even with Ollama stopped only ~9.2 GB is free. We set the floor to
9000 MB — comfortable for tabular LGBM/XGBoost and modest neural nets;
revisit if a competition needs >8 GB resident on the GPU.
"""
TRAINING_MIN_FREE_MB = 9000

log = logging.getLogger(__name__)


def get_free_vram_mb() -> int:
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free // (1024 * 1024))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_total_vram_mb() -> int:
    import pynvml
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.total // (1024 * 1024))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def ollama_is_loaded() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/ps", timeout=3)
        return len(r.json().get("models", [])) > 0
    except Exception:
        return False


def ollama_service_active() -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "ollama"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() == "active"
    except Exception:
        return False


def request_training_vram(timeout_s: int = 120, min_free_mb: int = TRAINING_MIN_FREE_MB) -> bool:
    """Stop Ollama if needed, wait for VRAM to free, return True on success."""
    if get_free_vram_mb() >= min_free_mb and not ollama_is_loaded():
        log.info("VRAM already free: %dMB", get_free_vram_mb())
        return True
    if ollama_service_active():
        log.info("stopping Ollama to free VRAM…")
        subprocess.run(["sudo", "-n", "/bin/systemctl", "stop", "ollama"], check=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        free = get_free_vram_mb()
        loaded = ollama_is_loaded()
        if free >= min_free_mb and not loaded:
            log.info("VRAM freed: %dMB available", free)
            return True
        time.sleep(2)
    log.error("VRAM not freed in %ds (free=%dMB)", timeout_s, get_free_vram_mb())
    return False


def release_training_vram(timeout_s: int = 30) -> bool:
    """Restart Ollama once VRAM is mostly idle."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if get_free_vram_mb() > 10000:
            break
        time.sleep(2)
    try:
        subprocess.run(["sudo", "-n", "/bin/systemctl", "start", "ollama"], check=True)
        log.info("Ollama restarted")
        return True
    except subprocess.CalledProcessError as e:
        log.error("failed to restart Ollama: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s vram %(message)s")
    print(f"total VRAM: {get_total_vram_mb()}MB")
    print(f"free VRAM:  {get_free_vram_mb()}MB")
    print(f"ollama service active: {ollama_service_active()}")
    print(f"ollama model loaded:   {ollama_is_loaded()}")
