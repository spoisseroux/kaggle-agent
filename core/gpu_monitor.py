"""Lightweight GPU stats wrapper around pynvml."""
from __future__ import annotations

import time
from typing import Iterator


def snapshot() -> dict:
    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power = None
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            proc_pids = [p.pid for p in procs]
        except Exception:
            proc_pids = []
        try:
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
        except Exception:
            name = None
        return {
            "name": name,
            "vram_total_mb": int(mem.total // (1024 * 1024)),
            "vram_used_mb": int(mem.used // (1024 * 1024)),
            "vram_free_mb": int(mem.free // (1024 * 1024)),
            "gpu_util_pct": int(util.gpu),
            "mem_util_pct": int(util.memory),
            "temp_c": temp,
            "power_w": power,
            "process_pids": proc_pids,
            "ts": time.time(),
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def stream_snapshots(interval_s: float = 3.0) -> Iterator[dict]:
    while True:
        yield snapshot()
        time.sleep(interval_s)


if __name__ == "__main__":
    import json
    print(json.dumps(snapshot(), indent=2))
