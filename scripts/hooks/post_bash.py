"""Claude Code PostToolUse hook for Bash — fires notify on meaningful events."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    output = event.get("output", "") or ""
    cmd = (event.get("tool_input", {}) or {}).get("command", "") or ""

    def fire(msg: str) -> None:
        os.system(f"python {REPO_ROOT}/core/notify.py {json.dumps(msg)}")

    m = re.search(r"CV[:\s]+([0-9.]+)", output)
    if m and "train.py" in cmd:
        fire(f"Training complete. CV: {m.group(1)}")
        return 0
    if "CUDA out of memory" in output:
        fire("OOM error during training. Check VRAM.")
        return 0
    if "Successfully submitted" in output:
        fire("Submission confirmed. Waiting for LB score…")
        return 0
    if re.search(r"Fold 1/", output) or re.search(r"Epoch 1/", output):
        fire("Training started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
