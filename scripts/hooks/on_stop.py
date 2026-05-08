"""Claude Code Stop hook — notifies user the agent is paused.

Has a 90-second cooldown so rapid consecutive stops don't spam.
Writes a timestamp file to /tmp so restarts don't carry over stale state.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

COOLDOWN_FILE = Path("/tmp/kaggle_agent_stop_notified")
COOLDOWN_SECS = 90


def main() -> int:
    now = time.time()

    # Check cooldown — don't fire if we already fired within COOLDOWN_SECS
    if COOLDOWN_FILE.exists():
        try:
            last = float(COOLDOWN_FILE.read_text().strip())
            if now - last < COOLDOWN_SECS:
                return 0  # Too soon — skip
        except Exception:
            pass  # Corrupt file — proceed anyway

    # Write timestamp before firing so concurrent calls see it
    try:
        COOLDOWN_FILE.write_text(str(now))
    except Exception:
        pass

    # Send the notification
    try:
        from core.notify import send_telegram
        from core.notify import _load_env
        _load_env()
        send_telegram("Agent paused — waiting for input.")
    except Exception as e:
        # Fallback to subprocess if import fails
        os.system(f"python {REPO_ROOT}/core/notify.py 'Agent paused — waiting for input.'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
