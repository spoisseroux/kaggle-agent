"""Stop hook — Windows-compatible bootstrap.

Claude Code runs hooks with Windows Python when the project is accessed via
a WSL UNC path. This script detects that case and delegates to WSL so the
real on_stop logic (with cooldown, imports, venv) runs correctly.
"""
import subprocess
import sys

subprocess.run(
    [
        "wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-c",
        "cd /home/keehar/kaggle-agent"
        " && source /home/keehar/kaggle-venv/bin/activate"
        " && python scripts/hooks/on_stop.py",
    ],
    check=False,
)
