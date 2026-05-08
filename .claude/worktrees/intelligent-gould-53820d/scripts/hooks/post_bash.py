"""PostToolUse (Bash) hook — Windows-compatible bootstrap.

Passes stdin through to WSL so the real post_bash logic (which uses httpx,
core.notify, etc.) runs with the correct Python environment.
"""
import subprocess
import sys

stdin_data = sys.stdin.buffer.read()

subprocess.run(
    [
        "wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-c",
        "cd /home/keehar/kaggle-agent"
        " && source /home/keehar/kaggle-venv/bin/activate"
        " && python scripts/hooks/post_bash.py",
    ],
    input=stdin_data,
    check=False,
)
