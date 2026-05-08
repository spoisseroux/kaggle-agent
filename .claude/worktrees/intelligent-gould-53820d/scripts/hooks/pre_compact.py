"""PreCompact hook — Windows-compatible bootstrap.

Delegates to WSL so the real pre_compact logic runs with the correct
Python environment and can import psycopg2, core.*, etc.
"""
import subprocess
import sys

subprocess.run(
    [
        "wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-c",
        "cd /home/keehar/kaggle-agent"
        " && source /home/keehar/kaggle-venv/bin/activate"
        " && python scripts/hooks/pre_compact.py",
    ],
    check=False,
)
