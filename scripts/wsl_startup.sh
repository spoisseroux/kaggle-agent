#!/usr/bin/env bash
# Run by Windows Task Scheduler at logon.
# Brings everything up: Tailscale, systemd services, the agent tmux session.
# Designed to run headlessly — all output goes to the log file.
# The terminal window (if any) can be closed immediately after this script
# is launched; the tmux session persists in the background.
set -euo pipefail

LOG=/home/keehar/kaggle-agent/logs/startup.log
mkdir -p "$(dirname "$LOG")"

exec >> "$LOG" 2>&1
echo "=== wsl_startup $(date -Iseconds) ==="

source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent

# Tailscale (idempotent — `up` is safe to re-run)
sudo -n tailscale up --ssh --accept-routes --accept-dns=false || true

# Local services
sudo -n /bin/systemctl start telegram-bot kaggle-api mlflow ollama || true

# Set system state to running after services are up
sleep 5
curl -s -X POST localhost:8765/system/resume || true

# Persistent tmux session running the agent loop
if ! tmux has-session -t kaggle-agent 2>/dev/null; then
    tmux new-session -d -s kaggle-agent -c /home/keehar/kaggle-agent \
        "bash scripts/start_agent.sh"
    echo "Created kaggle-agent tmux session"
else
    echo "kaggle-agent tmux session already exists"
fi

# Start the backend file watcher (restarts kaggle-api on source changes)
sudo -n /bin/systemctl start kaggle-api-watch || true

echo "wsl_startup complete"
