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

# Persistent tmux session running the agent loop.
# Use restart_agent.sh which kills any stale process + creates a fresh
# session — handles the "tmux session exists but claude inside died" case
# that previously left the tray red after a crash.
bash /home/keehar/kaggle-agent/scripts/restart_agent.sh
echo "kaggle-agent session bootstrapped via restart_agent.sh"

# Start the backend file watcher (restarts kaggle-api on source changes)
sudo -n /bin/systemctl start kaggle-api-watch || true

# Start hourly leaderboard rank monitor
sudo -n /bin/systemctl start kaggle-leaderboard-monitor.timer || true

python3 core/notify.py "🟢 Agent online — all services started" || true

echo "wsl_startup complete"
