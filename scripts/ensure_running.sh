#!/usr/bin/env bash
# ensure_running.sh — make sure all kaggle-agent services + the agent
# itself are running. Safe to call repeatedly — idempotent.
#
# Called by:
#   - tray/start_tray.bat on every tray launch
#   - any "wake the system" workflow (e.g. resuming from sleep)
#
# Unlike wsl_startup.sh this does NOT force-restart the agent — it only
# starts it if no claude process is currently running.
set -uo pipefail

LOG=/home/keehar/kaggle-agent/logs/ensure_running.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "=== ensure_running $(date -Iseconds) ==="

# 1. Tailscale (idempotent)
sudo -n tailscale up --ssh --accept-routes --accept-dns=false 2>/dev/null || true

# 2. systemd services — start is a no-op if already running
sudo -n /bin/systemctl start \
    telegram-bot \
    telegram-enforcer \
    kaggle-api \
    kaggle-api-watch \
    mlflow \
    ollama \
    kaggle-shutdown-notify \
    kaggle-message-responder 2>/dev/null || true

sudo -n /bin/systemctl start kaggle-leaderboard-monitor.timer 2>/dev/null || true

# 3. Mark the agent as running (the API stores its state in memory + a file)
sleep 2
curl -s -X POST http://localhost:8765/system/resume > /dev/null 2>&1 || true

# 4. Agent — only start if not already running.
# Match the unique --dangerously-skip-permissions arg pattern from start_agent.sh
if ! pgrep -f 'claude.*--dangerously-skip-permissions' > /dev/null 2>&1; then
    echo "Agent not running — bootstrapping via restart_agent.sh"
    bash /home/keehar/kaggle-agent/scripts/restart_agent.sh
else
    echo "Agent already running — leaving it alone"
fi

echo "ensure_running complete"
