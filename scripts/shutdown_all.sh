#!/usr/bin/env bash
# shutdown_all.sh — stop all kaggle-agent services and the agent tmux session.
#
# Called by:
#   - tray "Shutdown everything & quit" menu item
#   - POST /system/shutdown API endpoint
#
# Stops in reverse-dependency order. Does NOT exit the whole WSL — only
# the kaggle-agent stack. Tailscale and other unrelated services stay up.
set -uo pipefail  # do NOT use -e — we want to continue on partial failures

LOG=/home/keehar/kaggle-agent/logs/shutdown.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "=== shutdown_all $(date -Iseconds) ==="

# Final notify before telegram-bot dies (best-effort)
python3 /home/keehar/kaggle-agent/core/notify.py \
    "🛑 Shutting down agent and services (requested from tray)" \
    > /dev/null 2>&1 || true

# 1. Kill the agent tmux session + claude process first
echo "Stopping agent tmux session"
tmux kill-session -t kaggle-agent 2>/dev/null || true
tmux kill-session -t kaggle-login 2>/dev/null || true
pkill -f 'claude.*--dangerously-skip-permissions' 2>/dev/null || true

# 2. Stop application services
echo "Stopping application services"
sudo -n /bin/systemctl stop \
    kaggle-leaderboard-monitor.timer \
    kaggle-message-responder \
    kaggle-shutdown-notify \
    telegram-auto-submit \
    telegram-enforcer \
    telegram-bot \
    kaggle-api-watch \
    kaggle-api 2>/dev/null || true

# 3. Stop heavy resource services (Ollama eats VRAM, MLflow holds port 5000)
echo "Stopping heavy services (ollama, mlflow)"
sudo -n /bin/systemctl stop ollama mlflow 2>/dev/null || true

echo "shutdown complete"
