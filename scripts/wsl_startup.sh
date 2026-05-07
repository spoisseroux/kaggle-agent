#!/usr/bin/env bash
# Run by Windows Task Scheduler at logon.
# Brings everything up: Tailscale, systemd services, the agent tmux session.
set -euo pipefail

source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent

# Tailscale (idempotent — `up` is safe to re-run)
sudo -n tailscale up --ssh --accept-routes --accept-dns=false || true

# Local services
sudo -n /bin/systemctl start telegram-bot kaggle-api mlflow ollama || true

# Persistent tmux session running the agent loop
if ! tmux has-session -t kaggle-agent 2>/dev/null; then
  tmux new-session -d -s kaggle-agent -c /home/keehar/kaggle-agent \
    "bash scripts/start_agent.sh"
fi

echo "wsl_startup complete"
