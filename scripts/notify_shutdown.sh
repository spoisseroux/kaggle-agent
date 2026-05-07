#!/usr/bin/env bash
# Called by systemd before system halt/reboot/poweroff.
# Sends a Telegram notification so the user knows the agent went offline.
source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent
set -a && [ -f .env ] && source .env && set +a
python3 core/notify.py "🔴 Agent offline — system shutting down" || true
