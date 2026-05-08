#!/bin/bash
# Start the auto-submit daemon if not already running

DAEMON_SCRIPT="/home/keehar/kaggle-agent/scripts/telegram_auto_submit.py"
LOG_FILE="/tmp/auto_submit.log"

# Check if already running
if pgrep -f "telegram_auto_submit.py" > /dev/null; then
    echo "Auto-submit daemon already running"
    exit 0
fi

# Start daemon
PYTHONUNBUFFERED=1 nohup python "$DAEMON_SCRIPT" > "$LOG_FILE" 2>&1 &
NEW_PID=$!

sleep 2

# Verify it started
if ps -p $NEW_PID > /dev/null; then
    echo "Auto-submit daemon started successfully (PID: $NEW_PID)"
else
    echo "Failed to start daemon"
    exit 1
fi
