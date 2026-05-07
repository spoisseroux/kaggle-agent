#!/usr/bin/env bash
# Restart kaggle-api.service whenever Python source files in api/ or core/ change.
# Run via systemd/kaggle-api-watch.service.
set -euo pipefail

REPO=/home/keehar/kaggle-agent

echo "Backend watcher started — monitoring api/ and core/ for changes"

while true; do
    # Block until any .py file changes (modify, create, delete, or rename)
    inotifywait -r -e modify,create,delete,moved_to,moved_from \
        "$REPO/api/" \
        "$REPO/core/" \
        --include='.*\.py$' \
        -q 2>/dev/null || true

    echo "$(date -Iseconds): source change detected — restarting kaggle-api"
    # Brief debounce: editors often write the file in multiple steps
    sleep 1
    sudo -n /bin/systemctl restart kaggle-api || true
    # Cooldown to prevent thrashing if multiple files change at once
    sleep 3
done
