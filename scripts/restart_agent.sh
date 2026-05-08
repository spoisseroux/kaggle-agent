#!/usr/bin/env bash
# restart_agent.sh — cleanly restart the Claude Code agent in the kaggle-agent tmux session.
# Safe to run from tray, web UI, or terminal.
set -euo pipefail

SESSION=kaggle-agent

# ── 1. Create session if it doesn't exist ────────────────────────────────────
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No tmux session '$SESSION' found — creating one..."
    tmux new-session -d -s "$SESSION" -c /home/keehar/kaggle-agent
fi

# ── 2. Interrupt whatever is running in the pane (Ctrl-C) ────────────────────
# Send Ctrl-C twice to interrupt any running process cleanly
tmux send-keys -t "$SESSION" C-c ''
sleep 0.5
tmux send-keys -t "$SESSION" C-c ''
sleep 1

# ── 3. Ensure we're back at a shell prompt ────────────────────────────────────
# Send Enter to clear any partial input, then wait
tmux send-keys -t "$SESSION" '' Enter
sleep 0.5

# ── 4. Start the agent ───────────────────────────────────────────────────────
tmux send-keys -t "$SESSION" 'bash /home/keehar/kaggle-agent/scripts/start_agent.sh' Enter

echo "Agent restarting in tmux session '$SESSION'..."
