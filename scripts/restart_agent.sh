#!/usr/bin/env bash
# restart_agent.sh — cleanly restart the Claude Code agent in tmux.
#
# Why the previous Ctrl-C approach was broken: Claude Code traps signals
# and treats Ctrl-C as "interrupt current tool" not "exit". Sending a
# command after that ends up typed into Claude's prompt as a query,
# spawning nested claude processes instead of restarting.
#
# Correct approach: kill the running claude process, kill the tmux session
# if needed, and create a fresh session running start_agent.sh.
set -euo pipefail

SESSION=kaggle-agent
REPO=/home/keehar/kaggle-agent

# ── 1. Kill any running claude processes that match this repo's invocation ──
# Match the specific arg pattern from start_agent.sh so we don't kill any
# unrelated `claude` use the user has open.
pkill -f 'claude.*--dangerously-skip-permissions.*kaggle' 2>/dev/null || true
sleep 1

# ── 2. Kill the tmux session entirely (clears any nested shell state) ───────
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 0.5

# ── 3. Create a fresh session that runs start_agent.sh as its initial cmd ───
tmux new-session -d -s "$SESSION" -c "$REPO" "bash $REPO/scripts/start_agent.sh"

echo "Agent restarted in tmux session '$SESSION'"
