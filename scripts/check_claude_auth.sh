#!/usr/bin/env bash
# check_claude_auth.sh — run before starting the agent.
# Checks if the Claude OAuth token is expired and attempts a silent refresh.
# If refresh fails, sends a Telegram alert and exits 1 so start_agent.sh aborts.
set -euo pipefail

REPO=/home/keehar/kaggle-agent
CREDS=~/.claude/.credentials.json
VENV=/home/keehar/kaggle-venv

source "$VENV/bin/activate"
set -a; [ -f "$REPO/.env" ] && source "$REPO/.env"; set +a

notify() {
    python3 "$REPO/core/notify.py" "$1" 2>/dev/null || true
}

# ── 1. Check if credentials file exists ──────────────────────────────────
if [ ! -f "$CREDS" ]; then
    notify "🔑 Claude auth missing — no credentials file found.

Open a terminal in WSL and run:
  claude /login

Then restart the agent via the tray or:
  bash ~/kaggle-agent/scripts/wsl_startup.sh"
    echo "ERROR: no credentials file" >&2
    exit 1
fi

# ── 2. Read expiry and check ──────────────────────────────────────────────
EXPIRES_MS=$(python3 -c "
import json, sys
try:
    d = json.load(open('$CREDS'))
    print(d['claudeAiOauth']['expiresAt'])
except Exception as e:
    print(0)
" 2>/dev/null)

NOW_MS=$(python3 -c "import time; print(int(time.time()*1000))")
BUFFER_MS=$((5 * 60 * 1000))  # 5 min buffer

if [ "$EXPIRES_MS" -gt "$((NOW_MS + BUFFER_MS))" ]; then
    echo "Claude auth OK — token valid for $(( (EXPIRES_MS - NOW_MS) / 60000 )) more minutes"
    exit 0
fi

# ── 3. Token expired — try silent refresh ────────────────────────────────
echo "Claude token expired or expiring soon — attempting refresh..."

REFRESH_TOKEN=$(python3 -c "
import json
try:
    d = json.load(open('$CREDS'))
    print(d['claudeAiOauth']['refreshToken'])
except:
    print('')
" 2>/dev/null)

if [ -n "$REFRESH_TOKEN" ]; then
    # Attempt OAuth refresh via Anthropic token endpoint
    RESPONSE=$(python3 - <<'PYEOF'
import json, urllib.request, urllib.parse, os, sys

creds_path = os.path.expanduser("~/.claude/.credentials.json")
try:
    with open(creds_path) as f:
        creds = json.load(f)
    refresh_token = creds["claudeAiOauth"]["refreshToken"]
except Exception as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)

# Claude Code uses this endpoint for token refresh
data = urllib.parse.urlencode({
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
    "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
}).encode()

try:
    req = urllib.request.Request(
        "https://claude.ai/oauth/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        new_tokens = json.loads(resp.read())

    # Merge new tokens into credentials
    import time
    expires_in = new_tokens.get("expires_in", 3600)
    creds["claudeAiOauth"]["accessToken"] = new_tokens["access_token"]
    if "refresh_token" in new_tokens:
        creds["claudeAiOauth"]["refreshToken"] = new_tokens["refresh_token"]
    creds["claudeAiOauth"]["expiresAt"] = int((time.time() + expires_in) * 1000)

    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)

    print("REFRESHED")
except Exception as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    )

    if [ "$RESPONSE" = "REFRESHED" ]; then
        echo "Token refreshed successfully"
        notify "🔑 Claude auth token auto-refreshed — agent resuming normally"
        exit 0
    fi
fi

# ── 4. Refresh failed — alert user and exit ───────────────────────────────
echo "Token refresh failed — manual login required" >&2

notify "🔑 Claude login required — agent cannot start

The Claude Max OAuth token has expired and could not be auto-refreshed.

To fix:
1. Open WSL terminal (tray → 'Open tmux session')
2. Run: claude /login
3. Open the browser link it shows and log in with your Claude Max account
4. The agent will restart automatically after login

Agent is paused until login is complete."

exit 1
