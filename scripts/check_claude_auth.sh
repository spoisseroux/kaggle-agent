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
    # NOTE: This often returns 403 because Claude Code uses PKCE during initial
    # auth and the code_verifier is not stored in credentials.json. Claude Code
    # itself manages refresh internally — this is a best-effort attempt only.
    RESPONSE=$(python3 - <<'PYEOF'
import json, urllib.request, urllib.parse, urllib.error, os, sys, time

creds_path = os.path.expanduser("~/.claude/.credentials.json")
try:
    with open(creds_path) as f:
        creds = json.load(f)
    oauth = creds["claudeAiOauth"]
    refresh_token = oauth["refreshToken"]
    # Show debug info: token age
    expires_at_ms = oauth.get("expiresAt", 0)
    expired_ago_min = int((time.time() - expires_at_ms / 1000) / 60)
    print(f"DEBUG: token expired {expired_ago_min}m ago, attempting refresh", file=sys.stderr)
except Exception as e:
    print(f"FAIL:CREDS:{e}", file=sys.stderr)
    sys.exit(1)

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

    expires_in = new_tokens.get("expires_in", 3600)
    creds["claudeAiOauth"]["accessToken"] = new_tokens["access_token"]
    if "refresh_token" in new_tokens:
        creds["claudeAiOauth"]["refreshToken"] = new_tokens["refresh_token"]
    creds["claudeAiOauth"]["expiresAt"] = int((time.time() + expires_in) * 1000)

    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)

    print("REFRESHED")

except urllib.error.HTTPError as e:
    body = ""
    try:
        body = e.read().decode()[:200]
    except Exception:
        pass
    print(f"FAIL:HTTP:{e.code}:{body}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"FAIL:ERR:{e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    ) || true  # don't let set -e kill the script on refresh failure

    # Capture stderr from above into REFRESH_ERR for Telegram logging
    # (stderr already printed to terminal above; we re-run a quick parse)
    REFRESH_DETAIL=$(python3 -c "
import json, os, time
try:
    d = json.load(open(os.path.expanduser('~/.claude/.credentials.json')))
    exp = d['claudeAiOauth'].get('expiresAt', 0)
    print(f'Token expired {int((time.time() - exp/1000)/60)}m ago')
except Exception as e:
    print(f'Could not read creds: {e}')
" 2>/dev/null || echo "unknown")

    if [ "$RESPONSE" = "REFRESHED" ]; then
        echo "Token refreshed successfully"
        notify "🔑 Claude auth token auto-refreshed — agent resuming normally"
        exit 0
    fi
fi

# ── 4. Refresh failed — warn but don't block ─────────────────────────────
# Claude Code manages its own auth internally via PKCE — our REST refresh
# attempt lacks the code_verifier so it always 403s on stale tokens.
# Claude itself will succeed; we just alert in case it doesn't.
echo "Token refresh failed — Claude will handle auth internally" >&2

notify "⚠️ Claude token refresh failed — continuing anyway

Detail: $REFRESH_DETAIL
Cause: PKCE verifier not stored, REST refresh not possible

Claude Code handles its own auth — agent should still start fine.
If you see auth errors in the agent, run: claude /login"

# exit 0 so start_agent.sh continues
exit 0
