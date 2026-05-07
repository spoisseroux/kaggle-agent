#!/usr/bin/env bash
set -euo pipefail

source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent

# load .env into the shell
set -a
[ -f .env ] && source .env
set +a

ACTIVE_SLUG=$(python - <<'PY'
import json, pathlib
try:
    p = pathlib.Path('competitions/registry.json')
    print(json.loads(p.read_text()).get('active') or 'none')
except Exception:
    print('none')
PY
)

# Read the Claude model from .claude/kaggle_settings.json (set via tray or web UI)
MODEL=$(python - <<'PY'
import json, pathlib
try:
    p = pathlib.Path('.claude/kaggle_settings.json')
    m = json.loads(p.read_text()).get('model', '')
    print(m)
except Exception:
    print('')
PY
)

MODEL_FLAG=()
if [ -n "$MODEL" ]; then
    MODEL_FLAG=(--model "$MODEL")
fi

exec claude \
  "${MODEL_FLAG[@]}" \
  --dangerously-skip-permissions \
  --mcp-config .claude/mcp_config.json \
  -- \
  "You are the Kaggle agent. Active competition: $ACTIVE_SLUG.
   Read CLAUDE.md for full instructions.
   Check the ai-memory MCP for context from last session.
   Resume from where you left off."
