# Kaggle Agent — Complete PRD
# Single source of truth — no other files needed

**Hardware:** RTX 5070 12GB · 64GB RAM · Ryzen 5 7600X · WSL2/Ubuntu 24.04
**Stack:** Claude Code · PyTorch/CUDA · Ollama (Qwen3-14B) · Tailscale
         Telegram Bot · MLflow · FastAPI · Cloudflare Tunnel · HuggingFace
**Memory:** Tailscale VM named 'docker' (Postgres + Qdrant + MCP server)
**Integration:** agency-platform (Next.js + FastAPI + Postgres, Docker)

---

## 1. Confirmed System State — Do Not Reinstall These

```
WSL2:          Ubuntu 24.04, kernel 6.6.87.2-microsoft-standard-WSL2
Username:      keehar
Hostname:      DESKTOP-82IE70B
Home:          /home/keehar
Tailscale:     this machine name = 'kaggle'
               memory stack machine name = 'docker'

GPU:           NVIDIA GeForce RTX 5070, 12,227MB VRAM
CUDA:          13.2, Driver 596.21
nvidia-smi:    ✓ verified working

Python:        3.12.13 via uv
Venv:          /home/keehar/kaggle-venv
Activate:      source /home/keehar/kaggle-venv/bin/activate

PyTorch:       2.11.0+cu128 nightly — sm_120 verified, no warnings
               Reinstall cmd if ever needed:
               uv pip install --pre torch torchvision torchaudio \
                 --index-url https://download.pytorch.org/whl/nightly/cu128 \
                 --extra-index-url https://pypi.org/simple/

Installed packages (already in kaggle-venv — do not reinstall):
  torch, torchvision, torchaudio, lightgbm, xgboost, catboost,
  scikit-learn, pandas, polars, numpy, mlflow, optuna,
  python-telegram-bot, fastapi, uvicorn, jinja2, pynvml,
  python-dotenv, kaggle==2.1.2

Still needed (install during phases below):
  huggingface_hub, datasets, sdv, imbalanced-learn

System packages already installed:
  build-essential, git, curl, wget, tmux, jq, inotify-tools,
  python3-pip, nodejs, npm

Claude Code:   installed, authenticated

Existing files:
  ~/kaggle-agent/          directory exists
  ~/kaggle-agent/.env      exists (Telegram + Kaggle creds already set)
```

### CRITICAL — Kaggle auth
```
KAGGLE_TOKEN env var ONLY. Nothing else works.
kaggle.json → broken with KGAT_ tokens
KAGGLE_KEY  → wrong variable name, does not work
KAGGLE_TOKEN=KGAT_8faa4170883ec5f6b74adef04baffe57 → ✓ verified working

Every script, systemd EnvironmentFile, and shell must export KAGGLE_TOKEN.
Never use kaggle.json or KAGGLE_KEY anywhere in the codebase.
```

---

## 2. Architecture

```
[Tailscale network]
  ai-memory-stack (homelab)
    ├── postgres:5432     ← claude_memory DB
    ├── qdrant:6333       ← vector memory
    └── mcp:8000          ← MCP server (Claude Code connects natively)

[WSL2 — systemd services]
  kaggle-api      port 8765   ← FastAPI bridge to agency platform
  telegram-bot                ← bidirectional comms
  ollama          port 11434  ← Qwen3-14B, VRAM-managed
  mlflow          port 5000   ← experiment tracking
  tailscaled                  ← Tailscale daemon

[WSL2 — tmux session 'kaggle-agent']
  Claude Code agent loop

[Cloudflare Tunnel]
  kaggle.yourdomain.com → localhost:8765

[agency-platform — separate repo]
  pages/kaggle/*    ← 8 sub-services built in Session 2
```

---

## 3. Repository Structure

```
kaggle-agent/
├── CLAUDE.md                        ← agent brain (see §10)
├── .env                             ← secrets (gitignored)
├── .gitignore
├── .claude/
│   ├── mcp_config.json              ← Tailscale MCP server connection
│   └── settings.json                ← hooks config
├── api/
│   └── main.py                      ← FastAPI, port 8765
├── core/
│   ├── telegram_bot.py              ← systemd service
│   ├── message_bus.py               ← SQLite, chat messages
│   ├── ask_human.py                 ← blocking, waits for Telegram/web reply
│   ├── notify.py                    ← fire and forget notification
│   ├── competition_manager.py       ← CLI: new/switch/archive/list
│   ├── gpu_monitor.py               ← pynvml wrapper
│   ├── experiment_tracker.py        ← MLflow wrapper
│   ├── vram_manager.py              ← VRAM allocation daemon
│   ├── memory.py                    ← Postgres + Qdrant read/write
│   ├── ollama_client.py             ← Qwen3-14B with think mode
│   └── hf_search.py                 ← HuggingFace dataset search
├── competitions/
│   ├── registry.json
│   ├── active/
│   │   └── <slug>/
│   │       ├── CLAUDE.md
│   │       ├── src/
│   │       │   ├── data_pipeline.py
│   │       │   ├── features.py
│   │       │   ├── model.py
│   │       │   ├── train.py
│   │       │   └── predict.py
│   │       ├── configs/
│   │       ├── submissions/
│   │       │   └── scores.json
│   │       └── notebooks/
│   ├── archived/
│   └── template/                    ← copied for each new competition
├── scripts/
│   ├── wsl_startup.sh
│   ├── start_agent.sh
│   ├── new_competition.sh
│   ├── pre_train.sh                 ← stops Ollama, verifies VRAM free
│   └── post_train.sh                ← restarts Ollama after training
├── scripts/hooks/
│   └── post_bash.py                 ← meaningful event detection
├── systemd/
│   ├── kaggle-api.service
│   ├── telegram-bot.service
│   ├── mlflow.service
│   └── ollama.service
├── docs/
│   ├── README.md
│   ├── setup.md
│   ├── competitions.md
│   ├── data-pipeline.md
│   ├── memory.md
│   ├── vram-manager.md
│   ├── telegram.md
│   ├── api.md
│   ├── web-ui.md
│   ├── hooks.md
│   ├── troubleshooting.md
│   └── changelog.md
├── data/                            ← gitignored
└── mlruns/                          ← gitignored
```

---

## 4. .env (complete)

```bash
# Telegram
TELEGRAM_BOT_TOKEN=8716779851:AAGfA0mJ0DBEX8MOKP1fqbZJ0nGoHc43ByI
TELEGRAM_CHAT_ID=5332262167

# Kaggle — API requires KAGGLE_KEY + KAGGLE_USERNAME
KAGGLE_KEY=KGAT_8faa4170883ec5f6b74adef04baffe57
KAGGLE_USERNAME=spoisseroux

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODEL=claude-sonnet-4-5

# Tailscale memory stack — VM named 'docker' on Tailscale
TAILSCALE_MEMORY_HOST=docker
POSTGRES_DSN=postgresql://claude:${POSTGRES_PASSWORD}@docker:5432/claude_memory
QDRANT_URL=http://docker:6333
MCP_URL=http://docker:8000
MCP_BEARER_TOKEN=
POSTGRES_PASSWORD=

# HuggingFace — fill in after account setup
HF_TOKEN=
HF_HOME=/home/keehar/.cache/huggingface

# Local services
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_PORT=8765
TAILSCALE_HOSTNAME=kaggle
CLOUDFLARE_TUNNEL_URL=https://kaggle.yourdomain.com
ALLOWED_ORIGINS=https://your-agency-platform.com
```

---

## 5. .gitignore

```gitignore
.env
*.env.*
.env.local
data/
mlruns/
mlartifacts/
*.csv
*.parquet
*.h5
*.hdf5
*.pkl
*.pt
*.pth
*.onnx
*.bin
__pycache__/
*.py[cod]
.venv/
kaggle-venv/
*.egg-info/
dist/
build/
.ipynb_checkpoints/
*.ipynb
competitions/*/submissions/*.csv
!competitions/*/submissions/scores.json
*.db
*.sqlite
*.sqlite3
*.log
logs/
*.gguf
.DS_Store
Thumbs.db
```

---

## 6. MCP Config

`.claude/mcp_config.json`:
```json
{
  "mcpServers": {
    "ai-memory": {
      "type": "http",
      "url": "http://docker:8000",
      "headers": {
        "Authorization": "Bearer ${MCP_BEARER_TOKEN}"
      }
    }
  }
}
```

---

## 7. VRAM Manager

RTX 5070 VRAM budget:
```
Total:              12,227 MB
Windows reserve:     1,500 MB (never touch)
Safe WSL2 budget:   10,700 MB
Ollama (Qwen3-14B):  9,200 MB
Training minimum:    9,500 MB free before starting
```

`core/vram_manager.py`:
```python
import pynvml, subprocess, time, logging

TRAINING_MIN_FREE_MB = 9500

def get_free_vram_mb() -> int:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free // (1024 * 1024)

def ollama_is_loaded() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/ps", timeout=3)
        return len(r.json().get("models", [])) > 0
    except:
        return False

def request_training_vram(timeout_s: int = 120) -> bool:
    """Call before any GPU training. Stops Ollama, waits for VRAM to free."""
    if ollama_is_loaded():
        logging.info("Stopping Ollama to free VRAM...")
        subprocess.run(["sudo", "systemctl", "stop", "ollama"], check=True)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            free = get_free_vram_mb()
            if free >= TRAINING_MIN_FREE_MB:
                logging.info(f"VRAM freed: {free}MB available.")
                return True
            time.sleep(2)
        return False
    return get_free_vram_mb() >= TRAINING_MIN_FREE_MB

def release_training_vram():
    """Call after training. Restarts Ollama."""
    deadline = time.time() + 30
    while time.time() < deadline:
        if get_free_vram_mb() > 10000:
            break
        time.sleep(2)
    subprocess.run(["sudo", "systemctl", "start", "ollama"], check=True)
    logging.info("Ollama restarted.")
```

Every training script:
```python
from core.vram_manager import request_training_vram, release_training_vram

if not request_training_vram():
    raise RuntimeError("Could not acquire VRAM. Check GPU usage.")
try:
    train_model(config)
finally:
    release_training_vram()
```

---

## 8. Data Pipeline

Four stages run before any model training. Never train on raw data.

```
Stage 1: Refine    ← clean, fix errors, handle outliers
Stage 2: Enrich    ← merge HuggingFace external datasets
Stage 3: Synthetic ← generate additional samples (SMOTE, SDV)
Stage 4: Fuzz      ← add controlled noise to prevent overfitting
                      numeric: Gaussian noise sigma=0.01*std
                      categorical: swap 2% of values randomly
                      NEVER fuzz the target column
                      NEVER fuzz the test set
```

Each stage saved as versioned parquet and logged to `kaggle_datasets` table.

`core/hf_search.py` — search HuggingFace for external datasets during Stage 2:
```python
from huggingface_hub import HfApi
import os

def search_relevant_datasets(query: str, tags: list[str] = []) -> list:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    results = api.list_datasets(search=query, tags=tags,
                                sort="downloads", limit=10)
    return [{"id": d.id, "downloads": d.downloads, "tags": d.tags}
            for d in results]
```

---

## 9. Memory System (Tailscale Stack)

Uses existing homelab `ai-memory-stack` — no new containers in WSL2.

Kaggle tables to add via migration against existing `claude_memory` DB:

```sql
CREATE TABLE IF NOT EXISTS kaggle_competitions (
  id            SERIAL PRIMARY KEY,
  slug          TEXT UNIQUE NOT NULL,
  name          TEXT,
  metric        TEXT,
  higher_better BOOLEAN DEFAULT TRUE,
  deadline      TIMESTAMP,
  status        TEXT DEFAULT 'active',
  best_cv       FLOAT,
  best_lb       FLOAT,
  lb_rank       INTEGER,
  total_teams   INTEGER,
  submissions_used INTEGER DEFAULT 0,
  submissions_max  INTEGER,
  created_at    TIMESTAMP DEFAULT NOW(),
  updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_experiments (
  id              SERIAL PRIMARY KEY,
  competition_slug TEXT REFERENCES kaggle_competitions(slug),
  mlflow_run_id   TEXT,
  model_type      TEXT,
  feature_set     TEXT,
  data_version    TEXT,
  config          JSONB,
  cv_mean         FLOAT,
  cv_std          FLOAT,
  lb_score        FLOAT,
  training_time_s INTEGER,
  notes           TEXT,
  created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_features (
  id              SERIAL PRIMARY KEY,
  name            TEXT,
  competition_slug TEXT,
  code_snippet    TEXT,
  cv_delta        FLOAT,
  description     TEXT,
  tags            TEXT[],
  created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_submissions (
  id              SERIAL PRIMARY KEY,
  competition_slug TEXT REFERENCES kaggle_competitions(slug),
  filename        TEXT,
  cv_score        FLOAT,
  lb_score        FLOAT,
  lb_rank         INTEGER,
  total_teams     INTEGER,
  percentile      FLOAT,
  notes           TEXT,
  submitted_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kaggle_datasets (
  id              SERIAL PRIMARY KEY,
  competition_slug TEXT,
  version         TEXT,
  row_count       INTEGER,
  col_count       INTEGER,
  description     TEXT,
  hf_datasets_used TEXT[],
  created_at      TIMESTAMP DEFAULT NOW()
);
```

Qdrant collections (create on startup):
```python
collections = {
    "kaggle_experiments": 1536,  # find similar past approaches
    "kaggle_features":    1536,  # retrieve relevant feature code
    "kaggle_errors":      1536,  # find how similar errors were fixed
    "kaggle_insights":    1536,  # general wisdom across competitions
}
```

---

## 10. API Endpoints

```
GET  /health
     → { status, services: {postgres, qdrant, mlflow, ollama, telegram},
         active_competition, gpu, last_updated }

GET  /system/ssh-info
     → { tailscale_hostname: "kaggle",
         ssh_command: "ssh keehar@kaggle",
         tmux_command: "ssh keehar@kaggle -t tmux attach -t kaggle-agent",
         tailscale_connected: bool }

GET  /competitions
GET  /competitions/active
GET  /competitions/active/trajectory
     → { slug, metric, deadline_days, best_cv, best_lb, lb_rank,
         total_teams, percentile, cv_history, lb_history,
         cv_trend_slope, projected_percentile,
         submissions_used, submissions_max }
POST /competitions/new      { slug, metric, deadline }
POST /competitions/switch   { slug }

GET  /experiments?competition=<slug>&limit=50
GET  /submissions?competition=<slug>
GET  /leaderboard/<slug>
GET  /logs?lines=200
GET  /gpu/stream             SSE, 3s interval
GET  /chat/messages?limit=50
POST /chat/send              { text }
WS   /chat/ws
GET  /ollama/status
```

---

## 11. Web UI — Agency Platform

### Sidebar structure
```
▼ Kaggle    ← collapsible, status dot: green/yellow/red
    Dashboard
    Chat
    Experiments
    GPU
    Competitions
    Logs
    Submissions
    Leaderboard
```

### Offline states
- **Online** — green dot, all sub-tools accessible
- **Degraded** — yellow dot (e.g. Ollama stopped during training — expected)
- **Offline** — red dot, all sub-tools greyed + non-clickable
  Banner: "● Kaggle agent offline · Last seen X minutes ago"

Health check: `GET /health` every 10s, 3s timeout.
WebSocket disconnect → immediately sets offline without waiting for poll.

### Dashboard page
- Active competition name, metric, deadline countdown
- Best CV + best LB + LB rank + percentile
- CV over time chart (last 20 experiments)
- LB rank over time chart (submissions)
- Projected final percentile based on trend
- Submissions used / max
- SSH connect section:
  ```
  ssh keehar@kaggle                                    [Copy]
  ssh keehar@kaggle -t tmux attach -t kaggle-agent     [Copy]
  Tailscale: ● Connected
  ```

### Chat page
- WebSocket bubbles: agent messages left (gray), your messages right
- Source icon: phone = Telegram, monitor = web UI
- Unread badge on sidebar Chat item
- Send on Enter or button
- Auto-scroll to latest

### GPU page
- SSE live stats every 3s
- Utilization gauge, VRAM bar, temp with color (green/yellow/red)
- Current process if training
- Ollama status

### All other pages
- Experiments: sortable MLflow table, click to expand config
- Competitions: registry table + New Competition button
- Logs: last 200 lines, auto-refresh toggle, search
- Submissions: history with CV and LB scores
- Leaderboard: your rank + top 10 + delta from last check

### Env var
```bash
NEXT_PUBLIC_KAGGLE_API_URL=https://kaggle.yourdomain.com
```

---

## 12. Claude Code Hooks

`.claude/settings.json`:
```json
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python core/notify.py 'Agent paused — waiting for input.'"
      }]
    }],
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "python scripts/hooks/post_bash.py"
      }]
    }]
  }
}
```

`scripts/hooks/post_bash.py` — fires on meaningful events only:
```python
import sys, json, os, re

event = json.loads(sys.stdin.read())
output = event.get("output", "")
cmd = event.get("tool_input", {}).get("command", "")

if re.search(r"CV[:\s]+([0-9.]+)", output) and "train.py" in cmd:
    cv = re.search(r"CV[:\s]+([0-9.]+)", output).group(1)
    os.system(f'python core/notify.py "Training complete. CV: {cv}"')
elif "CUDA out of memory" in output:
    os.system('python core/notify.py "OOM error during training. Check VRAM."')
elif "Successfully submitted" in output:
    os.system('python core/notify.py "Submission confirmed. Waiting for LB score..."')
elif "Fold 1/" in output or "Epoch 1/" in output:
    os.system('python core/notify.py "Training started."')
```

---

## 13. Master CLAUDE.md

```markdown
# Kaggle Agent — Master Instructions

## Identity
You are an autonomous Kaggle competition agent running on a local RTX 5070
machine (WSL2). You run in a persistent tmux session. Your goal is to make
competition progress with minimal human interruption.

## Active competition
- Slug: {{ACTIVE_SLUG}}
- Metric: {{METRIC}} (higher is better: {{HIGHER_IS_BETTER}})
- Deadline: {{DEADLINE}}
- Best CV: {{BEST_CV}} | Best LB: {{BEST_LB}}
- Path: competitions/active/{{ACTIVE_SLUG}}/

## Workflow (always follow this order)
1. EDA — understand data, target distribution, missing values, feature types
2. Data pipeline — run all 4 stages: refine → enrich → synthetic → fuzz
3. Baseline — working end-to-end pipeline with a simple model
4. Feature engineering — iterate, log everything to MLflow
5. Model selection — LightGBM, XGBoost, CatBoost for tabular
6. Tuning — Optuna for hyperparameters
7. Ensembling — blend top-N by CV score
8. Submit — only with explicit human approval

## Local LLM routing (cost optimisation)
Use Ollama (qwen3:14b via core/ollama_client.py) for:
- Boilerplate training loops, CV scaffolding, sklearn pipelines
- Feature engineering implementation (once decided)
- Summarising experiment results in one sentence
- Writing docstrings and simple validation code
- Use think=True for complex feature pipelines or debugging

Use your own reasoning for:
- Deciding which approach to try next
- Interpreting unexpected results
- Architecture decisions
- Anything going to ask_human.py

## Memory usage
At start of each competition or session, query memory for relevant context:
- Search kaggle_experiments: similar past competition configs
- Search kaggle_features: relevant feature engineering that worked before
- Search kaggle_errors: how similar errors were resolved

Store after each experiment:
- Log to MLflow + kaggle_experiments table
- Embed and store in Qdrant if CV delta is meaningful

## What you do autonomously
- Write, run, modify Python code
- Run data pipeline stages
- Create experiment configs and run training
- Log all experiments to MLflow
- Send progress: python core/notify.py "message"

## When to use ask_human.py
Call: python core/ask_human.py "Your question here"
- Before ANY Kaggle leaderboard submission
- CV drops >2% unexpectedly after a change
- Two approaches tied, >1h compute to evaluate
- Stuck with no CV improvement for 3+ experiments
- Data quality issue that could invalidate all experiments
- Any action that cannot be undone

## VRAM — always use vram_manager
Before any GPU training:
  from core.vram_manager import request_training_vram, release_training_vram
  request_training_vram()  — stops Ollama, waits for VRAM to free
After training:
  release_training_vram()  — restarts Ollama
Never skip this. OOM mid-training wastes compute time.

## Documentation (mandatory)
When you create a file: add a section to the relevant docs/ file.
When you change how something works: update the relevant docs/ file.
When you fix a bug that could recur: add to docs/troubleshooting.md.
Always append to docs/changelog.md with today's date and what changed.

## Rules
- Always log to MLflow before and after training
- Never delete submissions/ contents
- Save YAML config before every experiment
- Data lives in data/<slug>/ — never commit it
- Training >30min → send notify at start
- KAGGLE_KEY + KAGGLE_USERNAME env vars only — never use kaggle.json
```

---

## 14. GitHub Auto-Commit Setup

Set up BEFORE starting the build so Claude Code commits after each phase.

```bash
# Configure git identity in WSL2
git config --global user.name "keehar"
git config --global user.email "your@email.com"

# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list
sudo apt update && sudo apt install -y gh

# Authenticate
gh auth login   # follow prompts, choose SSH

# Create private repo and push
cd ~/kaggle-agent
git init
git add .gitignore CLAUDE.md kaggle-agent-PRD-final.md
git commit -m "initial: PRD and gitignore"
gh repo create kaggle-agent --private --source=. --push
```

### Auto-commit instructions in CLAUDE.md (add to §13)

```markdown
## Git commits
After completing each phase, commit all new/changed files:
  git add -A
  git commit -m "phase <N> complete: <one line summary of what was built>"
  git push

After fixing a bug:
  git commit -am "fix: <what was broken and how it was fixed>"
  git push

Never commit: .env, data/, mlruns/, *.csv, *.parquet, *.pt, *.pkl
These are in .gitignore — leave them there.
```

---

## 15. Self-Healing via Telegram and Web Chat

The agent monitors the message bus for instructions, not just yes/no replies.
You can send any instruction from Telegram or the web chat and the agent
will pick it up and act on it — this is how you fix, extend, or redirect
the agent without touching the terminal.

### How it works

`core/ask_human.py` already blocks and waits for a reply.
The agent also checks the inbox at the top of its main loop for
unprompted instructions from you. Add this pattern to `CLAUDE.md`:

```markdown
## Receiving instructions via chat

At the start of each new task cycle, check for pending messages:
  from core.message_bus import get_pending_instructions
  instructions = get_pending_instructions()
  if instructions:
      act on them before continuing current work

This means you can be redirected mid-run via Telegram or web chat.
Examples of instructions you might receive:
  "stop training and try a different feature set"
  "the MLflow connection is broken, fix it"
  "switch to the titanic competition"
  "the Qdrant service on docker is down, work without it for now"
  "rebuild the telegram bot service, it crashed"
```

`core/message_bus.py` needs a `get_pending_instructions()` function that
returns messages from humans that arrived without the agent asking — i.e.
unsolicited messages in the inbox. The agent polls this at the top of each
major loop iteration.

### What this means practically

- You can **fix broken things** by sending "the kaggle API is returning 401,
  regenerate the token handling" in Telegram
- You can **change direction** mid-competition by sending "stop current
  experiment and try CatBoost instead"
- You can **extend the system** by sending "add a new endpoint to the API
  that returns the top 5 features by importance"
- The agent will pick it up, act on it, commit the change, and notify you

---

## 16. Auto-Start on Windows Boot

Yes — the system starts automatically when you log into Windows.

### How it works
Windows Task Scheduler runs `wsl_startup.sh` at login. This:
1. Starts Tailscale (connects to 'docker' memory stack)
2. Starts all systemd services (telegram-bot, kaggle-api, mlflow, ollama)
3. Recreates the tmux 'kaggle-agent' session if it doesn't exist

### Important detail
"At log on" = when you sign into Windows, not just power on.
For most people these are the same thing. If you use auto-login or
want it to start before you log in, change the trigger to "At startup"
and check "Run whether user is logged on or not" (requires admin password).

### If something doesn't start
SSH in from your phone: `ssh keehar@kaggle`
Then: `tmux attach -t kaggle-agent` to see the agent
Or check services: `sudo systemctl status kaggle-api telegram-bot`

---

## 17. Memory Stack Monitoring (Web UI)

Add a 9th sub-service to the Kaggle sidebar: **Memory**.

```
▼ Kaggle
    Dashboard
    Chat
    Experiments
    GPU
    Competitions
    Logs
    Submissions
    Leaderboard
    Memory          ← new
```

### What it shows

```
┌─────────────────────────────────────────────────────────────┐
│  Memory Stack (docker)                                       │
├──────────────────┬──────────────────┬────────────────────── │
│  Postgres        │  Qdrant          │  MCP Server           │
│  ● Online        │  ● Online        │  ● Online             │
│  docker:5432     │  docker:6333     │  docker:8000          │
│  claude_memory   │  4 collections   │  v1.2.0               │
│  12 kaggle tables│  48k vectors     │  last call: 2m ago    │
├──────────────────┴──────────────────┴────────────────────── │
│  Tailscale connection to 'docker': ● Connected              │
│  Round-trip latency: 4ms                                     │
├─────────────────────────────────────────────────────────────┤
│  Collections                                                 │
│  kaggle_experiments   12,430 vectors                        │
│  kaggle_features       3,891 vectors                        │
│  kaggle_errors           234 vectors                        │
│  kaggle_insights         891 vectors                        │
├─────────────────────────────────────────────────────────────┤
│  Recent memory activity                                      │
│  14:23  stored experiment lgbm_v7 (cv: 0.8821)             │
│  14:21  searched: "catboost tabular classification"         │
│  14:18  stored feature: price_per_sqft (delta: +0.003)     │
└─────────────────────────────────────────────────────────────┘
```

### Alert behaviour
If any of Postgres, Qdrant, or MCP goes offline:
- Red dot on Memory sidebar item
- Alert banner at top of any Kaggle page:
  ```
  ⚠ Memory stack offline — agent running without memory.
    Experiments will not be stored or retrieved.
    Check that 'docker' VM is running on Tailscale.
  ```
- Agent notified via message_bus so it knows to work without memory

### API endpoint
```
GET /memory/status
→ {
    postgres:  { status, host, db, tables_count, latency_ms },
    qdrant:    { status, host, collections: [{name, vectors_count}] },
    mcp:       { status, host, version, last_call },
    tailscale: { connected, latency_ms }
  }
```

---

## 18. Handoff Document (generated at end of Phase 5)

At the end of Phase 5, Claude Code generates `HANDOFF.md` in the repo root.
This is what you pass to the web UI session — it captures actual built state,
not just planned state.

Add to Phase 5 instructions:
```
Final step of Phase 5: generate HANDOFF.md containing:
- Actual API base URL (Cloudflare Tunnel URL)
- All working endpoint URLs with example responses
- WebSocket and SSE endpoint URLs
- Any deviations from the PRD (what was changed and why)
- Environment variables the frontend needs
- Current /health response (paste actual output)
- SSH commands (confirmed working)
- Any known issues or TODOs
```

Web UI session prompt uses `HANDOFF.md` instead of the full PRD:
```bash
cd ~/agency-platform
claude --dangerously-skip-permissions \
  --context-window-compaction-threshold 0.6 \
  "Read ~/kaggle-agent/HANDOFF.md for the exact API spec.
   Read ~/kaggle-agent/kaggle-agent-PRD-final.md sections 11 and 17
   for the UI spec including the Memory monitoring page.
   Build the Kaggle UI module. Start with useKaggleHealth hook.
   Touch only pages/kaggle/*, sidebar nav, .env.example."
```



`scripts/wsl_startup.sh`:
```bash
#!/bin/bash
source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent

sudo tailscale up --ssh --accept-routes --accept-dns=false
sudo systemctl start telegram-bot kaggle-api mlflow ollama

tmux has-session -t kaggle-agent 2>/dev/null || \
  tmux new-session -d -s kaggle-agent -c /home/keehar/kaggle-agent \
  "bash scripts/start_agent.sh"
```

`scripts/start_agent.sh`:
```bash
#!/bin/bash
source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent

ACTIVE_SLUG=$(python -c "
import json
try:
    r = json.load(open('competitions/registry.json'))
    print(r.get('active', 'none'))
except:
    print('none')
")

claude \
  --dangerously-skip-permissions \
  --context-window-compaction-threshold 0.6 \
  --mcp-config .claude/mcp_config.json \
  "You are the Kaggle agent. Active competition: $ACTIVE_SLUG.
   Read CLAUDE.md for full instructions.
   Check memory MCP for context from last session.
   Resume from where you left off."
```

Windows Task Scheduler:
```
Program:   C:\Windows\System32\wsl.exe
Arguments: -d Ubuntu-24.04 -- bash /home/keehar/kaggle-agent/scripts/wsl_startup.sh
Trigger:   At log on
```

---

## 15. Build Phases

### Phase 1 — Connectivity + message bus
- GitHub repo already created (done manually before this session)
- Install Tailscale, confirm `ping docker` works
- Confirm `curl http://docker:8000/health` reaches MCP server
- Create `.claude/mcp_config.json`
- Run DB migration (§9 SQL) against claude_memory on docker:5432
- `core/message_bus.py` — SQLite for chat + `get_pending_instructions()`
- `core/telegram_bot.py` — systemd service
- `core/ask_human.py` — blocking, polls message_bus
- `core/notify.py` — fire and forget
- `systemd/telegram-bot.service`
- Install: `uv pip install huggingface_hub python-telegram-bot[job-queue]`
- **Verify:** Telegram roundtrip works, `curl http://docker:8000/health` OK
- `git commit -m "phase 1 complete: connectivity + message bus"` and push

### Phase 2 — VRAM manager + Ollama
- Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
- Pull model: `ollama pull qwen3:14b`
- `core/vram_manager.py` (§7)
- `core/ollama_client.py` with think mode
- `systemd/ollama.service`, enable on startup
- **Verify:** `request_training_vram()` stops Ollama and confirms VRAM freed

### Phase 3 — Data pipeline + HuggingFace
- `uv pip install datasets sdv imbalanced-learn`
- `core/hf_search.py`
- `competitions/template/src/data_pipeline.py` — all 4 stages
- `scripts/pre_train.sh` and `scripts/post_train.sh`

### Phase 4 — Competition management + MLflow + API
- `core/competition_manager.py` — new/switch/archive/list/status
- `core/experiment_tracker.py` — MLflow wrapper
- `core/memory.py` — Postgres + Qdrant read/write
- `core/gpu_monitor.py` — pynvml wrapper
- MLflow local service + `systemd/mlflow.service`
- `api/main.py` — all endpoints from §10 including `/health` and `/system/ssh-info`
- `systemd/kaggle-api.service`
- Cloudflare Tunnel setup
- **Verify:** `curl localhost:8765/health` returns all services green

### Phase 5 — Hooks + startup + docs + handoff
- `.claude/settings.json` hooks (§12)
- `scripts/hooks/post_bash.py`
- `scripts/start_agent.sh` with compaction flag
- `scripts/wsl_startup.sh`
- Master `CLAUDE.md` populated from §13, including git commit instructions
  and get_pending_instructions() loop from §15
- Windows Task Scheduler instructions written to `docs/setup.md`
- All `docs/` files created and populated
- SSH confirmed: `sudo tailscale up --ssh`
- `GET /memory/status` endpoint added to api/main.py (§17)
- **Verify:** `curl localhost:8765/health` all green
- **Verify:** `curl localhost:8765/memory/status` shows docker services
- **Verify:** SSH from another device works
- **Verify:** send unsolicited Telegram message, confirm agent receives it
- Generate `HANDOFF.md` in repo root (§18)
- `git commit -m "phase 5 complete: hooks + startup + docs + handoff"` and push

---

## 20. Session 2 — Agency Platform Web UI

Run AFTER Phase 5 is verified and `HANDOFF.md` exists in `~/kaggle-agent/`.
Open a NEW Claude Code session in the agency-platform repo.

```bash
cd ~/agency-platform
claude --dangerously-skip-permissions \
  --context-window-compaction-threshold 0.6 \
  "Read ~/kaggle-agent/HANDOFF.md for the exact API spec and confirmed URLs.
   Read ~/kaggle-agent/kaggle-agent-PRD-final.md sections 11 and 17
   for the full UI spec including the Memory monitoring sub-service.
   Build the Kaggle UI module in this order:
   1. useKaggleHealth hook + offline state
   2. Collapsible Kaggle sidebar group with status dot
   3. Dashboard + trajectory charts + SSH connect section
   4. Chat + WebSocket + unread badge
   5. GPU + SSE
   6. Experiments, Competitions, Logs, Submissions, Leaderboard
   7. Memory monitoring page (section 17 of PRD)
   8. pages/kaggle/README.md
   Touch ONLY pages/kaggle/*, sidebar nav, .env.example.
   Do not modify existing routes, DB models, or backend files.
   Proceed automatically between steps. Stop only if unsure about
   an existing pattern in the codebase."
```

---

## 17. Competition Workflow (reference)

### Starting a competition
```
YOU:    Go to kaggle.com/competitions/<slug>
        Click "Join Competition" → accept rules
        Send in chat: "Enrolled in <slug>, deadline <date>, metric <metric>"

AGENT:  competition_manager.py new --slug <slug> ...
        kaggle competitions download -c <slug> -p data/<slug>/
        Unzip, inventory data
        Notify: "Data ready. Starting EDA."
        Begin autonomously
```

### Submission flow
```
AGENT:  ask_human.py "Submit? CV: 0.8934. File: lgbm_v7_blend.csv"
YOU:    Reply "yes"
AGENT:  kaggle competitions submit -c <slug> -f <file> -m "description"
        Notify when LB score available
```

### What you must always do manually
| Action | Where |
|---|---|
| Accept competition rules | kaggle.com |
| Phone verification (first time) | kaggle.com |
| Approve each LB submission | Telegram or web chat |
| Approve ensemble strategy | Telegram or web chat |
