# Setup

## Prereqs (already in place)
- WSL2 Ubuntu 24.04, Python 3.12 in `/home/keehar/kaggle-venv`
- PyTorch 2.11 nightly + CUDA 13.2 (sm_120 verified)
- Tailscale, with this host named `kaggle` and the memory VM `docker`
- `gh` (GitHub CLI), authenticated with SSH

## One-time sudoers setup
`vram_manager` and `wsl_startup.sh` need passwordless sudo for a small set
of commands. Run once:

```bash
echo 'keehar ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/systemctl, /usr/bin/cp, /usr/bin/dd, /usr/bin/tee, /usr/bin/apt, /usr/bin/apt-get, /bin/mkdir, /usr/bin/mkdir, /usr/sbin/service, /usr/bin/tailscale' \
  | sudo tee /etc/sudoers.d/kaggle-agent
sudo chmod 440 /etc/sudoers.d/kaggle-agent
```

## Install systemd services
The unit files live in `systemd/` and are installed once by copying them
into `/etc/systemd/system/`:

```bash
for u in telegram-bot.service ollama.service mlflow.service kaggle-api.service; do
  sudo cp systemd/$u /etc/systemd/system/$u
done
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot ollama mlflow kaggle-api
sudo systemctl start  telegram-bot ollama mlflow kaggle-api
```

Verify:
```bash
sudo systemctl is-active telegram-bot ollama mlflow kaggle-api
curl -sS localhost:8765/health | jq
```

## Cloudflare Tunnel
The agency-platform web UI talks to this box via a Cloudflare Tunnel that
maps `kaggle.<your-domain>` → `localhost:8765`. The repo ships
`cloudflared.deb`; install and configure once:

```bash
sudo dpkg -i cloudflared.deb       # one-time
cloudflared tunnel login            # browser flow once
cloudflared tunnel create kaggle
# point your DNS CNAME at <UUID>.cfargotunnel.com
cloudflared tunnel route dns kaggle kaggle.<your-domain>
cloudflared service install        # installs as a systemd service
```

After that, fill `CLOUDFLARE_TUNNEL_URL` and `ALLOWED_ORIGINS` in `.env`
and restart `kaggle-api` so CORS picks up the new origin.

## Windows Task Scheduler — auto-start at logon
Create a task with these settings:

```
Program:   C:\Windows\System32\wsl.exe
Arguments: -d Ubuntu-24.04 -- bash /home/keehar/kaggle-agent/scripts/wsl_startup.sh
Trigger:   At log on
```

For "before logon" semantics, change the trigger to **At startup** and check
**Run whether user is logged on or not** (admin password required).

## Recovering the agent remotely
- SSH from your phone via Tailscale: `ssh keehar@kaggle`
- Attach to the running session: `tmux attach -t kaggle-agent`
- Or check service status: `sudo systemctl status kaggle-api telegram-bot`
