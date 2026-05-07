# Troubleshooting

## Kaggle CLI auth
- Symptom: `Successfully submitted` never appears, or 401/403 from
  `kaggle competitions ...`.
- Cause: a `KGAT_*` token in `kaggle.json` or under `KAGGLE_KEY` is
  silently rejected.
- Fix: only `KAGGLE_TOKEN=KGAT_...` works. The systemd unit files load
  this from `.env` via `EnvironmentFile=`. The `/leaderboard/{slug}`
  endpoint copies `KAGGLE_TOKEN` into `KAGGLE_KEY` at call time because
  the older Kaggle CLI versions still read `KAGGLE_KEY`.

## VRAM threshold mismatch with PRD
- Symptom: `request_training_vram` returns False with `free=9190MB`.
- Cause: PRD §7 budgets 1.5 GB to Windows; on this box Windows reserves
  closer to 3.0 GB so the post-Ollama-stop free is ~9.2 GB.
- Fix already applied: `TRAINING_MIN_FREE_MB = 9000` in
  `core/vram_manager.py`. Raise it back to 9500 only if Windows reserve
  drops (e.g. close Docker Desktop, disable Tailscale-on-Windows).

## Telegram bot stops ingesting
- `sudo systemctl status telegram-bot` should show active/running.
- If `Restart=on-failure` is hitting too fast, `journalctl -u telegram-bot -n 50`.
- Most common: `TELEGRAM_BOT_TOKEN` rotated → update `.env` and
  `systemctl restart telegram-bot`.

## Memory stack offline
- The agent should keep working without it. Watch for the banner on the
  web UI's memory page. `core.memory.*` returns empty/`{"status":"offline"}`
  on connection error and never raises.
- Check from this machine: `ping docker`, `nc -z docker 5432`,
  `nc -z docker 6333`, `curl http://docker:8000/health`.

## Pynvml deprecation warning
- Cosmetic. Suppress via `PYTHONWARNINGS=ignore::FutureWarning` if it's
  noisy, or upgrade to `nvidia-ml-py` which is API-compatible.

## Cloudflare Tunnel CORS
- After setting `CLOUDFLARE_TUNNEL_URL`, also add the agency-platform
  origin to `ALLOWED_ORIGINS` in `.env` and `systemctl restart kaggle-api`.
- Without that, the browser will block requests with `CORS: missing
  Access-Control-Allow-Origin`.

## Adding a new sudoers entry
The current allowlist:
```
keehar ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/systemctl,
  /usr/bin/cp, /usr/bin/dd, /usr/bin/tee, /usr/bin/apt,
  /usr/bin/apt-get, /bin/mkdir, /usr/bin/mkdir, /usr/sbin/service,
  /usr/bin/tailscale
```
If `vram_manager.py` or `wsl_startup.sh` ever needs another binary, add
it here — and only the absolute path; sudoers requires it.
