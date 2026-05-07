# Telegram

Two flavours of agent ↔ human communication share the SQLite chat bus
(`message_bus.sqlite3`):

- **Agent → human:**
  - `core/notify.py` — fire-and-forget. Used inside hooks and at phase
    boundaries. Telegram delivery errors are swallowed.
  - `core/ask_human.py` — blocking. Waits for the next human reply (from
    Telegram or web) and returns it on stdout.
- **Human → agent:**
  - `core/telegram_bot.py` runs as a systemd service, long-polls Telegram,
    and writes inbound messages to the bus with `status='new'`.
  - The agent's loop calls `core.message_bus.get_pending_instructions()`
    at the top of each iteration to consume unsolicited redirects.
  - The web UI POSTs to `/chat/send` and reads via WebSocket
    (`/chat/ws`); both directions persist in the same bus.

## Setting it up
The bot token and chat ID are in `.env`. The systemd service is under
`systemd/telegram-bot.service`; install per `setup.md`.

## Re-using ask_human from a script
```bash
answer=$(python core/ask_human.py "Submit lgbm_v7? CV=0.8821" --timeout 600)
case "$answer" in
  yes|y|Y) ... ;;
  *)       echo "skipping submission" ;;
esac
```

`--timeout` is in seconds; omit for an unbounded wait.
