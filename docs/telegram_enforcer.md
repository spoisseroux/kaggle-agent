# Telegram Delivery Enforcer

## Problem
Claude Code assistant sometimes forgets to call `notify.py`, causing responses to only appear in the terminal (invisible to user) instead of being sent to Telegram.

## Solution
Automated background service that monitors the conversation and auto-forwards any assistant responses that weren't sent to Telegram.

## How It Works

1. **Monitors** the conversation JSONL file in real-time
2. **Detects** assistant turns that don't include a `notify.py` tool call
3. **Auto-sends** those messages to Telegram with a `🔔 Auto-forwarded` prefix
4. **Tracks** state to avoid duplicate sends

## Installation

### 1. Test the script
```bash
python scripts/telegram_enforcer.py --check-last
```

### 2. Install as systemd service (runs in background)
```bash
# Copy service file
sudo cp scripts/systemd/telegram-enforcer.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable telegram-enforcer
sudo systemctl start telegram-enforcer

# Check status
sudo systemctl status telegram-enforcer
```

### 3. View logs
```bash
# Real-time logs
sudo journalctl -u telegram-enforcer -f

# Last 100 lines
sudo journalctl -u telegram-enforcer -n 100
```

## Manual Usage

### One-time check
```bash
python scripts/telegram_enforcer.py --check-last
```

### Watch mode (foreground)
```bash
python scripts/telegram_enforcer.py --watch
```

Press `Ctrl+C` to stop.

## Configuration

The enforcer checks every 10 seconds for new messages. State is saved in:
```
.telegram_enforcer_state.json
```

This tracks:
- Last checked message index
- Last checked conversation ID

## Monitoring

Auto-forwarded messages are prefixed with:
```
🔔 Auto-forwarded (no notify.py call):
```

If you see these in Telegram, it means the assistant forgot to call notify.py.

## Disabling

To temporarily stop:
```bash
sudo systemctl stop telegram-enforcer
```

To permanently disable:
```bash
sudo systemctl disable telegram-enforcer
sudo systemctl stop telegram-enforcer
```

## How This Ensures 100% Delivery

1. **Deterministic**: Runs as a system service, always active
2. **Automatic**: No manual intervention needed
3. **Fail-safe**: Even if assistant forgets notify.py, message still gets sent
4. **Traceable**: Auto-forwarded messages are clearly marked
5. **Stateful**: Won't duplicate-send messages

## Limitations

- 10-second delay between assistant response and auto-send
- Very short messages (<10 chars) are skipped
- Requires systemd (Linux)

## Future Improvements

- Make it a Claude Code hook instead of external service
- Reduce latency (currently 10s polling)
- Add metrics (how often it fires)
