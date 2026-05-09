# Telegram Group Chat Setup

Enable the Kaggle Agent to work in a Telegram group chat with multiple team members.

## Quick Start

### 1. Add Bot to Group

1. Create a new Telegram group or use existing one
2. Add your bot to the group (search for bot username)
3. Make bot an admin (optional but recommended)

### 2. Get Group Chat ID

Run this command to see the group chat ID when someone sends a message:

```bash
# Temporarily run the bot with logging
python core/telegram_bot_group.py
```

Send a test message in the group. You'll see in the logs:
```
telegram_bot WARNING ignoring message from unauthorised chat_id=-123456789
```

The number after `chat_id=` is your group's chat ID (including the minus sign!).

### 3. Get User IDs (Optional - for user whitelist)

To restrict who can use the bot, get user IDs:

1. Have each team member send a message to the group
2. Check logs for: `Unauthorized user @username (123456789)`
3. The number in parentheses is the user ID

**OR** use [@userinfobot](https://t.me/userinfobot) - send `/start` and it replies with your user ID.

### 4. Update Environment Variables

Edit `/home/keehar/kaggle-agent/.env`:

```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=-123456789  # Group chat ID (note the minus!)

# Optional - restrict to specific users (comma-separated)
TELEGRAM_AUTHORIZED_USERS=123456789,987654321

# Optional - require @mention in groups (default: false)
TELEGRAM_REQUIRE_MENTION=true

# Optional - bot username for @mention detection
TELEGRAM_BOT_USERNAME=your_bot_username  # WITHOUT the @
```

### 5. Restart Bot Service

Replace the old bot with the group-enabled version:

```bash
# Stop old bot if running as service
sudo systemctl stop kaggle-telegram-bot

# Run new group-enabled bot
python core/telegram_bot_group.py
```

**OR** update your systemd service to use `telegram_bot_group.py` instead.

## Configuration Options

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | `123456:ABC-DEF1234...` |
| `TELEGRAM_CHAT_ID` | Group chat ID (with minus sign) | `-987654321` |

### Optional

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `TELEGRAM_AUTHORIZED_USERS` | Comma-separated user IDs | `""` (all) | `123456,789012` |
| `TELEGRAM_REQUIRE_MENTION` | Require @mention in groups | `false` | `true` |
| `TELEGRAM_BOT_USERNAME` | Bot username for @mention | `""` | `kaggle_agent_bot` |

## Usage in Group Chat

### Without @mention requirement

Just send messages normally:
```
You: "What's the current best CV?"
Bot: "Best CV: 0.345 (xgboost_optuna_v1)"
```

### With @mention requirement

You must @mention the bot:
```
You: "@kaggle_agent_bot what's the status?"
Bot: "System: idle, GPU: 0% utilization..."
```

### Commands

All users in authorized group can use:

- `/status` - Show system and GPU status
- `/help` - Show available commands
- `/queue` - Show pending message count

### User Attribution

Messages are attributed to the sender:

```
You: "Run experiments"
[Internal message bus]: "[your_username] Run experiments"
```

This way the agent knows who asked what, useful for team coordination.

## Features

✅ **Multi-user support** - Multiple team members can interact
✅ **User attribution** - Agent knows who sent each message
✅ **Authorization** - Whitelist specific users
✅ **@mention mode** - Reduce noise in busy groups
✅ **Commands** - /status, /help, /queue
✅ **Backward compatible** - Still works in private 1-on-1 chats

## Migration from Private Chat

If you're currently using private chat and want to switch to group:

1. Keep `TELEGRAM_CHAT_ID` as your private chat ID initially
2. Test the new bot: `python core/telegram_bot_group.py`
3. Verify it still works in private chat
4. Then follow setup steps above to switch to group

**Note**: You can only use ONE chat (private OR group) at a time per bot token. If you want both, create a second bot token.

## Troubleshooting

### "Unauthorized chat" warnings

- Check that `TELEGRAM_CHAT_ID` matches the group ID exactly (including minus sign)
- Group IDs are negative, private chats are positive

### "Unauthorized user" warnings

- Add user ID to `TELEGRAM_AUTHORIZED_USERS`
- Or remove the env var to allow all group members

### Bot doesn't respond in group

- Check `TELEGRAM_REQUIRE_MENTION` - if `true`, you must @mention
- Verify bot is not muted in the group
- Check bot has "Read messages" permission

### Messages from wrong person

- User attribution is shown in brackets: `[username] message`
- This is intentional for team coordination

## Example Group Workflow

**User A**: @kaggle_agent_bot analyze store-sales competition

**Bot**: Analyzing store-sales...
- Best CV: 0.345
- 2 bottlenecks found
- Proposed 3 experiments

**User B**: Which experiment should we try first?

**Bot**: Highest priority: exp_001 - Improve test set lag features (impact: +0.02-0.05 LB)

**User A**: @kaggle_agent_bot run exp_001

**Bot**: Starting experiment exp_001...

**User B**: /status

**Bot**: System: running experiment
GPU: 85% utilization
Estimated completion: 15 minutes

This enables true team collaboration!
