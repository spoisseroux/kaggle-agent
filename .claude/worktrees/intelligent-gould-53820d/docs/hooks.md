# Claude Code hooks

`.claude/settings.json` wires two hooks:

- **Stop** — fires once when the agent stops (waiting for a human reply or
  paused). Sends a Telegram notification via `core/notify.py`.
- **PostToolUse / Bash** — fires after every Bash tool call. The hook
  script `scripts/hooks/post_bash.py` inspects the captured stdout for
  meaningful events:

| Pattern | Notification |
|---|---|
| `CV: <num>` after `train.py` | `Training complete. CV: <num>` |
| `CUDA out of memory` | `OOM error during training. Check VRAM.` |
| `Successfully submitted` | `Submission confirmed. Waiting for LB score…` |
| `Fold 1/` or `Epoch 1/` | `Training started.` |

Hooks should be cheap: the post-bash script does string matching only and
returns within milliseconds. Adding a slow check here will throttle every
shell command the agent runs.
