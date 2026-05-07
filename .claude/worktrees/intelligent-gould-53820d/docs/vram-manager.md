# VRAM manager

The RTX 5070 has 12,227 MB of VRAM. Windows holds ~1.5 GB; Ollama with
Qwen3-14B (Q4) wants ~9.2 GB, which leaves no headroom for training.
`core/vram_manager.py` arbitrates between the two by stopping the Ollama
service before training starts, then restarting it afterwards.

## Budget
| Slice | Size |
|---|---|
| Total | 12,227 MB |
| Windows reserve | 1,500 MB |
| Safe WSL2 budget | 10,700 MB |
| Ollama (Qwen3-14B) | 9,200 MB |
| **Training minimum free** | **9,500 MB** |

## API
```python
from core.vram_manager import (
    request_training_vram, release_training_vram, get_free_vram_mb,
)

if not request_training_vram(timeout_s=120):
    raise RuntimeError("could not free VRAM for training")
try:
    train_model(config)
finally:
    release_training_vram()
```

The same logic is wrapped by `scripts/pre_train.sh` and
`scripts/post_train.sh` for shell use.

## How it stops Ollama
`request_training_vram()` calls `sudo systemctl stop ollama`. That works
because `/etc/sudoers.d/kaggle-agent` grants passwordless `systemctl` —
see `setup.md`.

## Failure modes
- If Ollama fails to release VRAM within `timeout_s`,
  `request_training_vram()` returns `False`. Don't start training; ping the
  human via `core/notify.py` and investigate (`nvidia-smi`).
- If `release_training_vram()` can't restart Ollama, the next agent step
  that calls Ollama will see `health()['ok'] == False`. The agent should
  fall back to its own reasoning until Ollama is back.
