#!/usr/bin/env bash
# Pre-training hook: stop Ollama and verify VRAM is free.
# Always source the venv before invoking this; train.py should call
# python core/vram_manager.py directly for fine-grained control. This
# wrapper exists for ad-hoc use from the shell.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/keehar/kaggle-venv/bin/activate

python - <<'PY'
import sys
from core.vram_manager import request_training_vram, get_free_vram_mb
ok = request_training_vram(timeout_s=120)
if not ok:
    print(f"FAILED to free VRAM (free={get_free_vram_mb()}MB)")
    sys.exit(1)
print(f"VRAM ready: {get_free_vram_mb()}MB free")
PY
