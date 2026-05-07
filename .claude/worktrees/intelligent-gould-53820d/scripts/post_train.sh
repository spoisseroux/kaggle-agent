#!/usr/bin/env bash
# Post-training hook: restart Ollama once GPU has settled.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/keehar/kaggle-venv/bin/activate

python - <<'PY'
import sys
from core.vram_manager import release_training_vram, get_free_vram_mb
ok = release_training_vram(timeout_s=30)
if not ok:
    print(f"FAILED to restart Ollama (free={get_free_vram_mb()}MB)")
    sys.exit(1)
print(f"Ollama restarted (free={get_free_vram_mb()}MB)")
PY
