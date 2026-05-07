#!/usr/bin/env bash
# Convenience wrapper. Usage:
#   bash scripts/new_competition.sh <slug> <metric> <YYYY-MM-DD> [name]
set -euo pipefail
source /home/keehar/kaggle-venv/bin/activate
cd /home/keehar/kaggle-agent
SLUG=${1:?slug required}
METRIC=${2:?metric required}
DEADLINE=${3:?deadline required}
NAME=${4:-$SLUG}
python core/competition_manager.py new \
  --slug "$SLUG" --metric "$METRIC" --deadline "$DEADLINE" --name "$NAME"
