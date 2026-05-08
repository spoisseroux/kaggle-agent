#!/bin/bash
# Wait until UTC midnight, then submit ensembles

cd /home/keehar/kaggle-agent

# Calculate seconds until UTC midnight
NOW_UTC=$(date -u +%s)
MIDNIGHT_UTC=$(date -u -d 'tomorrow 00:00:00' +%s)
SLEEP_SECONDS=$((MIDNIGHT_UTC - NOW_UTC))

python core/notify.py "⏰ Auto-submit scheduled

Will submit at UTC midnight
Time until submission: $((SLEEP_SECONDS / 60)) minutes

Sleeping until then..."

# Sleep until midnight
sleep $SLEEP_SECONDS

# Run the submission script
/home/keehar/kaggle-agent/scripts/auto_submit_ensembles.sh
