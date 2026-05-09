#!/bin/bash
# Auto-submit ensembles at midnight UTC

cd /home/keehar/kaggle-agent

# Send notification
python core/notify.py "🚀 Auto-submit started (UTC midnight)

Submitting 2 ensemble files:
1. ensemble_v1_0.7_v2_0.3.csv
2. ensemble_v1_v2_balanced.csv

This will take ~2 minutes..."

# Submit ensemble 1
echo "Submitting ensemble 70-30..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f competitions/active/store-sales-time-series-forecasting/submissions/ensemble_v1_0.7_v2_0.3.csv \
  -m "Ensemble: 70% v1 + 30% v2 (v1=0.526 LB, v2=0.551 LB)" 2>&1

sleep 5

# Submit ensemble 2
echo "Submitting ensemble 50-50..."
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f competitions/active/store-sales-time-series-forecasting/submissions/ensemble_v1_v2_balanced.csv \
  -m "Ensemble: 50% v1 + 50% v2 balanced blend" 2>&1

# Wait a bit for scores to process
sleep 30

# Check leaderboard
python core/notify.py "✅ Submissions complete!

Checking leaderboard for scores...
Results in ~1 minute"

sleep 30

# Get latest submissions
kaggle competitions submissions -c store-sales-time-series-forecasting | head -10 > /tmp/submissions.txt

# Run analysis script
python scripts/analyze_submission_results.py > /tmp/analysis.txt 2>&1

# Send results
python core/notify.py "📊 Submission Results:

$(cat /tmp/submissions.txt)

$(cat /tmp/analysis.txt)"

echo "Auto-submit complete"
