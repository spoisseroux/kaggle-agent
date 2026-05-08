#!/bin/bash
# Daily competition sweeper
# Fetches active Kaggle competitions and identifies interesting candidates
# Designed to run as a cron job or manual sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
MIN_DAYS_LEFT=7
MAX_DAYS_LEFT=90
MIN_PRIZE_USD=0        # 0 = include "Knowledge" rewards
MAX_PRIZE_USD=100000   # Skip very high-stakes competitions
MIN_TEAMS=50           # Must have some activity
MAX_TEAMS=5000         # Avoid overly popular (might be saturated)

# Notification helper
notify() {
    python core/notify.py "$1" || echo "$1"
}

echo "🔍 Researching new Kaggle competitions..."

# Create temporary Python script to parse competitions
CANDIDATES=$(python3 << 'EOF'
import subprocess
import csv
import json
from io import StringIO
from datetime import datetime

# Fetch all active competitions
result = subprocess.run(
    ["kaggle", "competitions", "list", "--csv"],
    capture_output=True,
    text=True,
    check=True
)

# Parse CSV
reader = csv.DictReader(StringIO(result.stdout))
now = datetime.now()

candidates = []

for row in reader:
    try:
        deadline = datetime.strptime(row["deadline"], "%Y-%m-%d %H:%M:%S")
        days_left = (deadline - now).days

        # Filter by timeline
        if days_left < 7 or days_left > 90:
            continue

        teams = int(row["teamCount"])

        # Filter by team count
        if teams < 50 or teams > 5000:
            continue

        # Parse prize (handle various formats)
        reward = row["reward"]
        prize_usd = 0
        if "Usd" in reward:
            prize_str = reward.replace(",", "").replace(" Usd", "")
            prize_usd = int(prize_str)
        elif reward in ["Knowledge", "Swag"]:
            prize_usd = 0

        # Filter by prize
        if prize_usd > 100000:  # Skip very high stakes
            continue

        # Good candidate!
        slug = row["ref"].split("/")[-1]
        candidates.append({
            "slug": slug,
            "category": row["category"],
            "days_left": days_left,
            "deadline": row["deadline"],
            "teams": teams,
            "reward": reward,
            "prize_usd": prize_usd,
        })

    except (ValueError, KeyError) as e:
        continue

# Sort by days_left (soonest first), then by prize (highest first)
candidates.sort(key=lambda x: (x["days_left"], -x["prize_usd"]))

# Output as JSON
print(json.dumps(candidates, indent=2))
EOF
)

# Check if we found any candidates
CANDIDATE_COUNT=$(echo "$CANDIDATES" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

if [ "$CANDIDATE_COUNT" -eq 0 ]; then
    notify "🔍 **Competition Sweep Complete**

No new interesting competitions found matching criteria:
- 7-90 days remaining
- 50-5000 teams
- Prize ≤ \$100K

Will check again tomorrow."
    exit 0
fi

# Format results for notification
NOTIFICATION=$(python3 << EOF
import sys
import json

candidates = json.loads('''$CANDIDATES''')

msg = f"🔍 **Found {len(candidates)} interesting competition(s)**\n\n"

for i, comp in enumerate(candidates[:5], 1):  # Show top 5
    prize_str = comp["reward"]
    if comp["prize_usd"] > 0:
        prize_str = f"\\\${comp['prize_usd']:,}"

    msg += f"{i}. **{comp['slug']}**\n"
    msg += f"   {comp['category']} | {comp['days_left']} days left\n"
    msg += f"   {comp['teams']:,} teams | Reward: {prize_str}\n\n"

if len(candidates) > 5:
    msg += f"... and {len(candidates) - 5} more\n\n"

msg += "React with competition number to start, or 'research NAME' to investigate."

print(msg)
EOF
)

# Send notification
notify "$NOTIFICATION"

# Save candidates to file for later reference
mkdir -p .claude
echo "$CANDIDATES" > .claude/competition_candidates_$(date +%Y%m%d).json

echo "✓ Competition sweep complete: $CANDIDATE_COUNT candidates found"
exit 0
