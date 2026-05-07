#!/bin/bash
# Research a Kaggle competition before starting work
# Usage: ./scripts/research_competition.sh <competition-slug>

set -e

COMPETITION="$1"

if [ -z "$COMPETITION" ]; then
    echo "Usage: $0 <competition-slug>"
    echo "Example: $0 titanic"
    exit 1
fi

echo "=========================================="
echo "RESEARCHING: $COMPETITION"
echo "=========================================="

# 1. Check leaderboard distribution
echo ""
echo "1. LEADERBOARD ANALYSIS"
echo "----------------------------------------"
echo "Top 20 scores:"
kaggle competitions leaderboard "$COMPETITION" --show 2>&1 | head -22

# Count perfect scores
PERFECT_COUNT=$(kaggle competitions leaderboard "$COMPETITION" --show 2>&1 | grep -c "1.00000" || true)
echo ""
echo "Perfect scores (1.0) found: $PERFECT_COUNT"

if [ "$PERFECT_COUNT" -gt 10 ]; then
    echo "⚠️  WARNING: Many perfect scores detected!"
    echo "   This suggests:"
    echo "   - Test set may be small/gameable"
    echo "   - Answers may be leaked/public"
    echo "   - Competition may be tutorial-level"
fi

# 2. Competition details
echo ""
echo "2. COMPETITION DETAILS"
echo "----------------------------------------"
kaggle competitions list | grep -i "$COMPETITION" || echo "No details found"

# 3. Check discussions
echo ""
echo "3. TOP DISCUSSIONS (check manually)"
echo "----------------------------------------"
echo "Run: kaggle competitions discussions $COMPETITION -p 1"
echo "Look for: leaks, tricks, meta strategies"

# 4. Check notebooks
echo ""
echo "4. TOP NOTEBOOKS (check manually)"
echo "----------------------------------------"
echo "Visit: https://www.kaggle.com/competitions/$COMPETITION/code"
echo "Analyze top 3-5 notebooks for winning approaches"

# 5. Web search suggestions
echo ""
echo "5. WEB SEARCH SUGGESTIONS"
echo "----------------------------------------"
echo "Search for:"
echo "  - 'kaggle $COMPETITION perfect score'"
echo "  - 'kaggle $COMPETITION test set leak'"
echo "  - 'kaggle $COMPETITION winning approach'"
echo "  - 'kaggle $COMPETITION trick'"

# 6. Decision framework
echo ""
echo "6. DECISION FRAMEWORK"
echo "----------------------------------------"
echo "RED FLAGS (stop/minimal effort):"
echo "  ✓ Many perfect scores (>10)"
echo "  ✓ Tiny test set (<500 samples)"
echo "  ✓ 'Getting started' or tutorial competition"
echo "  ✓ Very old competition (>2 years)"
echo ""
echo "GREEN FLAGS (worth optimizing):"
echo "  ✓ Active competition with prizes"
echo "  ✓ Large test set (>1000 samples)"
echo "  ✓ Realistic score distribution (no 1.0s)"
echo "  ✓ Recent start date"

echo ""
echo "=========================================="
echo "RESEARCH COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review findings above"
echo "2. Check discussions and notebooks manually"
echo "3. Decide: Worth optimizing or just for learning?"
echo "4. If learning: focus on technique, not leaderboard"
echo "5. If competing: proceed with full technical workflow"
