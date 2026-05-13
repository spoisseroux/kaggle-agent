#!/usr/bin/env python3
"""Session Start Hook - Query research DB for relevant context

Automatically runs at session start to inject relevant learnings from past work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.hypothesis_db import HypothesisDatabase

def main():
    """Query research DB and prepare context for new session"""

    # Get active competition from registry
    import json
    registry_path = Path(__file__).parent.parent.parent / "competitions/registry.json"
    with open(registry_path) as f:
        registry = json.load(f)

    active_comp = registry.get("active")
    if not active_comp:
        return  # No active competition

    db = HypothesisDatabase()

    # Get top insights for this competition
    insights = db.get_competition_insights(active_comp, min_confidence=0.7)

    # Get recent experiment history
    history = db.get_experiment_history(active_comp, limit=5)

    # Create context summary
    context_lines = []
    context_lines.append(f"📚 Research DB Context for {active_comp}:\n")

    if insights:
        context_lines.append(f"High-confidence insights ({len(insights)}):")
        for i, insight in enumerate(insights[:3], 1):
            context_lines.append(f"  {i}. {insight['insight'][:120]}...")
            context_lines.append(f"     Confidence: {insight['confidence']:.0%}, Category: {insight['category']}")
        context_lines.append("")

    if history:
        context_lines.append(f"Recent experiments:")
        for exp in history:
            succeeded = "✓" if exp.get('succeeded') else "✗"
            lb_score = exp.get('lb_score', 'N/A')
            context_lines.append(f"  {succeeded} {exp['experiment_id']}: CV {exp['cv_score']:.4f}, LB {lb_score}")
        context_lines.append("")

    context_lines.append("Query DB with: python -c \"from core.hypothesis_db import HypothesisDatabase; db=HypothesisDatabase(); ...\"")

    # Write to temp file for Claude to read
    context_file = Path("/tmp/session_context.txt")
    context_file.write_text("\n".join(context_lines))

    print("\n".join(context_lines))

if __name__ == "__main__":
    main()
