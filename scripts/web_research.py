#!/usr/bin/env python3
"""
Web research script for competitions.

This script is meant to be run BY Claude Code (not by subprocesses),
since only Claude Code has access to the WebSearch tool.

Usage:
    Called by research agent via: python scripts/web_research.py <competition_slug> <problem_type>

    Results saved to: .claude/web_research_{slug}.json
"""
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/web_research.py <competition_slug> <problem_type>")
        sys.exit(1)

    competition_slug = sys.argv[1]
    problem_type = sys.argv[2]

    # Placeholder - actual web search will be done by Claude Code when running this
    # The research agent will call this script and Claude Code will execute it
    # with access to WebSearch tool

    result = {
        "competition_slug": competition_slug,
        "problem_type": problem_type,
        "searched": False,
        "searches_performed": [],
        "insights": {
            "winning_solutions": [],
            "common_approaches": [],
            "known_tricks": [],
            "leaderboard_patterns": [],
            "discussion_highlights": [],
        },
        "sources": [],
    }

    # Save result
    output_path = Path(f".claude/web_research_{competition_slug}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Web research results saved to {output_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
