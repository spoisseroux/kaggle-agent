"""Multi-agent system for AutoKaggle-style workflow.

Specialized agents:
- Reader: Parse competition docs into structured summaries
- Planner: Decompose phases into executable tasks
- Developer: Implement code with iterative debugging
- Reviewer: Validate outputs and provide feedback
- Summarizer: Document phase execution
"""
from core.agents.reader import read_competition
from core.agents.planner import plan_phase
from core.agents.developer import develop_task
from core.agents.reviewer import review_code
from core.agents.summarizer import summarize_phase

__all__ = [
    "read_competition",
    "plan_phase",
    "develop_task",
    "review_code",
    "summarize_phase",
]
