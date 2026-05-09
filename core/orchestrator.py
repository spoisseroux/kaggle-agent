"""Multi-agent orchestrator for AutoKaggle-style workflow.

Coordinates the 5 specialized agents through a phase-based workflow:
1. Reader → parse competition docs
2. Planner → decompose phase into tasks
3. Developer → implement each task
4. Reviewer → validate outputs
5. Summarizer → document phase execution

Preserves backward compatibility with existing single-agent CLAUDE.md workflow.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.agents.reader import read_competition, save_competition_info
from core.agents.planner import plan_phase
from core.agents.developer import develop_task
from core.agents.reviewer import review_code
from core.agents.summarizer import summarize_phase
from core.notify import notify

log = logging.getLogger(__name__)


# Phase definitions (AutoKaggle-style)
PHASES = [
    "Background Understanding",
    "Preliminary EDA",
    "Data Cleaning",
    "In-Depth EDA",
    "Feature Engineering",
    "Model Building",
]


class MultiAgentOrchestrator:
    """Coordinate multi-agent workflow."""

    def __init__(self, competition_slug: str, data_dir: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            competition_slug: Kaggle competition slug
            data_dir: Override data directory (default: data/{slug})
        """
        self.slug = competition_slug
        self.data_dir = data_dir or Path(f"data/{competition_slug}")
        self.state = {
            "competition_slug": competition_slug,
            "data_dir": str(self.data_dir),
        }
        self.history = []
        self.tools_library = self._load_tools_library()

    def run(
        self,
        start_phase: int = 0,
        end_phase: Optional[int] = None,
        skip_reader: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute multi-agent workflow.

        Args:
            start_phase: Starting phase index (0 = Reader)
            end_phase: Ending phase index (None = all phases)
            skip_reader: Skip Reader agent (use existing competition_info)

        Returns:
            Final state with history of all phase executions
        """
        end_phase = end_phase or len(PHASES)

        log.info(f"Starting multi-agent orchestrator for {self.slug}")
        notify(f"🤖 Multi-agent mode: {self.slug}")

        # Phase 0: Reader
        if start_phase == 0 and not skip_reader:
            self._run_reader()

        # Load competition info if not run yet
        if "competition_info" not in self.state:
            self._load_competition_info()

        # Phases 1-6
        for phase_idx in range(max(start_phase, 1), end_phase):
            if phase_idx > len(PHASES):
                break

            phase = PHASES[phase_idx - 1]
            self._run_phase(phase)

        log.info("Multi-agent workflow complete")
        notify(f"✅ Multi-agent workflow complete for {self.slug}")

        return {
            "competition_slug": self.slug,
            "state": self.state,
            "history": self.history,
            "phases_completed": len(self.history),
        }

    def _run_reader(self) -> None:
        """Run Reader agent."""
        log.info("=== PHASE 0: READER ===")
        notify(f"📖 Reader agent analyzing {self.slug}")

        comp_info = read_competition(self.slug, self.data_dir)
        self.state["competition_info"] = comp_info

        # Save to file
        output_path = Path(f".claude/competition_info_{self.slug}.json")
        save_competition_info(comp_info, output_path)

        summary_msg = (
            f"Competition parsed:\n"
            f"- Type: {comp_info['problem_type']}\n"
            f"- Metric: {comp_info['eval_metric']}\n"
            f"- Data: {comp_info['data_shape']}"
        )
        notify(f"✓ {summary_msg}")

    def _load_competition_info(self) -> None:
        """Load existing competition info."""
        info_path = Path(f".claude/competition_info_{self.slug}.json")

        if info_path.exists():
            with info_path.open() as f:
                self.state["competition_info"] = json.load(f)
            log.info("Loaded existing competition info")
        else:
            log.warning("No competition info found - run Reader first")
            self.state["competition_info"] = {
                "problem_type": "unknown",
                "eval_metric": "unknown",
            }

    def _run_phase(self, phase: str) -> None:
        """Run a single phase with all agents."""
        log.info(f"=== PHASE: {phase} ===")
        notify(f"🔧 Starting {phase}")

        comp_info = self.state.get("competition_info", {})

        # 1. Planner
        log.info(f"[{phase}] Running Planner")
        plan = plan_phase(phase, comp_info, self.state)

        notify(f"  📋 {len(plan['tasks'])} tasks planned")
        log.info(f"Plan: {plan.get('approach', 'N/A')}")

        # 2. Developer + Reviewer loop for each task
        phase_results = []

        for i, task in enumerate(plan["tasks"], 1):
            log.info(f"[{phase}] Task {i}/{len(plan['tasks'])}: {task['name']}")
            notify(f"  ⚙️  Task {i}/{len(plan['tasks'])}: {task['name']}")

            try:
                # Develop
                dev_result = develop_task(task, self.tools_library, self.state)

                # Review
                review = review_code(
                    dev_result["code"],
                    task,
                    dev_result.get("output", {}),
                    dev_result.get("eval", {}),
                )

                # Handle review failures
                if not review.get("pass", True):
                    critical_issues = [
                        issue for issue in review.get("issues", [])
                        if issue.get("severity") == "critical"
                    ]

                    if critical_issues:
                        log.warning(f"Task {task['name']} has critical issues - attempting retry")

                        # One retry with feedback
                        dev_result = self._retry_task_with_feedback(task, review, dev_result)
                        review = review_code(
                            dev_result["code"],
                            task,
                            dev_result.get("output", {}),
                            dev_result.get("eval", {}),
                        )

                # Store task result
                task_result = {
                    "task": task,
                    "result": dev_result,
                    "review": review,
                }
                phase_results.append(task_result)

                # Update state with task outputs
                self._update_state_from_task(task, dev_result)

                status = "✓" if review.get("pass", True) else "⚠"
                notify(f"    {status} {task['name']} complete")

            except Exception as e:
                log.error(f"Task failed: {task['name']} - {e}")
                notify(f"    ✗ {task['name']} failed: {str(e)[:100]}")

                # Store failure
                phase_results.append({
                    "task": task,
                    "result": {"error": str(e)},
                    "review": {"pass": False},
                })

        # 3. Summarizer
        log.info(f"[{phase}] Running Summarizer")
        summary = summarize_phase(phase, phase_results, self.state)

        # Store phase history
        self.history.append({
            "phase": phase,
            "plan": plan,
            "results": phase_results,
            "summary": summary,
        })

        # Update state
        self.state[f"{phase}_summary"] = summary

        log.info(f"[{phase}] Complete")

    def _retry_task_with_feedback(
        self,
        task: Dict[str, Any],
        review: Dict[str, Any],
        prev_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Retry task incorporating reviewer feedback."""
        log.info(f"Retrying {task['name']} with reviewer feedback")

        # Augment task with feedback
        feedback_text = "\n".join([
            f"- {issue['description']}" for issue in review.get("issues", [])
        ])

        task_with_feedback = task.copy()
        task_with_feedback["methodology"] += f"\n\nREVIEWER FEEDBACK:\n{feedback_text}"

        # Retry development
        return develop_task(task_with_feedback, self.tools_library, self.state, max_attempts=3)

    def _update_state_from_task(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Update state with task outputs."""
        # Handle error case (result might be string or dict)
        if isinstance(result, str):
            log.warning(f"Result is string, not dict: {result[:100]}")
            return

        if not isinstance(result, dict):
            log.warning(f"Result is not dict: {type(result)}")
            return

        # Store code
        if "code" in result:
            code_key = task.get("name", "unknown").lower().replace(" ", "_") + "_code"
            self.state[code_key] = result["code"]

        # Store artifacts
        if "artifacts" in result:
            if "artifacts" not in self.state:
                self.state["artifacts"] = []
            self.state["artifacts"].extend(result["artifacts"])

        # Store metrics
        if "metrics" in result:
            if "metrics" not in self.state:
                self.state["metrics"] = {}
            self.state["metrics"].update(result["metrics"])

    def _load_tools_library(self) -> Dict[str, Any]:
        """Load ML tools library."""
        try:
            from core.tools import TOOLS_LIBRARY
            return TOOLS_LIBRARY
        except Exception as e:
            log.warning(f"Failed to load tools library: {e}")
            return {}

    def get_phase_summary(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific phase."""
        for entry in self.history:
            if entry["phase"] == phase:
                return entry.get("summary")
        return None

    def export_history(self, output_path: Path) -> None:
        """Export full workflow history to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump({
                "competition_slug": self.slug,
                "phases_completed": len(self.history),
                "history": self.history,
                "final_state": self.state,
            }, f, indent=2, default=str)

        log.info(f"Exported workflow history to {output_path}")


def run_multi_agent(
    slug: str,
    start_phase: int = 0,
    end_phase: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Quick helper to run multi-agent workflow.

    Args:
        slug: Competition slug
        start_phase: Starting phase (0=Reader, 1=Preliminary EDA, etc.)
        end_phase: Ending phase (None=all)
        data_dir: Override data directory

    Returns:
        Workflow results

    Example:
        result = run_multi_agent("store-sales-time-series-forecasting", start_phase=1, end_phase=3)
    """
    orch = MultiAgentOrchestrator(slug, data_dir)
    result = orch.run(start_phase=start_phase, end_phase=end_phase)

    # Export history
    output_path = Path(f".claude/multi_agent_history_{slug}.json")
    orch.export_history(output_path)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.orchestrator <competition_slug> [start_phase] [end_phase]")
        print("Example: python -m core.orchestrator titanic 0 2")
        sys.exit(1)

    slug = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end = int(sys.argv[3]) if len(sys.argv) > 3 else None

    result = run_multi_agent(slug, start_phase=start, end_phase=end)

    print(f"\nCompleted {result['phases_completed']} phases")
    print(f"History saved to: .claude/multi_agent_history_{slug}.json")
