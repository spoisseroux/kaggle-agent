#!/usr/bin/env python
"""Autonomous experiment loop orchestrator.

Runs continuous experiment cycles with pacing, reflection, and user monitoring.

Usage:
    python scripts/autonomous_loop.py [--competition SLUG] [--phase PHASE] [--dry-run]

The loop:
1. Checks experiment_pacer for permission
2. Loads experiment plan
3. Executes next experiment
4. Records result
5. Reflects and re-plans after N experiments
6. Checks for user messages
7. Repeats until stopped

Stopping conditions:
- User message received
- Pacer denies permission
- Emergency stop triggered
- Phase complete
- No more experiments in plan
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experiment_pacer import ExperimentPacer
from core.api_budget import get_tracker as get_api_tracker
from core.message_bus import get_pending_instructions

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def notify(message: str):
    """Send notification to user."""
    subprocess.run(["python", "core/notify.py", message], check=False)


def load_registry() -> dict:
    """Load competitions registry."""
    with open("competitions/registry.json") as f:
        return json.load(f)


def load_experiment_plan(slug: str) -> list[dict]:
    """Load experiment plan for competition."""
    plan_file = Path(f".claude/experiment_plan_{slug}.json")
    if not plan_file.exists():
        return []
    with open(plan_file) as f:
        data = json.load(f)
        return data.get("experiments", [])


def save_experiment_plan(slug: str, experiments: list[dict]):
    """Save updated experiment plan."""
    plan_file = Path(f".claude/experiment_plan_{slug}.json")
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_file, "w") as f:
        json.dump({"experiments": experiments, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def run_experiment(experiment: dict, competition_slug: str, dry_run: bool = False) -> dict:
    """
    Execute a single experiment.

    Returns:
        {
            "success": bool,
            "cv_score": float | None,
            "improved": bool,
            "error": str | None,
            "duration_sec": float
        }
    """
    start_time = time.time()

    if dry_run:
        log.info(f"[DRY RUN] Would run experiment: {experiment['name']}")
        time.sleep(2)  # Simulate work
        return {
            "success": True,
            "cv_score": 0.5 + (hash(experiment["name"]) % 100) / 1000,  # Fake score
            "improved": False,
            "error": None,
            "duration_sec": time.time() - start_time,
        }

    try:
        # TODO: Implement actual experiment execution
        # This would:
        # 1. Generate code with Ollama (experiment['approach'])
        # 2. Run training script
        # 3. Parse CV score from output
        # 4. Compare with best score so far
        # 5. Log to MLflow

        log.info(f"Executing experiment: {experiment['name']}")
        log.info(f"Approach: {experiment.get('approach', 'not specified')}")

        # Placeholder: actual implementation would call experiment runner
        # For now, return a mock result
        return {
            "success": False,
            "cv_score": None,
            "improved": False,
            "error": "Experiment execution not yet implemented",
            "duration_sec": time.time() - start_time,
        }

    except Exception as e:
        log.error(f"Experiment failed: {e}")
        return {
            "success": False,
            "cv_score": None,
            "improved": False,
            "error": str(e),
            "duration_sec": time.time() - start_time,
        }


def reflect_and_replan(competition_slug: str):
    """Run reflection and update experiment plan."""
    log.info("Running reflection and re-planning...")

    # Run reflection script (if exists)
    reflect_script = Path("scripts/reflect_experiments.py")
    if reflect_script.exists():
        result = subprocess.run(
            ["python", str(reflect_script), "--competition", competition_slug],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warning(f"Reflection failed: {result.stderr}")

    # Run planning script (if exists)
    plan_script = Path("scripts/plan_experiments.py")
    if plan_script.exists():
        result = subprocess.run(
            ["python", str(plan_script), "--competition", competition_slug],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warning(f"Planning failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous experiment loop")
    parser.add_argument("--competition", help="Competition slug (default: active from registry)")
    parser.add_argument("--phase", help="Current phase (eda, baseline, feature_engineering, etc.)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate experiments without executing")
    parser.add_argument("--max-iterations", type=int, help="Max iterations (for testing)")
    parser.add_argument("--reflect-interval", type=int, default=5, help="Reflect after N experiments")
    args = parser.parse_args()

    # Load competition
    registry = load_registry()
    competition_slug = args.competition or registry["active"]
    competition_data = registry["competitions"][competition_slug]

    log.info(f"Starting autonomous loop for: {competition_slug}")
    log.info(f"Phase: {args.phase or 'auto'}")
    log.info(f"Dry run: {args.dry_run}")

    # Initialize pacer
    competition_type = competition_data.get("type", "default")  # playground, featured, research
    deadline_str = competition_data.get("deadline")
    deadline = datetime.fromisoformat(deadline_str) if deadline_str else None

    pacer = ExperimentPacer(
        competition_type=competition_type,
        deadline=deadline,
    )

    notify(f"🤖 **Autonomous loop started**\n\nCompetition: {competition_slug}\nPhase: {args.phase or 'auto'}\nDry run: {args.dry_run}")

    iteration = 0
    experiments_run = 0
    consecutive_no_plan = 0

    while True:
        iteration += 1
        if args.max_iterations and iteration > args.max_iterations:
            log.info(f"Reached max iterations: {args.max_iterations}")
            break

        # Check for user messages
        instructions = get_pending_instructions()
        if instructions:
            log.info(f"User message received: {instructions}")
            # Format each message as a readable line, not a raw Python dict
            msg_lines = []
            for m in instructions:
                src = m.get("source", "?")
                txt = m.get("text", "").strip()
                prefix = "📱" if src == "telegram" else "💬"
                msg_lines.append(f"{prefix} {txt}")
            formatted = "\n\n".join(msg_lines)
            notify(f"📥 Loop paused — incoming message(s):\n\n{formatted}\n\nProcessing now...")
            break

        # Check if we can run
        allowed, reason = pacer.check_can_run(phase=args.phase)
        if not allowed:
            if "minimum interval" in reason.lower():
                wait_sec = int(reason.split("Wait ")[-1].replace("s.", ""))
                log.info(f"Waiting {wait_sec}s for minimum interval...")
                time.sleep(wait_sec + 1)
                continue
            else:
                log.info(f"Pacer denies permission: {reason}")
                notify(f"⏸️ **Loop paused**\n\n{reason}\n\nWill check again later or wait for instructions.")
                break

        # Load experiment plan
        plan = load_experiment_plan(competition_slug)
        if not plan:
            log.warning("No experiment plan found")
            consecutive_no_plan += 1

            if consecutive_no_plan >= 3:
                log.error("No plan after 3 attempts, stopping")
                notify(f"❌ **Loop stopped**\n\nNo experiment plan available after multiple attempts.\n\nPlease create a plan or provide instructions.")
                break

            # Try to create plan
            log.info("Attempting to create experiment plan...")
            reflect_and_replan(competition_slug)
            time.sleep(5)
            continue

        consecutive_no_plan = 0

        # Get next experiment
        next_experiment = None
        for exp in plan:
            if not exp.get("completed", False):
                next_experiment = exp
                break

        if not next_experiment:
            log.info("All experiments in plan completed")
            notify(f"✅ **Plan complete!**\n\nAll {len(plan)} experiments finished.\n\nReflecting and creating new plan...")
            reflect_and_replan(competition_slug)
            continue

        # Execute experiment
        log.info(f"[{experiments_run + 1}] Running: {next_experiment['name']}")
        result = run_experiment(next_experiment, competition_slug, dry_run=args.dry_run)

        # Record result
        pacer.record_experiment(
            phase=args.phase,
            improved=result["improved"],
            errored=not result["success"],
        )

        experiments_run += 1

        # Update plan (mark as completed)
        for exp in plan:
            if exp["name"] == next_experiment["name"]:
                exp["completed"] = True
                exp["result"] = result
                exp["completed_at"] = datetime.now(timezone.utc).isoformat()
                break
        save_experiment_plan(competition_slug, plan)

        # Send progress update
        status_emoji = "✓" if result["success"] else "✗"
        score_str = f"CV: {result['cv_score']:.4f}" if result["cv_score"] else "N/A"
        improved_str = " 🎯 NEW BEST!" if result["improved"] else ""

        notify(f"{status_emoji} **Experiment {experiments_run} complete**\n\n{next_experiment['name']}\n{score_str}{improved_str}\n\nTime: {result['duration_sec']:.1f}s")

        # Reflect and replan after interval
        if experiments_run % args.reflect_interval == 0:
            notify(f"🔄 **Reflection checkpoint** ({experiments_run} experiments)\n\nAnalyzing results and updating plan...")
            reflect_and_replan(competition_slug)

        # Check API budget status
        api_tracker = get_api_tracker()
        if warning := api_tracker.get_budget_warning():
            notify(warning)

        # Small delay between experiments
        time.sleep(5)

    # Final status
    pacer_status = pacer.get_status()
    notify(f"🛑 **Loop stopped**\n\nExperiments run: {experiments_run}\nDaily: {pacer_status['daily']['count']}/{pacer_status['daily']['limit']}\nAdaptive multiplier: {pacer_status['adaptive_multiplier']}x")

    log.info(f"Autonomous loop finished. Experiments run: {experiments_run}")


if __name__ == "__main__":
    main()
