"""Learning Multi-Agent Orchestrator - Dynamic experiment loop.

Coordinates 5 agents in a learning cycle:
1. Analyst - Understand current state
2. Strategist - Plan experiments
3. Engineer - Implement code
4. Evaluator - Assess results
5. Curator - Store learnings

Loop continues until compute budget exhausted or human intervention.
"""
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.analyst import analyze_state
from agents.strategist import strategize_experiments
from agents.engineer import implement_experiment
from agents.evaluator import evaluate_experiment
from agents.curator import curate_knowledge
from core.langfuse_tracker import track_phase
from core.notify import send_telegram


def run_learning_loop(
    competition: str,
    max_iterations: int = 10,
    compute_budget_hours: float = 6.0,
    auto_submit: bool = False
) -> dict:
    """
    Run the learning multi-agent loop.

    Args:
        competition: Competition slug
        max_iterations: Maximum number of experiment iterations
        compute_budget_hours: Available compute time
        auto_submit: If True, submit best result to Kaggle (requires approval)

    Returns:
        {
            "iterations_completed": int,
            "experiments_run": int,
            "successes": int,
            "failures": int,
            "best_cv": float | None,
            "learnings": list[str]
        }
    """
    print("\n" + "="*80)
    print(f"LEARNING MULTI-AGENT SYSTEM")
    print(f"Competition: {competition}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Compute Budget: {compute_budget_hours} hours")
    print("="*80 + "\n")

    send_telegram(f"🤖 Starting learning multi-agent loop on {competition}")
    send_telegram(f"  Max iterations: {max_iterations}, Budget: {compute_budget_hours}h")

    start_time = time.time()
    iterations_completed = 0
    experiments_run = 0
    successes = 0
    failures = 0
    learnings = []
    best_cv = None

    for iteration in range(max_iterations):
        iterations_completed = iteration + 1

        print(f"\n{'='*80}")
        print(f"ITERATION {iterations_completed}/{max_iterations}")
        print(f"{'='*80}\n")

        # Check compute budget
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = compute_budget_hours - elapsed_hours

        if remaining_hours <= 0:
            print("⏱ Compute budget exhausted")
            send_telegram(f"⏱ Compute budget exhausted after {iterations_completed} iterations")
            break

        print(f"⏱ Time remaining: {remaining_hours:.2f} hours\n")

        # STEP 1: ANALYZE
        print("📊 STEP 1: ANALYZE")
        track_phase("analyze", competition)
        try:
            analysis = analyze_state(competition)
        except Exception as e:
            print(f"ERROR in Analyst: {e}")
            send_telegram(f"❌ Analyst failed: {e}")
            break

        # Update best_cv tracking
        current_best_cv = analysis.get("current_best", {}).get("cv", {}).get("score")
        if current_best_cv and (best_cv is None or current_best_cv < best_cv):
            best_cv = current_best_cv

        # STEP 2: STRATEGIZE
        print("\n🎯 STEP 2: STRATEGIZE")
        track_phase("strategize", competition)
        try:
            experiments = strategize_experiments(
                analysis,
                competition,
                max_experiments=3,  # Propose top 3
                compute_budget_hours=remaining_hours
            )
        except Exception as e:
            print(f"ERROR in Strategist: {e}")
            send_telegram(f"❌ Strategist failed: {e}")
            break

        if not experiments:
            print("No experiments proposed - stopping")
            send_telegram("🛑 No more experiments to try")
            break

        # Run top priority experiment
        experiment = experiments[0]
        experiments_run += 1

        print(f"\n🔨 STEP 3: IMPLEMENT - {experiment['id']}: {experiment['name']}")
        track_phase("implement", competition)
        send_telegram(f"🔨 Implementing: {experiment['name']}")

        try:
            implementation_result = implement_experiment(experiment, competition)
        except Exception as e:
            print(f"ERROR in Engineer: {e}")
            send_telegram(f"❌ Engineer failed: {e}")
            failures += 1
            continue

        # STEP 4: EVALUATE
        print(f"\n🔍 STEP 4: EVALUATE")
        track_phase("evaluate", competition)

        try:
            evaluation = evaluate_experiment(experiment, implementation_result, competition)
        except Exception as e:
            print(f"ERROR in Evaluator: {e}")
            send_telegram(f"❌ Evaluator failed: {e}")
            failures += 1
            continue

        if evaluation.get("success"):
            successes += 1
            send_telegram(f"✅ Success: {experiment['name']}")
            send_telegram(f"  {evaluation.get('assessment', 'N/A')}")
        else:
            failures += 1
            send_telegram(f"⚠️ Partial/Failed: {experiment['name']}")
            send_telegram(f"  {evaluation.get('assessment', 'N/A')}")

        # STEP 5: LEARN (CURATE)
        print(f"\n📚 STEP 5: CURATE KNOWLEDGE")
        track_phase("curate", competition)

        try:
            curation_result = curate_knowledge(experiment, evaluation, competition)
            learnings.append(evaluation.get("learned", ""))
        except Exception as e:
            print(f"ERROR in Curator: {e}")
            # Don't break on curation failure - it's not critical

        # Summary for this iteration
        print(f"\n{'='*80}")
        print(f"ITERATION {iterations_completed} COMPLETE")
        print(f"  Experiment: {experiment['id']}")
        print(f"  Success: {evaluation.get('success')}")
        print(f"  Learned: {evaluation.get('learned', 'N/A')[:80]}")
        print(f"{'='*80}\n")

        # Small delay between iterations
        time.sleep(2)

    # Final summary
    total_time = (time.time() - start_time) / 3600
    success_rate = (successes / experiments_run * 100) if experiments_run > 0 else 0

    print("\n" + "="*80)
    print("LEARNING LOOP COMPLETE")
    print("="*80)
    print(f"Iterations: {iterations_completed}")
    print(f"Experiments Run: {experiments_run}")
    print(f"Successes: {successes} ({success_rate:.1f}%)")
    print(f"Failures: {failures}")
    print(f"Best CV: {best_cv:.4f if best_cv else 'N/A'}")
    print(f"Total Time: {total_time:.2f} hours")
    print(f"\nKey Learnings:")
    for i, learning in enumerate(learnings[:5], 1):
        print(f"  {i}. {learning[:100]}")
    print("="*80 + "\n")

    send_telegram(f"🏁 Learning loop complete!")
    send_telegram(f"  {experiments_run} experiments, {successes} successes ({success_rate:.0f}%)")
    send_telegram(f"  Best CV: {best_cv:.4f if best_cv else 'N/A'}")
    send_telegram(f"  Time: {total_time:.2f}h")

    return {
        "iterations_completed": iterations_completed,
        "experiments_run": experiments_run,
        "successes": successes,
        "failures": failures,
        "best_cv": best_cv,
        "learnings": learnings,
        "total_time_hours": total_time
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agents.orchestrator <competition_slug> [max_iterations] [compute_hours]")
        print("\nExample: python -m agents.orchestrator store-sales-time-series-forecasting 5 2.0")
        sys.exit(1)

    competition = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    compute_hours = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    result = run_learning_loop(
        competition=competition,
        max_iterations=max_iterations,
        compute_budget_hours=compute_hours
    )

    print("\nFinal Result:")
    print(f"  Iterations: {result['iterations_completed']}")
    print(f"  Experiments: {result['experiments_run']}")
    print(f"  Success Rate: {result['successes']}/{result['experiments_run']}")
    if result['best_cv']:
        print(f"  Best CV: {result['best_cv']:.4f}")
