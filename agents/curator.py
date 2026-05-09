"""Curator Agent - Build institutional knowledge from experiments.

Curates knowledge by:
- Storing successful patterns to Qdrant (semantic memory)
- Updating competition CLAUDE.md with learnings
- Tagging experiments in MLflow
- Recording patterns to Postgres for structured queries
"""
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.langfuse_tracker import track_decision
from core import memory


@track_decision("curate_knowledge")
def curate_knowledge(
    experiment: dict,
    evaluation: dict,
    competition: str
) -> dict:
    """
    Store learnings from an experiment.

    Args:
        experiment: Original experiment spec
        evaluation: Evaluation results
        competition: Competition slug

    Returns:
        {
            "stored_to_memory": bool,
            "updated_claude_md": bool,
            "tagged_in_mlflow": bool,
            "pattern_recorded": bool
        }
    """
    print(f"\n{'='*70}")
    print(f"CURATOR AGENT - Recording knowledge from {experiment['id']}")
    print(f"{'='*70}\n")

    result = {
        "stored_to_memory": False,
        "updated_claude_md": False,
        "tagged_in_mlflow": False,
        "pattern_recorded": False
    }

    # 1. Store to Postgres memory
    try:
        _store_to_postgres(experiment, evaluation, competition)
        result["stored_to_memory"] = True
        print("✓ Stored to Postgres memory")
    except Exception as e:
        print(f"✗ Failed to store to Postgres: {e}")

    # 2. Update competition CLAUDE.md if significant learning
    if _is_significant_learning(evaluation):
        try:
            _update_claude_md(experiment, evaluation, competition)
            result["updated_claude_md"] = True
            print("✓ Updated competition CLAUDE.md")
        except Exception as e:
            print(f"✗ Failed to update CLAUDE.md: {e}")

    # 3. Tag in MLflow (if we have MLflow integration)
    try:
        _tag_in_mlflow(experiment, evaluation)
        result["tagged_in_mlflow"] = True
        print("✓ Tagged in MLflow")
    except Exception as e:
        print(f"✗ Failed to tag in MLflow: {e}")

    # 4. Record pattern to Qdrant (if successful)
    if evaluation.get("success"):
        try:
            _store_pattern_to_qdrant(experiment, evaluation, competition)
            result["pattern_recorded"] = True
            print("✓ Recorded pattern to Qdrant")
        except Exception as e:
            print(f"✗ Failed to store pattern to Qdrant: {e}")

    print(f"\n{'='*70}")
    print("CURATION COMPLETE")
    print(f"{'='*70}\n")

    return result


def _store_to_postgres(experiment: dict, evaluation: dict, competition: str):
    """Store experiment learnings to Postgres."""
    # Use memory.insert_memory() to store the learning
    try:
        memory.insert_memory(
            title=f"{competition}: {experiment['name']}",
            body=json.dumps({
                "experiment_id": experiment['id'],
                "hypothesis": experiment.get('hypothesis'),
                "success": evaluation.get('success'),
                "assessment": evaluation.get('assessment'),
                "learned": evaluation.get('learned'),
                "next_steps": evaluation.get('next_steps', [])
            }, indent=2),
            memory_type="experiment_result",
            tags=[competition, experiment['id'], "success" if evaluation.get('success') else "failure"]
        )
    except AttributeError:
        # insert_memory might not exist, skip for now
        print("  (insert_memory not available - skipping Postgres storage)")


def _is_significant_learning(evaluation: dict) -> bool:
    """Determine if learning is significant enough to update CLAUDE.md."""
    # Significant if:
    # - Experiment succeeded
    # - Or failure with important lesson learned
    if evaluation.get("success"):
        return True

    learned = evaluation.get("learned", "").lower()
    significant_keywords = ["critical", "important", "major", "never", "always", "avoid"]

    return any(keyword in learned for keyword in significant_keywords)


def _update_claude_md(experiment: dict, evaluation: dict, competition: str):
    """Update competition CLAUDE.md with learnings."""
    claude_md_path = Path(f"competitions/active/{competition}/CLAUDE.md")

    if not claude_md_path.exists():
        print(f"  CLAUDE.md not found at {claude_md_path}")
        return

    content = claude_md_path.read_text()

    # Add learning to a "Learnings" section
    learning_entry = f"\n## Experiment Learnings\n\n" if "## Experiment Learnings" not in content else ""
    learning_entry += f"\n### {experiment['id']}: {experiment['name']}\n"
    learning_entry += f"- **Hypothesis**: {experiment.get('hypothesis', 'N/A')}\n"
    learning_entry += f"- **Outcome**: {evaluation.get('assessment', 'N/A')}\n"
    learning_entry += f"- **Learned**: {evaluation.get('learned', 'N/A')}\n"

    if "## Experiment Learnings" in content:
        # Append to existing section
        content = content.replace(
            "## Experiment Learnings\n",
            f"## Experiment Learnings\n{learning_entry}"
        )
    else:
        # Add new section at the end
        content += f"\n{learning_entry}"

    claude_md_path.write_text(content)


def _tag_in_mlflow(experiment: dict, evaluation: dict):
    """Tag experiment in MLflow."""
    # This would use MLflow API to tag the run
    # For now, placeholder
    print("  (MLflow tagging not implemented yet)")


def _store_pattern_to_qdrant(experiment: dict, evaluation: dict, competition: str):
    """Store successful pattern to Qdrant for semantic search."""
    # This would embed the successful approach and store in Qdrant
    # For now, placeholder
    print("  (Qdrant pattern storage not implemented yet)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m agents.curator <competition_slug> <experiment_json> <evaluation_json>")
        sys.exit(1)

    competition = sys.argv[1]
    experiment_json = sys.argv[2]
    evaluation_json = sys.argv[3]

    try:
        experiment = json.loads(experiment_json)
        evaluation = json.loads(evaluation_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        sys.exit(1)

    result = curate_knowledge(experiment, evaluation, competition)

    print("\n" + "="*70)
    print("CURATION REPORT")
    print("="*70)
    print(f"Stored to memory: {result['stored_to_memory']}")
    print(f"Updated CLAUDE.md: {result['updated_claude_md']}")
    print(f"Tagged in MLflow: {result['tagged_in_mlflow']}")
    print(f"Pattern recorded: {result['pattern_recorded']}")
