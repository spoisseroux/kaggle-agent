"""Planner Agent - Decompose phases into executable tasks.

Responsibilities:
- Break phases into ≤4 tasks with detailed methodologies
- Search for similar past approaches (semantic search)
- Estimate compute time per task
- Identify task dependencies

Cost: 80% Ollama, 20% Claude (escalate for ambiguous cases)
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional

from core.llm_interface import ask_ollama, ask_claude, should_escalate
from core.semantic_search import search_code

log = logging.getLogger(__name__)


# Phase to semantic search category mapping
PHASE_CATEGORIES = {
    "Background Understanding": None,  # No code search needed
    "Preliminary EDA": "visualization",
    "Data Cleaning": "utility",
    "In-Depth EDA": "visualization",
    "Feature Engineering": "feature_engineering",
    "Model Building": "model_training",
}


def plan_phase(
    phase: str,
    comp_info: Dict[str, Any],
    state: Dict[str, Any],
    max_tasks: int = 4,
) -> Dict[str, Any]:
    """
    Create task decomposition for a phase.

    Args:
        phase: Phase name (e.g., "Feature Engineering")
        comp_info: Competition information from Reader
        state: Current state (previous phase outputs, code, etc.)
        max_tasks: Maximum number of tasks to generate

    Returns:
        Plan dict with:
        - phase: phase name
        - tasks: [{name, methodology, expected_output, compute_estimate, dependencies}, ...]
        - approach: overall approach description
        - search_results: similar past approaches (for reference)
    """
    log.info(f"Planner agent planning {phase}")

    # 1. Search for similar past work (if applicable)
    similar_approaches = _search_similar_approaches(phase, comp_info)

    # 2. Generate plan with Ollama
    prompt = _build_planner_prompt(phase, comp_info, state, similar_approaches, max_tasks)
    system = "You are a data science competition planner. Decompose phases into concrete, executable tasks."

    try:
        response = ask_ollama(prompt, system=system, think=True)
        plan = _parse_plan_response(response, phase)

        # 3. Check for ambiguities/contradictions
        needs_escalation, reason = _check_plan_quality(plan, phase)

        if needs_escalation:
            log.info(f"Plan needs Claude review: {reason}")
            plan = _escalate_plan_to_claude(prompt, plan, reason)

        # Add metadata
        plan["search_results"] = similar_approaches
        plan["llm_used"] = "claude" if needs_escalation else "ollama"

        log.info(f"Planner complete: {len(plan['tasks'])} tasks generated")
        return plan

    except Exception as e:
        log.error(f"Planner failed: {e}")
        raise


def _search_similar_approaches(phase: str, comp_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Search for similar approaches from past competitions."""
    category = PHASE_CATEGORIES.get(phase)

    if not category:
        return []

    query = f"{phase} {comp_info.get('problem_type', '')} {comp_info.get('eval_metric', '')}"

    try:
        results = search_code(
            query=query,
            limit=3,
            category=category,
        )
        return results
    except Exception as e:
        log.warning(f"Semantic search failed: {e}")
        return []


def _build_planner_prompt(
    phase: str,
    comp_info: Dict[str, Any],
    state: Dict[str, Any],
    similar_approaches: List[Dict[str, Any]],
    max_tasks: int,
) -> str:
    """Build prompt for Planner."""
    # Format similar approaches
    similar_text = ""
    if similar_approaches:
        similar_text = "\n=== SIMILAR PAST APPROACHES ===\n"
        for i, approach in enumerate(similar_approaches, 1):
            similar_text += f"\n{i}. {approach['name']} (score: {approach['score']:.2f})\n"
            similar_text += f"   Competition: {approach['competition']}\n"
            similar_text += f"   Code:\n{approach['code'][:300]}...\n"

    # Format current state
    state_summary = json.dumps(state, indent=2, default=str)[:1000]

    prompt = f"""
Plan the "{phase}" phase for this Kaggle competition.

=== COMPETITION ===
Problem: {comp_info.get('problem_type', 'unknown')}
Metric: {comp_info.get('eval_metric', 'unknown')}
Data: {comp_info.get('data_shape', {})}
Target: {comp_info.get('target_info', {}).get('type', 'unknown')}
Known issues: {comp_info.get('known_issues', [])}

=== CURRENT STATE ===
{state_summary}

{similar_text}

---

Create a concrete plan with ≤{max_tasks} executable tasks.

Guidelines:
- Each task should be implementable in <30 min
- Include detailed methodology (specific steps)
- Avoid mixing concerns from different phases
- Prefer approaches that worked in similar competitions
- Estimate compute time (minutes)
- Identify task dependencies (which must complete first)

Generate a JSON object:
{{
  "phase": "{phase}",
  "approach": "overall strategy for this phase (1-2 sentences)",
  "tasks": [
    {{
      "name": "Task name (concise)",
      "methodology": "Detailed steps: 1) ... 2) ... 3) ...",
      "expected_output": "What this task produces (files, metrics, artifacts)",
      "compute_estimate": "X minutes",
      "dependencies": ["task_name1", "task_name2"] or []
    }}
  ]
}}

Return ONLY the JSON object, no other text.
"""
    return prompt


def _parse_plan_response(response: str, phase: str) -> Dict[str, Any]:
    """Parse Ollama response into plan dict."""
    # Clean response
    response_clean = response.strip()
    if response_clean.startswith("```json"):
        response_clean = response_clean.split("```json")[1].split("```")[0].strip()
    elif response_clean.startswith("```"):
        response_clean = response_clean.split("```")[1].split("```")[0].strip()

    try:
        plan = json.loads(response_clean)
        return plan
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse plan as JSON: {e}")
        log.error(f"Response: {response[:500]}")
        raise ValueError(f"Invalid plan JSON: {e}")


def _check_plan_quality(plan: Dict[str, Any], phase: str) -> tuple[bool, Optional[str]]:
    """
    Check if plan has issues requiring Claude review.

    Returns:
        (needs_escalation, reason)
    """
    tasks = plan.get("tasks", [])

    # Too many or too few tasks
    if len(tasks) == 0:
        return True, "No tasks generated"
    if len(tasks) > 6:
        return True, "Too many tasks (>6) - needs decomposition review"

    # Check for contradictory methodologies
    methodologies = [t.get("methodology", "") for t in tasks]
    combined = " ".join(methodologies).lower()

    contradictions = [
        ("daily", "weekly"),  # Different time granularities
        ("simple", "complex"),  # Contradictory complexity
        ("remove outliers", "keep outliers"),
        ("one-hot", "target encoding"),  # Should pick one approach
    ]

    for word1, word2 in contradictions:
        if word1 in combined and word2 in combined:
            return True, f"Plan mentions both '{word1}' and '{word2}' - potential contradiction"

    # Check for vague methodologies
    for task in tasks:
        method = task.get("methodology", "")
        if len(method) < 50:  # Too vague
            return True, f"Task '{task.get('name')}' has vague methodology"

        # Keywords indicating uncertainty
        uncertain_words = ["maybe", "possibly", "unclear", "tbd", "todo"]
        if any(word in method.lower() for word in uncertain_words):
            return True, f"Task '{task.get('name')}' has uncertain methodology"

    return False, None


def _escalate_plan_to_claude(
    original_prompt: str,
    ollama_plan: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    """Escalate plan to Claude for refinement."""
    escalation_prompt = f"""
The Planner agent generated this plan but it has issues:

Issue: {reason}

Ollama's plan:
{json.dumps(ollama_plan, indent=2)}

Original planning prompt:
{original_prompt}

---

Please provide a refined plan that fixes the issues. Return ONLY a JSON object with the same structure:
{{
  "phase": "...",
  "approach": "...",
  "tasks": [...]
}}
"""

    response = ask_claude(
        escalation_prompt,
        system="You are a data science competition strategist. Refine task plans to be clear and executable.",
        reason=reason,
    )

    # Parse Claude's response
    try:
        return _parse_plan_response(response, ollama_plan.get("phase", "unknown"))
    except Exception as e:
        log.error(f"Claude escalation failed to parse: {e}")
        # Return original plan as fallback
        return ollama_plan


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.agents.planner <phase_name>")
        print("Example: python -m core.agents.planner 'Feature Engineering'")
        sys.exit(1)

    phase = sys.argv[1]

    # Mock competition info
    comp_info = {
        "problem_type": "time_series",
        "eval_metric": "RMSLE",
        "data_shape": {"train_rows": 10000, "train_cols": 10},
        "target_info": {"type": "continuous"},
        "known_issues": [],
    }

    state = {}

    plan = plan_phase(phase, comp_info, state)
    print(json.dumps(plan, indent=2))
