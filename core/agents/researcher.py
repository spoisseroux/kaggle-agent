"""Research Agent - Analyze problem and recommend approaches.

Responsibilities:
- Research competition type and best approaches
- Find similar competitions and solutions
- Recommend model architectures
- Suggest feature engineering strategies
- Identify potential pitfalls

Cost: 100% Ollama (zero API cost)
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.llm_interface import ask_ollama
from core.semantic_search import search_code

log = logging.getLogger(__name__)


def research_competition(
    comp_info: Dict[str, Any],
    depth: str = "standard",
) -> Dict[str, Any]:
    """
    Research competition and recommend approaches.

    Args:
        comp_info: Competition info from Reader agent
        depth: "quick", "standard", or "deep"

    Returns:
        Research report with:
        - problem_analysis: Deep dive into problem characteristics
        - similar_competitions: List of similar past competitions
        - recommended_approaches: Ranked list of approaches to try
        - model_recommendations: Best models for this problem
        - feature_strategies: Feature engineering suggestions
        - pitfalls: Known issues and things to avoid
        - resource_estimate: Time/compute estimates
    """
    log.info(f"Research agent analyzing {comp_info.get('competition_slug', 'unknown')}")

    # 1. Analyze problem characteristics
    problem_analysis = _analyze_problem(comp_info)

    # 2. Search for similar competitions
    similar_comps = _find_similar_competitions(comp_info)

    # 3. Research best approaches
    approaches = _research_approaches(comp_info, problem_analysis, similar_comps)

    # 4. Recommend models
    models = _recommend_models(comp_info, problem_analysis)

    # 5. Feature engineering strategies
    features = _suggest_feature_strategies(comp_info, similar_comps)

    # 6. Identify pitfalls
    pitfalls = _identify_pitfalls(comp_info, problem_analysis)

    # 7. Estimate resources
    resources = _estimate_resources(comp_info, approaches)

    research_report = {
        "problem_analysis": problem_analysis,
        "similar_competitions": similar_comps,
        "recommended_approaches": approaches,
        "model_recommendations": models,
        "feature_strategies": features,
        "pitfalls": pitfalls,
        "resource_estimate": resources,
    }

    # Save report
    _save_research_report(comp_info.get("competition_slug", "unknown"), research_report)

    log.info(f"Research complete: {len(approaches)} approaches, {len(models)} models recommended")

    return research_report


def _analyze_problem(comp_info: Dict[str, Any]) -> Dict[str, Any]:
    """Deep analysis of problem characteristics."""
    prompt = f"""
Analyze this Kaggle competition problem in detail:

Problem Type: {comp_info.get('problem_type', 'unknown')}
Eval Metric: {comp_info.get('eval_metric', 'unknown')}
Data Shape: {comp_info.get('data_shape', {})}
Target: {comp_info.get('target_info', {})}
Known Issues: {comp_info.get('known_issues', [])}

Summary: {comp_info.get('summary', '')}

Provide deep analysis in JSON format:
{{
  "problem_category": "tabular|time_series|nlp|computer_vision|multi_modal",
  "complexity": "beginner|intermediate|advanced|expert",
  "key_challenges": ["challenge1", "challenge2"],
  "data_characteristics": {{
    "size": "small|medium|large",
    "quality": "clean|noisy|very_noisy",
    "balance": "balanced|imbalanced|highly_imbalanced",
    "missingness": "none|low|moderate|high"
  }},
  "success_factors": ["factor1", "factor2"],
  "competitive_landscape": "tutorial|active|highly_competitive"
}}

Return ONLY the JSON object.
"""

    system = "You are a data science competition analyst with expertise in identifying problem patterns."

    response = ask_ollama(prompt, system=system, think=True)

    try:
        # Parse JSON
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean.split("```json")[1].split("```")[0].strip()
        elif response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1].split("```")[0].strip()

        return json.loads(response_clean)

    except Exception as e:
        log.error(f"Failed to parse problem analysis: {e}")
        return {
            "problem_category": comp_info.get("problem_type", "unknown"),
            "complexity": "unknown",
            "key_challenges": [],
            "data_characteristics": {},
            "success_factors": [],
            "competitive_landscape": "unknown",
        }


def _find_similar_competitions(comp_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find similar past competitions using semantic search."""
    try:
        # Search for similar approaches
        query = f"{comp_info.get('problem_type', '')} {comp_info.get('eval_metric', '')} competition"

        results = search_code(
            query=query,
            limit=5,
            category="model_training",  # Find model training patterns
        )

        similar = []
        for result in results:
            similar.append({
                "competition": result.get("competition", "unknown"),
                "approach": result.get("name", ""),
                "similarity_score": result.get("score", 0.0),
                "code_preview": result.get("code", "")[:200],
            })

        return similar

    except Exception as e:
        log.warning(f"Semantic search failed: {e}")
        return []


def _research_approaches(
    comp_info: Dict[str, Any],
    problem_analysis: Dict[str, Any],
    similar_comps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Research and rank best approaches."""
    # Format similar competitions
    similar_text = "\n".join([
        f"- {s['competition']}: {s['approach']} (score: {s['similarity_score']:.2f})"
        for s in similar_comps[:3]
    ])

    prompt = f"""
Based on this competition analysis, recommend the best approaches to try:

Problem: {comp_info.get('problem_type', 'unknown')}
Metric: {comp_info.get('eval_metric', 'unknown')}
Complexity: {problem_analysis.get('complexity', 'unknown')}
Key Challenges: {problem_analysis.get('key_challenges', [])}

Similar past competitions:
{similar_text}

Winning approaches from top notebooks:
{comp_info.get('winning_approaches', [])}

Recommend 3-5 approaches ranked by priority. For each approach:
{{
  "approaches": [
    {{
      "name": "Approach name",
      "priority": 1,
      "description": "What to do",
      "rationale": "Why this works for this problem",
      "estimated_effort": "low|medium|high",
      "expected_improvement": "baseline|moderate|significant"
    }}
  ]
}}

Consider:
- What has worked on similar problems?
- What matches the eval metric?
- What's feasible given data size/quality?
- Start simple, iterate to complex

Return ONLY the JSON object.
"""

    system = "You are a Kaggle competitions expert. Recommend practical, high-ROI approaches."

    response = ask_ollama(prompt, system=system, think=True)

    try:
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean.split("```json")[1].split("```")[0].strip()
        elif response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1].split("```")[0].strip()

        result = json.loads(response_clean)
        return result.get("approaches", [])

    except Exception as e:
        log.error(f"Failed to parse approaches: {e}")
        return []


def _recommend_models(
    comp_info: Dict[str, Any],
    problem_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Recommend specific models and configurations."""
    prompt = f"""
Recommend the best machine learning models for this competition:

Problem: {comp_info.get('problem_type', 'unknown')}
Metric: {comp_info.get('eval_metric', 'unknown')}
Data Size: {comp_info.get('data_shape', {})}
Complexity: {problem_analysis.get('complexity', 'unknown')}

Recommend 3-5 models with specific configurations:
{{
  "models": [
    {{
      "name": "Model name (e.g., LightGBM, XGBoost)",
      "priority": 1,
      "rationale": "Why this model for this problem",
      "suggested_config": {{
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "other_params": "..."
      }},
      "ollama_model_recommendation": "qwen3:14b|deepseek-coder:33b|etc",
      "training_time_estimate": "minutes"
    }}
  ]
}}

Also recommend which Ollama model is best for implementing this approach.

Return ONLY the JSON object.
"""

    system = "You are a machine learning expert. Recommend models that balance performance and practicality."

    response = ask_ollama(prompt, system=system, think=True)

    try:
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean.split("```json")[1].split("```")[0].strip()
        elif response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1].split("```")[0].strip()

        result = json.loads(response_clean)
        return result.get("models", [])

    except Exception as e:
        log.error(f"Failed to parse model recommendations: {e}")
        return []


def _suggest_feature_strategies(
    comp_info: Dict[str, Any],
    similar_comps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suggest feature engineering strategies."""
    # Search for similar feature engineering
    try:
        query = f"{comp_info.get('problem_type', '')} feature engineering"
        feature_results = search_code(
            query=query,
            limit=5,
            category="feature_engineering",
        )

        strategies = []
        for result in feature_results:
            strategies.append({
                "strategy": result.get("name", ""),
                "competition": result.get("competition", ""),
                "relevance": result.get("score", 0.0),
                "code_example": result.get("code", "")[:200],
            })

        return strategies

    except Exception as e:
        log.warning(f"Feature search failed: {e}")
        return []


def _identify_pitfalls(
    comp_info: Dict[str, Any],
    problem_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Identify common pitfalls and things to avoid."""
    known_issues = comp_info.get("known_issues", [])

    prompt = f"""
Identify potential pitfalls and things to avoid for this competition:

Problem: {comp_info.get('problem_type', 'unknown')}
Known Issues: {known_issues}
Data Characteristics: {problem_analysis.get('data_characteristics', {})}

List 3-5 specific pitfalls:
{{
  "pitfalls": [
    {{
      "issue": "Name of the pitfall",
      "severity": "critical|high|medium|low",
      "description": "What could go wrong",
      "mitigation": "How to avoid it"
    }}
  ]
}}

Consider:
- Data leakage risks
- Overfitting potential
- Metric gaming
- Train/test distribution shift
- Known competition-specific issues

Return ONLY the JSON object.
"""

    system = "You are a cautious data scientist. Identify risks and suggest mitigations."

    response = ask_ollama(prompt, system=system, think=False)

    try:
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean.split("```json")[1].split("```")[0].strip()
        elif response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1].split("```")[0].strip()

        result = json.loads(response_clean)
        return result.get("pitfalls", [])

    except Exception as e:
        log.error(f"Failed to parse pitfalls: {e}")
        return []


def _estimate_resources(
    comp_info: Dict[str, Any],
    approaches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Estimate time and compute resources."""
    data_size = comp_info.get("data_shape", {})
    train_rows = data_size.get("train_rows", 0)

    # Rough estimates based on data size
    if train_rows < 1000:
        time_estimate = "1-2 hours"
        compute = "CPU sufficient"
    elif train_rows < 100000:
        time_estimate = "3-6 hours"
        compute = "CPU recommended, GPU optional"
    else:
        time_estimate = "6-12 hours"
        compute = "GPU recommended"

    total_approaches = len(approaches)

    return {
        "per_approach_time": time_estimate,
        "total_time_estimate": f"{total_approaches * 3}-{total_approaches * 6} hours",
        "compute_requirements": compute,
        "gpu_recommended": train_rows > 100000,
        "parallel_tasks": min(total_approaches, 3),
    }


def _save_research_report(slug: str, report: Dict[str, Any]) -> None:
    """Save research report to file."""
    output_dir = Path(f".claude/research")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"{slug}_research.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save Markdown summary
    md_path = output_dir / f"{slug}_research.md"
    md_content = _format_research_markdown(report)
    md_path.write_text(md_content)

    log.info(f"Saved research report to {json_path}")


def _format_research_markdown(report: Dict[str, Any]) -> str:
    """Format research report as markdown."""
    md = ["# Competition Research Report\n"]

    # Problem Analysis
    md.append("## Problem Analysis\n")
    analysis = report.get("problem_analysis", {})
    md.append(f"- **Category:** {analysis.get('problem_category', 'unknown')}")
    md.append(f"- **Complexity:** {analysis.get('complexity', 'unknown')}")
    md.append(f"- **Landscape:** {analysis.get('competitive_landscape', 'unknown')}\n")

    # Recommended Approaches
    md.append("## Recommended Approaches\n")
    for approach in report.get("recommended_approaches", []):
        md.append(f"### {approach.get('priority', '?')}. {approach.get('name', 'Unknown')}")
        md.append(f"**Rationale:** {approach.get('rationale', 'N/A')}")
        md.append(f"**Effort:** {approach.get('estimated_effort', 'unknown')}")
        md.append(f"**Expected Improvement:** {approach.get('expected_improvement', 'unknown')}\n")

    # Model Recommendations
    md.append("## Model Recommendations\n")
    for model in report.get("model_recommendations", []):
        md.append(f"### {model.get('priority', '?')}. {model.get('name', 'Unknown')}")
        md.append(f"**Rationale:** {model.get('rationale', 'N/A')}")
        md.append(f"**Ollama Model:** {model.get('ollama_model_recommendation', 'qwen3:14b')}\n")

    # Pitfalls
    md.append("## Potential Pitfalls\n")
    for pitfall in report.get("pitfalls", []):
        md.append(f"- **{pitfall.get('issue', 'Unknown')}** ({pitfall.get('severity', 'unknown')})")
        md.append(f"  - {pitfall.get('description', 'N/A')}")
        md.append(f"  - *Mitigation:* {pitfall.get('mitigation', 'N/A')}\n")

    return "\n".join(md)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.agents.researcher <competition_slug>")
        sys.exit(1)

    slug = sys.argv[1]

    # Load competition info
    comp_info_path = Path(f".claude/competition_info_{slug}.json")
    if not comp_info_path.exists():
        print(f"Error: No competition info found at {comp_info_path}")
        print("Run Reader agent first!")
        sys.exit(1)

    with comp_info_path.open() as f:
        comp_info = json.load(f)

    # Run research
    report = research_competition(comp_info, depth="standard")

    print("\n=== RESEARCH COMPLETE ===\n")
    print(f"Approaches: {len(report['recommended_approaches'])}")
    print(f"Models: {len(report['model_recommendations'])}")
    print(f"Pitfalls: {len(report['pitfalls'])}")
    print(f"\nReport saved to: .claude/research/{slug}_research.json")
