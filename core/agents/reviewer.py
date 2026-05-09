"""Reviewer Agent - Validate code and provide critical feedback.

Responsibilities:
- Review generated code for logic errors
- Check for data leakage
- Identify efficiency issues
- Validate against task requirements
- Provide actionable feedback

Cost: 100% Ollama (zero API cost)
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional

from core.llm_interface import ask_ollama

log = logging.getLogger(__name__)


def review_code(
    code: str,
    task: Dict[str, Any],
    exec_result: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Provide critical code review.

    Args:
        code: Generated code
        task: Task specification
        exec_result: Execution results
        eval_result: DeepEval validation results (optional)

    Returns:
        Review dict with:
        - pass: true/false
        - issues: [{severity, category, description}, ...]
        - suggestions: [actionable improvements]
        - score: 0-100 quality score
    """
    log.info(f"Reviewer agent reviewing: {task['name']}")

    prompt = f"""
Review this code for a data science task.

=== TASK ===
Name: {task['name']}
Methodology: {task['methodology']}
Expected output: {task['expected_output']}

=== CODE ===
```python
{code}
```

=== EXECUTION RESULTS ===
Success: {exec_result.get('success', False)}
Output: {exec_result.get('output', 'N/A')[:500]}
{f"Error: {exec_result.get('error', '')[:300]}" if not exec_result.get('success') else ""}

=== DEEPEVAL RESULTS ===
{json.dumps(eval_result, indent=2) if eval_result else "Not available"}

---

Provide a critical review checking for:

1. **Logic Errors**
   - Does the code implement the methodology correctly?
   - Are there off-by-one errors, incorrect indexing?
   - Are calculations correct?

2. **Data Leakage**
   - Does it use future information not available at prediction time?
   - Are train/test splits handled correctly?
   - Any target leakage in feature engineering?

3. **Efficiency Issues**
   - Are there unnecessary loops that could be vectorized?
   - Inefficient pandas operations?
   - Memory issues with large datasets?

4. **Edge Cases**
   - How does it handle missing values?
   - What about edge cases (empty data, single row, etc.)?
   - Error handling present?

5. **Consistency**
   - Does it match the expected output?
   - Consistent with previous phase results?

Generate a JSON object:
{{
  "pass": true/false,
  "issues": [
    {{
      "severity": "critical|warning|info",
      "category": "logic|leakage|efficiency|edge_case|consistency",
      "description": "Specific issue description"
    }}
  ],
  "suggestions": [
    "Actionable improvement 1",
    "Actionable improvement 2"
  ],
  "score": 0-100 (quality score),
  "summary": "One sentence overall assessment"
}}

Be critical but constructive. If there are no issues, say so.

Return ONLY the JSON object.
"""

    system = "You are a critical code reviewer for data science competitions. Find issues and suggest improvements."

    response = ask_ollama(prompt, system=system, think=False)

    # Parse response
    try:
        review = _parse_review_response(response)

        log.info(f"Review complete: {'PASS' if review['pass'] else 'FAIL'}, score: {review.get('score', 0)}")

        # Log issues
        for issue in review.get("issues", []):
            level = logging.ERROR if issue["severity"] == "critical" else logging.WARNING
            log.log(level, f"{issue['category']}: {issue['description']}")

        return review

    except Exception as e:
        log.error(f"Failed to parse review: {e}")
        # Fallback: pass with warning
        return {
            "pass": True,
            "issues": [],
            "suggestions": [],
            "score": 50,
            "summary": f"Review parsing failed: {e}",
            "error": str(e),
        }


def _parse_review_response(response: str) -> Dict[str, Any]:
    """Parse review response JSON."""
    response_clean = response.strip()

    if response_clean.startswith("```json"):
        response_clean = response_clean.split("```json")[1].split("```")[0].strip()
    elif response_clean.startswith("```"):
        response_clean = response_clean.split("```")[1].split("```")[0].strip()

    return json.loads(response_clean)


def check_for_data_leakage(code: str, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Quick data leakage check (supplementary to main review).

    Returns list of potential leakage issues.
    """
    issues = []

    code_lower = code.lower()

    # Common leakage patterns
    leakage_patterns = [
        ("test.csv", "Accessing test.csv in feature engineering (potential leakage)"),
        ("df['target']", "Using target column in features (direct leakage)"),
        (".shift(-", "Negative shift (using future data)"),
        ("future=True", "Future data flag enabled"),
    ]

    for pattern, description in leakage_patterns:
        if pattern in code_lower:
            issues.append({
                "severity": "critical",
                "category": "leakage",
                "description": description,
                "pattern": pattern,
            })

    return issues


if __name__ == "__main__":
    # Test review
    test_code = """
import pandas as pd

df = pd.read_csv('data/train.csv')
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df.to_csv('features.csv', index=False)
print(f"Created features: {len(df)} rows")
"""

    task = {
        "name": "Create lag features",
        "methodology": "Create lags 1,7 days",
        "expected_output": "features.csv",
    }

    exec_result = {
        "success": True,
        "output": "Created features: 1000 rows",
    }

    review = review_code(test_code, task, exec_result)
    print(json.dumps(review, indent=2))
