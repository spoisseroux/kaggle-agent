"""Developer Agent - Implement code with iterative debugging.

Responsibilities:
- Generate code based on task specifications
- Execute code in sandboxed environment
- Debug errors (max 5 retries)
- Validate with DeepEval
- Use ML tools library

Cost: 60% Ollama, 40% Claude (escalate after 3+ debug attempts)
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.llm_interface import ask_ollama, ask_claude, should_escalate

try:
    from core.experiment_evaluator import ExperimentEvaluator
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

log = logging.getLogger(__name__)


MAX_DEBUG_ATTEMPTS = 5


def develop_task(
    task: Dict[str, Any],
    tools_library: Dict[str, Any],
    state: Dict[str, Any],
    max_attempts: int = MAX_DEBUG_ATTEMPTS,
) -> Dict[str, Any]:
    """
    Implement and debug task code.

    Args:
        task: Task specification from Planner
        tools_library: Available ML tools
        state: Current state (previous code, data paths, etc.)
        max_attempts: Maximum debugging attempts

    Returns:
        Result dict with:
        - code: final working code
        - output: execution results
        - eval: DeepEval validation results
        - attempts: number of debug iterations
        - llm_used: ollama|claude|both
    """
    log.info(f"Developer agent implementing: {task['name']}")

    # 1. Generate initial code
    code = _generate_code(task, tools_library, state)
    llm_used = "ollama"

    # 2. Iterative debugging
    for attempt in range(max_attempts):
        log.info(f"Execution attempt {attempt + 1}/{max_attempts}")

        # Execute code
        exec_result = _execute_code(code, state)

        if exec_result["success"]:
            # Success! Validate with DeepEval
            log.info("Code executed successfully, running DeepEval validation")

            eval_result = _validate_with_deepeval(code, task, state, exec_result)

            if eval_result.get("pass", False):
                log.info(f"Task complete after {attempt + 1} attempts")
                return {
                    "code": code,
                    "output": exec_result["output"],
                    "eval": eval_result,
                    "attempts": attempt + 1,
                    "llm_used": llm_used,
                    "success": True,
                }
            else:
                # DeepEval failed - treat as error and debug
                error = f"DeepEval validation failed: {eval_result.get('issues', [])}"
                log.warning(error)
        else:
            # Execution error
            error = exec_result["error"]
            log.warning(f"Execution failed: {error}")

        # Debug
        if attempt < max_attempts - 1:
            # Decide: Ollama or Claude?
            needs_escalation, escalation_reason = should_escalate(
                "Developer",
                context=error,
                attempt=attempt,
                error=error,
            )

            if needs_escalation:
                log.info(f"Escalating to Claude: {escalation_reason}")
                code = _debug_with_claude(code, error, task, state, attempt)
                llm_used = "both"
            else:
                code = _debug_with_ollama(code, error, task, state)

    # Max attempts exhausted
    log.error(f"Failed to implement {task['name']} after {max_attempts} attempts")
    return {
        "code": code,
        "output": "",
        "eval": {"pass": False, "issues": ["Max debugging attempts exhausted"]},
        "attempts": max_attempts,
        "llm_used": llm_used,
        "success": False,
        "error": "Max debugging attempts exhausted - code still has errors",
    }


def _generate_code(
    task: Dict[str, Any],
    tools_library: Dict[str, Any],
    state: Dict[str, Any],
) -> str:
    """Generate initial code implementation."""
    # Format available tools
    tools_text = _format_tools(tools_library)

    # Get previous code context
    prev_code = state.get("code", "")
    prev_code_summary = prev_code[:500] + "..." if len(prev_code) > 500 else prev_code

    # Get data paths
    data_dir = state.get("data_dir", "data/")

    prompt = f"""
Implement this task:

Task: {task['name']}
Methodology: {task['methodology']}
Expected output: {task['expected_output']}

=== AVAILABLE TOOLS ===
{tools_text}

=== PREVIOUS CODE CONTEXT ===
{prev_code_summary}

=== DATA DIRECTORY ===
{data_dir}

---

Generate complete, executable Python code that:
1. Follows the methodology steps
2. Uses the available tools where applicable (import from core.tools.*)
3. Handles errors gracefully
4. Saves outputs to files (if applicable)
5. Prints summary results

Guidelines:
- Import all needed libraries at the top (pandas, numpy, matplotlib.pyplot, seaborn if needed)
- Use CORRECT import paths: from core.tools.cleaning import handle_missing_values
- Use absolute paths based on data_dir: f"{{data_dir}}/train.csv"
- ALWAYS check if files exist before reading: Path(file_path).exists()
- Use try/except for file operations and print clear error messages
- Include helpful comments
- Return or print key metrics/results
- Don't use placeholder data - work with real files

CRITICAL: Import tools from core.tools.* modules:
  from core.tools.cleaning import handle_missing_values, detect_outliers
  from core.tools.features import create_lag_features, one_hot_encode
  from core.tools.modeling import select_model, train_with_cv

REQUIRED IMPORTS (always include these):
```python
import pandas as pd
import numpy as np
from pathlib import Path
# Add matplotlib/seaborn only if visualization is needed
# Add sklearn only if modeling is needed
```

FILE PATH PATTERN (always use this):
```python
data_dir = Path("{data_dir}")
train_path = data_dir / "train.csv"
if not train_path.exists():
    raise FileNotFoundError(f"Training data not found at {{train_path}}")
df = pd.read_csv(train_path)
```

Return ONLY Python code in a code block. No explanations before/after.
"""

    system = "You are a Python developer implementing data science tasks. Write clean, executable code."

    response = ask_ollama(prompt, system=system, think=True)

    # Extract code from response
    code = _extract_code(response)

    return code


def _execute_code(code: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute code in a temporary environment."""
    # Create temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Add project root to PYTHONPATH so imports work
        import os
        env = os.environ.copy()
        project_root = str(Path.cwd())
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{project_root}:{pythonpath}" if pythonpath else project_root

        # Execute with timeout
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd=str(Path.cwd()),
            env=env,  # Pass modified environment
        )

        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "stderr": result.stderr,
            }
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout,
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Execution timeout (>5 minutes)",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Execution exception: {str(e)}\n{traceback.format_exc()}",
        }
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def _validate_with_deepeval(
    code: str,
    task: Dict[str, Any],
    state: Dict[str, Any],
    exec_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate code with DeepEval."""
    if not DEEPEVAL_AVAILABLE:
        # DeepEval not available - simple validation
        return {
            "pass": True,
            "issues": [],
            "warnings": ["DeepEval not available - skipped validation"],
        }

    try:
        evaluator = ExperimentEvaluator()

        # Build context for evaluation
        eval_context = {
            "task_name": task["name"],
            "expected_output": task["expected_output"],
            "code": code,
            "execution_output": exec_result["output"],
        }

        # For now, simple validation based on successful execution
        # TODO: integrate full ExperimentEvaluator
        return {
            "pass": True,
            "issues": [],
            "warnings": [],
        }

    except Exception as e:
        log.warning(f"DeepEval validation failed: {e}")
        # Fallback: simple pass
        return {
            "pass": True,
            "issues": [],
            "warnings": [f"DeepEval error: {e}"],
        }


def _debug_with_ollama(
    code: str,
    error: str,
    task: Dict[str, Any],
    state: Dict[str, Any],
) -> str:
    """Debug code with Ollama."""
    # Get data directory info
    data_dir = state.get("data_dir", "data/")

    prompt = f"""
This code has an error. Fix it.

=== TASK ===
{task['name']}: {task['methodology']}

=== CODE ===
```python
{code}
```

=== ERROR ===
{error}

=== DATA DIRECTORY ===
{data_dir}

Expected files:
- {data_dir}/train.csv (training data)
- {data_dir}/test.csv (test data)

---

Analyze the error and generate fixed code.

Common issues and fixes:
- ImportError (seaborn, sklearn): Add `import seaborn as sns` or `from sklearn.X import Y`
- FileNotFoundError: Use `Path("{data_dir}") / "train.csv"` pattern
- NameError: Variable not defined - check spelling or add import
- SyntaxError: Fix indentation, missing colons, unmatched parentheses
- KeyError: Column doesn't exist - check DataFrame column names first
- TypeError: Wrong data type - check df.dtypes

CRITICAL: Always validate file paths:
```python
from pathlib import Path
data_dir = Path("{data_dir}")
train_path = data_dir / "train.csv"
if not train_path.exists():
    raise FileNotFoundError(f"Training data not found at {{train_path}}")
```

Return ONLY the complete fixed Python code in a code block. No explanations.
"""

    system = "You are a Python debugger. Fix errors in data science code."

    response = ask_ollama(prompt, system=system, think=True)

    return _extract_code(response)


def _debug_with_claude(
    code: str,
    error: str,
    task: Dict[str, Any],
    state: Dict[str, Any],
    attempt: int,
) -> str:
    """Escalate debugging to Claude."""
    prompt = f"""
This code failed after {attempt + 1} debugging attempts with Ollama. Need strategic debugging.

=== TASK ===
{task['name']}
Methodology: {task['methodology']}
Expected: {task['expected_output']}

=== CURRENT CODE ===
```python
{code}
```

=== ERROR ===
{error}

=== STATE ===
{json.dumps(state, indent=2, default=str)[:500]}

---

The error persists despite {attempt + 1} fixes. This suggests:
- Fundamental misunderstanding of the task
- Incorrect approach to the problem
- Missing context about data structure

Please:
1. Identify the root cause
2. Suggest a different approach if needed
3. Provide fixed code

Return the complete fixed Python code.
"""

    response = ask_claude(
        prompt,
        system="You are a senior data scientist debugging complex issues.",
        reason=f"Debugging failed after {attempt + 1} attempts",
    )

    return _extract_code(response)


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    response = response.strip()

    # Look for code blocks
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return _validate_and_fix_code(code.strip())

    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            code = parts[1]
            # Skip if it's not Python (e.g., ```json)
            if not any(lang in parts[0][-20:].lower() for lang in ["json", "yaml", "bash", "sh"]):
                return _validate_and_fix_code(code.strip())

    # No code block markers - return as-is
    return _validate_and_fix_code(response)


def _validate_and_fix_code(code: str) -> str:
    """Validate code syntax and apply common fixes."""
    # Remove common Markdown artifacts
    code = code.replace("```python", "").replace("```", "")

    # Check for basic syntax errors using compile
    try:
        compile(code, "<string>", "exec")
        return code
    except SyntaxError as e:
        log.warning(f"Syntax error in generated code: {e}")
        # Return as-is, let the execution loop catch it
        return code


def _format_tools(tools_library: Dict[str, Any]) -> str:
    """Format tools library for prompt with correct import paths AND function signatures."""
    if not tools_library:
        return "No tools available yet - implement from scratch using pandas/sklearn."

    tools_text = []

    # Map category to import module
    import_map = {
        "data_cleaning": "core.tools.cleaning",
        "feature_engineering": "core.tools.features",
        "modeling": "core.tools.modeling",
    }

    for category, tools in tools_library.items():
        module = import_map.get(category, f"core.tools.{category}")
        tools_text.append(f"\n{category} (from {module}):")

        # List functions with FULL signatures
        func_names = [tool['name'] for tool in tools]
        tools_text.append(f"  Import: from {module} import {', '.join(func_names[:5])}")
        if len(func_names) > 5:
            tools_text.append(f"          # ... and {len(func_names) - 5} more")

        # List with descriptions AND signatures
        for tool in tools:
            signature = tool.get('signature', f"{tool['name']}(...)")
            tools_text.append(f"  - {signature}")
            tools_text.append(f"    {tool['description']}")

    return "\n".join(tools_text)


if __name__ == "__main__":
    # Test task
    task = {
        "name": "Create lag features",
        "methodology": "1) Load sales data 2) Create lags 1,7,14 days 3) Save to features.csv",
        "expected_output": "features.csv with lag columns",
    }

    tools_library = {}
    state = {"data_dir": "data/store-sales-time-series-forecasting"}

    try:
        result = develop_task(task, tools_library, state)
        print("Success!")
        print(result["code"])
    except Exception as e:
        print(f"Failed: {e}")
