"""Engineer Agent - Implement experiments with learning from past failures.

Implements code based on experiment specifications from Strategist, with:
- Memory of past implementation errors
- Validation before running
- Retry logic with incremental fixes
- DeepEval code quality checks (when available)
"""
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.langfuse_tracker import track_decision
from core.ollama_client import generate as ask_ollama
from core import memory


@track_decision("implement_experiment")
def implement_experiment(
    experiment: dict,
    competition: str,
    max_retries: int = 3
) -> dict:
    """
    Implement an experiment specification.

    Args:
        experiment: Experiment spec from Strategist
        competition: Competition slug
        max_retries: Maximum implementation attempts

    Returns:
        {
            "success": bool,
            "code_path": str | None,
            "validation_errors": list,
            "execution_results": dict | None,
            "attempts": int
        }
    """
    print(f"\n{'='*70}")
    print(f"ENGINEER AGENT - Implementing {experiment['id']}: {experiment['name']}")
    print(f"{'='*70}\n")

    # Query past errors for this type of experiment
    past_errors = _query_past_errors(competition, experiment['name'])

    attempts = 0
    validation_errors = []
    code_path = None

    while attempts < max_retries:
        attempts += 1
        print(f"\nAttempt {attempts}/{max_retries}")

        # Generate code
        code = _generate_code(experiment, competition, past_errors, validation_errors)

        if not code:
            print("ERROR: Failed to generate code")
            continue

        # Save code to file
        code_path = _save_code(experiment, competition, code, attempts)
        print(f"Code saved to: {code_path}")

        # Validate code (syntax check, imports, etc.)
        validation_errors = _validate_code(code_path)

        if validation_errors:
            print(f"Validation errors ({len(validation_errors)}):")
            for err in validation_errors[:3]:
                print(f"  - {err}")
            continue

        print("✓ Code validation passed")

        # Try to execute (dry run first if possible)
        try:
            print("Running code...")
            execution_results = _execute_code(code_path, competition, dry_run=False)

            if execution_results.get("success"):
                print(f"\n{'='*70}")
                print(f"IMPLEMENTATION SUCCESSFUL")
                print(f"{'='*70}\n")

                return {
                    "success": True,
                    "code_path": str(code_path),
                    "validation_errors": [],
                    "execution_results": execution_results,
                    "attempts": attempts
                }
            else:
                print(f"Execution failed: {execution_results.get('error', 'Unknown error')}")
                validation_errors.append(f"Runtime error: {execution_results.get('error')}")

        except Exception as e:
            print(f"Execution exception: {e}")
            validation_errors.append(f"Exception: {e}")

    # All attempts failed
    print(f"\n{'='*70}")
    print(f"IMPLEMENTATION FAILED after {attempts} attempts")
    print(f"{'='*70}\n")

    # Store error for future reference
    _store_error(competition, experiment['name'], validation_errors)

    return {
        "success": False,
        "code_path": str(code_path) if code_path else None,
        "validation_errors": validation_errors,
        "execution_results": None,
        "attempts": attempts
    }


def _generate_code(
    experiment: dict,
    competition: str,
    past_errors: list,
    current_errors: list
) -> Optional[str]:
    """Generate code using Ollama."""
    print("Generating code with Ollama...")

    # Build context from past errors
    error_context = ""
    if past_errors:
        error_context = "\n\nPAST ERRORS TO AVOID:\n" + "\n".join([f"  - {e}" for e in past_errors[:3]])
    if current_errors:
        error_context += "\n\nCURRENT ERRORS TO FIX:\n" + "\n".join([f"  - {e}" for e in current_errors[:3]])

    prompt = f"""Generate a Python script to implement this experiment for Kaggle competition '{competition}':

EXPERIMENT:
ID: {experiment['id']}
Name: {experiment['name']}
Hypothesis: {experiment['hypothesis']}
Approach: {experiment['approach']}
Expected Impact: {experiment.get('expected_impact', 'N/A')}
Success Criteria: {experiment.get('success_criteria', 'N/A')}
{error_context}

REQUIREMENTS:
1. Use existing code patterns from the competition directory
2. Save results to MLflow
3. Print clear progress messages
4. Handle errors gracefully
5. Return results dict with: {{"cv_score": float, "lb_score": float | None, "success": bool}}

IMPORTANT:
- Import from correct paths (competitions/active/{competition}/...)
- Use TimeSeriesSplit for time series data
- Log to MLflow before and after training
- Save submission file to submissions/ directory
- Do NOT submit to Kaggle (submission requires human approval)

Generate ONLY the Python code (no markdown, no explanations):
"""

    try:
        code = ask_ollama(prompt, model="qwen3:14b", think=True, max_tokens=4096, timeout_s=180.0)

        # Clean up markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

    except Exception as e:
        print(f"ERROR generating code: {e}")
        return None


def _save_code(experiment: dict, competition: str, code: str, attempt: int) -> Path:
    """Save generated code to file."""
    exp_dir = Path(f"competitions/active/{competition}/experiments")
    exp_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{experiment['id']}_attempt{attempt}.py"
    code_path = exp_dir / filename

    code_path.write_text(code)
    return code_path


def _validate_code(code_path: Path) -> list[str]:
    """Validate code syntax and imports."""
    errors = []

    # Syntax check with py_compile
    try:
        import py_compile
        py_compile.compile(str(code_path), doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(f"Syntax error: {e}")

    # Check for common issues (simple pattern matching)
    code = code_path.read_text()

    if "kaggle competitions submit" in code:
        errors.append("Code contains submission command - this requires human approval")

    # More validation could be added here

    return errors


def _execute_code(code_path: Path, competition: str, dry_run: bool = False) -> dict:
    """Execute the generated code."""
    if dry_run:
        return {"success": True, "message": "Dry run - not executed"}

    try:
        # Run the code with a timeout
        result = subprocess.run(
            [sys.executable, str(code_path)],
            cwd=Path(__file__).parent.parent,  # Run from repo root
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )

        if result.returncode == 0:
            # Try to parse results from output
            # Look for JSON in output
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if line.strip().startswith("{") and "cv_score" in line:
                    try:
                        results = json.loads(line)
                        return {"success": True, **results}
                    except json.JSONDecodeError:
                        pass

            return {
                "success": True,
                "message": "Code executed successfully",
                "output": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            }
        else:
            return {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
                "stdout": result.stdout[-500:] if result.stdout else ""
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Execution timeout (>10 min)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _query_past_errors(competition: str, experiment_name: str) -> list[str]:
    """Query past errors for similar experiments."""
    # This would query a memory store of past errors
    # For now, return empty list
    return []


def _store_error(competition: str, experiment_name: str, errors: list):
    """Store errors for future reference."""
    # This would store to memory
    # For now, just print
    print(f"Storing error memory for future attempts...")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m agents.engineer <competition_slug> <experiment_json>")
        sys.exit(1)

    competition = sys.argv[1]
    experiment_json = sys.argv[2]

    try:
        experiment = json.loads(experiment_json)
    except json.JSONDecodeError:
        print("ERROR: Invalid experiment JSON")
        sys.exit(1)

    result = implement_experiment(experiment, competition)

    print("\n" + "="*70)
    print("IMPLEMENTATION RESULT")
    print("="*70)
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    if result['code_path']:
        print(f"Code: {result['code_path']}")
    if result['execution_results']:
        print(f"Results: {json.dumps(result['execution_results'], indent=2)}")
