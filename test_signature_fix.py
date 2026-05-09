"""Test that signature fix resolves code generation issues."""
import logging
from pathlib import Path
from core.agents.developer import develop_task
from core.tools import TOOLS_LIBRARY

logging.basicConfig(level=logging.INFO)

# Simple task: handle missing values
task = {
    "name": "Handle missing values",
    "methodology": "1) Load train.csv 2) Use handle_missing_values tool with auto strategy 3) Print summary",
    "expected_output": "Cleaned dataframe with no missing values",
}

state = {
    "data_dir": "data/titanic",
    "code": "",
}

print("=" * 80)
print("Testing Developer agent with qwen2.5-coder:7b")
print("Task: Handle missing values using tools library")
print("=" * 80)
print()

result = develop_task(task, TOOLS_LIBRARY, state, max_attempts=3)

print()
print("=" * 80)
print("RESULT:")
print(f"Success: {result['success']}")
print(f"Attempts: {result['attempts']}")
print(f"LLM used: {result['llm_used']}")
print("=" * 80)
print()

if result['success']:
    print("✓ CODE GENERATED:")
    print(result['code'])
    print()
    print("✓ OUTPUT:")
    print(result['output'])
else:
    print("✗ FAILED:")
    print(result.get('error', 'Unknown error'))
    if 'output' in result:
        print("\nLast output:")
        print(result['output'])
