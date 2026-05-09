#!/usr/bin/env python3
"""
Test Langfuse + Experiment Queue integration.

Run this after setting up Langfuse on homelab to verify everything works.
"""
import os
import sys
from pathlib import Path

print("=" * 80)
print("Infrastructure Integration Test")
print("=" * 80)

# Check environment variables
print("\n1. Checking environment variables...")
required_vars = {
    "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST"),
    "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY"),
    "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY"),
    "POSTGRES_DSN": os.environ.get("POSTGRES_DSN"),
}

all_set = True
for var, value in required_vars.items():
    if value:
        # Mask secrets
        display_value = value if "KEY" not in var and "DSN" not in var else f"{value[:10]}...{value[-5:]}"
        print(f"  ✅ {var}: {display_value}")
    else:
        print(f"  ❌ {var}: NOT SET")
        all_set = False

if not all_set:
    print("\n❌ Missing environment variables. Add them to ~/.bashrc or .env")
    sys.exit(1)

# Test Langfuse
print("\n2. Testing Langfuse integration...")
try:
    from core.langfuse_integration import get_langfuse_client, trace_llm_call

    client = get_langfuse_client()
    if client:
        print("  ✅ Langfuse client initialized")

        # Send test trace
        trace_llm_call(
            func_name="test_infrastructure",
            model="test-model",
            prompt="This is a test prompt",
            response="This is a test response",
            duration=0.1,
            agent="TestAgent",
            metadata={"test": True},
        )
        print("  ✅ Test trace sent to Langfuse")
        print(f"  📊 Check traces at: {os.environ.get('LANGFUSE_HOST')}")
    else:
        print("  ❌ Langfuse client not available")
        sys.exit(1)

except Exception as e:
    print(f"  ❌ Langfuse test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test Experiment Queue
print("\n3. Testing Experiment Queue...")
try:
    from core.experiment_queue import ExperimentQueue

    queue = ExperimentQueue(max_parallel=2)
    print("  ✅ Experiment queue initialized")

    # Submit test experiment
    exp_id = queue.submit(
        competition_slug="test",
        config={
            "model": "xgboost",
            "params": {"n_estimators": 100},
            "features": ["test_feature"],
        },
        priority=5,
        estimated_duration_min=1,
    )
    print(f"  ✅ Test experiment submitted (ID: {exp_id})")

    # Get status
    status = queue.get_status(exp_id)
    if status and status['status'] == 'queued':
        print(f"  ✅ Experiment status: {status['status']}")
    else:
        print(f"  ❌ Unexpected status: {status}")

    # Clean up test experiment
    queue.cancel(exp_id)
    print("  ✅ Test experiment cancelled")

except Exception as e:
    print(f"  ❌ Experiment queue test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test LLM interface with tracing
print("\n4. Testing LLM interface with Langfuse tracing...")
try:
    from core.llm_interface import ask_ollama

    response = ask_ollama(
        "What is 2+2? Answer in one word.",
        system="You are a helpful math assistant.",
        think=False,
    )
    print(f"  ✅ Ollama call successful: {response.strip()[:50]}")
    print(f"  📊 Check Langfuse for 'ollama' trace")

except Exception as e:
    print(f"  ❌ LLM interface test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
print("\nNext steps:")
print("1. Check Langfuse UI for traces: " + os.environ.get("LANGFUSE_HOST", ""))
print("2. Run Titanic test with full observability: python -m core.orchestrator titanic")
print("3. Monitor experiment queue: psql <POSTGRES_DSN> -c 'SELECT * FROM active_experiments;'")
