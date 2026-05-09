"""
Langfuse integration for LLM observability.

Tracks all LLM calls (Ollama and Claude) with:
- Prompts and responses
- Latencies
- Token counts (estimated for Ollama)
- Costs
- Agent context
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional, Dict, Any
from functools import wraps

log = logging.getLogger(__name__)

# Try to import Langfuse
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    log.warning("Langfuse not installed - install with: pip install langfuse")


# Initialize Langfuse client
_langfuse_client = None

def get_langfuse_client() -> Optional[Langfuse]:
    """Get or create Langfuse client."""
    global _langfuse_client

    if not LANGFUSE_AVAILABLE:
        return None

    if _langfuse_client is None:
        # Check for required env vars
        host = os.environ.get("LANGFUSE_HOST")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

        if not all([host, public_key, secret_key]):
            log.warning("Langfuse env vars not set - skipping observability")
            return None

        _langfuse_client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        log.info(f"Langfuse client initialized: {host}")

    return _langfuse_client


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return len(text) // 4


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate cost in USD.

    Pricing (as of 2026):
    - Claude Sonnet 4.5: $3/M input, $15/M output
    - Ollama: $0 (local)
    """
    if "claude" in model.lower() or "sonnet" in model.lower():
        # Claude pricing
        input_cost = (prompt_tokens / 1_000_000) * 3.0
        output_cost = (completion_tokens / 1_000_000) * 15.0
        return input_cost + output_cost
    else:
        # Ollama is free
        return 0.0


def trace_llm_call(
    func_name: str,
    model: str,
    prompt: str,
    response: str,
    duration: float,
    agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Trace LLM call to Langfuse.

    Args:
        func_name: Function name (e.g., "ask_ollama", "ask_claude")
        model: Model name
        prompt: Input prompt
        response: LLM response
        duration: Call duration in seconds
        agent: Agent name (Reader, Planner, etc.)
        metadata: Additional metadata
    """
    client = get_langfuse_client()
    if not client:
        return

    try:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response)
        total_tokens = prompt_tokens + completion_tokens
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        trace_metadata = {
            "agent": agent,
            "function": func_name,
            "duration_s": duration,
            "cost_usd": cost,
            **(metadata or {}),
        }

        # Create generation using context manager (Langfuse SDK v4+)
        with client.start_as_current_observation(
            as_type="generation",
            name=f"{agent or 'unknown'}/{func_name}",
            model=model,
            metadata=trace_metadata,
        ) as generation:
            generation.update(
                input=prompt[:1000],  # Limit size
                output=response[:1000],
                usage={
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": total_tokens,
                    "unit": "TOKENS",
                },
            )

        log.debug(f"Traced {func_name}: {total_tokens} tokens, ${cost:.4f}, {duration:.2f}s")

    except Exception as e:
        log.warning(f"Failed to trace LLM call: {e}")


def langfuse_observe(agent: Optional[str] = None):
    """
    Decorator to automatically trace LLM calls.

    Usage:
        @langfuse_observe(agent="Developer")
        def generate_code(prompt):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not LANGFUSE_AVAILABLE:
                return func(*args, **kwargs)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Try to extract prompt and response from args/kwargs
                # This is a simple heuristic - adjust based on function signature
                prompt = args[0] if args else kwargs.get('prompt', '')
                response = result if isinstance(result, str) else str(result)
                model = kwargs.get('model', 'unknown')

                trace_llm_call(
                    func_name=func.__name__,
                    model=model,
                    prompt=str(prompt)[:1000],  # Limit size
                    response=str(response)[:1000],
                    duration=duration,
                    agent=agent,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                log.error(f"Function {func.__name__} failed after {duration:.2f}s: {e}")
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Test Langfuse integration
    print("Testing Langfuse integration...")

    if LANGFUSE_AVAILABLE:
        client = get_langfuse_client()
        if client:
            print(f"✅ Langfuse client initialized")

            # Test trace
            trace_llm_call(
                func_name="test_call",
                model="qwen3:14b",
                prompt="What is 2+2?",
                response="4",
                duration=0.5,
                agent="Test",
            )
            print("✅ Test trace sent")
        else:
            print("❌ Langfuse client not available (check env vars)")
    else:
        print("❌ Langfuse not installed")
