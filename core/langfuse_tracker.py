"""Langfuse tracking for single-agent mode.

Wraps key operations with Langfuse traces so we can observe:
- What experiments are being run
- How decisions are being made
- What's working vs not working
"""
import os
import functools
from typing import Any, Callable, Optional

# Try to import Langfuse, but make it optional
try:
    from langfuse import observe, Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create no-op decorator if Langfuse not available
    def observe(name=None, as_type=None):
        def decorator(func):
            return func
        return decorator

# Initialize Langfuse
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY", ""))
os.environ.setdefault("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY", ""))
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")


def track_experiment(name: str, metadata: Optional[dict] = None):
    """
    Decorator to track experiments in Langfuse.

    Usage:
        @track_experiment("xgboost_training", {"version": "v10"})
        def train_model():
            # training code
            return {"cv_score": 0.370}
    """
    def decorator(func: Callable) -> Callable:
        @observe(name=name, as_type="generation")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function (Langfuse observe handles tracing)
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def track_decision(decision_type: str, context: Optional[dict] = None):
    """
    Track decision-making moments.

    Usage:
        @track_decision("feature_selection", {"current_features": 29})
        def decide_features():
            # analysis code
            return {"action": "add_lag_14", "reason": "gap in temporal coverage"}
    """
    def decorator(func: Callable) -> Callable:
        @observe(name=f"decision_{decision_type}", as_type="span")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def manual_trace(name: str, input_data: Any, output_data: Any, metadata: Optional[dict] = None):
    """
    Manually create a Langfuse trace for ad-hoc tracking.

    Usage:
        manual_trace(
            "feature_engineering",
            input_data={"base_features": 20},
            output_data={"final_features": 29, "cv_delta": 0.05},
            metadata={"approach": "added_lags_and_rolling"}
        )
    """
    if not LANGFUSE_AVAILABLE or not is_configured():
        return

    langfuse = Langfuse()
    trace = langfuse.trace(
        name=name,
        input=input_data,
        output=output_data,
        metadata=metadata or {}
    )
    trace.update()
    langfuse.flush()


def track_phase(phase_name: str, competition: str):
    """
    Track completion of a workflow phase.

    Usage:
        track_phase("feature_engineering", "store-sales")
    """
    if not LANGFUSE_AVAILABLE or not is_configured():
        return

    langfuse = Langfuse()
    trace = langfuse.trace(
        name=f"phase_{phase_name}",
        metadata={
            "competition": competition,
            "phase": phase_name
        }
    )
    trace.update()
    langfuse.flush()


# Helper to check if Langfuse is properly configured
def is_configured() -> bool:
    """Check if Langfuse keys are set."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
