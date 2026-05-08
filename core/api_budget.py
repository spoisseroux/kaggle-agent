"""Smart API budget system for Claude Code usage.

Instead of a simple rate limit (N calls/hour), use a token-based budget that:
1. Tracks actual token usage (input + output)
2. Different "costs" for different operation types
3. Prioritizes critical operations over nice-to-have
4. Soft warnings before hard limits
5. Daily/hourly budgets with rollover

This allows burst usage when needed while preventing runaway costs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# Token budgets (configurable via environment or config file)
DEFAULT_HOURLY_TOKENS = int(os.environ.get("CLAUDE_HOURLY_TOKEN_BUDGET", "200_000"))
DEFAULT_DAILY_TOKENS = int(os.environ.get("CLAUDE_DAILY_TOKEN_BUDGET", "3_000_000"))

OperationType = Literal[
    "strategic",      # High priority: deciding next approach, interpreting surprises
    "code_review",    # Medium priority: reviewing Ollama-generated code
    "debugging",      # Medium priority: understanding unexpected behavior
    "documentation",  # Low priority: writing docs, comments
    "routine",        # Low priority: routine checks, confirmations
]

# Token multipliers for priority (lower = higher priority)
PRIORITY_COST_MULTIPLIER = {
    "strategic": 0.5,      # Strategic decisions get 2x effective budget
    "code_review": 1.0,    # Normal cost
    "debugging": 1.0,      # Normal cost
    "documentation": 2.0,  # Docs cost 2x (discourage using Claude for this)
    "routine": 2.0,        # Routine operations cost 2x
}


@dataclass
class BudgetState:
    hourly_used: int
    hourly_limit: int
    hourly_reset_at: float
    daily_used: int
    daily_limit: int
    daily_reset_at: float
    total_calls: int
    total_tokens: int


class APIBudgetTracker:
    """Tracks and enforces Claude API token budgets."""

    def __init__(self, state_file: Path | None = None):
        self.state_file = state_file or Path.home() / ".claude" / "projects" / "-home-keehar-kaggle-agent" / "api_budget_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> BudgetState:
        """Load state from disk or initialize fresh."""
        if not self.state_file.exists():
            return self._fresh_state()

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            now = time.time()
            # Reset if timestamps passed
            if now > data["hourly_reset_at"]:
                data["hourly_used"] = 0
                data["hourly_reset_at"] = now + 3600

            if now > data["daily_reset_at"]:
                data["daily_used"] = 0
                # Reset at next UTC midnight
                tomorrow = datetime.now(timezone.utc).date()
                data["daily_reset_at"] = datetime.combine(
                    tomorrow, datetime.min.time(), tzinfo=timezone.utc
                ).timestamp() + 86400

            return BudgetState(**data)
        except Exception as e:
            log.warning(f"Failed to load budget state: {e}. Starting fresh.")
            return self._fresh_state()

    def _fresh_state(self) -> BudgetState:
        """Create a fresh budget state."""
        now = time.time()
        tomorrow = datetime.now(timezone.utc).date()
        next_midnight = datetime.combine(
            tomorrow, datetime.min.time(), tzinfo=timezone.utc
        ).timestamp() + 86400

        return BudgetState(
            hourly_used=0,
            hourly_limit=DEFAULT_HOURLY_TOKENS,
            hourly_reset_at=now + 3600,
            daily_used=0,
            daily_limit=DEFAULT_DAILY_TOKENS,
            daily_reset_at=next_midnight,
            total_calls=0,
            total_tokens=0,
        )

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(
                    {
                        "hourly_used": self.state.hourly_used,
                        "hourly_limit": self.state.hourly_limit,
                        "hourly_reset_at": self.state.hourly_reset_at,
                        "daily_used": self.state.daily_used,
                        "daily_limit": self.state.daily_limit,
                        "daily_reset_at": self.state.daily_reset_at,
                        "total_calls": self.state.total_calls,
                        "total_tokens": self.state.total_tokens,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            log.warning(f"Failed to save budget state: {e}")

    def check_budget(
        self,
        estimated_tokens: int,
        operation_type: OperationType = "routine",
    ) -> tuple[bool, str]:
        """
        Check if operation is within budget.

        Returns:
            (allowed, reason) where reason is empty if allowed, else explanation
        """
        # Apply priority multiplier
        effective_cost = int(estimated_tokens * PRIORITY_COST_MULTIPLIER[operation_type])

        # Reload state (might have been updated by another process)
        self.state = self._load_state()

        # Check hourly budget
        if self.state.hourly_used + effective_cost > self.state.hourly_limit:
            mins_left = int((self.state.hourly_reset_at - time.time()) / 60)
            return False, f"Hourly budget exceeded. Resets in {mins_left} minutes."

        # Check daily budget
        if self.state.daily_used + effective_cost > self.state.daily_limit:
            hours_left = int((self.state.daily_reset_at - time.time()) / 3600)
            return False, f"Daily budget exceeded. Resets in {hours_left} hours."

        return True, ""

    def record_usage(
        self,
        tokens_used: int,
        operation_type: OperationType = "routine",
    ) -> None:
        """Record actual token usage after an API call."""
        effective_cost = int(tokens_used * PRIORITY_COST_MULTIPLIER[operation_type])

        self.state.hourly_used += effective_cost
        self.state.daily_used += effective_cost
        self.state.total_calls += 1
        self.state.total_tokens += tokens_used

        self._save_state()

    def get_status(self) -> dict:
        """Get current budget status for display."""
        self.state = self._load_state()

        hourly_pct = 100 * self.state.hourly_used / self.state.hourly_limit
        daily_pct = 100 * self.state.daily_used / self.state.daily_limit

        hourly_mins_left = int((self.state.hourly_reset_at - time.time()) / 60)
        daily_hours_left = int((self.state.daily_reset_at - time.time()) / 3600)

        # Determine status
        if hourly_pct > 90 or daily_pct > 90:
            status = "critical"
        elif hourly_pct > 70 or daily_pct > 70:
            status = "warning"
        else:
            status = "ok"

        return {
            "status": status,
            "hourly": {
                "used": self.state.hourly_used,
                "limit": self.state.hourly_limit,
                "percent": round(hourly_pct, 1),
                "resets_in_min": hourly_mins_left,
            },
            "daily": {
                "used": self.state.daily_used,
                "limit": self.state.daily_limit,
                "percent": round(daily_pct, 1),
                "resets_in_hours": daily_hours_left,
            },
            "lifetime": {
                "calls": self.state.total_calls,
                "tokens": self.state.total_tokens,
            },
        }

    def get_budget_warning(self) -> str | None:
        """Get a warning message if budget is running low, else None."""
        status = self.get_status()

        if status["status"] == "critical":
            return (
                f"⚠️ *API Budget Critical*\n"
                f"Hourly: {status['hourly']['percent']}% used "
                f"(resets in {status['hourly']['resets_in_min']}m)\n"
                f"Daily: {status['daily']['percent']}% used "
                f"(resets in {status['daily']['resets_in_hours']}h)\n"
                f"Switching to Ollama-only mode."
            )
        elif status["status"] == "warning":
            return (
                f"⚡ API budget at {max(status['hourly']['percent'], status['daily']['percent'])}%. "
                f"Using Ollama for non-critical operations."
            )

        return None


# Global tracker instance
_tracker: APIBudgetTracker | None = None


def get_tracker() -> APIBudgetTracker:
    """Get or create the global budget tracker."""
    global _tracker
    if _tracker is None:
        _tracker = APIBudgetTracker()
    return _tracker


def check_and_record(
    estimated_tokens: int,
    operation_type: OperationType = "routine",
) -> tuple[bool, str]:
    """
    Convenience function for checking budget and recording usage.

    Usage:
        allowed, reason = check_and_record(5000, "strategic")
        if not allowed:
            # Fall back to Ollama or wait
            return use_ollama_instead()

    Returns:
        (allowed, reason)
    """
    tracker = get_tracker()
    allowed, reason = tracker.check_budget(estimated_tokens, operation_type)

    if allowed:
        # Pre-record the estimated cost (will be corrected later with actual)
        tracker.record_usage(estimated_tokens, operation_type)

    return allowed, reason


if __name__ == "__main__":
    # Demo/testing
    tracker = get_tracker()
    status = tracker.get_status()

    print(json.dumps(status, indent=2))

    if warning := tracker.get_budget_warning():
        print(f"\n{warning}")
