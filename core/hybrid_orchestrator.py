"""Hybrid orchestrator - intelligently switches between single and multi-agent modes."""
import sys
import time
from pathlib import Path
from typing import Optional

from core.task_classifier import classify_task, should_use_multiagent as classifier_check
from core.usage_tracker import get_tracker, record_request, is_rate_limited
from core.notify import send_telegram
from core.orchestrator import run_multi_agent


class HybridOrchestrator:
    """
    Smart orchestrator that decides when to use single-agent vs multi-agent mode.
    
    Decision logic:
    1. Check API rate limits - if approaching/exceeded, use multi-agent
    2. Classify task complexity - complex tasks use multi-agent
    3. Track mode switches to avoid thrashing
    """
    
    def __init__(self):
        self.tracker = get_tracker()
        self.mode_history = []  # Track last N mode decisions
        self.max_history = 10
    
    def decide_mode(self, user_input: str) -> tuple[str, str]:
        """
        Decide which mode to use.
        
        Returns:
            (mode, reason) where mode is "single" or "multi"
        """
        # Check rate limits first (highest priority)
        limited, wait_seconds = is_rate_limited()
        if limited:
            return "multi", f"Rate limited - waiting {wait_seconds}s, using multi-agent to conserve API usage"
        
        # Check if approaching limits
        should_conserve, conserve_reason = self.tracker.should_use_multiagent()
        if should_conserve:
            return "multi", conserve_reason
        
        # Classify task complexity
        use_multi, task_reason = classifier_check(user_input)
        if use_multi:
            return "multi", task_reason
        else:
            return "single", task_reason
    
    def execute(
        self,
        user_input: str,
        competition_slug: Optional[str] = None,
        phase: Optional[int] = None
    ) -> dict:
        """
        Execute a task using the appropriate mode.
        
        Args:
            user_input: User's request
            competition_slug: Optional competition to work on
            phase: Optional phase to run (for multi-agent mode)
        
        Returns:
            Result dict with mode, reason, and output
        """
        mode, reason = self.decide_mode(user_input)
        
        # Log decision
        self.mode_history.append((mode, reason, time.time()))
        if len(self.mode_history) > self.max_history:
            self.mode_history.pop(0)
        
        # Notify user of mode choice
        mode_emoji = "🤖" if mode == "multi" else "⚡"
        send_telegram(f"{mode_emoji} Using {mode}-agent mode: {reason}")
        
        result = {
            "mode": mode,
            "reason": reason,
            "input": user_input,
        }
        
        if mode == "multi":
            # Use multi-agent system
            if competition_slug:
                agent_result = run_multi_agent(
                    competition_slug,
                    start_phase=phase or 0,
                    end_phase=phase or 8
                )
                result["output"] = agent_result
            else:
                send_telegram("⚠️ Multi-agent mode requires competition_slug")
                result["output"] = {"error": "Missing competition_slug"}
        else:
            # Single-agent mode - just return control to Claude Code
            result["output"] = {
                "message": "Executing in single-agent mode",
                "note": "Claude Code will handle this directly"
            }
        
        return result
    
    def get_usage_report(self) -> str:
        """Get comprehensive usage and decision report."""
        usage_summary = self.tracker.get_usage_summary()
        
        # Recent mode decisions
        recent_modes = self.mode_history[-5:] if self.mode_history else []
        mode_summary = "\n".join([
            f"  {i+1}. {mode.upper()}: {reason}"
            for i, (mode, reason, _) in enumerate(recent_modes)
        ])
        
        limited, wait = is_rate_limited()
        limit_status = f"Rate limited - wait {wait}s" if limited else "Normal"
        
        return f"""{usage_summary}

Rate Limit Status: {limit_status}

Recent Mode Decisions:
{mode_summary if mode_summary else "  (none yet)"}"""
    
    def wait_for_rate_limit_clear(self, max_wait: int = 300):
        """
        Wait for rate limit to clear, with progress updates.
        
        Args:
            max_wait: Maximum seconds to wait (default 5 min)
        """
        limited, wait_seconds = is_rate_limited()
        if not limited:
            return
        
        actual_wait = min(wait_seconds, max_wait)
        send_telegram(f"⏳ Rate limited - waiting {actual_wait}s before continuing...")
        
        # Wait in chunks, updating every 30s
        remaining = actual_wait
        while remaining > 0:
            time.sleep(min(30, remaining))
            remaining -= 30
            if remaining > 0:
                send_telegram(f"⏳ Still waiting - {remaining}s remaining...")
        
        send_telegram("✅ Rate limit cleared - resuming work")


# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> HybridOrchestrator:
    """Get global hybrid orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HybridOrchestrator()
    return _orchestrator


def decide_and_execute(
    user_input: str,
    competition_slug: Optional[str] = None,
    phase: Optional[int] = None
) -> dict:
    """
    Convenience function to decide mode and execute.
    
    This is the main entry point for hybrid mode.
    """
    return get_orchestrator().execute(user_input, competition_slug, phase)


def get_usage_report() -> str:
    """Get usage and decision report."""
    return get_orchestrator().get_usage_report()


if __name__ == "__main__":
    # CLI interface for testing
    if len(sys.argv) < 2:
        print("Usage: python -m core.hybrid_orchestrator '<user_input>' [competition_slug] [phase]")
        sys.exit(1)
    
    user_input = sys.argv[1]
    competition = sys.argv[2] if len(sys.argv) > 2 else None
    phase = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = decide_and_execute(user_input, competition, phase)
    print(f"\nMode: {result['mode']}")
    print(f"Reason: {result['reason']}")
    print(f"\nResult: {result['output']}")
