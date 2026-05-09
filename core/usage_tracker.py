"""Track Claude Code API usage and manage rate limits."""
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

USAGE_FILE = Path.home() / ".claude" / "usage_tracker.json"
USAGE_FILE.parent.mkdir(exist_ok=True)

# Rate limit thresholds (conservative estimates)
TOKENS_PER_MINUTE_LIMIT = 40000  # Anthropic's typical limit
TOKENS_PER_DAY_LIMIT = 1000000  # Typical daily limit
REQUESTS_PER_MINUTE_LIMIT = 50

# Warning thresholds (switch to multi-agent before hitting limits)
TOKEN_WARNING_THRESHOLD = 0.7  # 70% of limit
REQUEST_WARNING_THRESHOLD = 0.8  # 80% of limit


class UsageTracker:
    """Track API usage and predict when to switch modes."""
    
    def __init__(self):
        self.usage_file = USAGE_FILE
        self.load_usage()
    
    def load_usage(self):
        """Load usage data from disk."""
        if self.usage_file.exists():
            with open(self.usage_file) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "minute_window": [],  # [(timestamp, tokens, requests), ...]
                "day_window": [],
                "total_tokens": 0,
                "total_requests": 0,
                "last_rate_limit": None,
            }
    
    def save_usage(self):
        """Save usage data to disk."""
        with open(self.usage_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_request(self, tokens_used: int):
        """Record a Claude Code API request."""
        now = time.time()
        
        # Add to windows
        self.data["minute_window"].append((now, tokens_used, 1))
        self.data["day_window"].append((now, tokens_used, 1))
        
        # Update totals
        self.data["total_tokens"] += tokens_used
        self.data["total_requests"] += 1
        
        # Clean old entries
        self._clean_windows(now)
        
        self.save_usage()
    
    def _clean_windows(self, now: float):
        """Remove old entries from sliding windows."""
        # Keep last 60 seconds for minute window
        self.data["minute_window"] = [
            (ts, tok, req) for ts, tok, req in self.data["minute_window"]
            if now - ts < 60
        ]
        
        # Keep last 24 hours for day window
        self.data["day_window"] = [
            (ts, tok, req) for ts, tok, req in self.data["day_window"]
            if now - ts < 86400
        ]
    
    def get_current_usage(self):
        """Get current usage stats."""
        now = time.time()
        self._clean_windows(now)
        
        minute_tokens = sum(tok for _, tok, _ in self.data["minute_window"])
        minute_requests = sum(req for _, _, req in self.data["minute_window"])
        
        day_tokens = sum(tok for _, tok, _ in self.data["day_window"])
        day_requests = sum(req for _, _, req in self.data["day_window"])
        
        return {
            "minute_tokens": minute_tokens,
            "minute_requests": minute_requests,
            "day_tokens": day_tokens,
            "day_requests": day_requests,
            "minute_token_pct": minute_tokens / TOKENS_PER_MINUTE_LIMIT,
            "minute_request_pct": minute_requests / REQUESTS_PER_MINUTE_LIMIT,
            "day_token_pct": day_tokens / TOKENS_PER_DAY_LIMIT,
        }
    
    def should_use_multiagent(self) -> tuple[bool, str]:
        """
        Determine if we should use multi-agent mode to conserve API usage.
        
        Returns:
            (use_multiagent, reason)
        """
        usage = self.get_current_usage()
        
        # Check minute limits
        if usage["minute_token_pct"] > TOKEN_WARNING_THRESHOLD:
            return True, f"Token usage at {usage['minute_token_pct']:.0%} of minute limit"
        
        if usage["minute_request_pct"] > REQUEST_WARNING_THRESHOLD:
            return True, f"Request rate at {usage['minute_request_pct']:.0%} of minute limit"
        
        # Check day limits
        if usage["day_token_pct"] > TOKEN_WARNING_THRESHOLD:
            return True, f"Daily tokens at {usage['day_token_pct']:.0%} of limit"
        
        return False, "API usage normal"
    
    def is_rate_limited(self) -> tuple[bool, int]:
        """
        Check if we're currently rate limited.
        
        Returns:
            (is_limited, wait_seconds)
        """
        if self.data.get("last_rate_limit"):
            # Assume 60s cooldown after rate limit
            limit_time = datetime.fromisoformat(self.data["last_rate_limit"])
            now = datetime.now()
            elapsed = (now - limit_time).total_seconds()
            
            if elapsed < 60:
                return True, int(60 - elapsed)
        
        usage = self.get_current_usage()
        
        # Hard limits
        if usage["minute_tokens"] >= TOKENS_PER_MINUTE_LIMIT:
            return True, 60
        
        if usage["minute_requests"] >= REQUESTS_PER_MINUTE_LIMIT:
            return True, 60
        
        return False, 0
    
    def record_rate_limit(self):
        """Record that we hit a rate limit."""
        self.data["last_rate_limit"] = datetime.now().isoformat()
        self.save_usage()
    
    def get_usage_summary(self) -> str:
        """Get human-readable usage summary."""
        usage = self.get_current_usage()
        
        return f"""API Usage:
  Last minute: {usage['minute_tokens']:,} tokens ({usage['minute_token_pct']:.0%}), {usage['minute_requests']} requests ({usage['minute_request_pct']:.0%})
  Last 24h: {usage['day_tokens']:,} tokens ({usage['day_token_pct']:.0%})
  Total: {self.data['total_tokens']:,} tokens, {self.data['total_requests']:,} requests"""


# Global tracker instance
_tracker = None

def get_tracker() -> UsageTracker:
    """Get global usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


def record_request(tokens: int):
    """Convenience function to record a request."""
    get_tracker().record_request(tokens)


def should_use_multiagent() -> tuple[bool, str]:
    """Convenience function to check if multi-agent mode should be used."""
    return get_tracker().should_use_multiagent()


def is_rate_limited() -> tuple[bool, int]:
    """Convenience function to check rate limit status."""
    return get_tracker().is_rate_limited()
