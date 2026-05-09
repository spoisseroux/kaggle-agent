"""Classify tasks as quick (single-agent) or complex (multi-agent)."""
import re
from typing import Literal

TaskType = Literal["quick", "complex"]


class TaskClassifier:
    """Classify user requests to determine execution mode."""
    
    # Keywords that suggest quick single-agent tasks
    QUICK_INDICATORS = [
        "debug", "fix", "check", "show", "explain", "why",
        "what's", "how come", "status", "summary", "list",
        "compare", "diff", "inspect", "analyze result",
        "small change", "quick", "simple"
    ]
    
    # Keywords that suggest complex multi-agent tasks
    COMPLEX_INDICATORS = [
        "build", "create", "implement", "develop", "generate",
        "full workflow", "end-to-end", "competition run",
        "feature engineering", "optimize", "tune", "ensemble",
        "research", "experiment batch", "iterate", "pipeline"
    ]
    
    # Explicit mode requests
    MODE_KEYWORDS = {
        "single-agent": ["yourself", "single agent", "single-agent", "just you", "no agents"],
        "multi-agent": ["multi agent", "multi-agent", "use agents", "use ollama", "autonomous workflow"]
    }
    
    def classify(self, user_input: str) -> tuple[TaskType, str, float]:
        """
        Classify a user request.
        
        Returns:
            (task_type, reason, confidence)
        """
        lower_input = user_input.lower()
        
        # Check for explicit mode requests
        for mode, keywords in self.MODE_KEYWORDS.items():
            if any(kw in lower_input for kw in keywords):
                if mode == "single-agent":
                    return "quick", f"Explicit request: {mode}", 1.0
                else:
                    return "complex", f"Explicit request: {mode}", 1.0
        
        # Count indicators
        quick_score = sum(1 for kw in self.QUICK_INDICATORS if kw in lower_input)
        complex_score = sum(1 for kw in self.COMPLEX_INDICATORS if kw in lower_input)
        
        # Length heuristic (very short messages usually quick)
        word_count = len(user_input.split())
        if word_count < 10:
            quick_score += 1
        
        # Code blocks suggest implementation work (complex)
        if "```" in user_input:
            complex_score += 2
        
        # File operations
        if re.search(r'\b(file|script|model|pipeline)\b', lower_input):
            if any(verb in lower_input for verb in ['create', 'build', 'implement', 'write']):
                complex_score += 1
        
        # Questions are usually quick
        if "?" in user_input and word_count < 20:
            quick_score += 1
        
        # Determine classification
        confidence = abs(complex_score - quick_score) / max(complex_score + quick_score, 1)
        
        if complex_score > quick_score:
            return "complex", f"Indicators: {complex_score} complex vs {quick_score} quick", confidence
        elif quick_score > complex_score:
            return "quick", f"Indicators: {quick_score} quick vs {complex_score} complex", confidence
        else:
            # Tie - default to quick for responsiveness
            return "quick", "Unclear - defaulting to quick", 0.3
    
    def should_use_multiagent(
        self, 
        user_input: str,
        force_reason: str = None
    ) -> tuple[bool, str]:
        """
        Determine if multi-agent mode should be used.
        
        Args:
            user_input: The user's message
            force_reason: Optional reason to force multi-agent (e.g., API limits)
        
        Returns:
            (use_multiagent, reason)
        """
        if force_reason:
            return True, force_reason
        
        task_type, reason, confidence = self.classify(user_input)
        
        if task_type == "complex":
            return True, f"Complex task detected: {reason}"
        else:
            return False, f"Quick task detected: {reason}"


# Global classifier instance
_classifier = None


def get_classifier() -> TaskClassifier:
    """Get global task classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = TaskClassifier()
    return _classifier


def classify_task(user_input: str) -> tuple[TaskType, str, float]:
    """Convenience function to classify a task."""
    return get_classifier().classify(user_input)


def should_use_multiagent(user_input: str, force_reason: str = None) -> tuple[bool, str]:
    """Convenience function to determine execution mode."""
    return get_classifier().should_use_multiagent(user_input, force_reason)
