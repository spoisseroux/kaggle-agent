"""Experiment pacing and resource limit enforcement.

Enforces limits from config/experiment_limits.yaml:
- Experiment counts (hourly, daily, per-phase)
- Minimum intervals between experiments
- Adaptive scaling (deadline-aware, hot streaks, resource constraints)
- Emergency stop conditions

Integrates with:
- core/api_budget.py for API token tracking
- core/gpu_monitor.py for GPU temperature (if exists)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from core.api_budget import get_tracker as get_api_tracker

log = logging.getLogger(__name__)

PhaseType = Literal["eda", "baseline", "feature_engineering", "model_selection", "tuning", "ensembling"]


@dataclass
class ExperimentState:
    """Tracks experiment counts and timestamps."""
    hourly_count: int = 0
    hourly_reset_at: float = 0.0
    daily_count: int = 0
    daily_reset_at: float = 0.0
    phase_counts: dict[str, int] = field(default_factory=dict)
    last_experiment_at: float = 0.0
    recent_improvements: list[bool] = field(default_factory=list)  # Last N experiments
    consecutive_errors: int = 0
    total_errors: int = 0
    total_experiments: int = 0


class ExperimentPacer:
    """Enforces experiment pacing limits with adaptive scaling."""

    def __init__(
        self,
        config_path: Path | str = "config/experiment_limits.yaml",
        competition_type: str = "default",
        deadline: datetime | None = None,
        state_file: Path | str | None = None,
    ):
        self.config_path = Path(config_path)
        self.competition_type = competition_type
        self.deadline = deadline

        if state_file is None:
            state_file = Path.home() / ".claude" / "projects" / "-home-keehar-kaggle-agent" / "experiment_pacer_state.json"
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()
        self.state = self._load_state()

    def _load_config(self) -> dict:
        """Load experiment limits configuration."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Merge defaults with competition-specific overrides
        limits = config["defaults"].copy()

        if self.competition_type in config.get("competition_overrides", {}):
            overrides = config["competition_overrides"][self.competition_type]
            limits.update(overrides)

        # Store full config for adaptive scaling
        limits["adaptive"] = config.get("adaptive", {})
        limits["emergency_stop"] = config.get("emergency_stop", {})
        limits["phase_limits"] = config.get("phase_limits", {})

        return limits

    def _load_state(self) -> ExperimentState:
        """Load state from disk or initialize fresh."""
        if not self.state_file.exists():
            return self._fresh_state()

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            now = time.time()
            state = ExperimentState(**data)

            # Reset if timestamps passed
            if now > state.hourly_reset_at:
                state.hourly_count = 0
                state.hourly_reset_at = now + 3600

            if now > state.daily_reset_at:
                state.daily_count = 0
                # Reset at next UTC midnight
                tomorrow = datetime.now(timezone.utc).date()
                next_midnight = datetime.combine(
                    tomorrow, datetime.min.time(), tzinfo=timezone.utc
                ).timestamp() + 86400
                state.daily_reset_at = next_midnight

            return state
        except Exception as e:
            log.warning(f"Failed to load experiment state: {e}. Starting fresh.")
            return self._fresh_state()

    def _fresh_state(self) -> ExperimentState:
        """Create a fresh experiment state."""
        now = time.time()
        tomorrow = datetime.now(timezone.utc).date()
        next_midnight = datetime.combine(
            tomorrow, datetime.min.time(), tzinfo=timezone.utc
        ).timestamp() + 86400

        return ExperimentState(
            hourly_reset_at=now + 3600,
            daily_reset_at=next_midnight,
        )

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump({
                    "hourly_count": self.state.hourly_count,
                    "hourly_reset_at": self.state.hourly_reset_at,
                    "daily_count": self.state.daily_count,
                    "daily_reset_at": self.state.daily_reset_at,
                    "phase_counts": self.state.phase_counts,
                    "last_experiment_at": self.state.last_experiment_at,
                    "recent_improvements": self.state.recent_improvements,
                    "consecutive_errors": self.state.consecutive_errors,
                    "total_errors": self.state.total_errors,
                    "total_experiments": self.state.total_experiments,
                }, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save experiment state: {e}")

    def _get_adaptive_multiplier(self) -> float:
        """Calculate adaptive scaling multiplier based on context."""
        if not self.config.get("adaptive", {}).get("enabled", True):
            return 1.0

        multiplier = 1.0
        adaptive_config = self.config["adaptive"]

        # Deadline-aware scaling
        if self.deadline:
            # Ensure deadline is timezone-aware
            if self.deadline.tzinfo is None:
                deadline_aware = self.deadline.replace(tzinfo=timezone.utc)
            else:
                deadline_aware = self.deadline
            days_left = (deadline_aware - datetime.now(timezone.utc)).total_seconds() / 86400
            thresholds = adaptive_config.get("days_left_thresholds", [])
            for threshold in sorted(thresholds, key=lambda x: x["days"], reverse=True):
                if days_left <= threshold["days"]:
                    multiplier *= threshold["multiplier"]
                    break

        # Hot streak detection
        hot_streak_threshold = adaptive_config.get("hot_streak_threshold", 3)
        if len(self.state.recent_improvements) >= hot_streak_threshold:
            if all(self.state.recent_improvements[-hot_streak_threshold:]):
                hot_streak_mult = adaptive_config.get("hot_streak_multiplier", 1.5)
                multiplier *= hot_streak_mult

        # API budget constraints
        api_tracker = get_api_tracker()
        api_status = api_tracker.get_status()
        hourly_pct = api_status["hourly"]["percent"] / 100
        daily_pct = api_status["daily"]["percent"] / 100

        critical_threshold = adaptive_config.get("api_budget_critical_threshold", 0.9)
        if hourly_pct > critical_threshold or daily_pct > critical_threshold:
            critical_mult = adaptive_config.get("api_budget_critical_multiplier", 0.3)
            multiplier *= critical_mult

        # GPU temperature constraints (if available)
        try:
            from core.gpu_monitor import get_gpu_temp
            temp = get_gpu_temp()
            temp_warning = adaptive_config.get("gpu_temp_warning_threshold", 80)
            if temp > temp_warning:
                temp_mult = adaptive_config.get("gpu_temp_warning_multiplier", 0.5)
                multiplier *= temp_mult
        except (ImportError, Exception):
            # GPU monitoring not available, skip
            pass

        return multiplier

    def check_can_run(self, phase: PhaseType | None = None) -> tuple[bool, str]:
        """
        Check if an experiment can run now.

        Returns:
            (allowed, reason) - reason is empty if allowed, else explanation why not
        """
        # Reload state (might have been updated by another process)
        self.state = self._load_state()

        # Check emergency stop conditions first
        emergency, reason = self._check_emergency_stop()
        if emergency:
            return False, f"EMERGENCY STOP: {reason}"

        # Apply adaptive multiplier
        multiplier = self._get_adaptive_multiplier()

        # Check hourly limit
        max_hourly = int(self.config["max_experiments_per_hour"] * multiplier)
        if self.state.hourly_count >= max_hourly:
            mins_left = int((self.state.hourly_reset_at - time.time()) / 60)
            return False, f"Hourly limit reached ({self.state.hourly_count}/{max_hourly}). Resets in {mins_left}min."

        # Check daily limit
        max_daily = int(self.config["max_experiments_per_day"] * multiplier)
        if self.state.daily_count >= max_daily:
            hours_left = int((self.state.daily_reset_at - time.time()) / 3600)
            return False, f"Daily limit reached ({self.state.daily_count}/{max_daily}). Resets in {hours_left}h."

        # Check minimum interval
        min_interval = self.config["min_experiment_interval_min"] * 60  # Convert to seconds
        if self.state.last_experiment_at > 0:
            time_since_last = time.time() - self.state.last_experiment_at
            if time_since_last < min_interval:
                wait_sec = int(min_interval - time_since_last)
                return False, f"Minimum interval not met. Wait {wait_sec}s."

        # Check phase-specific limits
        if phase and phase in self.config.get("phase_limits", {}):
            phase_config = self.config["phase_limits"][phase]
            phase_count = self.state.phase_counts.get(phase, 0)
            max_phase = phase_config.get("max_experiments", 999999)
            if phase_count >= max_phase:
                return False, f"Phase '{phase}' limit reached ({phase_count}/{max_phase})."

        # Check consecutive improvements stopping condition
        max_without_improvement = self.config.get("max_experiments_without_improvement", 8)
        if len(self.state.recent_improvements) >= max_without_improvement:
            if not any(self.state.recent_improvements[-max_without_improvement:]):
                return False, f"No improvement in last {max_without_improvement} experiments. Pausing."

        return True, ""

    def _check_emergency_stop(self) -> tuple[bool, str]:
        """Check for emergency stop conditions."""
        emergency_config = self.config.get("emergency_stop", {})

        # Consecutive errors
        max_consecutive = emergency_config.get("max_consecutive_errors", 5)
        if self.state.consecutive_errors >= max_consecutive:
            return True, f"{self.state.consecutive_errors} consecutive errors"

        # Error rate
        if self.state.total_experiments > 10:  # Need minimum sample
            error_rate = self.state.total_errors / self.state.total_experiments
            max_error_rate = emergency_config.get("max_error_rate", 0.5)
            if error_rate > max_error_rate:
                return True, f"Error rate {error_rate:.1%} > {max_error_rate:.1%}"

        # API rate limit hits (tracked in api_budget.py if needed)
        # TODO: Could add rate limit hit counter

        # Disk space
        try:
            import shutil
            disk_free_mb = shutil.disk_usage(".").free / (1024 * 1024)
            min_disk_mb = emergency_config.get("disk_space_mb", 1000)
            if disk_free_mb < min_disk_mb:
                return True, f"Disk space critically low: {disk_free_mb:.0f}MB < {min_disk_mb}MB"
        except Exception:
            pass

        return False, ""

    def record_experiment(
        self,
        phase: PhaseType | None = None,
        improved: bool = False,
        errored: bool = False,
    ) -> None:
        """Record that an experiment was run."""
        self.state.hourly_count += 1
        self.state.daily_count += 1
        self.state.total_experiments += 1
        self.state.last_experiment_at = time.time()

        if phase:
            self.state.phase_counts[phase] = self.state.phase_counts.get(phase, 0) + 1

        # Track improvements (keep last 10)
        self.state.recent_improvements.append(improved)
        if len(self.state.recent_improvements) > 10:
            self.state.recent_improvements.pop(0)

        # Track errors
        if errored:
            self.state.consecutive_errors += 1
            self.state.total_errors += 1
        else:
            self.state.consecutive_errors = 0

        self._save_state()

    def get_status(self) -> dict:
        """Get current pacing status for display."""
        self.state = self._load_state()

        multiplier = self._get_adaptive_multiplier()
        max_hourly = int(self.config["max_experiments_per_hour"] * multiplier)
        max_daily = int(self.config["max_experiments_per_day"] * multiplier)

        return {
            "hourly": {
                "count": self.state.hourly_count,
                "limit": max_hourly,
                "percent": round(100 * self.state.hourly_count / max_hourly, 1) if max_hourly > 0 else 0,
            },
            "daily": {
                "count": self.state.daily_count,
                "limit": max_daily,
                "percent": round(100 * self.state.daily_count / max_daily, 1) if max_daily > 0 else 0,
            },
            "adaptive_multiplier": round(multiplier, 2),
            "recent_improvements": self.state.recent_improvements[-5:],
            "consecutive_errors": self.state.consecutive_errors,
            "total_experiments": self.state.total_experiments,
        }


if __name__ == "__main__":
    # Demo/testing
    pacer = ExperimentPacer(competition_type="playground")
    allowed, reason = pacer.check_can_run(phase="feature_engineering")

    print(f"Can run: {allowed}")
    if not allowed:
        print(f"Reason: {reason}")

    print(f"\nStatus: {json.dumps(pacer.get_status(), indent=2)}")
