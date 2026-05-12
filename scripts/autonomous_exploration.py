#!/usr/bin/env python3
"""Autonomous Exploration Framework

Runs systematic A/B tests without human intervention.
Tracks patterns, writes to research DB, escalates only critical decisions.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.hypothesis_db import HypothesisDatabase
    HAS_DB = True
except ImportError:
    HAS_DB = False

DATA_DIR = Path("data/store-sales-time-series-forecasting")
PREDICTIONS_DIR = Path("competitions/active/store-sales-time-series-forecasting/predictions")
RESULTS_FILE = Path("experiments/autonomous_exploration_results.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)


class ExplorationTracker:
    """Track exploration results and identify patterns."""

    def __init__(self):
        self.results = []

    def add_result(self, test_name, cv_score, baseline_cv, params, metadata):
        """Add test result."""
        delta = cv_score - baseline_cv
        pct_change = (delta / baseline_cv) * 100

        result = {
            'test_name': test_name,
            'cv_score': float(cv_score),
            'baseline_cv': float(baseline_cv),
            'delta': float(delta),
            'pct_change': float(pct_change),
            'improved': bool(cv_score < baseline_cv),
            'params': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                      for k, v in params.items()},
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }

        self.results.append(result)

        # Save after each test
        self.save()

    def save(self):
        """Save results to file."""
        with open(RESULTS_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_summary(self):
        """Get summary statistics."""
        if not self.results:
            return {}

        improved = [r for r in self.results if r['improved']]
        degraded = [r for r in self.results if not r['improved']]

        return {
            'total_tests': len(self.results),
            'improved': len(improved),
            'degraded': len(degraded),
            'best_cv': min(r['cv_score'] for r in self.results),
            'best_test': min(self.results, key=lambda x: x['cv_score'])['test_name'],
            'avg_delta': np.mean([r['delta'] for r in self.results]),
            'pattern': self._identify_pattern()
        }

    def _identify_pattern(self):
        """Identify patterns across tests."""
        if len(self.results) < 3:
            return "Insufficient data"

        improved_count = sum(1 for r in self.results if r['improved'])

        if improved_count == 0:
            return "All tests degraded - baseline is optimal"
        elif improved_count == len(self.results):
            return "All tests improved - baseline was suboptimal"
        else:
            return f"Mixed results - {improved_count}/{len(self.results)} improved"


def test_ridge_weights():
    """Test different Ridge ensemble weights."""
    print("\n" + "="*70)
    print("EXPLORATION 1: Ridge Weight Variations")
    print("="*70)

    tracker = ExplorationTracker()

    # Load predictions
    lgb_val = np.load(PREDICTIONS_DIR / "v51_lgbm_val.npy")
    xgb_val = np.load(PREDICTIONS_DIR / "v51_xgb_val.npy")
    y_val = np.load(PREDICTIONS_DIR / "v51_y_val.npy")

    baseline_cv = 0.3925  # v50 Ridge with learned weights

    # Test different weight combinations
    weight_tests = [
        (0.60, 0.40, "60/40 LGB/XGB"),
        (0.70, 0.30, "70/30 LGB/XGB"),
        (0.80, 0.20, "80/20 LGB/XGB"),
        (0.90, 0.10, "90/10 LGB/XGB"),
        (0.50, 0.50, "50/50 LGB/XGB"),
    ]

    print(f"Baseline (v50 learned weights): CV {baseline_cv:.4f}")
    print(f"Testing {len(weight_tests)} weight combinations...\n")

    for lgb_w, xgb_w, name in weight_tests:
        # Ridge with fixed weights
        meta_train = np.column_stack([lgb_val, xgb_val])
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(meta_train, y_val)

        # Override weights manually
        ridge_manual = Ridge(alpha=1.0, fit_intercept=True)
        ridge_manual.coef_ = np.array([lgb_w, xgb_w])
        ridge_manual.intercept_ = ridge.intercept_

        preds = lgb_w * lgb_val + xgb_w * xgb_val + ridge.intercept_
        preds = np.maximum(preds, 0)
        cv = np.sqrt(mean_squared_log_error(y_val, preds))

        tracker.add_result(
            test_name=f"ridge_weights_{name.replace('/', '_')}",
            cv_score=cv,
            baseline_cv=baseline_cv,
            params={'lgb_weight': lgb_w, 'xgb_weight': xgb_w},
            metadata={'test_type': 'ridge_weights'}
        )

        delta = cv - baseline_cv
        symbol = "✓" if cv < baseline_cv else "✗"
        print(f"{symbol} {name:<20} CV {cv:.4f}  ({delta:+.4f})")

    summary = tracker.get_summary()
    print(f"\nPattern: {summary['pattern']}")
    print(f"Best: {summary['best_test']} → CV {summary['best_cv']:.4f}")

    return tracker


def test_seed_ensembles():
    """Test seed-based ensembles."""
    print("\n" + "="*70)
    print("EXPLORATION 2: Seed Ensembles")
    print("="*70)

    tracker = ExplorationTracker()

    # Load data
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])

    # Quick feature creation
    train = train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    train["day_of_week"] = train["date"].dt.dayofweek
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)
    train["Lag_3"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(3)
    train["Lag_7"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].shift(7)

    for window in [7, 14, 30, 60, 90]:
        train[f"Roll_mean_{window}"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    train["Roll_std_7"] = train.groupby(["store_nbr", "family"], observed=True)["sales"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )

    national_holidays = holidays[holidays["locale"] == "National"]["date"].unique()
    train["is_holiday"] = train["date"].isin(national_holidays).astype(int)
    train = train.dropna()

    feature_cols = ["onpromotion", "day_of_week", "is_weekend", "Lag_3", "Lag_7",
                    "Roll_mean_7", "Roll_mean_14", "Roll_mean_30", "Roll_mean_60",
                    "Roll_mean_90", "Roll_std_7", "is_holiday"]

    cutoff_date = train["date"].max() - pd.Timedelta(days=30)
    val_mask = train["date"] >= cutoff_date
    train_mask = ~val_mask

    X_train = train.loc[train_mask, feature_cols]
    y_train = train.loc[train_mask, "sales"]
    X_val = train.loc[val_mask, feature_cols]
    y_val = train.loc[val_mask, "sales"]

    baseline_cv = 0.3572  # v19 single seed

    # Test different seed combinations
    seed_tests = [
        ([42, 123], "2-seed ensemble"),
        ([42, 123, 456], "3-seed ensemble"),
        ([42, 123, 456, 789, 2024], "5-seed ensemble"),
    ]

    print(f"Baseline (single seed=42): CV {baseline_cv:.4f}")
    print(f"Testing {len(seed_tests)} seed ensemble strategies...\n")

    for seeds, name in seed_tests:
        predictions = []

        for seed in seeds:
            model = lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.05,
                num_leaves=64,
                max_depth=6,
                random_state=seed,
                n_estimators=400,
                verbosity=-1
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

            preds = model.predict(X_val)
            predictions.append(preds)

        # Average predictions
        avg_preds = np.mean(predictions, axis=0)
        avg_preds = np.maximum(avg_preds, 0)
        cv = np.sqrt(mean_squared_log_error(y_val, avg_preds))

        tracker.add_result(
            test_name=f"seed_ensemble_{len(seeds)}seeds",
            cv_score=cv,
            baseline_cv=baseline_cv,
            params={'seeds': seeds, 'n_seeds': len(seeds)},
            metadata={'test_type': 'seed_ensemble'}
        )

        delta = cv - baseline_cv
        symbol = "✓" if cv < baseline_cv else "✗"
        print(f"{symbol} {name:<20} CV {cv:.4f}  ({delta:+.4f})")

    summary = tracker.get_summary()
    print(f"\nPattern: {summary['pattern']}")

    return tracker


def main():
    """Run autonomous exploration."""
    print("="*70)
    print("AUTONOMOUS EXPLORATION")
    print("="*70)
    print(f"Started: {datetime.now()}")
    print()

    all_results = []

    # Exploration 1: Ridge weights
    print("Phase 1/2: Testing Ridge weight variations...")
    ridge_tracker = test_ridge_weights()
    all_results.extend(ridge_tracker.results)

    # Exploration 2: Seed ensembles
    print("\nPhase 2/2: Testing seed ensembles...")
    seed_tracker = test_seed_ensembles()
    all_results.extend(seed_tracker.results)

    # Overall summary
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)

    improved = [r for r in all_results if r['improved']]
    best = min(all_results, key=lambda x: x['cv_score'])

    print(f"Total tests: {len(all_results)}")
    print(f"Improved: {len(improved)}")
    print(f"Degraded: {len(all_results) - len(improved)}")
    print(f"\nBest result: {best['test_name']}")
    print(f"  CV: {best['cv_score']:.4f}")
    print(f"  vs baseline: {best['delta']:+.4f} ({best['pct_change']:+.2f}%)")

    if len(improved) == 0:
        print("\n⚠️ FINDING: All tests degraded - baseline is optimal")
    else:
        print(f"\n✓ FINDING: {len(improved)}/{len(all_results)} tests improved")

    # Write to hypothesis DB
    if HAS_DB:
        try:
            db = HypothesisDatabase()
            for result in all_results:
                hyp_id = db.add_hypothesis(
                    competition="store-sales-time-series-forecasting",
                    experiment_id=result['test_name'],
                    hypothesis=f"Exploration test: {result['test_name']}",
                    rationale="Autonomous exploration to find patterns",
                    category="exploration"
                )
                db.record_result(
                    hypothesis_id=hyp_id,
                    cv_score=result['cv_score'],
                    lb_score=None,
                    baseline_cv=result['baseline_cv'],
                    succeeded=result['improved'],
                    params=result['params'],
                    metadata=result['metadata']
                )
            print("\n✓ Logged all results to hypothesis database")
        except Exception as e:
            print(f"\nNote: Could not log to DB: {e}")

    print(f"\nResults saved: {RESULTS_FILE}")
    print(f"Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
