#!/usr/bin/env python3
"""Analyze validation window bias and find optimal validation period

Current validation (Jul 16 - Aug 15) has +33% elevated sales.
This analysis finds 30-day windows closest to overall training mean.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/store-sales-time-series-forecasting")

def main():
    print("="*70)
    print("VALIDATION WINDOW ANALYSIS")
    print("="*70)

    # Load training data
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )

    # Overall stats
    overall_mean = train["sales"].mean()
    overall_median = train["sales"].median()
    overall_std = train["sales"].std()

    print(f"\nOverall Training Statistics:")
    print(f"  Date range: {train['date'].min()} to {train['date'].max()}")
    print(f"  Mean: {overall_mean:.2f}")
    print(f"  Median: {overall_median:.2f}")
    print(f"  Std: {overall_std:.2f}")
    print()

    # Current validation window
    current_val_start = pd.Timestamp("2017-07-16")
    current_val_end = pd.Timestamp("2017-08-15")
    current_val = train[(train["date"] >= current_val_start) & (train["date"] <= current_val_end)]
    current_val_mean = current_val["sales"].mean()

    print(f"Current Validation Window (Jul 16 - Aug 15):")
    print(f"  Mean: {current_val_mean:.2f}")
    print(f"  Deviation: {((current_val_mean - overall_mean) / overall_mean * 100):+.1f}%")
    print(f"  CV-LB gap on v67: 16.4% (CV 0.3908 → LB 0.45416)")
    print()

    # Analyze all possible 30-day windows
    print("Analyzing all 30-day validation windows...")
    print()

    # Get unique dates
    dates = sorted(train["date"].unique())

    # We need at least 60 days before validation for training
    # And validation should end at least 15 days before max date (test set)
    min_val_start = dates[60]  # 60 days after start
    max_val_end = train["date"].max() - pd.Timedelta(days=15)  # 15 days before test

    windows = []

    for i, date in enumerate(dates):
        if date < min_val_start:
            continue

        val_start = date
        val_end = date + pd.Timedelta(days=30)

        if val_end > max_val_end:
            break

        # Get validation window data
        val_data = train[(train["date"] >= val_start) & (train["date"] <= val_end)]

        if len(val_data) < 1000:  # Skip if too few samples
            continue

        val_mean = val_data["sales"].mean()
        val_median = val_data["sales"].median()
        deviation = abs(val_mean - overall_mean)
        deviation_pct = (val_mean - overall_mean) / overall_mean * 100

        windows.append({
            "start": val_start,
            "end": val_end,
            "mean": val_mean,
            "median": val_median,
            "deviation": deviation,
            "deviation_pct": deviation_pct,
            "samples": len(val_data)
        })

    # Sort by deviation (lowest = most representative)
    windows_df = pd.DataFrame(windows).sort_values("deviation")

    print("Top 10 Most Representative Validation Windows:")
    print("="*70)
    print(f"{'Rank':<6} {'Start':<12} {'End':<12} {'Mean':<10} {'Dev %':<10} {'Samples'}")
    print("-"*70)

    for i, row in windows_df.head(10).iterrows():
        print(f"{len(windows_df) - list(windows_df.index).index(i):<6} "
              f"{row['start'].strftime('%Y-%m-%d'):<12} "
              f"{row['end'].strftime('%Y-%m-%d'):<12} "
              f"{row['mean']:<10.2f} "
              f"{row['deviation_pct']:<+10.1f} "
              f"{row['samples']}")

    print()
    print("Current validation window rank:")
    current_matches = windows_df[
        (windows_df['start'] == current_val_start) &
        (windows_df['end'] <= current_val_end + pd.Timedelta(days=1))
    ]
    if len(current_matches) > 0:
        current_rank = len(windows_df) - list(windows_df.index).index(current_matches.index[0])
        print(f"  Rank: {current_rank} / {len(windows_df)}")
        print(f"  Deviation: {((current_val_mean - overall_mean) / overall_mean * 100):+.1f}%")
    else:
        print(f"  Not in analyzed range (too close to test set)")
        print(f"  Deviation: {((current_val_mean - overall_mean) / overall_mean * 100):+.1f}%")
    print()

    # Recommendation
    best_window = windows_df.iloc[0]
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nBest validation window: {best_window['start'].strftime('%Y-%m-%d')} to {best_window['end'].strftime('%Y-%m-%d')}")
    print(f"  Mean: {best_window['mean']:.2f} (vs overall {overall_mean:.2f})")
    print(f"  Deviation: {best_window['deviation_pct']:+.2f}% (vs current {((current_val_mean - overall_mean) / overall_mean * 100):+.1f}%)")
    print(f"  Expected impact:")
    print(f"    - More accurate CV estimates (closer to LB)")
    print(f"    - Feature engineering won't overfit to validation anomalies")
    print(f"    - Models will generalize better")
    print()

    # Save results
    windows_df.to_csv("analysis/validation_windows.csv", index=False)
    print(f"Full results saved to: analysis/validation_windows.csv")
    print()

    # Next steps
    print("NEXT STEPS:")
    print("1. Retrain v67 on best validation window")
    print("2. Compare CV on new window to LB 0.454")
    print("3. If CV-LB gap closes → use this window for feature engineering")
    print("4. Retry v81/v86 features on representative validation window")


if __name__ == "__main__":
    main()
