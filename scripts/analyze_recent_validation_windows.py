#!/usr/bin/env python3
"""Find recent validation windows with good distribution

v87 lesson: Validation needs BOTH:
1. Representative distribution (close to overall mean)
2. Temporal proximity (recent data before test)

This analyzes 2016-2017 windows only.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/store-sales-time-series-forecasting")

def main():
    print("="*70)
    print("RECENT VALIDATION WINDOW ANALYSIS (2016-2017)")
    print("="*70)

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"store_nbr": "category", "family": "category"},
        parse_dates=["date"],
    )

    overall_mean = train["sales"].mean()
    overall_median = train["sales"].median()

    print(f"\nOverall Training Statistics:")
    print(f"  Mean: {overall_mean:.2f}")
    print(f"  Median: {overall_median:.2f}")
    print()

    # Current validation
    current_val = train[
        (train["date"] >= "2017-07-16") &
        (train["date"] <= "2017-08-15")
    ]
    current_mean = current_val["sales"].mean()

    print("Current Validation (Jul 16 - Aug 15, 2017):")
    print(f"  Mean: {current_mean:.2f}")
    print(f"  Deviation: {((current_mean - overall_mean) / overall_mean * 100):+.1f}%")
    print(f"  CV: 0.3908, LB: 0.45416 (16.4% gap)")
    print()

    # Analyze 2016-2017 windows only
    dates = sorted(train["date"].unique())

    # Start from Jan 2016
    start_date = pd.Timestamp("2016-01-01")
    # End before test set (Aug 16, 2017)
    end_date = pd.Timestamp("2017-08-01")  # Leave room for 30-day window

    windows = []

    for date in dates:
        if date < start_date or date > end_date:
            continue

        val_start = date
        val_end = date + pd.Timedelta(days=30)

        if val_end > pd.Timestamp("2017-08-15"):  # Don't overlap with test
            break

        val_data = train[(train["date"] >= val_start) & (train["date"] <= val_end)]

        if len(val_data) < 1000:
            continue

        val_mean = val_data["sales"].mean()
        val_median = val_data["sales"].median()
        deviation = abs(val_mean - overall_mean)
        deviation_pct = (val_mean - overall_mean) / overall_mean * 100

        # Days before test period starts (Aug 16, 2017)
        days_before_test = (pd.Timestamp("2017-08-16") - val_end).days

        windows.append({
            "start": val_start,
            "end": val_end,
            "mean": val_mean,
            "median": val_median,
            "deviation": deviation,
            "deviation_pct": deviation_pct,
            "days_before_test": days_before_test,
            "samples": len(val_data)
        })

    windows_df = pd.DataFrame(windows)

    # Sort by deviation (best distribution)
    windows_df_by_dist = windows_df.sort_values("deviation")

    print("Top 10 by Distribution (closest to overall mean):")
    print("="*80)
    print(f"{'Start':<12} {'End':<12} {'Mean':<8} {'Dev %':<8} {'Days Before Test':<17} {'Samples'}")
    print("-"*80)

    for _, row in windows_df_by_dist.head(10).iterrows():
        print(f"{row['start'].strftime('%Y-%m-%d'):<12} "
              f"{row['end'].strftime('%Y-%m-%d'):<12} "
              f"{row['mean']:<8.2f} "
              f"{row['deviation_pct']:<+8.1f} "
              f"{row['days_before_test']:<17} "
              f"{row['samples']}")

    print()

    # Sort by recency (closest to test)
    windows_df_by_recency = windows_df.sort_values("days_before_test")

    print("Top 10 by Recency (closest to test period):")
    print("="*80)
    print(f"{'Start':<12} {'End':<12} {'Mean':<8} {'Dev %':<8} {'Days Before Test':<17} {'Samples'}")
    print("-"*80)

    for _, row in windows_df_by_recency.head(10).iterrows():
        print(f"{row['start'].strftime('%Y-%m-%d'):<12} "
              f"{row['end'].strftime('%Y-%m-%d'):<12} "
              f"{row['mean']:<8.2f} "
              f"{row['deviation_pct']:<+8.1f} "
              f"{row['days_before_test']:<17} "
              f"{row['samples']}")

    print()

    # Find best trade-off: within 90 days of test, lowest deviation
    recent_windows = windows_df[windows_df["days_before_test"] <= 90]
    if len(recent_windows) > 0:
        recent_windows = recent_windows.sort_values("deviation")

        print("Best Trade-Off (within 90 days of test, best distribution):")
        print("="*80)
        print(f"{'Start':<12} {'End':<12} {'Mean':<8} {'Dev %':<8} {'Days Before Test':<17} {'Samples'}")
        print("-"*80)

        for _, row in recent_windows.head(10).iterrows():
            print(f"{row['start'].strftime('%Y-%m-%d'):<12} "
                  f"{row['end'].strftime('%Y-%m-%d'):<12} "
                  f"{row['mean']:<8.2f} "
                  f"{row['deviation_pct']:<+8.1f} "
                  f"{row['days_before_test']:<17} "
                  f"{row['samples']}")

        print()

        # Recommendation
        best = recent_windows.iloc[0]
        print("="*70)
        print("RECOMMENDATION")
        print("="*70)
        print(f"\nBest recent validation window:")
        print(f"  Period: {best['start'].strftime('%Y-%m-%d')} to {best['end'].strftime('%Y-%m-%d')}")
        print(f"  Mean: {best['mean']:.2f} (vs overall {overall_mean:.2f})")
        print(f"  Deviation: {best['deviation_pct']:+.1f}% (vs current +32.0%)")
        print(f"  Days before test: {best['days_before_test']}")
        print()
        print("Expected impact:")
        print(f"  - Better than current window ({best['deviation_pct']:+.1f}% vs +32.0% deviation)")
        print(f"  - Recent enough to capture 2017 trends")
        print(f"  - Should reduce CV-LB gap from 16.4%")
        print()

        # Save
        windows_df.to_csv("analysis/recent_validation_windows.csv", index=False)
        print("Full results saved to: analysis/recent_validation_windows.csv")

    else:
        print("No windows found within 90 days of test")


if __name__ == "__main__":
    main()
