"""Ensemble multiple models for better predictions

Combines predictions from:
- XGBoost (0.321 RMSLE)
- LightGBM (0.401 RMSLE)
- Optuna-tuned LightGBM (TBD)

Uses weighted average based on CV scores.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_log_error

SUBMISSION_DIR = Path("competitions/active/store-sales-time-series-forecasting/submissions")


def load_submission(filename):
    """Load a submission CSV and return predictions."""
    path = SUBMISSION_DIR / filename
    if not path.exists():
        print(f"Warning: {filename} not found")
        return None
    df = pd.read_csv(path)
    return df


def ensemble_submissions(submissions, weights=None, method="weighted"):
    """
    Ensemble multiple submission files.

    Args:
        submissions: List of (filename, weight) tuples
        method: 'weighted' or 'rank_average'

    Returns:
        Ensembled submission DataFrame
    """
    dfs = []
    actual_weights = []

    for filename, weight in submissions:
        df = load_submission(filename)
        if df is not None:
            dfs.append(df)
            actual_weights.append(weight)
        else:
            print(f"Skipping {filename}")

    if not dfs:
        raise ValueError("No valid submissions found!")

    # Normalize weights
    actual_weights = np.array(actual_weights)
    actual_weights = actual_weights / actual_weights.sum()

    print(f"\nEnsembling {len(dfs)} models:")
    for (filename, _), weight in zip(submissions, actual_weights):
        print(f"  {filename}: {weight:.3f}")

    # Get base structure from first submission
    result = dfs[0][['id']].copy()

    if method == "weighted":
        # Weighted average
        predictions = np.zeros(len(result))
        for df, weight in zip(dfs, actual_weights):
            predictions += df['sales'].values * weight
        result['sales'] = predictions

    elif method == "rank_average":
        # Rank averaging (more robust to outliers)
        ranks = np.zeros(len(result))
        for df in dfs:
            ranks += df['sales'].rank().values
        ranks /= len(dfs)
        # Convert ranks back to predictions (use median of group)
        result['sales'] = dfs[0]['sales'].values  # Placeholder
        print("Warning: rank_average not fully implemented, using first model")

    return result


def main():
    """Create ensemble submission."""
    print("=== Model Ensemble ===\n")

    # List available submissions
    print("Available submissions:")
    for f in sorted(SUBMISSION_DIR.glob("*.csv")):
        print(f"  {f.name}")

    # Define ensemble components with weights (inverse of RMSLE works well)
    # Lower RMSLE = better model = higher weight
    submissions = [
        ("xgb_v1_submission.csv", 1.0 / 0.321),      # XGBoost: best model
        ("lgbm_v1_submission.csv", 1.0 / 0.401),     # LightGBM baseline
    ]

    # Check for Optuna-tuned model
    optuna_files = list(SUBMISSION_DIR.glob("lgbm_optuna_*.csv"))
    if optuna_files:
        # Use the best one (lowest RMSLE in filename)
        optuna_file = sorted(optuna_files)[0]
        rmsle = float(optuna_file.stem.split("_")[-1])
        submissions.append((optuna_file.name, 1.0 / rmsle))
        print(f"\nFound Optuna model: {optuna_file.name} (RMSLE: {rmsle:.4f})")

    # Create ensemble
    ensemble_df = ensemble_submissions(submissions, method="weighted")

    # Save
    output_path = SUBMISSION_DIR / "ensemble_weighted.csv"
    ensemble_df.to_csv(output_path, index=False)
    print(f"\n✅ Ensemble saved to: {output_path}")
    print(f"Total predictions: {len(ensemble_df)}")

    # Show stats
    print(f"\nEnsemble statistics:")
    print(f"  Mean: {ensemble_df['sales'].mean():.2f}")
    print(f"  Median: {ensemble_df['sales'].median():.2f}")
    print(f"  Min: {ensemble_df['sales'].min():.2f}")
    print(f"  Max: {ensemble_df['sales'].max():.2f}")

    return ensemble_df


if __name__ == "__main__":
    main()
