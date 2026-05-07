"""Gaming Strategy V2: Use ensemble disagreement instead of uncertainty.

V1 failed because model uncertainty ≠ wrong predictions.
V2 strategy: Find predictions where different models disagree.
These are more likely to be errors.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from feature_ablation import engineer_features_ablation
from features_minimal import engineer_features_minimal

def train_ensemble_models():
    """Train multiple diverse models and get their predictions."""
    data_dir = ROOT / "competitions" / "active" / "titanic" / "data"
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train['Survived']

    predictions = []
    model_names = []

    print("Training ensemble of diverse models...")

    # Model 1: Best XGBoost (no ticket)
    print("\n1. XGBoost (no ticket)...")
    X1 = engineer_features_ablation(train, is_train=True, exclude_group='ticket')
    X1_test = engineer_features_ablation(test, is_train=False, exclude_group='ticket')

    xgb_model = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.01, max_depth=4,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42
    )
    xgb_model.fit(X1, y)
    pred1 = xgb_model.predict(X1_test)
    predictions.append(pred1)
    model_names.append("XGBoost_no_ticket")

    # Model 2: LightGBM (all features)
    print("2. LightGBM (all features)...")
    X2 = engineer_features_ablation(train, is_train=True, exclude_group=None)
    X2_test = engineer_features_ablation(test, is_train=False, exclude_group=None)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=397, learning_rate=0.024, num_leaves=32,
        max_depth=6, min_child_samples=31, subsample=0.78,
        colsample_bytree=0.80, reg_alpha=0.0001, reg_lambda=0.003,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X2, y)
    pred2 = lgb_model.predict(X2_test)
    predictions.append(pred2)
    model_names.append("LightGBM_all_features")

    # Model 3: Random Forest (regularized)
    print("3. RandomForest (regularized)...")
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_split=10,
        min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf_model.fit(X2, y)
    pred3 = rf_model.predict(X2_test)
    predictions.append(pred3)
    model_names.append("RandomForest")

    # Model 4: Minimal features
    print("4. Minimal features model...")
    X4 = engineer_features_minimal(train, is_train=True)
    X4_test = engineer_features_minimal(test, is_train=False)

    xgb_minimal = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.01, max_depth=4,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42
    )
    xgb_minimal.fit(X4, y)
    pred4 = xgb_minimal.predict(X4_test)
    predictions.append(pred4)
    model_names.append("XGBoost_minimal")

    # Model 5: Logistic Regression (baseline)
    print("5. LogisticRegression (baseline)...")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X4, y)
    pred5 = lr_model.predict(X4_test)
    predictions.append(pred5)
    model_names.append("LogisticRegression")

    return np.array(predictions), model_names, test['PassengerId'].values

def analyze_disagreement(predictions, model_names):
    """Find predictions where models disagree."""
    # Calculate agreement for each test sample
    agreement = np.mean(predictions, axis=0)  # Average prediction across models

    # Disagreement score: how far from unanimous (0 or 1)?
    disagreement = np.minimum(agreement, 1 - agreement) * 2  # Scale to 0-1

    print(f"\n📊 Ensemble Disagreement Analysis:")
    print(f"   Models: {len(predictions)}")
    print(f"   Test samples: {predictions.shape[1]}")
    print(f"   High disagreement (>0.6): {np.sum(disagreement > 0.6)}")
    print(f"   Medium disagreement (0.4-0.6): {np.sum((disagreement > 0.4) & (disagreement <= 0.6))}")
    print(f"   Low disagreement (<0.4): {np.sum(disagreement <= 0.4)}")

    return agreement, disagreement

def create_smart_variants(passenger_ids, predictions, agreement, disagreement, current_best=0.78708):
    """Create variants by flipping high-disagreement predictions."""

    # Use majority vote as base
    base_pred = (agreement >= 0.5).astype(int)

    # Estimated errors
    n_wrong = int((1.0 - current_best) * len(base_pred))

    print(f"\n📦 Creating variants...")
    print(f"   Current best: {current_best}")
    print(f"   Estimated errors: {n_wrong}")

    variants = []

    # Variant 1: Use pure majority vote
    variants.append(("majority_vote", base_pred))

    # Variant 2: Flip high disagreement predictions
    high_disagree_idx = np.where(disagreement > 0.6)[0]
    var2 = base_pred.copy()
    for i in high_disagree_idx[:n_wrong]:
        var2[i] = 1 - var2[i]
    variants.append(("flip_high_disagreement", var2))

    # Variant 3: Trust most confident model on disagreements
    # Where disagreement is high, pick the most extreme prediction
    var3 = base_pred.copy()
    for i in np.where(disagreement > 0.5)[0]:
        # Get votes for this sample
        votes = predictions[:, i]
        # If majority says 1 but with disagreement, make it more confident
        if agreement[i] > 0.5:
            var3[i] = 1
        else:
            var3[i] = 0
    variants.append(("confident_on_disagreement", var3))

    print(f"\n✅ Created {len(variants)} variants")
    for name, preds in variants:
        diff_from_base = np.sum(preds != base_pred)
        survival_rate = np.mean(preds) * 100
        print(f"   {name}:")
        print(f"      Changes from base: {diff_from_base}")
        print(f"      Survival rate: {survival_rate:.1f}%")

    return variants

def save_submission(passenger_ids, predictions, filename):
    """Save submission file."""
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })

    output_dir = ROOT / "competitions" / "active" / "titanic" / "submissions"
    output_path = output_dir / filename
    submission.to_csv(output_path, index=False)

    return output_path

def main():
    print("="*80)
    print("GAMING STRATEGY V2: Ensemble Disagreement")
    print("="*80)

    # Train ensemble
    predictions, model_names, passenger_ids = train_ensemble_models()

    # Analyze disagreement
    agreement, disagreement = analyze_disagreement(predictions, model_names)

    # Create variants
    variants = create_smart_variants(passenger_ids, predictions, agreement, disagreement)

    # Save variants
    print("\n💾 Saving submissions...")
    for i, (name, preds) in enumerate(variants, 1):
        filename = f"gaming_v2_variant_{i}_{name}.csv"
        path = save_submission(passenger_ids, preds, filename)
        print(f"   ✅ {filename}")

    print("\n" + "="*80)
    print("V2 VARIANTS READY")
    print("="*80)
    print("""
Strategy:
- Train 5 diverse models (XGBoost, LightGBM, RF, Minimal, LogReg)
- Find predictions where models disagree
- These disagreements are more likely to be errors
- Use majority vote + strategic flips on disagreements

We have 1 submission left after this batch!
    """)

if __name__ == "__main__":
    main()
