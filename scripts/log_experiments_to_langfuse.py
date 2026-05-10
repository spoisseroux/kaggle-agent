import os
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk-lf-e1a19aa5-4fe1-49aa-bc81-32e34b3d2e1d'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk-lf-5594c0b8-eb71-4a7f-a49f-4b60c0cc842e'
os.environ['LANGFUSE_HOST'] = 'http://docker:3000'

from langfuse import Langfuse

client = Langfuse()

experiments = [
    {"name": "v18-xgboost-optuna", "model": "XGBoost", "cv": 0.4872, "lb": 0.5334, "status": "no_improvement"},
    {"name": "v19-lightgbm-BEST", "model": "LightGBM", "cv": 0.3572, "lb": 0.4983, "status": "BEST"},
    {"name": "v20-lightgbm-optuna", "model": "LightGBM", "cv": 0.3516, "lb": 0.5116, "status": "overfit"},
    {"name": "v21-catboost", "model": "CatBoost", "cv": 0.4458, "lb": None, "status": "middle"},
    {"name": "v22-lightgbm-month", "model": "LightGBM", "cv": 0.3584, "lb": None, "status": "no_help"},
    {"name": "v23-lightgbm-day_of_month", "model": "LightGBM", "cv": 0.3773, "lb": None, "status": "hurts"},
    {"name": "v24-ensemble", "model": "Ensemble", "cv": 0.3572, "lb": None, "status": "100pct_lgbm"},
    {"name": "v25-lightgbm-deeper", "model": "LightGBM", "cv": 0.3660, "lb": None, "status": "worse"},
]

for exp in experiments:
    event = client.create_event(
        name=f"store-sales-{exp['name']}",
        input={
            "competition": "store-sales-time-series-forecasting",
            "model": exp["model"],
            "version": exp["name"]
        },
        output={
            "cv": exp["cv"],
            "lb": exp["lb"],
            "status": exp["status"]
        },
        metadata={
            "date": "2026-05-09",
            "session": "autonomous-v18-v25"
        }
    )
    print(f"✅ Logged {exp['name']}")

client.flush()
print(f"\n✅ All {len(experiments)} experiments logged to Langfuse!")
print("Check http://docker:3000")
