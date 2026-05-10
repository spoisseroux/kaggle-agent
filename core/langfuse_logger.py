"""Langfuse experiment logger for Kaggle competitions.

Logs experiments as traces to Langfuse for observability.
Uses REST API for reliable ingestion.
"""
import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Load credentials
def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_load_env()

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://docker:3000")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")


def log_experiment(
    name: str,
    competition: str,
    model_type: str,
    cv_score: float,
    lb_score: Optional[float] = None,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
) -> bool:
    """
    Log a Kaggle experiment to Langfuse as a trace.
    
    Args:
        name: Experiment name (e.g., "v19-LightGBM")
        competition: Competition slug
        model_type: Model type (e.g., "LightGBM", "XGBoost")
        cv_score: Cross-validation score
        lb_score: Leaderboard score (if available)
        input_data: Additional input metadata
        output_data: Additional output metadata
        metadata: Additional trace metadata
        version: Version identifier
        
    Returns:
        True if successful, False otherwise
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("⚠️  Langfuse credentials not configured, skipping logging")
        return False
    
    trace_id = f"{competition}-{name}-{int(datetime.now().timestamp())}"
    
    # Build input data
    input_dict = {
        "model": model_type,
        "competition": competition,
        **(input_data or {})
    }
    
    # Build output data
    output_dict = {
        "cv_score": cv_score,
        "lb_score": lb_score,
        **(output_data or {})
    }
    
    # Build metadata
    meta_dict = {
        "competition": competition,
        "model": model_type,
        "version": version or name,
        **(metadata or {})
    }
    
    # Create trace via REST API
    batch = [{
        "id": trace_id,
        "type": "trace-create",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "body": {
            "id": trace_id,
            "name": name,
            "userId": "kaggle-agent",
            "metadata": meta_dict,
            "input": input_dict,
            "output": output_dict
        }
    }]
    
    try:
        response = requests.post(
            f"{LANGFUSE_HOST}/api/public/ingestion",
            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
            json={"batch": batch},
            timeout=10
        )
        
        if response.status_code == 207:
            result = response.json()
            if result.get("successes"):
                print(f"✅ Logged to Langfuse: {name}")
                return True
            else:
                print(f"⚠️  Langfuse logging had errors: {result.get('errors')}")
                return False
        else:
            print(f"⚠️  Langfuse API returned {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  Failed to log to Langfuse: {e}")
        return False


def log_kaggle_model(
    name: str,
    competition: str,
    model_type: str,
    cv_score: float,
    lb_score: Optional[float] = None,
    features: Optional[int] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> bool:
    """
    Convenience function for logging Kaggle model experiments.
    
    Args:
        name: Model version (e.g., "v19", "v20-optuna")
        competition: Competition slug
        model_type: Model type
        cv_score: Cross-validation score
        lb_score: Leaderboard score
        features: Number of features
        hyperparameters: Model hyperparameters
        status: Status label (e.g., "BEST", "overfit", "no_help")
        
    Returns:
        True if successful
    """
    input_data = {}
    if features:
        input_data["features"] = features
    if hyperparameters:
        input_data["hyperparameters"] = hyperparameters
        
    metadata = {}
    if status:
        metadata["status"] = status
        
    return log_experiment(
        name=f"{name}-{model_type}",
        competition=competition,
        model_type=model_type,
        cv_score=cv_score,
        lb_score=lb_score,
        input_data=input_data,
        metadata=metadata,
        version=name
    )
