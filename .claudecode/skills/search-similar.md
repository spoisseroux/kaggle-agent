---
name: search-similar
description: Semantic search for similar code patterns across competitions
---

# /search-similar - Semantic Code Search

Search past Kaggle competition code using natural language queries.
Finds similar feature engineering, model training, and validation patterns.

## Usage

```
/search-similar "lag features for time series"
/search-similar "rolling window aggregations" --category feature_engineering
/search-similar "xgboost hyperparameter tuning" --competition store-sales
/search-similar --code "def create_lag_features(df): ..."
```

## What It Does

1. **Generates embeddings** using RTX 5070 (3x faster than homelab)
2. **Searches Qdrant** on homelab for similar code
3. **Returns ranked results** with similarity scores
4. **Filters by category** or competition if requested

## Categories

- `feature_engineering` - Feature creation, transformations
- `model_training` - Model fitting, training loops
- `validation` - CV strategies, TimeSeriesSplit
- `hyperparameter_tuning` - Optuna, GridSearch
- `visualization` - Plots, EDA
- `utility` - Helper functions

## Examples

**Find similar feature engineering:**
```python
from core.semantic_search import find_similar_features

results = find_similar_features("lag features for sales", limit=5)
for r in results:
    print(f"{r['name']} - Score: {r['score']:.2f}")
    print(f"Competition: {r['competition']}")
    print(r['code'][:200])
```

**Search by category:**
```python
from core.semantic_search import CodeSearch

searcher = CodeSearch()
results = searcher.search_by_category(
    category="model_training",
    query="lightgbm with custom eval metric",
    limit=3
)
```

**Find similar to your code:**
```python
from core.semantic_search import CodeSearch

my_code = """
def create_rolling_features(df, windows=[7, 14, 30]):
    for w in windows:
        df[f'Roll_mean_{w}'] = df['sales'].rolling(w).mean()
    return df
"""

searcher = CodeSearch()
similar = searcher.search_similar_code(my_code, limit=5)
```

## Performance

- **Embedding generation**: ~50 texts/second on RTX 5070
- **Search latency**: ~50ms (homelab Qdrant via Tailscale)
- **Indexed**: All Python code from past competitions

## Indexing New Code

To index a new competition:
```bash
python -m core.code_indexer store-sales-time-series-forecasting
```

To index all competitions:
```bash
python -m core.code_indexer
```
