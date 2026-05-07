# Data pipeline

Four stages, run in order. Never train on raw data.

## Stage 1 — refine
- Drop fully-empty rows / cols
- Strip whitespace from string columns; coerce numeric-looking columns
- Winsorise numeric columns at ±5σ (target column excluded)

## Stage 2 — enrich
- Pull external HuggingFace datasets via `core.hf_search.search_relevant_datasets`
- Left-join on caller-supplied keys

## Stage 3 — synthetic
- `method=smote` for class imbalance (numeric features only — encode first)
- `method=sdv` for general tabular synthetic via Gaussian copulas

## Stage 4 — fuzz
- Numeric: Gaussian noise N(0, σ·std) per column (σ default 0.01)
- Categorical: swap 2% of values with another random category
- **Never** fuzz the target column
- **Never** fuzz the test set

Each stage writes a versioned parquet under
`data/<slug>/processed/<vN>_<stage>/<stage>.parquet` and inserts a row in
the `kaggle_datasets` table.

## CLI
```bash
python -m competitions.template.src.data_pipeline \
  --slug <slug> \
  --raw-train data/<slug>/train.csv \
  --target <colname> \
  --stages refine,enrich,synthetic,fuzz \
  --hf-dataset-ids someorg/foo,someorg/bar \
  --join-keys id \
  --synthetic-method smote
```

When training, always go through `scripts/pre_train.sh` first so Ollama is
stopped and ≥9.5 GB VRAM is free. After training, `scripts/post_train.sh`
restarts Ollama.
