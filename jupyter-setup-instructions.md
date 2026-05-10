# JupyterLab Setup for Kaggle Agent

## 1. Add to your `.env` file

Add this line to your `.env` file (wherever your docker-compose uses it):

```bash
JUPYTER_TOKEN=your-secure-random-token-here
```

Generate a secure token:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 2. Update docker-compose.yml

Replace your current `docker-compose.yml` with the one I created at:
`/home/keehar/kaggle-agent/docker-compose-with-jupyter.yml`

Or manually add the `jupyterlab` service section.

## 3. Start JupyterLab

```bash
cd /opt/docker-config/claude-memory  # or wherever your docker-compose.yml is
docker-compose up -d jupyterlab
```

## 4. Access JupyterLab

Open in browser: `http://localhost:8888`

Token: Use the `JUPYTER_TOKEN` you set in `.env`

## 5. Verify Setup

Once in JupyterLab:
1. Open a terminal
2. Run: `nvidia-smi` (should show GPU)
3. Run: `python -c "import lightgbm, xgboost, catboost; print('OK')"`

## What's Available

- **Full kaggle-agent repo** at `/home/jovyan/work`
- **GPU access** for training
- **All ML packages**: LightGBM, XGBoost, CatBoost, Optuna, MLflow
- **Database access**: Postgres and Qdrant
- **Kaggle API** configured with your credentials

## Usage from Claude Code

I can now:
1. Create notebooks for EDA and analysis
2. Update existing notebooks with new experiments
3. Generate visualizations and save them
4. Create Kaggle-submission-ready notebooks

## Example

Create a new notebook:
```python
# I can write to: /home/keehar/kaggle-agent/notebooks/store-sales-eda.ipynb
```

You access it at: `work/notebooks/store-sales-eda.ipynb` in JupyterLab
