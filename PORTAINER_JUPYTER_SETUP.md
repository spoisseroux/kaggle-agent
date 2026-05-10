# JupyterLab Setup in Portainer - Step by Step

## Part 1: Add Environment Variable

### Step 1: Generate Token
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```
Copy the output (will look like: `xKj9mP2vQwE7rN4sL8hF6gT1yU3oI5aD`)

### Step 2: Add to Stack in Portainer
1. Open Portainer web UI
2. Go to **Stacks** → Click on **ai-memory-stack**
3. Click **Editor** tab
4. Scroll to the **Environment variables** section at the bottom
5. Click **+ Add environment variable**
6. Add:
   - Name: `JUPYTER_TOKEN`
   - Value: `<paste-your-generated-token>`
7. Click **Save** (don't deploy yet)

---

## Part 2: Add JupyterLab Service to Stack

### In the same Editor view, scroll to the `services:` section

### After the `langfuse-worker:` service, add this entire block:

```yaml
  jupyterlab:
    image: jupyter/tensorflow-notebook:latest
    container_name: kaggle-jupyterlab
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      GRANT_SUDO: "yes"
      POSTGRES_DSN: postgresql://claude:${POSTGRES_PASSWORD}@postgres:5432/claude_memory
      QDRANT_URL: http://qdrant:6333
      MLFLOW_TRACKING_URI: http://host.docker.internal:5000
      KAGGLE_USERNAME: ${KAGGLE_USERNAME}
      KAGGLE_KEY: ${KAGGLE_KEY}
      NB_UID: 1000
      NB_GID: 1000
      CHOWN_HOME: "yes"
    volumes:
      - /home/keehar/kaggle-agent:/home/jovyan/work
      - model_cache:/models
      - jupyter_config:/home/jovyan/.jupyter
      - jupyter_conda:/opt/conda
    ports:
      - "8888:8888"
    user: root
    working_dir: /home/jovyan/work
    command: >
      bash -c "
      echo 'Installing Kaggle ML packages...' &&
      pip install --quiet --no-cache-dir \
        lightgbm xgboost catboost \
        optuna mlflow langfuse \
        kaggle plotly seaborn scikit-learn \
        psycopg2-binary qdrant-client tqdm &&
      echo 'Setting permissions...' &&
      chown -R 1000:1000 /home/jovyan &&
      echo 'Starting JupyterLab...' &&
      start-notebook.sh \
        --ServerApp.token='${JUPYTER_TOKEN}' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.ip='0.0.0.0'
      "
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      - postgres
      - qdrant
```

### Then scroll down to the `volumes:` section at the bottom and add:

```yaml
  jupyter_config:
  jupyter_conda:
```

So your volumes section should look like:
```yaml
volumes:
  pgdata:
  model_cache:
  jupyter_config:
  jupyter_conda:
```

---

## Part 3: Deploy the Stack

### Step 1: Deploy
1. Still in the Stack Editor
2. Click **Update the stack** button at the bottom
3. Check **Re-pull images and re-deploy**
4. Click **Update**

### Step 2: Monitor Deployment
1. Watch the deployment logs in Portainer
2. Look for: `Starting JupyterLab...`
3. Should take 2-3 minutes to install packages and start

### Step 3: Verify Container is Running
1. Go to **Containers** in Portainer
2. Find `kaggle-jupyterlab`
3. Status should be **running** (green)
4. Click on it to see logs

---

## Part 4: Access JupyterLab

### Step 1: Open in Browser
```
http://localhost:8888
```
Or if accessing remotely:
```
http://<your-server-ip>:8888
```

### Step 2: Login
- **Password or token:** Paste your `JUPYTER_TOKEN` value
- Click **Log in**

### Step 3: Verify Setup
1. You should see the JupyterLab interface
2. Left sidebar shows `work` folder (your kaggle-agent repo)
3. Click **File** → **New** → **Terminal**
4. Run test commands:
   ```bash
   nvidia-smi  # Should show your RTX 5070
   python -c "import lightgbm, xgboost; print('ML packages OK')"
   ```

---

## Part 5: First Notebook (I'll create this for you)

Once JupyterLab is running, let me know and I'll create:
1. `notebooks/store-sales-eda.ipynb` - Full EDA of the competition
2. `notebooks/model-comparison.ipynb` - Compare v19 vs v32
3. `notebooks/recursive-forecasting-demo.ipynb` - Show how v32 works

---

## Troubleshooting

### If container won't start:
1. In Portainer → Containers → kaggle-jupyterlab → Logs
2. Look for errors
3. Common issues:
   - GPU driver not available → Remove the `deploy:` section temporarily
   - Permission errors → Check `/home/keehar/kaggle-agent` exists in WSL
   - Port conflict → Change `8888:8888` to `8889:8888`

### If can't access on port 8888:
1. In Portainer → Containers → kaggle-jupyterlab
2. Check **Port mapping** shows `8888`
3. Try: `curl http://localhost:8888` from server terminal

### To restart JupyterLab:
1. Portainer → Containers → kaggle-jupyterlab
2. Click **Restart**

---

## What You Get

✅ Full Jupyter environment with GPU  
✅ Your entire kaggle-agent repo accessible  
✅ All ML packages pre-installed  
✅ Database access (Postgres, Qdrant)  
✅ MLflow integration  
✅ Kaggle API configured  

I can then create notebooks, update them, add visualizations, and you can view/edit them in the browser!
