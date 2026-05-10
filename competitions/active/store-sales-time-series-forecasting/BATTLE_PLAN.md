# Battle Plan: Reach #1 on Store Sales Leaderboard

**Current Status:**
- Best Model: v19 LightGBM
- Current LB: 0.498 (RMSLE)
- Target: Top 1% (~0.40-0.45 estimated)

**Gap Analysis:**
- Need ~10-20% improvement to reach top tier
- Current approach: Basic features + LightGBM
- Missing: Advanced architectures, deep features, ensembles

---

## Phase 1: Neural Network Architectures (PRIORITY)

### A. N-BEATS (Neural Basis Expansion Analysis for Time Series)
**Why:** SOTA for univariate time series, handles seasonality well

**Implementation:**
```python
# v26: N-BEATS baseline
- Architecture: 2 stacks (trend + seasonality)
- Lookback: 30 days
- Forecast horizon: 16 days
- Per store-family model or global with embeddings
```

**Expected Impact:** 5-15% improvement (LB ~0.42-0.47)
**Time:** ~2 hours (GPU training)
**Risk:** High memory usage, might need batching

---

### B. Temporal Fusion Transformer (TFT)
**Why:** Handles multiple time series + covariates, attention mechanism

**Implementation:**
```python
# v27: TFT with all features
- Static: store_nbr, family (embeddings)
- Time-varying known: date features, holidays, onpromotion
- Time-varying unknown: lagged sales
- Attention heads: 4
- Hidden size: 64
```

**Expected Impact:** 10-20% improvement (LB ~0.40-0.45)
**Time:** ~4 hours (GPU training)
**Risk:** Complex, prone to overfitting

---

### C. LSTM/GRU Baseline
**Why:** Simple, proven for time series, good benchmark

**Implementation:**
```python
# v28: Bidirectional LSTM
- 2 layers, 128 units each
- Dropout 0.2
- Sequence length: 30 days
- Multi-output (16-day forecast)
```

**Expected Impact:** 5-10% improvement (LB ~0.45-0.47)
**Time:** ~1 hour
**Risk:** Low, might underperform vs tree models

---

## Phase 2: Advanced Feature Engineering

### A. Fourier Transform Features
**Why:** Capture periodic patterns (weekly, monthly)

```python
# v29: LightGBM + Fourier features
- FFT on rolling windows (7, 14, 30 days)
- Extract dominant frequencies
- Sin/cos transforms for cyclical features
```

**Expected Impact:** 3-7% improvement on v19
**Time:** ~30 min

---

### B. External Data Integration (SAFE)
**Why:** Oil prices affect Ecuador economy, store metadata

```python
# v30: LightGBM + oil + store data
- Oil price (lag 1, 7, 30 days)
- Oil price rolling stats
- Store city, state, type, cluster
- Store-specific encoding (not label encoder!)
```

**Expected Impact:** 2-5% improvement
**Time:** ~45 min

---

### C. Target Encoding (Time-Aware)
**Why:** Capture store-family interaction effects safely

```python
# v31: LightGBM + target encoding
- Time-based split for encoding (no leakage)
- Store-family historical mean (expanding window)
- Smoothing for low-frequency combinations
```

**Expected Impact:** 3-8% improvement
**Time:** ~1 hour

---

## Phase 3: Stacking & Ensembles

### A. Multi-Level Stacking
**Why:** Combine different model types for diversity

```python
# v32: Stack LightGBM + N-BEATS + TFT
Level 1:
- LightGBM v19 (tree-based)
- N-BEATS v26 (neural, univariate)
- TFT v27 (neural, multivariate)

Level 2:
- Ridge regression on L1 predictions
- Optimize weights on validation set
```

**Expected Impact:** 5-15% over best single model
**Time:** ~1 hour (models already trained)

---

### B. Weighted Ensemble (Simple)
**Why:** Quick wins without retraining

```python
# v33: Weighted average
- v19 LightGBM: weight TBD
- v26 N-BEATS: weight TBD
- Optimize on validation RMSLE
```

**Expected Impact:** 2-5% over best single model
**Time:** ~10 min

---

## Phase 4: Post-Processing

### A. Quantile Clipping
**Why:** Extreme predictions hurt RMSLE

```python
# v34: v32 + quantile clipping
- Clip predictions to 1st-99th percentile of training
- Per store-family quantiles
```

**Expected Impact:** 1-3% improvement
**Time:** ~5 min

---

### B. Negative Sales Handling
**Why:** RMSLE requires positive predictions

```python
# Already doing: np.maximum(preds, 0)
# Try: np.maximum(preds, 0.01) - avoid log(0)
```

**Expected Impact:** 0.5-1% improvement
**Time:** ~2 min

---

## Execution Order (Next 2-3 Days)

### Day 1 (Tonight):
1. **v26: N-BEATS** - 2 hours
2. **v28: LSTM baseline** - 1 hour
3. **v29: Fourier features** - 30 min

### Day 2 (Tomorrow):
1. Submit v21-v25 queue (use 5 daily submissions)
2. **v27: TFT** - 4 hours (while waiting for LB results)
3. **v30: External data** - 45 min
4. **v31: Target encoding** - 1 hour

### Day 3 (Day after):
1. Analyze v21-v25 LB results
2. **v32: Multi-level stack** - 1 hour
3. **v33: Weighted ensemble** - 10 min
4. **v34: Post-processing** - 30 min
5. Submit best 3-5 models

---

## Success Metrics

| Milestone | LB Target | Status |
|-----------|-----------|--------|
| Beat v19 baseline | < 0.498 | ⏳ Pending |
| Top 10% | < 0.47 | ⏳ Pending |
| Top 5% | < 0.45 | ⏳ Pending |
| Top 1% | < 0.42 | 🎯 Goal |

---

## Risk Mitigation

1. **Overfitting Neural Networks:**
   - Use dropout (0.2-0.3)
   - Early stopping (patience=20)
   - Monitor CV-LB gap closely

2. **Memory Issues:**
   - Train per store-family if needed
   - Use gradient checkpointing
   - Reduce batch size

3. **Time Constraints:**
   - Parallelize training where possible
   - Use GPU for all neural models
   - Cache preprocessed data

---

## Contingency Plans

**If neural networks underperform:**
- Focus on advanced tree-based features
- Try AutoML (AutoGluon, FLAML)
- Deep dive into top public notebooks

**If hitting daily submission limits:**
- Build internal validation leaderboard
- Use CV-LB correlation from v19 (9.3% gap)
- Submit only highest-confidence models

**If stuck at plateau:**
- Analyze top LB submissions for patterns
- Try completely different approach (Prophet, AutoARIMA)
- Focus on edge cases (zeros, promotions, holidays)
