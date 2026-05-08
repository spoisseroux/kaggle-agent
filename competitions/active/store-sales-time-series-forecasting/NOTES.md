# Store Sales Competition - Session Notes (2026-05-08)

## Best Model So Far
**XGBoost v1**: LB 0.526, CV 0.321
- Simple mean-fill for test lags (last 30 days avg)
- Full feature set (7 lags, 10 rolling features, seasonality)

## Leaderboard Results

| Model | CV RMSLE | LB Score | Notes |
|-------|----------|----------|-------|
| XGBoost v1 | 0.321 | **0.526** | Simple lag-fill - BEST |
| XGBoost v2 | 0.321 | 0.551 | Day-by-day lag propagation |
| LightGBM v1 | - | 0.572 | Similar to XGBoost v1 |
| XGBoost v3 | 0.315 | 1.00+ | External data breaks it |
| XGBoost v3-fixed | 0.319 | 1.01 | Even without transactions |

## Key Findings

1. **Simple > Complex**: V1's simple mean-fill approach beats V2's "proper" day-by-day lag propagation (0.526 vs 0.551)

2. **External Data Problem**: Adding oil prices, store metadata consistently breaks LB:
   - Improves CV (0.315) but destroys LB (1.00+)
   - Suggests data leakage or fundamental mismatch with test set

3. **Transaction Data**: Doesn't cover test period (ends 2017-08-15, test starts 2017-08-16)

4. **CV/LB Gap**: Persistent across all models (CV ~0.32, LB ~0.53-0.57)
   - Suggests validation period doesn't match test distribution
   - Or test set has different characteristics

## Target
- Current best: 0.526
- Leaderboard top: ~0.377
- **Gap to close: 0.149**

## Ready for Tomorrow
Created ensemble submissions (hit daily limit):
1. `ensemble_v1_0.7_v2_0.3.csv` - 70% v1 + 30% v2
2. `ensemble_v1_v2_balanced.csv` - 50% v1 + 50% v2

## Next Steps
1. Submit ensembles when limit resets
2. Try feature selection (remove weakest features from v1)
3. Investigate why external features break LB
4. Try completely different approaches (neural nets, Prophet, etc.)
5. Look at actual top public solutions for ideas
