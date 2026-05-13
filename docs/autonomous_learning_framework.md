# Autonomous Learning Framework for Kaggle Competitions

**Based on lessons from Store Sales autonomous session (May 12, 2026)**

## Executive Summary

This framework enables systematic, autonomous exploration of Kaggle competitions with minimal human intervention. Built from 4+ hours of autonomous work that tested 4 major hypotheses across 3 experiments, achieving 100% negative results but high learning value.

**Key principle:** Negative results are progress. Knowing what doesn't work is as valuable as finding what does.

---

## Phase 0: Pre-Session Setup

### Before Going Autonomous

**1. Establish Baseline**
- ✅ Working end-to-end pipeline
- ✅ Validated LB submission (know your starting point)
- ✅ Clear performance metric (CV and LB relationship)
- ✅ Reproducible environment

**Why:** Can't measure improvement without a baseline. Store Sales had v50 (LB 0.455) as anchor.

**2. Define Search Space**
- Research phase complete (know the landscape)
- Hypothesis list prioritized
- Clear success criteria for each hypothesis
- Estimated compute budget per experiment

**Why:** Unfocused exploration wastes time. We spent 2 hours researching before experimenting.

**3. Set Validation Gates**
- Correlation threshold (detect alignment bugs)
- Distribution threshold (detect scaling issues)
- CV threshold (stop bad experiments early)
- LB submission criteria

**Why:** Automated gates catch catastrophic bugs. Would have saved v32/v33/v63 failures.

---

## Phase 1: Hypothesis Generation

### Research-Driven Hypothesis Creation

**Sources (in priority order):**
1. Top public notebooks (sort by votes)
2. Competition discussions (recent + highly voted)
3. Similar past competitions (transfer learning)
4. Domain literature (if applicable)

**Extraction Process:**
```python
# Use analyze_top_notebooks.py pattern
for notebook in top_notebooks:
    extract:
        - Models used (architectures)
        - Feature engineering patterns
        - Validation strategies
        - Score mentions
        - Critical code patterns
    
    store:
        - .claude/notebook_patterns.json
        - Per-competition hypothesis list
```

**Hypothesis Template:**
```yaml
hypothesis: "EWM features add signal on top of proven baseline"
rationale: "Comprehensive guide (2895 votes) uses EWM. Exponential decay weights recent data."
category: "feature_engineering"
expected_impact: "+5-10% CV improvement"
compute_cost: "low (feature creation only)"
risk: "low (additive, not replacement)"
validation:
  - correlation_check: "> 0.95"
  - distribution_check: "within ±10%"
  - cv_improvement: "> 1%"
fallback: "if fails, test standalone EWM"
```

**Prioritization Criteria:**
1. **High confidence** (multiple sources mention it)
2. **Low risk** (additive changes, not rewrites)
3. **Fast validation** (can test in <30 min)
4. **Clear falsifiability** (know when to stop)

---

## Phase 2: Systematic Experimentation

### Experiment Structure

**Pre-Experiment Checklist:**
- [ ] Hypothesis clearly stated
- [ ] Expected outcome quantified
- [ ] Validation gates defined
- [ ] Baseline comparison planned
- [ ] Rollback strategy ready

**Experiment Template:**
```python
#!/usr/bin/env python3
"""Model v{N} - {Hypothesis Name}

Hypothesis: {One sentence hypothesis}
Rationale: {Why this should work}
Expected: {Quantified prediction}

Previous: v{N-1} CV {score}
Target: CV < {target}
"""

def main():
    print("="*70)
    print(f"v{N} - {Hypothesis}")
    print("="*70)
    
    # 1. Load baseline for comparison
    baseline = load_baseline()
    
    # 2. Implement change
    results = implement_hypothesis()
    
    # 3. Validate
    gates = validate_all(results, baseline)
    
    # 4. Report
    report_with_decision(gates)
    
    return results
```

**Validation Gate Implementation:**
```python
def validate_all(experiment, baseline):
    gates = {}
    
    # Gate 1: Correlation (detect alignment bugs)
    correlation = np.corrcoef(experiment.preds, baseline.preds)[0,1]
    gates['correlation'] = {
        'value': correlation,
        'pass': correlation > 0.95,
        'critical': correlation < 0,  # Negative = alignment bug
        'action': 'fix' if correlation < 0 else 'continue'
    }
    
    # Gate 2: Distribution (detect scaling issues)
    exp_mean = np.mean(experiment.preds)
    base_mean = np.mean(baseline.preds)
    diff_pct = (exp_mean - base_mean) / base_mean * 100
    gates['distribution'] = {
        'value': diff_pct,
        'pass': abs(diff_pct) < 20,
        'warning': abs(diff_pct) > 10,
        'action': 'review' if abs(diff_pct) > 10 else 'continue'
    }
    
    # Gate 3: CV Improvement
    improvement = (baseline.cv - experiment.cv) / baseline.cv * 100
    gates['cv'] = {
        'value': experiment.cv,
        'improvement': improvement,
        'pass': improvement > 1,  # At least 1% better
        'action': 'submit' if improvement > 1 else 'stop'
    }
    
    return gates
```

### Decision Tree

```
Experiment Complete
    ├─ Correlation < 0? 
    │   └─ YES → FIX (alignment bug, like v63)
    │   └─ NO → Continue
    │
    ├─ Distribution > 20% off?
    │   └─ YES → FIX (scaling issue, like v66)
    │   └─ NO → Continue
    │
    ├─ CV > baseline?
    │   └─ YES → STOP (hypothesis rejected)
    │   └─ NO → Continue
    │
    └─ CV improvement > 1%?
        └─ YES → SUBMIT (hypothesis confirmed)
        └─ NO → STOP (marginal, not worth LB slot)
```

---

## Phase 3: Learning Integration

### After Each Experiment

**1. Update Knowledge Base**
```python
# Store in hypothesis_db
db.record_result(
    hypothesis_id=hyp_id,
    cv_score=result.cv,
    lb_score=None,  # Fill after submission
    baseline_cv=baseline.cv,
    succeeded=(result.cv < baseline.cv),
    params=experiment.params,
    metadata={
        'correlation': gates['correlation']['value'],
        'distribution_shift': gates['distribution']['value'],
        'failure_mode': identify_failure_mode(gates)
    }
)
```

**2. Pattern Recognition**
```python
# After N experiments, look for patterns
def analyze_session_patterns(experiments):
    patterns = {
        'failure_modes': Counter(),
        'improvement_strategies': [],
        'dead_ends': []
    }
    
    for exp in experiments:
        if exp.failed:
            mode = identify_failure_mode(exp)
            patterns['failure_modes'][mode] += 1
            
            # If same failure mode 3+ times, mark as dead end
            if patterns['failure_modes'][mode] >= 3:
                patterns['dead_ends'].append(exp.category)
    
    return patterns
```

**3. Hypothesis Pruning**
```python
# If a category fails 3 times, stop exploring it
if category in dead_ends:
    print(f"⚠️  {category} marked as dead end - skipping related hypotheses")
    continue
```

**Example from Store Sales:**
- EWM features: v68 fail, v69 fail → Category: "EWM-based features" marked dead
- Long lags: v65 fail, v68 fail → Category: "long lags only" marked dead
- Recursive: v66 fail, v70 fail → Category: "recursive forecasting" marked dead

---

## Phase 4: Meta-Learning

### Competition-Level Insights

**Track Across Experiments:**
1. **Narrow vs Wide Optimum**
   - Narrow: Most changes hurt (Store Sales)
   - Wide: Many approaches work similarly
   
2. **Feature Sensitivity**
   - Additive: New features improve
   - Saturated: New features hurt (Store Sales v19)
   
3. **Public vs Private Techniques**
   - Match: Public notebooks work
   - Mismatch: Gap unexplained (Store Sales 21% gap)

**Decision Rules:**

```python
if competition_type == "narrow_optimum":
    strategy = "conservative_tuning"
    # Focus on: hyperparameters, ensemble weights
    # Avoid: adding features, changing architecture
    
elif competition_type == "wide_optimum":
    strategy = "aggressive_exploration"
    # Try: new models, feature combinations, architectures
    
elif competition_type == "public_private_gap":
    strategy = "accept_limits"
    # Public notebooks don't contain winning techniques
    # Stop at best public approach, move to new competition
```

**Store Sales Classification:**
- Type: Narrow optimum + Public/private gap
- Signal: 14/15 feature additions failed, 21% unexplained gap
- Decision: Accept v50 (LB 0.455) as near-optimal for public techniques

---

## Phase 5: Communication Protocol

### Autonomous → Human Messages

**Progress Updates (Every Major Phase):**
```python
# At phase start
notify("🔬 Starting [Phase Name]: [Brief description] (~[ETA])")

# At phase completion
notify("✅ [Phase Name] complete: [Key finding]. Starting [Next Phase].")
```

**Experiment Results:**
```python
# Immediate notification after validation
if experiment.failed:
    notify(f"❌ v{N} FAILED: {hypothesis}\n"
           f"CV: {cv} ({pct_change}% vs baseline)\n"
           f"Issue: {failure_mode}\n"
           f"Learning: {what_we_learned}")
else:
    notify(f"✅ v{N} SUCCESS: {hypothesis}\n"
           f"CV: {cv} ({improvement}% improvement)\n"
           f"Ready for LB validation")
```

**Session Summaries:**
```python
# Every 2-3 hours or after 3 experiments
notify(f"📊 SESSION CHECKPOINT\n"
       f"Duration: {hours}h\n"
       f"Experiments: {n_tested} tested, {n_success} succeeded\n"
       f"Hypotheses: {n_rejected} rejected, {n_confirmed} confirmed\n"
       f"Next: {next_hypothesis}\n"
       f"ETA: ~{eta}")
```

**Critical Issues (Immediate):**
```python
# Stop and ask for help
if issue_type == "critical":
    ask_human(f"🚨 CRITICAL ISSUE\n"
              f"{description}\n"
              f"Current state: {state}\n"
              f"Options:\n"
              f"1) {option1}\n"
              f"2) {option2}\n"
              f"3) Stop and review\n\n"
              f"What should I do?")
```

---

## Phase 6: Session Design Patterns

### Pattern 1: Breadth-First Exploration

**When:** Early in competition, many untested hypotheses
**Strategy:** Test multiple low-cost hypotheses quickly

```python
hypotheses = [
    {"name": "EWM features", "cost": "low", "priority": 1},
    {"name": "Long lags", "cost": "low", "priority": 2},
    {"name": "Recursive", "cost": "high", "priority": 3}
]

# Test top 3-5 hypotheses in one session
for hyp in hypotheses[:5]:
    if hyp['cost'] == 'low':
        result = test_hypothesis(hyp)
        if result.failed:
            mark_dead_end(hyp.category)
```

**Store Sales Example:** Tested 3 hypotheses (EWM, recursive, feature addition) in one session.

### Pattern 2: Depth-First Investigation

**When:** One hypothesis shows promise but needs refinement
**Strategy:** Iterate on variations of promising approach

```python
if hypothesis.showed_promise:
    variations = generate_variations(hypothesis)
    # Test: standalone, additive, different params
    for var in variations:
        result = test_variation(var)
        if result.improvement > best:
            best = var
```

**Store Sales Example:** 
- v68 (EWM standalone) → FAIL
- v69 (EWM additive) → FAIL
- Conclusion: EWM fundamentally doesn't work, stop variations

### Pattern 3: Systematic Ablation

**When:** Baseline is strong but unclear which features matter
**Strategy:** Remove features one at a time

```python
baseline_features = get_features(v19)
for feature in baseline_features:
    subset = baseline_features - {feature}
    result = test_with_features(subset)
    impact = baseline.cv - result.cv
    feature_importance[feature] = impact
```

**Store Sales v62:** Attempted but flawed (wrong hyperparameters). Fixed approach would work.

### Pattern 4: Emergency Pivot

**When:** All hypotheses in category failing
**Strategy:** Switch to completely different approach

```python
if all_experiments_in_category_failed(experiments, category):
    print(f"⚠️  Category {category} exhausted - pivoting")
    
    pivot_options = [
        "different_model_family",
        "external_data",
        "ensemble_methods",
        "post_processing"
    ]
    
    next_category = choose_pivot(pivot_options, tried_categories)
```

**Store Sales:** After EWM/recursive/long lags all failed → Should pivot to ensemble techniques or accept current performance

---

## Phase 7: Resource Management

### Compute Budget

**Allocation:**
```yaml
per_experiment:
  training: "30 min max"
  validation: "5 min"
  analysis: "5 min"
  total: "40 min"

per_session:
  experiments: "5-8"
  research: "1-2 hours"
  documentation: "30 min"
  total: "4-6 hours"

stop_conditions:
  - "3 consecutive failures"
  - "Same failure mode 3 times"
  - "6 hours elapsed"
  - "No improvement >1% found"
```

### API Cost Management

**Token Budget:**
```python
# Use hybrid orchestrator for automatic mode switching
from core.hybrid_orchestrator import decide_and_execute

# Automatically switches to Ollama when approaching limits
result = decide_and_execute(task, competition_slug)
```

**Guidelines:**
- Research/analysis: Use Claude (strategic thinking)
- Code generation: Use Ollama (boilerplate)
- Experiment running: Use Ollama (execution)
- Result interpretation: Use Claude (insights)

### Storage Management

**File Organization:**
```
competitions/active/<slug>/
├── src/
│   ├── model_v*.py           # Experiment code
│   └── model_baseline.py     # Reference baseline
├── predictions/
│   └── v*_test.npy           # For correlation checks
├── submissions/
│   └── v*_*.csv              # Submission files
└── analysis/
    └── session_<date>.json   # Experiment results
```

**Cleanup Rules:**
- Keep: All submission files, final models, predictions for baseline
- Delete: Intermediate checkpoints, temp files, failed model weights
- Archive: Session summaries, hypothesis DB entries

---

## Lessons Learned (Store Sales Case Study)

### What Worked ✅

1. **Validation Gates**
   - Would have caught v63 alignment bug immediately
   - Would have flagged v66 scaling issue early
   - Saved ~2-3 LB submission slots

2. **Systematic Research**
   - 6 notebooks analyzed → clear pattern extraction
   - Hypothesis generation from evidence, not guessing
   - Reproducible research process

3. **Negative Results as Progress**
   - 3/3 experiments failed but learned what doesn't work
   - EWM, long lags, recursive all marked as dead ends
   - Prevented future wasted efforts

4. **Documentation**
   - Full session summary created
   - Findings documented in competition CLAUDE.md
   - Tools created for future use

### What Didn't Work ❌

1. **Assumption: Public Notebooks = Competitive Techniques**
   - Store Sales: 2895-vote notebook techniques all failed
   - Lesson: High votes ≠ high performance
   - Better: Focus on recent high-scoring submissions

2. **Insufficient Variation Testing**
   - EWM: Only tested 2 variations before concluding
   - Better: Test 3-5 variations before marking dead end

3. **No LB Validation During Session**
   - Tested 3 hypotheses but no LB feedback
   - Could have pivoted faster with LB data
   - Better: Submit after every 2 experiments if CV shows promise

4. **Compute Overinvestment in Recursive**
   - v70 took 20+ minutes for same failure as v66
   - Should have stopped after v66 LB failure
   - Better: Quick batch CV test before full recursive

### Improved Process

**Future Autonomous Sessions Should:**

1. **Test faster** (batch CV before expensive implementations)
2. **Validate earlier** (submit after 2nd experiment, not 3rd)
3. **Pivot quicker** (2 failures in category → try different category)
4. **Budget LB slots** (reserve 2 submissions per session)
5. **Set hard stops** (6 hours OR 3 failures OR no >1% improvement)

---

## Template: Autonomous Session Plan

```markdown
# Autonomous Session: [Competition] [Date]

## Pre-Session State
- Baseline: v{N} (CV: {cv}, LB: {lb})
- Remaining LB submissions: {count}
- Compute budget: {hours} hours
- API tokens remaining: {tokens}

## Hypotheses to Test (Priority Order)
1. {Hypothesis 1}
   - Expected: {metric}
   - Cost: {time}
   - Risk: {low/medium/high}
   
2. {Hypothesis 2}
   ...

## Success Criteria
- CV improvement: > 1%
- Correlation: > 0.95
- Distribution: within ±10%

## Stop Conditions
- 3 consecutive failures
- 6 hours elapsed
- No >1% improvement found
- API limits reached

## Validation Schedule
- Experiment 1: Test only (no LB)
- Experiment 2: Submit if CV improved
- Experiment 3: Submit if CV improved
- After 3: Pivot or stop

## Deliverables
- [ ] Session summary (.claude/)
- [ ] Updated CLAUDE.md
- [ ] Hypothesis DB entries
- [ ] At least 1 LB submission (if any success)

## Post-Session Review
- Hypotheses tested: {count}
- Succeeded: {count}
- Failed: {count}
- LB submissions: {count}
- Key learning: {summary}
- Next session focus: {direction}
```

---

## Integration with Existing Workflow

### CLAUDE.md Requirements

**Add to competition CLAUDE.md:**
```markdown
## Autonomous Learning Configuration

**Session Template:** docs/autonomous_learning_framework.md
**Validation Gates:**
- Correlation threshold: 0.95
- Distribution threshold: ±20%
- CV improvement: >1%

**Dead Ends (Do Not Explore):**
- {Category 1}: Tested in v{N}, v{M} - failed
- {Category 2}: Tested in v{X}, v{Y} - failed

**Promising Directions:**
- {Direction 1}: Untested, high priority
- {Direction 2}: Partial success in v{Z}, needs refinement

**LB Budget:**
- Used: {count} / {total}
- Reserved for autonomous: 3-5 slots
```

### Hybrid Orchestrator Integration

**Automatic Mode Selection:**
```python
# Already implemented in core/hybrid_orchestrator.py
from core.hybrid_orchestrator import decide_and_execute

# Handles:
# - API rate limits → Switches to Ollama
# - Task complexity → Routes to appropriate model
# - Queue management → Waits and resumes

result = decide_and_execute(
    task="test hypothesis: EWM features",
    competition_slug="store-sales"
)
```

---

## Future Enhancements

### Short-Term (Implement Next Session)

1. **LangGraph Integration**
   - Use `scripts/langgraph_experiment_workflow.py`
   - Automatic validation gate enforcement
   - State management across experiments

2. **Experiment Queueing**
   - Load multiple hypotheses
   - Run overnight batch
   - Morning: Review results, submit best

3. **Smart Pivoting**
   - After 2 failures: Suggest pivot
   - After 3 failures: Force pivot
   - Pattern recognition: Similar failures → Stop category

### Long-Term (Build Over Time)

1. **Cross-Competition Transfer Learning**
   - Store patterns from each competition
   - "Time series with lag features" → Recommend similar approaches
   - Build up repertoire of working patterns

2. **Meta-Learning from Past Sessions**
   - Which research sources are most reliable?
   - Which hypotheses types have highest success rate?
   - Optimal experiment order (fast failures first)

3. **Automated Hypothesis Generation**
   - Analyze top submissions (not just notebooks)
   - Extract patterns from code diffs
   - Generate variations automatically

---

## Conclusion

Autonomous learning is most effective when:
1. ✅ Clear baseline established
2. ✅ Validation gates prevent catastrophic failures
3. ✅ Negative results documented as progress
4. ✅ Hard stop conditions prevent infinite loops
5. ✅ Learning integrated back into knowledge base

**Store Sales demonstrated:** Even with 0% experiment success rate, autonomous session provided high value through systematic elimination of dead ends and creation of reusable tools.

**Next competition:** Apply this framework from day 1, not after reaching plateau.
