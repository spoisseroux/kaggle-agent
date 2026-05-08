# Telegram Message Formatting Guide

## Rules for Mobile-Friendly Messages

1. **Short paragraphs** (1-2 sentences max)
2. **Line breaks** between ideas  
3. **Simple bullets** using dashes (-)
4. **Headers** using emojis
5. **Max ~10 lines** per message
6. **No bold/italic** (doesn't render in plain text mode)

## Good Example
```
🏋 Training Complete

Results:
- CV: 0.315 (improved!)
- Time: 12 min
- Features: 39

Next: Creating submission file
ETA: 5 min
```

## Bad Example
```
Training complete with CV 0.315 which is an improvement over 
the previous 0.321 using 39 features including oil prices 
with rolling averages and store metadata encoded as categorical 
variables, took 12 minutes on RTX 5070 with early stopping 
at iteration 450, now generating submission file which should 
take about 5 minutes.
```

## Message Templates

### Progress Update
```
[Emoji] [What I'm doing]

Status: [current step]
Progress: [X/Y or percentage]

ETA: [time]
```

### Results
```
[Emoji] [Action] Complete

Key results:
- Metric 1: value
- Metric 2: value
- Metric 3: value

Next: [what's next]
```

### Question/Decision
```
[Emoji] Need Input

Situation: [brief context]

Options:
1) Option A - [why]
2) Option B - [why]
3) Option C - [why]

Which one?
```

### Error/Issue
```
❌ Issue Found

Problem: [brief description]

Impact: [what it affects]

Fix: [what I'm doing]
ETA: [time]
```

## Emoji Key
- 🚀 Starting something
- 🏋 Training/heavy work
- ✅ Success/complete
- 📊 Results/analysis
- ❌ Error/problem
- 🔍 Investigating
- 💡 Insight/learning
- 📨 Message received
- ⏰ Scheduled/waiting
- 🎯 Target/goal
