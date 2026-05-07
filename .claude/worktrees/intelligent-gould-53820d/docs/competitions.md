# Competitions

## Lifecycle
```
new       → register slug, copy template, create per-comp CLAUDE.md
switch    → set `active` in registry.json
archive   → move active/<slug> → archived/<slug>, mark status=archived
list      → dump registry.json
status    → show active competition fields
```

CLI:
```bash
python core/competition_manager.py new \
  --slug house-prices --metric rmse --deadline 2026-08-15 \
  --name "House Prices" --higher-better false --submissions-max 5
python core/competition_manager.py status
python core/competition_manager.py switch <slug>
python core/competition_manager.py archive <slug>
```

Convenience wrapper:
```bash
bash scripts/new_competition.sh <slug> <metric> <YYYY-MM-DD> ["display name"]
```

## Per-competition layout
```
competitions/active/<slug>/
  CLAUDE.md              ← short brief written by the agent
  src/                   ← copied from template/src
  configs/               ← YAML per experiment
  submissions/           ← CSVs go here, scores.json is the index
  notebooks/             ← throwaway exploration (gitignored)
  data/                  ← raw + processed (gitignored)
```

## Manual steps you always do
| Step | Where |
|---|---|
| Accept competition rules | kaggle.com |
| Phone verification (first time) | kaggle.com |
| Approve each LB submission | Telegram or web chat |
| Approve ensemble strategy | Telegram or web chat |
