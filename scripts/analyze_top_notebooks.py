#!/usr/bin/env python3
"""Analyze top Kaggle notebooks to extract winning patterns.

Systematically extracts:
- Feature engineering techniques
- Model architectures
- Validation strategies
- Score mentions
- Critical code patterns
"""
import json
import re
from pathlib import Path
from collections import defaultdict

NOTEBOOKS_DIR = Path("/tmp/notebooks")

def extract_code_cells(notebook_path):
    """Extract all code cells from notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)

    return code_cells

def extract_markdown_cells(notebook_path):
    """Extract all markdown cells."""
    with open(notebook_path) as f:
        nb = json.load(f)

    markdown_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                text = ''.join(source)
            else:
                text = source
            markdown_cells.append(text)

    return markdown_cells

def find_patterns(code_cells, markdown_cells):
    """Extract key patterns from cells."""
    patterns = defaultdict(list)

    all_code = '\n'.join(code_cells)
    all_text = '\n'.join(markdown_cells)

    # Feature engineering patterns
    if 'shift(' in all_code:
        shifts = re.findall(r'shift\((\d+)\)', all_code)
        patterns['lag_features'] = list(set(shifts))

    if 'rolling(' in all_code or 'rolling' in all_code:
        windows = re.findall(r'rolling\((?:window=)?(\d+)\)', all_code)
        patterns['rolling_windows'] = list(set(windows))

    if 'ewm(' in all_code:
        patterns['exponential_smoothing'] = True

    # Model types
    if 'LGBMRegressor' in all_code or 'lgb.train' in all_code:
        patterns['models'].append('LightGBM')
    if 'XGBRegressor' in all_code or 'xgb.train' in all_code:
        patterns['models'].append('XGBoost')
    if 'CatBoostRegressor' in all_code:
        patterns['models'].append('CatBoost')
    if 'LSTM' in all_code or 'GRU' in all_code:
        patterns['models'].append('RNN')
    if 'Prophet' in all_code:
        patterns['models'].append('Prophet')

    # Validation strategy
    if 'TimeSeriesSplit' in all_code:
        patterns['validation'] = 'TimeSeriesSplit'
    elif 'train_test_split' in all_code:
        patterns['validation'] = 'holdout'

    # Recursive forecasting
    if any(keyword in all_code for keyword in ['recursive', 'iterative', 'day-by-day', 'step-by-step']):
        patterns['recursive_forecasting'] = True

    # Score mentions
    scores = re.findall(r'(?:LB|leaderboard|score)[:\s]+([0-9]\.[0-9]+)', all_text, re.IGNORECASE)
    if scores:
        patterns['scores'] = scores

    # Store-family specific
    if 'groupby' in all_code and ('store' in all_code or 'family' in all_code):
        patterns['hierarchical'] = True

    return dict(patterns)

def main():
    notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))

    print("=" * 80)
    print("TOP NOTEBOOK ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(notebooks)} notebooks...\n")

    all_findings = {}

    for nb_path in notebooks:
        print(f"\n{'='*80}")
        print(f"Notebook: {nb_path.name}")
        print("=" * 80)

        try:
            code_cells = extract_code_cells(nb_path)
            markdown_cells = extract_markdown_cells(nb_path)

            patterns = find_patterns(code_cells, markdown_cells)
            all_findings[nb_path.name] = patterns

            # Print findings
            if patterns.get('lag_features'):
                print(f"  Lag features: {sorted(set(int(x) for x in patterns['lag_features']))}")

            if patterns.get('rolling_windows'):
                print(f"  Rolling windows: {sorted(set(int(x) for x in patterns['rolling_windows']))}")

            if patterns.get('exponential_smoothing'):
                print(f"  Uses exponential smoothing: Yes")

            if patterns.get('models'):
                print(f"  Models: {', '.join(patterns['models'])}")

            if patterns.get('validation'):
                print(f"  Validation: {patterns['validation']}")

            if patterns.get('recursive_forecasting'):
                print(f"  Recursive forecasting: YES ⭐")

            if patterns.get('hierarchical'):
                print(f"  Hierarchical (store-family): Yes")

            if patterns.get('scores'):
                print(f"  Scores mentioned: {patterns['scores']}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("CROSS-NOTEBOOK PATTERNS")
    print("=" * 80)

    # Most common models
    all_models = []
    for findings in all_findings.values():
        all_models.extend(findings.get('models', []))

    if all_models:
        model_counts = defaultdict(int)
        for model in all_models:
            model_counts[model] += 1

        print("\nMost common models:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} notebooks")

    # Recursive forecasting count
    recursive_count = sum(1 for f in all_findings.values() if f.get('recursive_forecasting'))
    print(f"\nRecursive forecasting: {recursive_count}/{len(all_findings)} notebooks")

    # Common lag features
    all_lags = set()
    for findings in all_findings.values():
        if findings.get('lag_features'):
            all_lags.update(int(x) for x in findings['lag_features'])

    if all_lags:
        print(f"\nLag features used: {sorted(all_lags)}")

    # Save full findings
    output_file = Path(".claude/notebook_patterns.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_findings, f, indent=2)

    print(f"\nFull analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
