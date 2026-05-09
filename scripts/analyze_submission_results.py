#!/usr/bin/env python3
"""Analyze submission results and recommend next experiments

Compares LB scores across submissions and identifies patterns.
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_latest_submissions(n=10):
    """Get latest submissions from Kaggle"""
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "store-sales-time-series-forecasting"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error getting submissions: {result.stderr}")
        return None

    lines = result.stdout.strip().split('\n')
    return lines[:n+1]  # +1 for header


def parse_submissions(lines):
    """Parse submission output into structured data"""
    if not lines or len(lines) < 2:
        return []

    submissions = []
    for line in lines[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 4:
            submissions.append({
                'filename': parts[0],
                'date': ' '.join(parts[1:3]),
                'description': ' '.join(parts[3:-2]) if len(parts) > 5 else '',
                'status': parts[-2],
                'score': parts[-1]
            })

    return submissions


def analyze_results(submissions):
    """Analyze submission results and make recommendations"""
    if not submissions:
        return "No submissions found"

    # Find ensembles
    ensembles = [s for s in submissions if 'ensemble' in s['filename'].lower()]
    v1_subs = [s for s in submissions if 'v1' in s['filename'].lower() and 'ensemble' not in s['filename'].lower()]
    v2_subs = [s for s in submissions if 'v2' in s['filename'].lower() and 'ensemble' not in s['filename'].lower()]

    analysis = ["=" * 70, "SUBMISSION ANALYSIS", "=" * 70, ""]

    # Individual models
    if v1_subs:
        v1_score = v1_subs[0]['score']
        analysis.append(f"XGBoost v1 (baseline): {v1_score}")

    if v2_subs:
        v2_score = v2_subs[0]['score']
        analysis.append(f"XGBoost v2 (fixed lags): {v2_score}")

    analysis.append("")

    # Ensembles
    if ensembles:
        analysis.append("Ensemble Results:")
        for ens in ensembles:
            analysis.append(f"  {ens['filename']}: {ens['score']}")
        analysis.append("")

    # Find best
    try:
        scored = [s for s in submissions if s['score'] != 'None' and s['score'] != 'pending']
        if scored:
            best = min(scored, key=lambda x: float(x['score']))
            analysis.append(f"Best submission: {best['filename']} @ {best['score']}")
    except:
        pass

    analysis.append("")
    analysis.append("=" * 70)
    analysis.append("RECOMMENDATIONS")
    analysis.append("=" * 70)

    # Make recommendations based on results
    if ensembles and v1_subs:
        try:
            ens_score = float(ensembles[0]['score'])
            v1_score = float(v1_subs[0]['score'])

            if ens_score < v1_score:
                improvement = v1_score - ens_score
                analysis.append(f"✅ Ensemble improved LB by {improvement:.4f}")
                analysis.append("   → Continue with ensemble approaches")
                analysis.append("   → Try stacking with LightGBM")
                analysis.append("   → Test different blend ratios")
            else:
                analysis.append(f"❌ Ensemble did not improve LB (v1 still best)")
                analysis.append("   → CV/LB gap remains (0.321 CV vs 0.526 LB)")
                analysis.append("   → Focus on improving test set lag handling")
                analysis.append("   → Try recursive multi-step forecasting")
                analysis.append("   → Investigate data leakage in validation")
        except:
            analysis.append("⏳ Scores pending, will analyze when available")

    return '\n'.join(analysis)


def main():
    print("Fetching latest submissions...")
    lines = get_latest_submissions(10)

    if not lines:
        print("Failed to fetch submissions")
        return

    print('\n'.join(lines))
    print()

    submissions = parse_submissions(lines)
    analysis = analyze_results(submissions)
    print(analysis)

    # Save analysis
    output_path = Path("competitions/active/store-sales-time-series-forecasting/submission_analysis.txt")
    output_path.write_text(analysis)
    print(f"\nAnalysis saved to: {output_path}")


if __name__ == "__main__":
    main()
