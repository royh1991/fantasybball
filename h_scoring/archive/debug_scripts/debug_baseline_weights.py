"""
Debug script to investigate baseline weight calculation discrepancy.

The user observed:
- FT_PCT has 22.6% weight (highest)
- DD has 6.0% weight (low)

But intuitively:
- DD should be more scarce (high CV)
- FT_PCT should be less scarce (low CV)

This script will trace through the actual CV calculation and compare to
the baseline weights that are set.
"""

import pandas as pd
import numpy as np
import json
import os

from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator


def main():
    print("=" * 80)
    print("DEBUGGING BASELINE WEIGHT CALCULATION")
    print("=" * 80)

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print(f"\nLoading: {data_file}")
    print(f"Loading: {variance_file}")

    league_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Initialize
    scoring = PlayerScoring(league_data, player_variances, roster_size=13)
    cov_calc = CovarianceCalculator(league_data, scoring)

    categories = scoring.all_cats

    print("\n" + "=" * 80)
    print("MANUAL CV CALCULATION")
    print("=" * 80)

    # Manually calculate CV for each category
    cv_results = []

    for cat in categories:
        cat_values = cov_calc.season_averages[cat].values
        mean_val = np.mean(cat_values)
        std_val = np.std(cat_values)

        if mean_val > 0:
            cv = std_val / mean_val
        else:
            cv = 0

        cv_results.append({
            'category': cat,
            'mean': mean_val,
            'std': std_val,
            'cv': cv
        })

    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('cv', ascending=False)

    print("\nCoefficient of Variation (CV) by Category:")
    print(cv_df.to_string(index=False))

    # Calculate what weights SHOULD be based on CV
    cv_values = np.array([x['cv'] for x in cv_results])
    cv_weights = cv_values / cv_values.sum()

    print("\n" + "=" * 80)
    print("EXPECTED WEIGHTS FROM CV")
    print("=" * 80)

    expected_weights = []
    for i, cat in enumerate(categories):
        expected_weights.append({
            'category': cat,
            'cv': cv_values[i],
            'expected_weight': cv_weights[i],
            'expected_pct': cv_weights[i] * 100
        })

    expected_df = pd.DataFrame(expected_weights)
    expected_df = expected_df.sort_values('expected_weight', ascending=False)

    print("\nExpected Weights (sorted by weight):")
    print(expected_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ACTUAL BASELINE WEIGHTS FROM get_setup_params()")
    print("=" * 80)

    # Get actual baseline weights
    setup_params = cov_calc.get_setup_params()
    actual_weights = setup_params['baseline_weights']

    actual_weights_list = []
    for i, cat in enumerate(categories):
        actual_weights_list.append({
            'category': cat,
            'actual_weight': actual_weights[i],
            'actual_pct': actual_weights[i] * 100
        })

    actual_df = pd.DataFrame(actual_weights_list)
    actual_df = actual_df.sort_values('actual_weight', ascending=False)

    print("\nActual Baseline Weights (sorted by weight):")
    print(actual_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("COMPARISON: EXPECTED VS ACTUAL")
    print("=" * 80)

    comparison = []
    for i, cat in enumerate(categories):
        comparison.append({
            'category': cat,
            'expected_pct': cv_weights[i] * 100,
            'actual_pct': actual_weights[i] * 100,
            'diff_pct': (actual_weights[i] - cv_weights[i]) * 100
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('diff_pct', ascending=False)

    print("\nComparison (sorted by difference):")
    print(comparison_df.to_string(index=False))

    # Check if weights match
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if np.allclose(cv_weights, actual_weights, rtol=1e-3):
        print("\n✓ Weights MATCH CV calculation")
    else:
        print("\n✗ Weights DO NOT MATCH CV calculation")
        print("\nPossible causes:")
        print("1. Different method being used (not 'scarcity')")
        print("2. Weights being transformed after CV calculation")
        print("3. Different data being used for CV vs baseline weights")
        print("4. Bug in calculate_baseline_weights() implementation")

    # Highlight specific examples user mentioned
    print("\n" + "=" * 80)
    print("USER'S SPECIFIC EXAMPLES")
    print("=" * 80)

    ft_pct_idx = categories.index('FT_PCT')
    dd_idx = categories.index('DD')

    print(f"\nFT_PCT:")
    print(f"  CV: {cv_values[ft_pct_idx]:.4f}")
    print(f"  Expected weight: {cv_weights[ft_pct_idx]*100:.1f}%")
    print(f"  Actual weight: {actual_weights[ft_pct_idx]*100:.1f}%")

    print(f"\nDD:")
    print(f"  CV: {cv_values[dd_idx]:.4f}")
    print(f"  Expected weight: {cv_weights[dd_idx]*100:.1f}%")
    print(f"  Actual weight: {actual_weights[dd_idx]*100:.1f}%")

    if cv_values[dd_idx] > cv_values[ft_pct_idx]:
        print(f"\n✓ User is CORRECT: DD has higher CV ({cv_values[dd_idx]:.4f}) than FT_PCT ({cv_values[ft_pct_idx]:.4f})")
        print(f"  Therefore DD SHOULD have higher weight than FT_PCT")

        if actual_weights[dd_idx] < actual_weights[ft_pct_idx]:
            print(f"\n⚠️  BUT actual weights are INVERTED:")
            print(f"  DD weight ({actual_weights[dd_idx]*100:.1f}%) < FT_PCT weight ({actual_weights[ft_pct_idx]*100:.1f}%)")
            print(f"\n  This is the DISCREPANCY we need to investigate!")


if __name__ == "__main__":
    main()
