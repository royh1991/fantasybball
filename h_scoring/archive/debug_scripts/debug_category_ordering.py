"""
Check if there's a category ordering mismatch bug.

Hypothesis: The baseline_weights array might be in one category order,
but we're indexing it with a different category order, causing weights
to be assigned to the wrong categories!
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator


def check_category_ordering():
    """Check if category ordering is consistent."""

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    league_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Initialize
    scoring = PlayerScoring(league_data, player_variances, roster_size=13)
    cov_calc = CovarianceCalculator(league_data, scoring)

    print("=" * 80)
    print("CATEGORY ORDERING CHECK")
    print("=" * 80)

    print("\nCategory order from scoring.all_cats:")
    print(scoring.all_cats)

    print("\nCategory order from cov_calc.categories:")
    print(cov_calc.categories)

    if scoring.all_cats != cov_calc.categories:
        print("\n⚠️  WARNING: Category orders don't match!")
    else:
        print("\n✓ Category orders match")

    # Now manually calculate CV for each category IN THE ORDER of cov_calc.categories
    print("\n" + "=" * 80)
    print("CV CALCULATION IN CORRECT ORDER")
    print("=" * 80)

    cv_values = []
    for cat in cov_calc.categories:
        cat_values = cov_calc.season_averages[cat].values
        mean_val = np.mean(cat_values)
        std_val = np.std(cat_values)

        if mean_val > 0:
            cv = std_val / mean_val
        else:
            cv = 0

        cv_values.append(cv)
        print(f"{cat:<12} CV = {cv:.4f}")

    # Normalize CV to weights
    cv_array = np.array(cv_values)
    expected_weights = cv_array / cv_array.sum()

    print("\n" + "=" * 80)
    print("EXPECTED WEIGHTS (from CV)")
    print("=" * 80)

    for i, cat in enumerate(cov_calc.categories):
        print(f"{cat:<12} {expected_weights[i]:.6f} ({expected_weights[i]*100:.2f}%)")

    # Get actual baseline weights
    baseline_weights = cov_calc.calculate_baseline_weights(method='scarcity')

    print("\n" + "=" * 80)
    print("ACTUAL BASELINE WEIGHTS (from calculate_baseline_weights)")
    print("=" * 80)

    for i, cat in enumerate(cov_calc.categories):
        print(f"{cat:<12} {baseline_weights[i]:.6f} ({baseline_weights[i]*100:.2f}%)")

    # Check if they match
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    matches = np.allclose(expected_weights, baseline_weights)
    if matches:
        print("\n✓ Expected weights MATCH actual baseline weights")
    else:
        print("\n✗ Expected weights DO NOT MATCH actual baseline weights")

        print("\nDifferences:")
        for i, cat in enumerate(cov_calc.categories):
            diff = baseline_weights[i] - expected_weights[i]
            if abs(diff) > 1e-6:
                print(f"{cat:<12} diff = {diff:+.6f}")

    # Now check setup_params
    print("\n" + "=" * 80)
    print("SETUP_PARAMS BASELINE WEIGHTS")
    print("=" * 80)

    setup_params = cov_calc.get_setup_params()
    setup_baseline = setup_params['baseline_weights']

    for i, cat in enumerate(setup_params['categories']):
        print(f"{cat:<12} {setup_baseline[i]:.6f} ({setup_baseline[i]*100:.2f}%)")

    # Check if categories order in setup_params matches
    print("\n" + "=" * 80)
    print("CATEGORY ORDER IN SETUP_PARAMS")
    print("=" * 80)

    if setup_params['categories'] == cov_calc.categories:
        print("\n✓ setup_params['categories'] matches cov_calc.categories")
    else:
        print("\n✗ setup_params['categories'] DOES NOT MATCH cov_calc.categories")
        print("\nsetup_params['categories']:")
        print(setup_params['categories'])
        print("\ncov_calc.categories:")
        print(cov_calc.categories)


if __name__ == "__main__":
    check_category_ordering()
