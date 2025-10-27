#!/usr/bin/env python
"""
Debug variance calculations to understand H-score issues.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_final import HScoreOptimizerFinal


def debug_variances():
    """Debug variance calculations."""
    print("=" * 80)
    print("DEBUGGING VARIANCE CALCULATIONS")
    print("=" * 80)

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    weekly_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    league_data = pd.read_csv(weekly_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Initialize
    scoring = PlayerScoring(league_data, player_variances)
    cov_calc = CovarianceCalculator(league_data, scoring)
    setup_params = cov_calc.get_setup_params()

    optimizer = HScoreOptimizerFinal(setup_params, scoring)

    print("\n1. COVARIANCE MATRIX DIAGONAL (Original)")
    print("-" * 80)
    categories = setup_params['categories']
    cov_original = optimizer.cov_matrix_original

    print(f"{'Category':<10} {'Cov Diag':>12} {'Sqrt':>12}")
    print("-" * 40)
    for i, cat in enumerate(categories):
        diag_val = cov_original[i, i]
        print(f"{cat:<10} {diag_val:>12.2f} {np.sqrt(diag_val):>12.2f}")

    print("\n2. CATEGORY VARIANCES (Calculated)")
    print("-" * 80)
    print(f"{'Category':<10} {'Variance':>12} {'StdDev':>12}")
    print("-" * 40)
    for cat in categories:
        var = optimizer.category_variances.get(cat, 0)
        print(f"{cat:<10} {var:>12.2f} {np.sqrt(var):>12.2f}")

    print("\n3. IMPACT ON WIN PROBABILITIES")
    print("-" * 80)

    # Test with different differential values
    test_diffs = [1, 3, 5, 10, 20]

    print(f"{'Category':<10}", end='')
    for diff in test_diffs:
        print(f"  Diff={diff:>2}", end='')
    print()
    print("-" * 60)

    for cat in categories:
        cat_idx = categories.index(cat)
        var = optimizer.category_variances.get(cat, 26.0)
        std_dev = np.sqrt(var)

        print(f"{cat:<10}", end='')
        for diff in test_diffs:
            z_score = diff / std_dev
            win_prob = norm.cdf(z_score)
            print(f"  {win_prob:>6.3f}", end='')
        print()

    print("\n4. COMPARING PLAYER WIN PROBS (First Pick)")
    print("-" * 80)

    test_players = ['Victor Wembanyama', 'Nikola Jokić', 'Shai Gilgeous-Alexander']

    for player in test_players:
        print(f"\n{player}:")

        # Get player X-scores
        candidate_x = np.array([scoring.calculate_x_score(player, cat)
                               for cat in categories])

        # Get opponent
        opponent_x = optimizer._calculate_average_opponent_x([], player, 0)

        # Calculate X_delta
        weights = setup_params['baseline_weights']
        x_delta = optimizer.calculate_x_delta(weights, n_remaining=12)

        # Team projection (just player + x_delta for first pick)
        team_projection = candidate_x + x_delta

        # Calculate win probabilities
        win_probs = optimizer.calculate_win_probabilities(team_projection, opponent_x)

        print(f"  Categories with >0.7 win prob:")
        high_wins = [(cat, win_probs[i]) for i, cat in enumerate(categories) if win_probs[i] > 0.7]
        for cat, prob in high_wins:
            diff = team_projection[categories.index(cat)] - opponent_x[categories.index(cat)]
            print(f"    {cat}: {prob:.3f} (diff={diff:.2f})")

        print(f"  Total win prob sum: {np.sum(win_probs):.3f}")

        # Get actual H-score
        h_score, _ = optimizer.evaluate_player(
            player, [], [], 0, total_picks=13, format='each_category'
        )
        print(f"  H-score after optimization: {h_score:.3f}")

    print("\n5. CHECKING X_DELTA IMPACT")
    print("-" * 80)

    # Check if X_delta is still dominating
    weights = setup_params['baseline_weights']
    x_delta = optimizer.calculate_x_delta(weights, n_remaining=12)

    print(f"X_delta values: {[f'{x:.2f}' for x in x_delta]}")
    print(f"X_delta norm: {np.linalg.norm(x_delta):.2f}")

    # Compare to player norms
    for player in test_players:
        candidate_x = np.array([scoring.calculate_x_score(player, cat)
                               for cat in categories])
        print(f"{player} X norm: {np.linalg.norm(candidate_x):.2f}")

    print(f"\nX_delta is {np.linalg.norm(x_delta) / np.linalg.norm(candidate_x):.1f}x the size of a typical player")


if __name__ == '__main__':
    from scipy.stats import norm
    debug_variances()