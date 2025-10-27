#!/usr/bin/env python
"""
Debug X_delta calculation to understand why bad players get high H-scores.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer import HScoreOptimizer


def main():
    """Debug X_delta calculation."""
    print("=" * 80)
    print("X_DELTA DEBUGGING")
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

    # Initialize scoring
    scoring = PlayerScoring(league_data, player_variances)

    # Get player list
    player_list = league_data['PLAYER_NAME'].unique().tolist()

    # Setup covariance
    cov_calc = CovarianceCalculator(scoring, player_list)
    setup_params = cov_calc.calculate_all()

    # Initialize optimizer
    optimizer = HScoreOptimizer(setup_params, scoring)

    # Test players
    test_players = ['Franz Wagner', 'Anthony Edwards']

    for player_name in test_players:
        print(f"\n{'='*60}")
        print(f"Player: {player_name}")
        print("="*60)

        # Get X-scores
        candidate_x = np.array([
            scoring.calculate_x_score(player_name, cat)
            for cat in setup_params['categories']
        ])

        print("\nX-scores:")
        for i, cat in enumerate(setup_params['categories']):
            print(f"  {cat:8s}: {candidate_x[i]:8.3f}")

        # Calculate X_delta with baseline weights
        weights = optimizer.baseline_weights.copy()
        n_remaining = 12  # First pick

        print(f"\nBaseline weights:")
        for i, cat in enumerate(setup_params['categories']):
            print(f"  {cat:8s}: {weights[i]:.4f}")

        # Calculate X_delta
        x_delta = optimizer.calculate_x_delta(weights, n_remaining)

        print(f"\nX_delta (n_remaining={n_remaining}):")
        for i, cat in enumerate(setup_params['categories']):
            print(f"  {cat:8s}: {x_delta[i]:8.3f}")

        # Total projection
        team_projection = candidate_x + x_delta

        print(f"\nTeam projection (candidate + X_delta):")
        for i, cat in enumerate(setup_params['categories']):
            print(f"  {cat:8s}: {team_projection[i]:8.3f}")

        # Calculate win probabilities against zero opponent
        opponent_x = np.zeros(len(setup_params['categories']))
        variance = 2 * 13 + n_remaining * 1.0
        std_dev = np.sqrt(variance)

        z_scores = team_projection / std_dev
        win_probs = norm.cdf(z_scores)

        print(f"\nWin probabilities (vs zero opponent):")
        for i, cat in enumerate(setup_params['categories']):
            print(f"  {cat:8s}: {win_probs[i]:.4f}")

        print(f"\nTotal win probability sum: {np.sum(win_probs):.4f}")


if __name__ == "__main__":
    from scipy.stats import norm
    main()