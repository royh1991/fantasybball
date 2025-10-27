#!/usr/bin/env python
"""
Debug why H-scores are producing incorrect rankings.
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


def debug_player_evaluation():
    """Debug player evaluation to understand why rankings are wrong."""
    print("=" * 80)
    print("DEBUGGING H-SCORE RANKINGS")
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

    # Test specific players
    test_players = [
        'Nikola Jokić',
        'Luka Dončić',
        'Trey Murphy III',
        'Norman Powell'
    ]

    print("\n1. PLAYER X-SCORES")
    print("-" * 80)

    for player in test_players:
        x_scores = [scoring.calculate_x_score(player, cat)
                   for cat in setup_params['categories']]
        x_total = sum(x_scores)
        print(f"\n{player}:")
        print(f"  X-scores: {[f'{x:.2f}' for x in x_scores]}")
        print(f"  X-total: {x_total:.2f}")
        print(f"  X-norm: {np.linalg.norm(x_scores):.2f}")

    print("\n2. OPPONENT MODELING")
    print("-" * 80)

    # Check what opponent X-scores look like
    opponent_x = optimizer._calculate_average_opponent_x([], test_players[0], 0)
    print(f"Opponent X (first pick): {[f'{x:.2f}' for x in opponent_x]}")
    print(f"Opponent X norm: {np.linalg.norm(opponent_x):.2f}")

    print("\n3. X_DELTA CALCULATION")
    print("-" * 80)

    weights = setup_params['baseline_weights']
    x_delta = optimizer.calculate_x_delta(weights, n_remaining=12)
    print(f"X_delta: {[f'{x:.2f}' for x in x_delta]}")
    print(f"X_delta norm: {np.linalg.norm(x_delta):.2f}")

    print("\n4. DETAILED EVALUATION FOR TOP PLAYERS")
    print("-" * 80)

    for player in test_players:
        print(f"\n{player}:")

        # Get player X-scores
        candidate_x = np.array([scoring.calculate_x_score(player, cat)
                               for cat in setup_params['categories']])

        # Empty team (first pick scenario)
        current_team_x = np.zeros(len(setup_params['categories']))

        # Get opponent X
        opponent_x = optimizer._calculate_average_opponent_x([], player, 0)

        # Calculate X_delta
        x_delta = optimizer.calculate_x_delta(weights, n_remaining=12, candidate_player=player)

        # Project team with candidate
        team_projection = current_team_x + candidate_x + x_delta

        print(f"  Candidate X norm: {np.linalg.norm(candidate_x):.2f}")
        print(f"  X_delta norm: {np.linalg.norm(x_delta):.2f}")
        print(f"  Team projection norm: {np.linalg.norm(team_projection):.2f}")
        print(f"  Opponent norm: {np.linalg.norm(opponent_x):.2f}")

        # Calculate differential
        differential = team_projection - opponent_x
        print(f"  Differential: {[f'{d:.2f}' for d in differential]}")
        print(f"  Differential norm: {np.linalg.norm(differential):.2f}")

        # Calculate win probabilities (simplified)
        variance = 2 * 13  # Simplified variance
        z_scores = differential / np.sqrt(variance)
        win_probs = norm.cdf(z_scores)

        print(f"  Win probabilities: {[f'{p:.3f}' for p in win_probs]}")
        print(f"  Average win prob: {np.mean(win_probs):.3f}")

        # Get actual H-score
        h_score, optimal_weights = optimizer.evaluate_player(
            player, [], [], 0, total_picks=13, format='each_category'
        )
        print(f"  H-score: {h_score:.3f}")
        print(f"  Optimal weights: {[f'{w:.3f}' for w in optimal_weights]}")

    print("\n5. CATEGORY-BY-CATEGORY COMPARISON")
    print("-" * 80)

    categories = setup_params['categories']
    jokic_x = np.array([scoring.calculate_x_score('Nikola Jokić', cat) for cat in categories])
    murphy_x = np.array([scoring.calculate_x_score('Trey Murphy III', cat) for cat in categories])

    print(f"{'Category':<10} {'Jokić':>10} {'Murphy':>10} {'Diff':>10}")
    print("-" * 40)
    for i, cat in enumerate(categories):
        diff = jokic_x[i] - murphy_x[i]
        print(f"{cat:<10} {jokic_x[i]:>10.2f} {murphy_x[i]:>10.2f} {diff:>10.2f}")

    print("\n6. CHECKING OPTIMIZER OBJECTIVE FUNCTION")
    print("-" * 80)

    # Test the objective function directly for Jokić
    candidate_x = np.array([scoring.calculate_x_score('Nikola Jokić', cat)
                           for cat in categories])
    current_team_x = np.zeros(len(categories))
    opponent_x = optimizer._calculate_average_opponent_x([], 'Nikola Jokić', 0)

    # Test with baseline weights
    objective_baseline = optimizer.calculate_objective(
        weights, candidate_x, current_team_x, opponent_x, 12, format='each_category'
    )
    print(f"Jokić objective with baseline weights: {objective_baseline:.3f}")

    # Test with uniform weights
    uniform_weights = np.ones(len(categories)) / len(categories)
    objective_uniform = optimizer.calculate_objective(
        uniform_weights, candidate_x, current_team_x, opponent_x, 12, format='each_category'
    )
    print(f"Jokić objective with uniform weights: {objective_uniform:.3f}")


if __name__ == '__main__':
    from scipy.stats import norm
    debug_player_evaluation()