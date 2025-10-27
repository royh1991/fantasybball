#!/usr/bin/env python
"""
Detailed debugging of the optimizer to understand why bad players get high H-scores.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer import HScoreOptimizer


def debug_objective_calculation(optimizer, player_name, scoring, setup_params):
    """Debug the objective calculation for a specific player."""

    print(f"\n{'='*60}")
    print(f"DEBUGGING: {player_name}")
    print('='*60)

    # Get X-scores
    candidate_x = np.array([
        scoring.calculate_x_score(player_name, cat)
        for cat in setup_params['categories']
    ])

    print("\nX-scores:")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {candidate_x[i]:8.3f}")
    print(f"  SUM: {np.sum(candidate_x):.3f}")

    # Empty team, first pick
    current_team_x = np.zeros(len(setup_params['categories']))
    n_remaining = 12

    # Check what opponent_x would be (after our fix)
    # Simulate the evaluate_player logic
    picks_made = 0
    if picks_made == 0:
        opponent_x = np.ones(optimizer.n_cats) * 0.5  # From our fix

    print(f"\nOpponent X-scores (modeled):")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {opponent_x[i]:8.3f}")
    print(f"  SUM: {np.sum(opponent_x):.3f}")

    # Use baseline weights
    weights = optimizer.baseline_weights.copy()

    # Calculate X_delta
    x_delta = optimizer.calculate_x_delta(weights, n_remaining)

    print(f"\nX_delta:")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {x_delta[i]:8.3f}")
    print(f"  SUM: {np.sum(x_delta):.3f}")

    # Team projection
    team_projection = current_team_x + candidate_x + x_delta

    print(f"\nTeam projection (candidate + X_delta):")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {team_projection[i]:8.3f}")
    print(f"  SUM: {np.sum(team_projection):.3f}")

    # Differential
    differential = team_projection - opponent_x

    print(f"\nDifferential (team - opponent):")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {differential[i]:8.3f}")
    print(f"  SUM: {np.sum(differential):.3f}")

    # Win probabilities
    variance = 2 * 13 + n_remaining * 1.0
    win_probs = optimizer.calculate_win_probabilities(team_projection, opponent_x, variance)

    print(f"\nWin probabilities:")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {win_probs[i]:.4f}")
    print(f"  SUM (objective): {np.sum(win_probs):.4f}")

    return np.sum(win_probs)


def main():
    """Debug the optimizer."""

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
    optimizer = HScoreOptimizer(setup_params, scoring)

    print("="*80)
    print("OPTIMIZER DEBUGGING")
    print("="*80)

    # Debug Franz Wagner vs Anthony Edwards
    franz_obj = debug_objective_calculation(optimizer, 'Franz Wagner', scoring, setup_params)
    anthony_obj = debug_objective_calculation(optimizer, 'Anthony Edwards', scoring, setup_params)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Franz Wagner initial objective: {franz_obj:.4f}")
    print(f"Anthony Edwards initial objective: {anthony_obj:.4f}")
    print()
    print("The issue: Franz has LOWER X-scores but gets HIGHER objective!")
    print("This suggests the X_delta calculation is problematic.")

    # Let's check what happens without X_delta
    print("\n" + "="*80)
    print("WITHOUT X_DELTA")
    print("="*80)

    # Franz without X_delta
    candidate_x_franz = np.array([scoring.calculate_x_score('Franz Wagner', cat)
                                  for cat in setup_params['categories']])
    opponent_x = np.ones(optimizer.n_cats) * 0.5
    variance = 2 * 13 + 12 * 1.0

    diff_franz = candidate_x_franz - opponent_x
    win_probs_franz = norm.cdf(diff_franz / np.sqrt(variance))

    print(f"Franz Wagner (no X_delta): {np.sum(win_probs_franz):.4f}")

    # Anthony without X_delta
    candidate_x_ant = np.array([scoring.calculate_x_score('Anthony Edwards', cat)
                               for cat in setup_params['categories']])
    diff_ant = candidate_x_ant - opponent_x
    win_probs_ant = norm.cdf(diff_ant / np.sqrt(variance))

    print(f"Anthony Edwards (no X_delta): {np.sum(win_probs_ant):.4f}")

    print("\nWithout X_delta, Anthony Edwards correctly scores higher!")
    print("The X_delta calculation is causing the inversion.")


if __name__ == "__main__":
    main()