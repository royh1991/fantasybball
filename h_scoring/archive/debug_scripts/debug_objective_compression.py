#!/usr/bin/env python
"""
Debug why H-scores are so compressed.
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
from draft_assistant import DraftAssistant


def debug_objective_details(assistant, player_name):
    """Debug the objective calculation in detail."""

    optimizer = assistant.optimizer
    scoring = assistant.scoring
    categories = assistant.setup_params['categories']

    # Get X-scores
    candidate_x = np.array([
        scoring.calculate_x_score(player_name, cat)
        for cat in categories
    ])

    # Setup for first pick
    current_team_x = np.zeros(len(categories))
    opponent_x = np.ones(len(categories)) * 0.5  # From our fix
    n_remaining = 12

    # Use baseline weights
    weights = optimizer.baseline_weights.copy()

    # Calculate X_delta
    x_delta, diagnostics = optimizer._compute_xdelta_exact(
        jC=weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=0
    )

    # Team projection
    team_projection = current_team_x + candidate_x + x_delta

    # Calculate win probabilities
    variance = 2 * 13 + n_remaining * 1.0
    std_dev = np.sqrt(variance)

    differential = team_projection - opponent_x
    z_scores = differential / std_dev
    win_probs = norm.cdf(z_scores)

    print(f"\n{player_name}:")
    print(f"  X-score sum: {np.sum(candidate_x):.2f}")
    print(f"  X_delta norm: {np.linalg.norm(x_delta):.2f}")
    print(f"  X_delta diagnostics: multiplier={diagnostics['multiplier']:.0f}, norm_unit={diagnostics['norm_xdelta_unit']:.4f}")

    print(f"\n  Win probabilities by category:")
    for i, cat in enumerate(categories):
        print(f"    {cat:8s}: {win_probs[i]:.3f} (z={z_scores[i]:.2f}, diff={differential[i]:.2f})")

    print(f"\n  Objective (sum): {np.sum(win_probs):.4f}")
    print(f"  Variance: {variance:.2f}, Std Dev: {std_dev:.2f}")

    return np.sum(win_probs), np.sum(candidate_x)


def main():
    """Debug objective compression."""
    print("=" * 80)
    print("DEBUGGING OBJECTIVE COMPRESSION")
    print("=" * 80)

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    weekly_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    # Initialize
    assistant = DraftAssistant(weekly_file, variance_file)

    # Test players
    test_players = [
        'Nikola Jokić',      # Elite (X=22.1)
        'Victor Wembanyama',  # Star (X=13.4)
        'Scottie Barnes',     # Good (X=7.6)
        'Daniel Gafford',     # Role (X=-2.7)
    ]

    results = []
    for player in test_players:
        if player in assistant.league_data['PLAYER_NAME'].values:
            obj, x_sum = debug_objective_details(assistant, player)
            results.append((player, obj, x_sum))

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nObjective values (with baseline weights, before optimization):")
    for player, obj, x_sum in results:
        print(f"  {player:<25} Obj: {obj:.4f}  X-sum: {x_sum:6.2f}")

    print(f"\nProblem identified:")
    print(f"  1. Variance is fixed at {2*13 + 12:.0f} for all players")
    print(f"  2. This gives std_dev = {np.sqrt(2*13 + 12):.2f}")
    print(f"  3. Most differentials fall within ±2 std devs")
    print(f"  4. This compresses win probs to 0.05-0.95 range")
    print(f"  5. Sum of 11 categories → most players get 5-8 total")

    print(f"\nThe issue is that X_delta is adding similar adjustments to all players,")
    print(f"reducing the differentiation between elite and average players.")

    print(f"\nPossible solutions:")
    print(f"  1. Scale the objective by player quality (e.g., multiply by X-score sum)")
    print(f"  2. Use a different variance model that scales with player quality")
    print(f"  3. Reduce the impact of X_delta")
    print(f"  4. Use a different objective function entirely")


if __name__ == "__main__":
    main()