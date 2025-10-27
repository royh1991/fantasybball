#!/usr/bin/env python
"""
Debug why top players like Wembanyama and Luka are ranked low.
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
from draft_assistant import DraftAssistant


def debug_player(assistant, player_name):
    """Debug H-score calculation for a specific player."""
    print(f"\n{'='*60}")
    print(f"DEBUGGING: {player_name}")
    print('='*60)

    # Get X-scores
    categories = assistant.setup_params['categories']
    candidate_x = np.array([
        assistant.scoring.calculate_x_score(player_name, cat)
        for cat in categories
    ])

    # Get G-scores
    g_scores = assistant.scoring.calculate_all_g_scores(player_name)

    print(f"\nG-score: {g_scores['TOTAL']:.4f}")
    print(f"X-score sum: {np.sum(candidate_x):.4f}")

    # Evaluate with optimizer
    h_score, optimal_weights = assistant.optimizer.evaluate_player(
        player_name,
        [],  # Empty team
        [],  # No opponents
        0,   # First pick
        total_picks=13,
        last_weights=None,
        format='each_category'
    )

    print(f"H-score: {h_score:.4f}")

    print("\nOptimal weights:")
    sorted_weights = sorted(
        zip(categories, optimal_weights),
        key=lambda x: x[1],
        reverse=True
    )
    for cat, weight in sorted_weights[:5]:
        print(f"  {cat:8s}: {weight:.4f}")

    print("\nX-scores by category:")
    sorted_x = sorted(
        zip(categories, candidate_x),
        key=lambda x: x[1],
        reverse=True
    )
    for cat, x_score in sorted_x[:5]:
        print(f"  {cat:8s}: {x_score:8.3f}")

    return h_score, g_scores['TOTAL']


def main():
    """Debug top players."""
    print("=" * 80)
    print("DEBUGGING TOP PLAYER H-SCORES")
    print("=" * 80)

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    weekly_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    # Initialize
    assistant = DraftAssistant(weekly_file, variance_file)

    # Debug key players
    test_players = [
        'Nikola Jokić',       # #1 G-score, #2 H-score - OK
        'Luka Dončić',        # #2 G-score, #15 H-score - BAD
        'Victor Wembanyama',  # #4 G-score, #14 H-score - BAD
        'Giannis Antetokounmpo',  # #6 G-score, #18 H-score - BAD
        'Scottie Barnes',     # #22 G-score, #5 H-score - BAD
        'Pascal Siakam',      # #30 G-score, #9 H-score - BAD
        'Daniel Gafford',     # Low G-score, high H-score
        'Jamal Murray',       # High G-score, low H-score
    ]

    results = []
    for player in test_players:
        if player in assistant.league_data['PLAYER_NAME'].values:
            h_score, g_score = debug_player(assistant, player)
            results.append({
                'Player': player,
                'H-Score': h_score,
                'G-Score': g_score
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"{r['Player']:<25} H: {r['H-Score']:.4f}  G: {r['G-Score']:.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
The H-scores are too compressed (all between 7.1-7.5).
This suggests the optimizer is converging to similar solutions for all players.

Possible issues:
1. X_delta might still be dominating despite the fix
2. The gradient descent might not be finding good local optima
3. The opponent modeling (0.5 for all categories) might be too uniform
4. The objective function (sum of win probabilities) might need scaling

The fact that role players rank above superstars suggests the algorithm
is not properly capturing player value differences.
    """)


if __name__ == "__main__":
    main()