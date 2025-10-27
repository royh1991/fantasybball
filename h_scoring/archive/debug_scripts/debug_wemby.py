#!/usr/bin/env python
"""
Debug why Wembanyama and other elite big men are ranked too low.
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
from modules.h_optimizer_final import HScoreOptimizerFinal


def debug_wembanyama():
    """Debug Wembanyama's H-score calculation."""
    print("=" * 80)
    print("DEBUGGING WEMBANYAMA AND ELITE BIG MEN")
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

    # Compare elite big men vs guards
    big_men = ['Victor Wembanyama', 'Giannis Antetokounmpo', 'Anthony Davis', 'Nikola Jokić']
    guards = ['Shai Gilgeous-Alexander', 'Luka Dončić', 'James Harden', 'Stephen Curry']

    categories = setup_params['categories']

    print("\n1. PLAYER STATS COMPARISON")
    print("-" * 80)

    # Collect stats for all players
    player_stats = {}
    for player in big_men + guards:
        x_scores = np.array([scoring.calculate_x_score(player, cat) for cat in categories])
        g_scores = scoring.calculate_all_g_scores(player)
        player_stats[player] = {
            'x_scores': x_scores,
            'g_total': g_scores['TOTAL'],
            'type': 'Big' if player in big_men else 'Guard'
        }

    # Show X-scores by category
    print(f"\n{'Category':<10}", end='')
    for player in ['Wembanyama', 'Giannis', 'SGA', 'Harden']:
        print(f"{player[:10]:>12}", end='')
    print()
    print("-" * 60)

    display_players = ['Victor Wembanyama', 'Giannis Antetokounmpo',
                      'Shai Gilgeous-Alexander', 'James Harden']

    for i, cat in enumerate(categories):
        print(f"{cat:<10}", end='')
        for player in display_players:
            x_score = player_stats[player]['x_scores'][i]
            print(f"{x_score:>12.2f}", end='')
        print()

    print(f"\n{'X-norm':<10}", end='')
    for player in display_players:
        x_norm = np.linalg.norm(player_stats[player]['x_scores'])
        print(f"{x_norm:>12.2f}", end='')
    print()

    print(f"{'G-score':<10}", end='')
    for player in display_players:
        print(f"{player_stats[player]['g_total']:>12.2f}", end='')
    print()

    print("\n2. OPPONENT MODELING CHECK")
    print("-" * 80)

    # Check opponent X for first pick
    opponent_x = optimizer._calculate_average_opponent_x([], 'Victor Wembanyama', 0)
    print(f"Opponent X-scores: {[f'{x:.2f}' for x in opponent_x]}")
    print(f"Opponent X norm: {np.linalg.norm(opponent_x):.2f}")

    print("\n3. WIN PROBABILITY ANALYSIS")
    print("-" * 80)

    for player in ['Victor Wembanyama', 'Shai Gilgeous-Alexander']:
        print(f"\n{player}:")

        # Get player X-scores
        candidate_x = player_stats[player]['x_scores']

        # Calculate X_delta
        weights = setup_params['baseline_weights']
        x_delta = optimizer.calculate_x_delta(weights, n_remaining=12)

        # Team projection
        team_projection = candidate_x + x_delta

        # Differential vs opponent
        differential = team_projection - opponent_x

        # Win probabilities
        variance = 2 * 13  # Simplified
        z_scores = differential / np.sqrt(variance)
        z_scores = np.clip(z_scores, -10, 10)  # Clip for stability
        win_probs = norm.cdf(z_scores)

        print(f"  Strong categories (>0.8 win prob):")
        for i, cat in enumerate(categories):
            if win_probs[i] > 0.8:
                print(f"    {cat}: {win_probs[i]:.3f} (diff: {differential[i]:.2f})")

        print(f"  Weak categories (<0.3 win prob):")
        for i, cat in enumerate(categories):
            if win_probs[i] < 0.3:
                print(f"    {cat}: {win_probs[i]:.3f} (diff: {differential[i]:.2f})")

        print(f"  Average win prob: {np.mean(win_probs):.3f}")
        print(f"  Sum of win probs: {np.sum(win_probs):.3f}")

    print("\n4. DETAILED H-SCORE CALCULATION")
    print("-" * 80)

    for player in ['Victor Wembanyama', 'Shai Gilgeous-Alexander', 'Nikola Jokić']:
        print(f"\n{player}:")

        # Evaluate with detailed output
        h_score, optimal_weights = optimizer.evaluate_player(
            player, [], [], 0, total_picks=13, format='each_category'
        )

        print(f"  H-score: {h_score:.3f}")
        print(f"  Top 3 weighted categories:")
        weight_cats = [(w, cat) for w, cat in zip(optimal_weights, categories)]
        weight_cats.sort(reverse=True)
        for w, cat in weight_cats[:3]:
            print(f"    {cat}: {w:.3f}")

    print("\n5. TESTING DIFFERENT OPPONENT STRENGTHS")
    print("-" * 80)

    # Temporarily modify opponent calculation to test impact
    original_opponent_x = optimizer._calculate_average_opponent_x([], 'Victor Wembanyama', 0)

    # Test with weaker opponents (50% strength)
    weaker_opponent_x = original_opponent_x * 0.5

    # Test with stronger opponents (150% strength)
    stronger_opponent_x = original_opponent_x * 1.5

    print(f"\n{'Player':<25} {'Weak Opp':>10} {'Normal':>10} {'Strong Opp':>10}")
    print("-" * 60)

    for player in ['Victor Wembanyama', 'Shai Gilgeous-Alexander', 'Nikola Jokić']:
        candidate_x = np.array([scoring.calculate_x_score(player, cat) for cat in categories])
        x_delta = optimizer.calculate_x_delta(setup_params['baseline_weights'], n_remaining=12)
        team_projection = candidate_x + x_delta

        # Calculate objectives with different opponent strengths
        weak_obj = 0
        normal_obj = 0
        strong_obj = 0

        for opp_x, obj_ref in [(weaker_opponent_x, 'weak'),
                               (original_opponent_x, 'normal'),
                               (stronger_opponent_x, 'strong')]:
            differential = team_projection - opp_x
            variance = 2 * 13
            z_scores = np.clip(differential / np.sqrt(variance), -10, 10)
            win_probs = norm.cdf(z_scores)
            obj_val = np.sum(win_probs)

            if obj_ref == 'weak':
                weak_obj = obj_val
            elif obj_ref == 'normal':
                normal_obj = obj_val
            else:
                strong_obj = obj_val

        print(f"{player:<25} {weak_obj:>10.3f} {normal_obj:>10.3f} {strong_obj:>10.3f}")

    print("\n6. CHECKING VARIANCE CALCULATION")
    print("-" * 80)

    # The variance used in win probability might be wrong
    print("Current variance formula: 2 * N (where N = 13)")
    print("This gives variance = 26 for all categories")
    print("\nBut categories have different natural variances!")

    # Check actual category variances
    print("\nActual within-player variances by category:")
    for cat in categories:
        var_info = scoring._calculate_league_stats(cat)
        print(f"  {cat:<10}: within={var_info['within_variance']:.2f}, between={var_info['between_variance']:.2f}")


if __name__ == '__main__':
    debug_wembanyama()