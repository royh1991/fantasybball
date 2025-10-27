"""
Test that the X_delta fix makes Sabonis beat KD on pick #1.

The bug was that X_delta was the same for all candidates.
The fix makes X_delta adapt:
- After drafting Sabonis (7.50 DD), X_delta DD should be LOWER
- After drafting KD (-0.16 DD), X_delta DD should be HIGHER
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_final import HScoreOptimizerFinal


def main():
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
    setup_params = cov_calc.get_setup_params()

    categories = setup_params['categories']
    dd_idx = categories.index('DD')

    # Create optimizer with fix
    optimizer = HScoreOptimizerFinal(setup_params, scoring, omega=0.7, gamma=0.25)

    print("=" * 80)
    print("TESTING X_DELTA FIX: SABONIS VS KD")
    print("=" * 80)

    # Get X-scores for both candidates
    sabonis_x = np.array([
        scoring.calculate_x_score("Domantas Sabonis", cat)
        for cat in categories
    ])

    kd_x = np.array([
        scoring.calculate_x_score("Kevin Durant", cat)
        for cat in categories
    ])

    print(f"\nSabonis DD X-score: {sabonis_x[dd_idx]:.2f}")
    print(f"KD DD X-score: {kd_x[dd_idx]:.2f}")

    # Current team (empty on pick #1)
    current_team_x = np.zeros(len(categories))

    # Calculate X_delta for SABONIS
    baseline_weights = setup_params['baseline_weights']

    x_delta_sabonis = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=sabonis_x,
        current_team_x=current_team_x
    )

    # Calculate X_delta for KD
    x_delta_kd = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=kd_x,
        current_team_x=current_team_x
    )

    print("\n" + "=" * 80)
    print("X_DELTA COMPARISON")
    print("=" * 80)

    print(f"\n{'Category':<12} {'Sabonis X_delta':<18} {'KD X_delta':<18} {'Difference':<12}")
    print("-" * 70)

    for i, cat in enumerate(categories):
        diff = x_delta_sabonis[i] - x_delta_kd[i]
        marker = ""
        if cat == 'DD':
            marker = " ← KEY CATEGORY"
        print(f"{cat:<12} {x_delta_sabonis[i]:>16.2f} {x_delta_kd[i]:>16.2f} {diff:>11.2f}{marker}")

    print("\n" + "=" * 80)
    print("DD CATEGORY ANALYSIS")
    print("=" * 80)

    print(f"\nSabonis DD X_delta: {x_delta_sabonis[dd_idx]:.2f}")
    print(f"KD DD X_delta: {x_delta_kd[dd_idx]:.2f}")
    print(f"Difference: {x_delta_kd[dd_idx] - x_delta_sabonis[dd_idx]:.2f}")

    if x_delta_sabonis[dd_idx] < x_delta_kd[dd_idx]:
        print("\n✓ FIX WORKING!")
        print(f"  - Sabonis (elite DD: {sabonis_x[dd_idx]:.2f}) → X_delta DD is LOWER ({x_delta_sabonis[dd_idx]:.2f})")
        print(f"  - KD (weak DD: {kd_x[dd_idx]:.2f}) → X_delta DD is HIGHER ({x_delta_kd[dd_idx]:.2f})")
        print("  - This correctly accounts for Sabonis already providing elite DD!")
    else:
        print("\n✗ FIX NOT WORKING")
        print(f"  - Sabonis X_delta DD ({x_delta_sabonis[dd_idx]:.2f}) should be LOWER than KD ({x_delta_kd[dd_idx]:.2f})")

    # Now evaluate full H-scores
    print("\n" + "=" * 80)
    print("H-SCORE COMPARISON")
    print("=" * 80)

    my_team = []
    opponent_teams = [[]]  # Empty opponents on pick #1

    sabonis_h, sabonis_weights = optimizer.evaluate_player(
        "Domantas Sabonis", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    kd_h, kd_weights = optimizer.evaluate_player(
        "Kevin Durant", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    sabonis_g = scoring.calculate_all_g_scores("Domantas Sabonis")['TOTAL']
    kd_g = scoring.calculate_all_g_scores("Kevin Durant")['TOTAL']

    print(f"\n{'Player':<20} {'G-score':<10} {'H-score':<10} {'Winner':<10}")
    print("-" * 50)
    print(f"{'Sabonis':<20} {sabonis_g:>8.2f} {sabonis_h:>8.2f} {'<-- SHOULD WIN' if sabonis_h > kd_h else ''}")
    print(f"{'KD':<20} {kd_g:>8.2f} {kd_h:>8.2f} {'<-- SHOULD WIN' if kd_h > sabonis_h else ''}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if sabonis_h > kd_h:
        print("\n✓✓✓ SUCCESS! Sabonis now beats KD!")
        print(f"\nSabonis H-score: {sabonis_h:.2f}")
        print(f"KD H-score: {kd_h:.2f}")
        print(f"Difference: {sabonis_h - kd_h:.2f}")
        print(f"\nG-score difference: {sabonis_g - kd_g:.2f}")
        print("\nThe X_delta fix correctly values Sabonis's elite DD specialization!")
    else:
        print("\n✗✗✗ STILL BROKEN")
        print(f"\nKD H-score: {kd_h:.2f}")
        print(f"Sabonis H-score: {sabonis_h:.2f}")
        print(f"Difference: {kd_h - sabonis_h:.2f}")
        print("\nKD still winning despite Sabonis having 2.84 higher G-score")

    # Show total projections
    print("\n" + "=" * 80)
    print("TOTAL DD PROJECTIONS")
    print("=" * 80)

    sabonis_total_dd = current_team_x[dd_idx] + sabonis_x[dd_idx] + x_delta_sabonis[dd_idx]
    kd_total_dd = current_team_x[dd_idx] + kd_x[dd_idx] + x_delta_kd[dd_idx]

    print(f"\nSabonis: {current_team_x[dd_idx]:.2f} (current) + {sabonis_x[dd_idx]:.2f} (player) + {x_delta_sabonis[dd_idx]:.2f} (future) = {sabonis_total_dd:.2f}")
    print(f"KD:      {current_team_x[dd_idx]:.2f} (current) + {kd_x[dd_idx]:.2f} (player) + {x_delta_kd[dd_idx]:.2f} (future) = {kd_total_dd:.2f}")

    print(f"\nDifference: {sabonis_total_dd - kd_total_dd:.2f} DD")

    if abs(sabonis_total_dd - kd_total_dd) < abs(sabonis_x[dd_idx] - kd_x[dd_idx]):
        print("\n✓ X_delta correctly compensates for candidate strength")
        print(f"  - Player DD difference: {sabonis_x[dd_idx] - kd_x[dd_idx]:.2f}")
        print(f"  - Total DD difference:  {sabonis_total_dd - kd_total_dd:.2f}")
        print(f"  - X_delta reduced the gap by accounting for future picks")


if __name__ == "__main__":
    main()
