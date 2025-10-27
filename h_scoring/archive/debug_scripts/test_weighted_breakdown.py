"""
Show the weighted contribution of each category to understand the tie.
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
    baseline_weights = setup_params['baseline_weights']

    # Create optimizer
    optimizer = HScoreOptimizerFinal(setup_params, scoring, omega=0.7, gamma=0.25)

    # Get X-scores
    sabonis_x = np.array([
        scoring.calculate_x_score("Domantas Sabonis", cat)
        for cat in categories
    ])

    kd_x = np.array([
        scoring.calculate_x_score("Kevin Durant", cat)
        for cat in categories
    ])

    current_team_x = np.zeros(len(categories))
    opponent_x = np.zeros(len(categories))

    # Calculate X_delta for both
    x_delta_sabonis = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=sabonis_x,
        current_team_x=current_team_x
    )

    x_delta_kd = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=kd_x,
        current_team_x=current_team_x
    )

    # Team projections
    team_proj_sabonis = current_team_x + sabonis_x + x_delta_sabonis
    team_proj_kd = current_team_x + kd_x + x_delta_kd

    # Win probabilities
    win_probs_sabonis = optimizer.calculate_win_probabilities(team_proj_sabonis, opponent_x)
    win_probs_kd = optimizer.calculate_win_probabilities(team_proj_kd, opponent_x)

    print("=" * 90)
    print("WEIGHTED OBJECTIVE BREAKDOWN")
    print("=" * 90)

    print(f"\n{'Category':<12} {'Weight':<10} {'Sabonis WP':<12} {'KD WP':<12} {'Sab Contrib':<14} {'KD Contrib':<14} {'Diff':<10}")
    print("-" * 90)

    sabonis_total = 0
    kd_total = 0

    for i, cat in enumerate(categories):
        sab_contrib = baseline_weights[i] * win_probs_sabonis[i]
        kd_contrib = baseline_weights[i] * win_probs_kd[i]
        diff = sab_contrib - kd_contrib

        sabonis_total += sab_contrib
        kd_total += kd_contrib

        marker = ""
        if abs(diff) > 0.01:
            if diff > 0:
                marker = " ← Sabonis edge"
            else:
                marker = " ← KD edge"

        print(f"{cat:<12} {baseline_weights[i]*100:>8.1f}% {win_probs_sabonis[i]:>11.1%} {win_probs_kd[i]:>11.1%} {sab_contrib:>13.4f} {kd_contrib:>13.4f} {diff:>9.4f}{marker}")

    print("-" * 90)
    print(f"{'TOTAL':<12} {'100.0%':<10} {np.sum(win_probs_sabonis):>11.2f} {np.sum(win_probs_kd):>11.2f} {sabonis_total:>13.4f} {kd_total:>13.4f} {sabonis_total - kd_total:>9.4f}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    print(f"\nSabonis weighted H-score: {sabonis_total:.4f}")
    print(f"KD weighted H-score: {kd_total:.4f}")
    print(f"Difference: {sabonis_total - kd_total:.4f}")

    if abs(sabonis_total - kd_total) < 0.001:
        print("\n⚠️  NEARLY TIED!")
        print("\nThis means:")
        print("  - Sabonis's elite DD (22.6% weight, 86% win prob) = 0.195 contribution")
        print("  - KD's balanced stats across many categories = similar total")
        print("\nThe weighting is working, but there may be an issue with opponent_x = 0.")
        print("Against no opponent, win probabilities are just based on your team's projection.")
    elif sabonis_total > kd_total:
        print("\n✓ SABONIS WINS!")
        print(f"\nSabonis is correctly valued {sabonis_total - kd_total:.4f} points higher.")
    else:
        print("\n✗ KD WINS")
        print(f"\nKD is valued {kd_total - sabonis_total:.4f} points higher despite lower G-score.")


if __name__ == "__main__":
    main()
