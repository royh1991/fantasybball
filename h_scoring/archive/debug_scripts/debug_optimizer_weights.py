"""
Debug the weight optimization to understand why KD beats Sabonis.

Even with X_delta adapting, KD (6.72) > Sabonis (6.59).
Let's see what weights the optimizer finds and why.
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

    print("=" * 80)
    print("OPTIMIZER WEIGHT ANALYSIS")
    print("=" * 80)

    # Get candidate X-scores
    sabonis_x = np.array([
        scoring.calculate_x_score("Domantas Sabonis", cat)
        for cat in categories
    ])

    kd_x = np.array([
        scoring.calculate_x_score("Kevin Durant", cat)
        for cat in categories
    ])

    current_team_x = np.zeros(len(categories))
    opponent_x = np.zeros(len(categories))  # Simplified

    # Evaluate both with full optimization
    my_team = []
    opponent_teams = [[]]

    print("\n" + "=" * 80)
    print("SABONIS EVALUATION")
    print("=" * 80)

    sabonis_h, sabonis_weights = optimizer.evaluate_player(
        "Domantas Sabonis", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    print(f"\nH-score: {sabonis_h:.2f}")
    print(f"\n{'Category':<12} {'Baseline Wt':<14} {'Optimal Wt':<14} {'Change':<10} {'X-score':<10}")
    print("-" * 70)

    for i, cat in enumerate(categories):
        change = sabonis_weights[i] - baseline_weights[i]
        change_pct = (change / baseline_weights[i]) * 100 if baseline_weights[i] > 0 else 0
        marker = ""
        if abs(change_pct) > 50:
            marker = " <- BIG CHANGE"
        print(f"{cat:<12} {baseline_weights[i]*100:>12.1f}% {sabonis_weights[i]*100:>12.1f}% {change_pct:>8.1f}% {sabonis_x[i]:>9.2f}{marker}")

    # Calculate win probabilities
    x_delta_sabonis = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=sabonis_x,
        current_team_x=current_team_x
    )

    team_proj_sabonis = current_team_x + sabonis_x + x_delta_sabonis
    win_probs_sabonis = optimizer.calculate_win_probabilities(team_proj_sabonis, opponent_x)

    print(f"\nTotal expected categories won: {np.sum(win_probs_sabonis):.2f}")

    print("\n" + "=" * 80)
    print("KD EVALUATION")
    print("=" * 80)

    kd_h, kd_weights = optimizer.evaluate_player(
        "Kevin Durant", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    print(f"\nH-score: {kd_h:.2f}")
    print(f"\n{'Category':<12} {'Baseline Wt':<14} {'Optimal Wt':<14} {'Change':<10} {'X-score':<10}")
    print("-" * 70)

    for i, cat in enumerate(categories):
        change = kd_weights[i] - baseline_weights[i]
        change_pct = (change / baseline_weights[i]) * 100 if baseline_weights[i] > 0 else 0
        marker = ""
        if abs(change_pct) > 50:
            marker = " <- BIG CHANGE"
        print(f"{cat:<12} {baseline_weights[i]*100:>12.1f}% {kd_weights[i]*100:>12.1f}% {change_pct:>8.1f}% {kd_x[i]:>9.2f}{marker}")

    x_delta_kd = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=kd_x,
        current_team_x=current_team_x
    )

    team_proj_kd = current_team_x + kd_x + x_delta_kd
    win_probs_kd = optimizer.calculate_win_probabilities(team_proj_kd, opponent_x)

    print(f"\nTotal expected categories won: {np.sum(win_probs_kd):.2f}")

    print("\n" + "=" * 80)
    print("WHY IS KD WINNING?")
    print("=" * 80)

    wp_differences = win_probs_sabonis - win_probs_kd

    print("\nSabonis's biggest advantages:")
    sabonis_best = np.argsort(-wp_differences)[:3]
    for idx in sabonis_best:
        if wp_differences[idx] > 0:
            print(f"  {categories[idx]:<12} +{wp_differences[idx]:.3f} ({win_probs_sabonis[idx]:.1%} vs {win_probs_kd[idx]:.1%})")

    print("\nKD's biggest advantages:")
    kd_best = np.argsort(wp_differences)[:3]
    for idx in kd_best:
        if wp_differences[idx] < 0:
            print(f"  {categories[idx]:<12} {wp_differences[idx]:.3f} ({win_probs_sabonis[idx]:.1%} vs {win_probs_kd[idx]:.1%})")


if __name__ == "__main__":
    main()
