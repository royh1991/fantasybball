"""
Compare LeBron vs KAT in:
1. Original paper-faithful algorithm (before fixes)
2. Fixed algorithm (after all fixes)
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful
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

    # Initialize scoring (this now has the variance fix)
    scoring = PlayerScoring(league_data, player_variances, roster_size=13)
    cov_calc = CovarianceCalculator(league_data, scoring)
    setup_params = cov_calc.get_setup_params()

    categories = setup_params['categories']

    print("=" * 100)
    print("LEBRON JAMES vs KARL-ANTHONY TOWNS")
    print("=" * 100)

    # Get G-scores
    lebron_g = scoring.calculate_all_g_scores("LeBron James")
    kat_g = scoring.calculate_all_g_scores("Karl-Anthony Towns")

    lebron_g_total = sum(lebron_g.values())
    kat_g_total = sum(kat_g.values())

    print("\nG-SCORES:")
    print(f"  LeBron: {lebron_g_total:.2f}")
    print(f"  KAT: {kat_g_total:.2f}")
    print(f"  Winner: {'LeBron' if lebron_g_total > kat_g_total else 'KAT'} (+{abs(lebron_g_total - kat_g_total):.2f})")

    # Get X-scores (now using fixed variance)
    lebron_x = np.array([scoring.calculate_x_score("LeBron James", cat) for cat in categories])
    kat_x = np.array([scoring.calculate_x_score("Karl-Anthony Towns", cat) for cat in categories])

    print("\n" + "=" * 100)
    print("X-SCORES (AFTER VARIANCE FIX)")
    print("=" * 100)

    print(f"\n{'Category':<12} {'LeBron':<10} {'KAT':<10} {'Difference':<12}")
    print("-" * 50)
    for i, cat in enumerate(categories):
        diff = lebron_x[i] - kat_x[i]
        marker = ""
        if abs(diff) > 0.5:
            marker = " <- " + ("LeBron" if diff > 0 else "KAT")
        print(f"{cat:<12} {lebron_x[i]:>8.2f} {kat_x[i]:>8.2f} {diff:>11.2f}{marker}")

    # Test with ORIGINAL paper-faithful algorithm
    print("\n" + "=" * 100)
    print("ORIGINAL PAPER-FAITHFUL ALGORITHM (No fixes)")
    print("=" * 100)

    optimizer_original = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    my_team = []
    opponent_teams = [[]]

    lebron_h_original, lebron_weights_original = optimizer_original.evaluate_player(
        "LeBron James", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    kat_h_original, kat_weights_original = optimizer_original.evaluate_player(
        "Karl-Anthony Towns", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    print(f"\nH-scores (original):")
    print(f"  LeBron: {lebron_h_original:.4f}")
    print(f"  KAT: {kat_h_original:.4f}")
    print(f"  Difference: {lebron_h_original - kat_h_original:+.4f}")
    print(f"  Winner: {'LeBron' if lebron_h_original > kat_h_original else 'KAT'}")

    # Test with FIXED algorithm
    print("\n" + "=" * 100)
    print("FIXED ALGORITHM (All fixes applied)")
    print("=" * 100)

    optimizer_fixed = HScoreOptimizerFinal(setup_params, scoring, omega=0.7, gamma=0.25)

    lebron_h_fixed, lebron_weights_fixed = optimizer_fixed.evaluate_player(
        "LeBron James", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    kat_h_fixed, kat_weights_fixed = optimizer_fixed.evaluate_player(
        "Karl-Anthony Towns", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    print(f"\nH-scores (fixed):")
    print(f"  LeBron: {lebron_h_fixed:.4f}")
    print(f"  KAT: {kat_h_fixed:.4f}")
    print(f"  Difference: {lebron_h_fixed - kat_h_fixed:+.4f}")
    print(f"  Winner: {'LeBron' if lebron_h_fixed > kat_h_fixed else 'KAT'}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\n{'Metric':<30} {'LeBron':<15} {'KAT':<15} {'Winner':<15}")
    print("-" * 75)
    print(f"{'G-score':<30} {lebron_g_total:>13.2f} {kat_g_total:>13.2f} {'LeBron' if lebron_g_total > kat_g_total else 'KAT':<15}")
    print(f"{'H-score (original)':<30} {lebron_h_original:>13.4f} {kat_h_original:>13.4f} {'LeBron' if lebron_h_original > kat_h_original else 'KAT':<15}")
    print(f"{'H-score (fixed)':<30} {lebron_h_fixed:>13.4f} {kat_h_fixed:>13.4f} {'LeBron' if lebron_h_fixed > kat_h_fixed else 'KAT':<15}")

    print("\n" + "=" * 100)
    print("KEY STATS COMPARISON")
    print("=" * 100)

    # Get raw stats
    lebron_data = league_data[league_data['PLAYER_NAME'] == 'LeBron James']
    kat_data = league_data[league_data['PLAYER_NAME'] == 'Karl-Anthony Towns']

    key_stats = ['DD', 'BLK', 'PTS', 'REB', 'AST']

    print(f"\n{'Stat':<12} {'LeBron (avg)':<15} {'KAT (avg)':<15} {'Difference':<15}")
    print("-" * 60)

    for stat in key_stats:
        lebron_avg = lebron_data[stat].mean()
        kat_avg = kat_data[stat].mean()
        diff = lebron_avg - kat_avg

        winner = "LeBron" if diff > 0 else "KAT"
        print(f"{stat:<12} {lebron_avg:>13.2f} {kat_avg:>13.2f} {diff:>+13.2f} ({winner})")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    if lebron_h_original > kat_h_original and lebron_h_fixed > kat_h_fixed:
        print("\nBoth algorithms prefer LeBron")
    elif lebron_h_original < kat_h_original and lebron_h_fixed < kat_h_fixed:
        print("\nBoth algorithms prefer KAT")
    else:
        print("\nAlgorithms DISAGREE!")
        if lebron_h_original > kat_h_original:
            print("  Original: LeBron")
            print("  Fixed: KAT")
        else:
            print("  Original: KAT")
            print("  Fixed: LeBron")


if __name__ == "__main__":
    main()
