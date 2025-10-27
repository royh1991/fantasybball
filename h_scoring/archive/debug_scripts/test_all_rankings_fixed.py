"""
Test H-score rankings after fixing variance units.
Compare top 20 players by G-score, ADP, and H-score.
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

    # Create optimizer
    optimizer = HScoreOptimizerFinal(setup_params, scoring, omega=0.7, gamma=0.25)

    print("=" * 100)
    print("TOP PLAYERS AFTER FIXING VARIANCE UNITS")
    print("=" * 100)

    # Get top players by G-score
    g_rankings = scoring.rank_players_by_g_score(top_n=30)

    results = []

    my_team = []
    opponent_teams = [[]]

    for idx, row in g_rankings.iterrows():
        player_name = row['PLAYER_NAME']
        g_score = row['TOTAL']

        try:
            h_score, _ = optimizer.evaluate_player(
                player_name, my_team, opponent_teams, picks_made=0, total_picks=13
            )

            # Get key stats
            dd_x = scoring.calculate_x_score(player_name, 'DD')
            blk_x = scoring.calculate_x_score(player_name, 'BLK')

            results.append({
                'player': player_name,
                'g_score': g_score,
                'h_score': h_score,
                'dd_x': dd_x,
                'blk_x': blk_x
            })
        except Exception as e:
            print(f"Error for {player_name}: {e}")

    # Sort by H-score
    results.sort(key=lambda x: x['h_score'], reverse=True)

    print(f"\n{'H-Rank':<8} {'Player':<30} {'H-score':<10} {'G-score':<10} {'G-Rank':<8} {'DD':<8} {'BLK':<8}")
    print("-" * 100)

    for i, r in enumerate(results[:20], 1):
        # Find G-rank
        g_rank = next((j for j, row in enumerate(g_rankings.iterrows(), 1)
                      if row[1]['PLAYER_NAME'] == r['player']), "?")

        marker = ""
        if 'Towns' in r['player']:
            marker = " <- KAT"
        elif 'Sabonis' in r['player']:
            marker = " <- Sabonis"
        elif 'Durant' in r['player']:
            marker = " <- KD"

        print(f"{i:<8} {r['player']:<30} {r['h_score']:>8.4f} {r['g_score']:>9.2f} {g_rank:<8} {r['dd_x']:>7.2f} {r['blk_x']:>7.2f}{marker}")

    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    sabonis_rank = next((i for i, r in enumerate(results, 1) if 'Sabonis' in r['player']), None)
    kat_rank = next((i for i, r in enumerate(results, 1) if 'Towns' in r['player']), None)
    kd_rank = next((i for i, r in enumerate(results, 1) if 'Durant' in r['player']), None)

    print(f"\nH-score Rankings:")
    if sabonis_rank:
        print(f"  Sabonis: #{sabonis_rank}")
    if kat_rank:
        print(f"  KAT: #{kat_rank}")
    if kd_rank:
        print(f"  KD: #{kd_rank}")

    sabonis_data = next((r for r in results if 'Sabonis' in r['player']), None)
    kat_data = next((r for r in results if 'Towns' in r['player']), None)

    if sabonis_data and kat_data:
        print(f"\nSabonis vs KAT:")
        print(f"  Sabonis G-score: {sabonis_data['g_score']:.2f} (higher by {sabonis_data['g_score'] - kat_data['g_score']:.2f})")
        print(f"  Sabonis H-score: {sabonis_data['h_score']:.4f}")
        print(f"  KAT H-score: {kat_data['h_score']:.4f}")
        h_diff = sabonis_data['h_score'] - kat_data['h_score']
        print(f"  Difference: {h_diff:.4f} ({'Sabonis ahead' if h_diff > 0 else 'KAT ahead'})")


if __name__ == "__main__":
    main()
