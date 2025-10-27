"""
Simple DD investigation - just look at the raw numbers.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring


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

    print("=" * 90)
    print("DD ANALYSIS")
    print("=" * 90)

    players_to_check = [
        "Domantas Sabonis",
        "Giannis Antetokounmpo",
        "Anthony Davis",
        "Joel Embiid",
        "Karl-Anthony Towns",
        "Nikola Jokic",
        "LeBron James"
    ]

    results = []

    for player in players_to_check:
        try:
            # Get player data
            player_data = league_data[league_data['PLAYER_NAME'] == player]
            
            if len(player_data) == 0:
                continue

            # Raw stats
            dd_per_game = player_data['DD'].mean()
            dd_std = player_data['DD'].std()
            dd_var = player_data['DD'].var()

            # Scores
            x_score = scoring.calculate_x_score(player, 'DD')
            g_score = scoring.calculate_g_score(player, 'DD')

            results.append({
                'player': player,
                'dd_mean': dd_per_game,
                'dd_std': dd_std,
                'dd_var': dd_var,
                'x_score': x_score,
                'g_score': g_score
            })

        except Exception as e:
            print(f"Error for {player}: {e}")

    # Sort by DD mean
    results.sort(key=lambda x: x['dd_mean'], reverse=True)

    print(f"\n{'Rank':<6} {'Player':<25} {'DD/Game':<10} {'Std Dev':<10} {'X-score':<10} {'G-score':<10}")
    print("-" * 85)

    for i, r in enumerate(results, 1):
        marker = ""
        if 'Towns' in r['player']:
            marker = " <- KAT (5th in raw DD!)"
        elif 'Sabonis' in r['player']:
            marker = " <- Sabonis"
            
        print(f"{i:<6} {r['player']:<25} {r['dd_mean']:>8.2f} {r['dd_std']:>9.2f} {r['x_score']:>9.2f} {r['g_score']:>9.2f}{marker}")

    print("\n" + "=" * 90)
    print("WHY THE GAP?")
    print("=" * 90)

    sabonis = next((r for r in results if 'Sabonis' in r['player']), None)
    kat = next((r for r in results if 'Towns' in r['player']), None)

    if sabonis and kat:
        print(f"\nSabonis:")
        print(f"  DD per game: {sabonis['dd_mean']:.3f}")
        print(f"  DD std dev: {sabonis['dd_std']:.3f}")
        print(f"  X-score: {sabonis['x_score']:.2f}")

        print(f"\nKAT:")
        print(f"  DD per game: {kat['dd_mean']:.3f}")
        print(f"  DD std dev: {kat['dd_std']:.3f}")
        print(f"  X-score: {kat['x_score']:.2f}")

        raw_diff = sabonis['dd_mean'] - kat['dd_mean']
        x_diff = sabonis['x_score'] - kat['x_score']
        
        print(f"\nRaw DD difference: {raw_diff:.3f} DD/game ({raw_diff/sabonis['dd_mean']*100:.1f}% more)")
        print(f"X-score difference: {x_diff:.2f}")
        
        if sabonis['dd_std'] < kat['dd_std']:
            print(f"\nSabonis is MORE CONSISTENT (lower std dev: {sabonis['dd_std']:.3f} vs {kat['dd_std']:.3f})")
            print(f"X-score formula divides by std dev, so consistency helps")
            print(f"Amplification factor: {x_diff / raw_diff:.1f}x")

    print("\n" + "=" * 90)
    print("LEAGUE CONTEXT")
    print("=" * 90)

    # Get all players' DD averages
    all_dd_means = []
    for player_name in league_data['PLAYER_NAME'].unique():
        player_data = league_data[league_data['PLAYER_NAME'] == player_name]
        if len(player_data) > 0:
            all_dd_means.append(player_data['DD'].mean())

    league_dd_mean = np.mean(all_dd_means)
    league_dd_std = np.std(all_dd_means)

    print(f"\nLeague average DD: {league_dd_mean:.3f}")
    print(f"League std dev (between players): {league_dd_std:.3f}")

    if sabonis and kat:
        print(f"\nSabonis is {(sabonis['dd_mean'] - league_dd_mean)/league_dd_std:.2f} std devs above league avg")
        print(f"KAT is {(kat['dd_mean'] - league_dd_mean)/league_dd_std:.2f} std devs above league avg")


if __name__ == "__main__":
    main()
