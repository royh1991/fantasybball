"""
Investigate DD calculation - why is Sabonis DD X-score so high (7.50)
and KAT's so low (2.41) when KAT should be ~4th best?

Let's look at the raw data.
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

    print("=" * 80)
    print("DD CALCULATION INVESTIGATION")
    print("=" * 80)

    # Get top players by DD
    players_to_check = [
        "Nikola Jokic",
        "Domantas Sabonis",
        "Giannis Antetokounmpo",
        "Karl-Anthony Towns",
        "Anthony Davis",
        "Joel Embiid",
        "Luka Doncic",
        "LeBron James"
    ]

    print("\n" + "=" * 80)
    print("RAW DD DATA")
    print("=" * 80)

    print(f"\n{'Player':<25} {'DD/Game':<12} {'DD Variance':<15} {'X-score':<10} {'G-score':<10}")
    print("-" * 80)

    dd_data = []

    for player in players_to_check:
        try:
            # Get raw stats
            player_data = league_data[league_data['PLAYER_NAME'] == player]

            if len(player_data) == 0:
                print(f"{player:<25} NOT FOUND")
                continue

            # Calculate per-game DD average
            dd_per_game = player_data['DD'].mean()

            # Get variance
            if player in player_variances and 'DD' in player_variances[player]:
                dd_variance = player_variances[player]['DD']
            else:
                dd_variance = None

            # Get X-score and G-score
            x_score = scoring.calculate_x_score(player, 'DD')
            g_score = scoring.calculate_g_score(player, 'DD')

            dd_data.append({
                'player': player,
                'dd_per_game': dd_per_game,
                'dd_variance': dd_variance,
                'x_score': x_score,
                'g_score': g_score
            })

            var_str = f"{dd_variance:.2f}" if dd_variance is not None else "N/A"
            print(f"{player:<25} {dd_per_game:>10.2f} {var_str:>13} {x_score:>9.2f} {g_score:>9.2f}")

        except Exception as e:
            print(f"{player:<25} ERROR: {e}")

    # Sort by DD per game
    dd_data.sort(key=lambda x: x['dd_per_game'], reverse=True)

    print("\n" + "=" * 80)
    print("SORTED BY DD PER GAME")
    print("=" * 80)

    print(f"\n{'Rank':<6} {'Player':<25} {'DD/Game':<12} {'X-score':<10} {'Gap to #1':<12}")
    print("-" * 70)

    for i, data in enumerate(dd_data, 1):
        gap = dd_data[0]['dd_per_game'] - data['dd_per_game']
        marker = ""
        if data['player'] == 'Karl-Anthony Towns':
            marker = " ← KAT"
        elif data['player'] == 'Domantas Sabonis':
            marker = " ← Sabonis"

        print(f"{i:<6} {data['player']:<25} {data['dd_per_game']:>10.2f} {data['x_score']:>9.2f} {gap:>11.2f}{marker}")

    # Check X-score formula
    print("\n" + "=" * 80)
    print("X-SCORE FORMULA CHECK")
    print("=" * 80)

    # Get league stats
    league_stats = scoring._calculate_league_stats()
    dd_league_mean = league_stats['DD']['between_mean']
    dd_league_var_within = league_stats['DD']['within_variance']

    print(f"\nLeague DD mean: {dd_league_mean:.3f}")
    print(f"League DD within-variance: {dd_league_var_within:.3f}")
    print(f"League DD within-std: {np.sqrt(dd_league_var_within):.3f}")

    print("\nX-score formula: (player_mean - league_mean) / sqrt(player_within_variance)")

    print(f"\n{'Player':<25} {'DD Mean':<12} {'- League':<12} {'÷ sqrt(Var)':<15} {'= X-score':<12} {'Actual':<10}")
    print("-" * 95)

    for data in dd_data[:5]:
        player = data['player']
        player_mean = data['dd_per_game']
        player_var = data['dd_variance']

        if player_var is not None and player_var > 0:
            numerator = player_mean - dd_league_mean
            denominator = np.sqrt(player_var)
            calculated_x = numerator / denominator
        else:
            numerator = player_mean - dd_league_mean
            denominator = None
            calculated_x = None

        calc_str = f"{calculated_x:.2f}" if calculated_x is not None else "N/A"
        denom_str = f"{denominator:.2f}" if denominator is not None else "N/A"

        print(f"{player:<25} {player_mean:>10.2f} {numerator:>11.2f} {denom_str:>14} {calc_str:>11} {data['x_score']:>9.2f}")

    # Check for variance issues
    print("\n" + "=" * 80)
    print("VARIANCE ANALYSIS")
    print("=" * 80)

    sabonis_data = next((d for d in dd_data if d['player'] == 'Domantas Sabonis'), None)
    kat_data = next((d for d in dd_data if d['player'] == 'Karl-Anthony Towns'), None)

    if sabonis_data and kat_data:
        print(f"\nSabonis:")
        print(f"  DD per game: {sabonis_data['dd_per_game']:.2f}")
        print(f"  DD variance: {sabonis_data['dd_variance']:.2f}")
        print(f"  DD std dev: {np.sqrt(sabonis_data['dd_variance']):.2f}")
        print(f"  X-score: {sabonis_data['x_score']:.2f}")

        print(f"\nKAT:")
        print(f"  DD per game: {kat_data['dd_per_game']:.2f}")
        print(f"  DD variance: {kat_data['dd_variance']:.2f}")
        print(f"  DD std dev: {np.sqrt(kat_data['dd_variance']):.2f}")
        print(f"  X-score: {kat_data['x_score']:.2f}")

        print(f"\nWhy is Sabonis's X-score so much higher?")

        if sabonis_data['dd_variance'] < kat_data['dd_variance']:
            ratio = kat_data['dd_variance'] / sabonis_data['dd_variance']
            print(f"  Sabonis has LOWER variance (more consistent)")
            print(f"  KAT's variance is {ratio:.2f}x higher")
            print(f"  This penalizes KAT's X-score")

        dd_diff = sabonis_data['dd_per_game'] - kat_data['dd_per_game']
        print(f"\n  Raw DD difference: {dd_diff:.2f} DD/game")

        x_diff = sabonis_data['x_score'] - kat_data['x_score']
        print(f"  X-score difference: {x_diff:.2f}")
        print(f"  X-score amplifies raw difference by {x_diff / dd_diff:.1f}x due to variance")

    # Show game-by-game samples
    print("\n" + "=" * 80)
    print("GAME-BY-GAME CONSISTENCY CHECK")
    print("=" * 80)

    for player_name in ['Domantas Sabonis', 'Karl-Anthony Towns']:
        player_games = league_data[league_data['PLAYER_NAME'] == player_name]['DD']

        if len(player_games) > 0:
            print(f"\n{player_name}:")
            print(f"  Games with DD: {(player_games >= 1).sum()} / {len(player_games)}")
            print(f"  Mean: {player_games.mean():.2f}")
            print(f"  Std: {player_games.std():.2f}")
            print(f"  Variance: {player_games.var():.2f}")
            print(f"  Min: {player_games.min():.0f}, Max: {player_games.max():.0f}")

            # Show distribution
            print(f"  Distribution:")
            for dd_count in range(4):
                count = (player_games == dd_count).sum()
                pct = count / len(player_games) * 100
                print(f"    {dd_count} DDs: {count:>3} games ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
