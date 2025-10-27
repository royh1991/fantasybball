"""
Diagnose Double-Double Rates

Compare historical DD rates vs simulated DD rates for key players
to understand why the model is underestimating.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel, parse_date
from datetime import datetime


def get_dd_from_game(pts, reb, ast, stl, blk):
    """Calculate if a game has a double-double."""
    dd_stats = [pts, reb, ast, stl, blk]
    return 1 if sum(s >= 10 for s in dd_stats) >= 2 else 0


def analyze_player_dd_rate(player_name, historical_data, n_simulations=1000):
    """Analyze a player's historical vs simulated DD rate."""

    # Get player data
    player_data = historical_data[historical_data['PLAYER_NAME'].str.lower() == player_name.lower()].copy()

    if len(player_data) == 0:
        return None

    # Parse dates
    player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
    player_data = player_data.dropna(subset=['parsed_date'])
    player_data = player_data.sort_values('parsed_date')

    # Get 2024-25 season data
    cutoff_date = datetime(2024, 10, 1)
    season_2024_25 = player_data[player_data['parsed_date'] >= cutoff_date]

    if len(season_2024_25) == 0:
        return None

    # Calculate historical DD rate
    season_2024_25['DD_actual'] = season_2024_25.apply(
        lambda row: get_dd_from_game(
            row.get('PTS', 0),
            row.get('REB', 0),
            row.get('AST', 0),
            row.get('STL', 0),
            row.get('BLK', 0)
        ),
        axis=1
    )

    historical_dd_rate = season_2024_25['DD_actual'].mean()
    historical_dd_count = season_2024_25['DD_actual'].sum()
    n_games = len(season_2024_25)

    # Get average stats for context
    avg_pts = season_2024_25['PTS'].mean()
    avg_reb = season_2024_25['REB'].mean()
    avg_ast = season_2024_25['AST'].mean()
    avg_stl = season_2024_25['STL'].mean()
    avg_blk = season_2024_25['BLK'].mean()

    # Fit model and simulate
    model = FantasyProjectionModel(evolution_rate=0.5)
    success = model.fit_player(historical_data, player_name)

    if not success:
        return None

    # Run simulations
    simulated_dds = []
    for _ in range(n_simulations):
        game = model.simulate_game()
        simulated_dds.append(game['DD'])

    simulated_dd_rate = np.mean(simulated_dds)

    return {
        'player_name': player_name,
        'n_games_2024_25': n_games,
        'historical_dd_count': historical_dd_count,
        'historical_dd_rate': historical_dd_rate,
        'simulated_dd_rate': simulated_dd_rate,
        'dd_rate_gap': historical_dd_rate - simulated_dd_rate,
        'dd_rate_ratio': historical_dd_rate / simulated_dd_rate if simulated_dd_rate > 0 else np.inf,
        'avg_pts': avg_pts,
        'avg_reb': avg_reb,
        'avg_ast': avg_ast,
        'avg_stl': avg_stl,
        'avg_blk': avg_blk,
        'stats_above_10': sum([avg_pts >= 10, avg_reb >= 10, avg_ast >= 10, avg_stl >= 10, avg_blk >= 10])
    }


def main():
    print("="*80)
    print("DOUBLE-DOUBLE DIAGNOSTIC")
    print("="*80)

    # Load historical data
    print("\nLoading historical data...")
    historical = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    print(f"  Loaded {len(historical)} game logs")

    # Key players to analyze (known DD producers)
    players_to_analyze = [
        'Nikola Jokić',
        'Domantas Sabonis',
        'Anthony Davis',
        'LeBron James',
        'Giannis Antetokounmpo',
        'Karl-Anthony Towns',
        'Bam Adebayo',
        'Pascal Siakam',
        'Jayson Tatum',
        'Julius Randle'
    ]

    results = []

    print("\nAnalyzing DD rates for key players...")
    for player_name in players_to_analyze:
        print(f"  Analyzing {player_name}...")
        result = analyze_player_dd_rate(player_name, historical, n_simulations=1000)
        if result:
            results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('dd_rate_gap', ascending=False)

    # Display results
    print("\n" + "="*80)
    print("RESULTS: Historical vs Simulated DD Rates")
    print("="*80)

    for _, row in results_df.iterrows():
        print(f"\n{row['player_name']}")
        print(f"  2024-25 Season: {row['n_games_2024_25']} games")
        print(f"  Avg Stats: {row['avg_pts']:.1f} PTS / {row['avg_reb']:.1f} REB / {row['avg_ast']:.1f} AST / {row['avg_stl']:.1f} STL / {row['avg_blk']:.1f} BLK")
        print(f"  Stats >= 10: {row['stats_above_10']}")
        print(f"  Historical DD: {row['historical_dd_count']:.0f} in {row['n_games_2024_25']} games ({row['historical_dd_rate']:.1%})")
        print(f"  Simulated DD:  {row['simulated_dd_rate']:.1%}")
        print(f"  GAP: {row['dd_rate_gap']:.1%} (ratio: {row['dd_rate_ratio']:.2f}x)")

        if row['dd_rate_gap'] > 0.1:
            print(f"  ⚠️  SEVERE UNDERESTIMATION")
        elif row['dd_rate_gap'] > 0.05:
            print(f"  ⚠️  MODERATE UNDERESTIMATION")
        else:
            print(f"  ✅  Good match")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Players analyzed: {len(results_df)}")
    print(f"Mean historical DD rate: {results_df['historical_dd_rate'].mean():.1%}")
    print(f"Mean simulated DD rate: {results_df['simulated_dd_rate'].mean():.1%}")
    print(f"Mean gap: {results_df['dd_rate_gap'].mean():.1%}")
    print(f"Mean ratio: {results_df['dd_rate_ratio'].mean():.2f}x")

    # Save results
    output_dir = Path('/Users/rhu/fantasybasketball2/fantasy_2026/debug_outputs')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'dd_diagnostic.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
