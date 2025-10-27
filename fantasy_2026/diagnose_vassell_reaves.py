"""
Diagnostic: Why are Vassell and Reaves 3PM predictions so low?

Examines historical data to understand why the model predicts ~1 3PM/game
when their recent seasons show 2.5+ 3PM/game.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel, load_data, parse_date

def analyze_player(player_name, historical):
    """Analyze a player's historical 3PM data."""

    print(f"\n{'='*80}")
    print(f"{player_name}")
    print(f"{'='*80}\n")

    # Get player's data
    player_data = historical[historical['PLAYER_NAME'] == player_name].copy()

    if len(player_data) == 0:
        print(f"ERROR: No data found for {player_name}")
        return

    print(f"Total games in historical data: {len(player_data)}")

    # Parse dates
    player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
    player_data = player_data.sort_values('parsed_date')

    # Add season labels
    def get_season(date):
        if date >= datetime(2024, 10, 1):
            return '2024-25'
        elif date >= datetime(2023, 10, 1):
            return '2023-24'
        elif date >= datetime(2022, 10, 1):
            return '2022-23'
        elif date >= datetime(2021, 10, 1):
            return '2021-22'
        elif date >= datetime(2020, 12, 1):
            return '2020-21'
        elif date >= datetime(2019, 10, 1):
            return '2019-20'
        else:
            return 'Older'

    player_data['season'] = player_data['parsed_date'].apply(get_season)

    # Per-season averages
    print("\nPer-Season 3PM Averages:")
    print(f"{'Season':<12} {'Games':<8} {'3PM/G':<10} {'3PA/G':<10} {'3P%':<10}")
    print(f"{'-'*60}")

    for season in ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21', '2019-20']:
        season_data = player_data[player_data['season'] == season]
        if len(season_data) > 0:
            # Filter for games with 3PA > 0 for percentage
            season_with_attempts = season_data[season_data['FG3A'] > 0]

            avg_3pm = season_data['FG3M'].mean()
            avg_3pa = season_data['FG3A'].mean()

            if len(season_with_attempts) > 0:
                avg_3pct = season_with_attempts['FG3_PCT'].mean()
                pct_str = f"{avg_3pct:.3f}"
            else:
                pct_str = "N/A"

            print(f"{season:<12} {len(season_data):<8} {avg_3pm:<10.2f} {avg_3pa:<10.2f} {pct_str:<10}")

    # Recent 10 games (what the model uses with decay_factor=0.9)
    recent_10 = player_data.tail(10)
    print(f"\nMost Recent 10 Games (what model uses):")
    print(f"{'Date':<12} {'3PM':<6} {'3PA':<6} {'3P%':<8}")
    print(f"{'-'*40}")

    for _, game in recent_10.iterrows():
        date_str = game['parsed_date'].strftime('%Y-%m-%d')
        fg3m = game['FG3M']
        fg3a = game['FG3A']
        pct_str = f"{game['FG3_PCT']:.3f}" if fg3a > 0 else "0/0"
        print(f"{date_str:<12} {fg3m:<6.0f} {fg3a:<6.0f} {pct_str:<8}")

    # Calculate weighted average (with decay_factor=0.9)
    print(f"\nWeighted Average (decay_factor=0.9):")
    n = len(recent_10)
    weights = 0.9 ** np.arange(n-1, -1, -1)
    weights = weights / weights.sum()

    weighted_3pm = (recent_10['FG3M'].values * weights).sum()
    weighted_3pa = (recent_10['FG3A'].values * weights).sum()

    print(f"  3PM/game: {weighted_3pm:.2f}")
    print(f"  3PA/game: {weighted_3pa:.2f}")

    # Check what the model actually fitted
    print(f"\nFitting model to check predictions...")
    model = FantasyProjectionModel(evolution_rate=0.5)

    try:
        model.fit_player(player_data, player_name)

        # Check 3PM distribution
        if 'FG3M' in model.distributions:
            dist = model.distributions['FG3M']
            print(f"\nModel's FG3M distribution:")
            print(f"  Posterior mean: {dist['posterior_mean']:.2f}")
            print(f"  Posterior variance: {dist['posterior_var']:.4f}")
            print(f"  Observation variance: {dist['obs_var']:.4f}")

        # Check 3P%
        if 'FG3_PCT' in model.percentages:
            pct = model.percentages['FG3_PCT']
            print(f"\nModel's FG3_PCT distribution:")
            print(f"  Posterior mean: {pct['posterior_mean']:.3f}")
            print(f"  Posterior variance: {pct['posterior_var']:.6f}")
            print(f"  Observation variance: {pct['obs_var']:.6f}")

        # Check 3PA distribution
        if 'FG3A' in model.distributions:
            dist = model.distributions['FG3A']
            print(f"\nModel's FG3A distribution:")
            print(f"  Posterior mean: {dist['posterior_mean']:.2f}")
            print(f"  Posterior variance: {dist['posterior_var']:.4f}")

        # Simulate 1000 games to see actual output
        sim_3pm = []
        sim_3pa = []
        for _ in range(1000):
            game = model.simulate_game()
            sim_3pm.append(game.get('FG3M', 0))
            sim_3pa.append(game.get('FG3A', 0))

        print(f"\nSimulated game distribution (1000 sims):")
        print(f"  3PM: {np.mean(sim_3pm):.2f} ± {np.std(sim_3pm):.2f}")
        print(f"  3PA: {np.mean(sim_3pa):.2f} ± {np.std(sim_3pa):.2f}")

        # Check if FG3A is being underestimated
        print(f"\n{'='*80}")
        print(f"DIAGNOSIS:")
        print(f"{'='*80}")

        actual_recent = recent_10['FG3M'].mean()
        predicted = np.mean(sim_3pm)

        print(f"Recent 10 games avg: {actual_recent:.2f} 3PM/game")
        print(f"Model prediction:    {predicted:.2f} 3PM/game")
        print(f"Gap:                 {actual_recent - predicted:.2f} 3PM/game ({(actual_recent - predicted)/actual_recent*100:.1f}%)")

        if np.mean(sim_3pa) < recent_10['FG3A'].mean() * 0.8:
            print(f"\n⚠️  3PA UNDERESTIMATED!")
            print(f"   Recent avg: {recent_10['FG3A'].mean():.2f} attempts")
            print(f"   Model:      {np.mean(sim_3pa):.2f} attempts")

        if np.mean(sim_3pm) / max(np.mean(sim_3pa), 1) < recent_10[recent_10['FG3A'] > 0]['FG3_PCT'].mean() * 0.8:
            print(f"\n⚠️  3P% UNDERESTIMATED!")
            print(f"   Recent avg: {recent_10[recent_10['FG3A'] > 0]['FG3_PCT'].mean():.3f}")
            print(f"   Model:      {np.mean(sim_3pm) / max(np.mean(sim_3pa), 1):.3f}")

    except Exception as e:
        print(f"ERROR fitting model: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("="*80)
    print("VASSELL & REAVES 3PM DIAGNOSTIC")
    print("="*80)

    # Load data
    print("\nLoading data...")
    roster, matchups, historical, mapping, espn_projections = load_data()

    print(f"Historical data: {len(historical)} games")
    print(f"Date range: {historical['GAME_DATE'].min()} to {historical['GAME_DATE'].max()}")

    # Check if we have these players
    print(f"\nSearching for players...")
    vassell_games = len(historical[historical['PLAYER_NAME'] == 'Devin Vassell'])
    reaves_games = len(historical[historical['PLAYER_NAME'] == 'Austin Reaves'])

    print(f"  Devin Vassell: {vassell_games} games")
    print(f"  Austin Reaves: {reaves_games} games")

    if vassell_games == 0:
        print("\nChecking alternative spellings for Vassell...")
        vassell_matches = historical[historical['PLAYER_NAME'].str.contains('Vassell', case=False, na=False)]['PLAYER_NAME'].unique()
        print(f"  Found: {vassell_matches}")

    if reaves_games == 0:
        print("\nChecking alternative spellings for Reaves...")
        reaves_matches = historical[historical['PLAYER_NAME'].str.contains('Reaves', case=False, na=False)]['PLAYER_NAME'].unique()
        print(f"  Found: {reaves_matches}")

    # Analyze each player
    analyze_player('Devin Vassell', historical)
    analyze_player('Austin Reaves', historical)

if __name__ == '__main__':
    main()
