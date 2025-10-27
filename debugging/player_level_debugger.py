"""
Player-Level Simulation Debugger

Logs every individual player's simulated games across all 500 matchup simulations.
Shows detailed statistics for each player.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import (
    load_data, fit_player_models, FantasyProjectionModel
)


def simulate_player_games_with_logging(player_name: str, model: FantasyProjectionModel,
                                       n_simulations: int, n_games: int) -> pd.DataFrame:
    """Simulate all games for a player across all simulations."""
    all_games = []

    for sim_num in range(n_simulations):
        for game_num in range(n_games):
            game_stats = model.simulate_game()
            game_stats['sim_num'] = sim_num
            game_stats['game_num'] = game_num
            game_stats['player_name'] = player_name
            all_games.append(game_stats)

    return pd.DataFrame(all_games)


def simulate_team_week_with_player_logs(roster_names: List[str], player_models: Dict,
                                        n_simulations: int = 500, n_games: int = 3):
    """Simulate team weeks and log all individual player games."""

    all_player_games = []
    team_weekly_totals = []

    for sim_num in range(n_simulations):
        # Simulate this week
        weekly_games = []

        for player_name in roster_names:
            if player_name not in player_models:
                continue

            model = player_models[player_name]

            # Simulate 3 games for this player in this simulation
            for game_num in range(n_games):
                game_stats = model.simulate_game()
                game_stats['sim_num'] = sim_num
                game_stats['game_num'] = game_num
                game_stats['player_name'] = player_name
                weekly_games.append(game_stats)
                all_player_games.append(game_stats)

        # Aggregate weekly totals for this simulation
        weekly_df = pd.DataFrame(weekly_games)
        weekly_total = {
            'sim_num': sim_num,
            'FGM': weekly_df['FGM'].sum(),
            'FGA': weekly_df['FGA'].sum(),
            'FTM': weekly_df['FTM'].sum(),
            'FTA': weekly_df['FTA'].sum(),
            'FG3M': weekly_df['FG3M'].sum(),
            'FG3A': weekly_df['FG3A'].sum(),
            'PTS': weekly_df['PTS'].sum(),
            'REB': weekly_df['REB'].sum(),
            'AST': weekly_df['AST'].sum(),
            'STL': weekly_df['STL'].sum(),
            'BLK': weekly_df['BLK'].sum(),
            'TOV': weekly_df['TOV'].sum(),
            'DD': weekly_df['DD'].sum()
        }
        team_weekly_totals.append(weekly_total)

    return pd.DataFrame(all_player_games), pd.DataFrame(team_weekly_totals)


def create_player_summary_stats(player_games_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for each player across all simulations."""

    stats_cols = ['FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'DD']

    summaries = []

    for player_name in player_games_df['player_name'].unique():
        player_data = player_games_df[player_games_df['player_name'] == player_name]

        summary = {'player_name': player_name}

        # Calculate per-game stats (averaged across all games)
        for stat in stats_cols:
            summary[f'{stat}_mean'] = player_data[stat].mean()
            summary[f'{stat}_std'] = player_data[stat].std()
            summary[f'{stat}_min'] = player_data[stat].min()
            summary[f'{stat}_max'] = player_data[stat].max()

        # Calculate shooting percentages
        summary['FG_PCT_mean'] = player_data['FGM'].sum() / player_data['FGA'].sum()
        summary['FT_PCT_mean'] = player_data['FTM'].sum() / player_data['FTA'].sum() if player_data['FTA'].sum() > 0 else 0
        summary['FG3_PCT_mean'] = player_data['FG3M'].sum() / player_data['FG3A'].sum() if player_data['FG3A'].sum() > 0 else 0

        # Calculate weekly averages (3 games)
        weekly_data = player_data.groupby('sim_num')[stats_cols].sum()
        for stat in stats_cols:
            summary[f'{stat}_weekly_mean'] = weekly_data[stat].mean()
            summary[f'{stat}_weekly_std'] = weekly_data[stat].std()

        summaries.append(summary)

    return pd.DataFrame(summaries)


def create_player_comparison_viz(team_a_summary: pd.DataFrame, team_b_summary: pd.DataFrame,
                                 team_a_name: str, team_b_name: str, output_dir: str):
    """Create detailed player comparison visualizations."""

    # Sort by weekly points
    team_a_sorted = team_a_summary.sort_values('PTS_weekly_mean', ascending=True)
    team_b_sorted = team_b_summary.sort_values('PTS_weekly_mean', ascending=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    stat_configs = [
        ('PTS_weekly_mean', 'PTS_weekly_std', 'Points per Week', 0),
        ('REB_weekly_mean', 'REB_weekly_std', 'Rebounds per Week', 1),
        ('AST_weekly_mean', 'AST_weekly_std', 'Assists per Week', 2),
        ('FG3M_weekly_mean', 'FG3M_weekly_std', '3-Pointers per Week', 3),
        ('STL_weekly_mean', 'STL_weekly_std', 'Steals per Week', 4),
        ('BLK_weekly_mean', 'BLK_weekly_std', 'Blocks per Week', 5),
    ]

    for mean_col, std_col, title, idx in stat_configs:
        ax = axes[idx // 3, idx % 3]

        # Sort by this stat
        team_a_sorted = team_a_summary.sort_values(mean_col, ascending=True)
        team_b_sorted = team_b_summary.sort_values(mean_col, ascending=True)

        y_a = np.arange(len(team_a_sorted))
        y_b = np.arange(len(team_b_sorted)) + len(team_a_sorted) + 1

        # Plot Team A
        ax.barh(y_a, team_a_sorted[mean_col], xerr=team_a_sorted[std_col],
               color='blue', alpha=0.7, label=team_a_name)

        # Plot Team B
        ax.barh(y_b, team_b_sorted[mean_col], xerr=team_b_sorted[std_col],
               color='red', alpha=0.7, label=team_b_name)

        # Labels
        all_labels = list(team_a_sorted['player_name']) + [''] + list(team_b_sorted['player_name'])
        all_positions = list(y_a) + [len(team_a_sorted)] + list(y_b)

        ax.set_yticks(all_positions)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_xlabel(title, fontsize=10)
        ax.set_title(f'{title} (± 1 std dev)', fontsize=11, fontweight='bold')
        ax.axhline(y=len(team_a_sorted) + 0.5, color='black', linestyle='--', linewidth=2)
        ax.grid(axis='x', alpha=0.3)
        if idx == 0:
            ax.legend()

    plt.suptitle(f'Player-by-Player Weekly Averages: {team_a_name} vs {team_b_name}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/player_level_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/player_level_comparison.png")


def main():
    import os

    # Allow specifying matchup via command line or use default
    if len(sys.argv) > 1:
        target_team = sys.argv[1]
    else:
        target_team = 'Team Boricua Squad'

    print("="*80)
    print("PLAYER-LEVEL SIMULATION DEBUGGER")
    print("="*80)

    # Load data and fit models
    print("\nLoading data...")
    roster, matchups, historical, mapping, espn_projections = load_data()

    print("\nFitting player models...")
    player_models = fit_player_models(roster, historical, mapping, espn_projections)

    # Get the specific matchup
    matchup_row = matchups[
        (matchups['home_team_name'] == target_team) |
        (matchups['away_team_name'] == target_team)
    ]

    if len(matchup_row) == 0:
        print(f"\nERROR: Could not find matchup for '{target_team}'")
        print("\nAvailable teams:")
        for _, m in matchups.iterrows():
            print(f"  - {m['home_team_name']} vs {m['away_team_name']}")
        return

    matchup = matchup_row.iloc[0]
    team_a_id = matchup['home_team_id']
    team_b_id = matchup['away_team_id']
    team_a_name = matchup['home_team_name']
    team_b_name = matchup['away_team_name']

    # Create matchup-specific output directory
    safe_name = f"{team_a_name}_vs_{team_b_name}".replace(' ', '_').replace('.', '')
    output_dir = f'/Users/rhu/fantasybasketball2/debugging/{safe_name}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAnalyzing: {team_a_name} vs {team_b_name}")
    print(f"Output directory: {output_dir}")

    # Get rosters
    team_a_roster = roster[(roster['fantasy_team_id'] == team_a_id) &
                          (roster['currently_rostered'] == True)]['player_name'].tolist()
    team_b_roster = roster[(roster['fantasy_team_id'] == team_b_id) &
                          (roster['currently_rostered'] == True)]['player_name'].tolist()

    # Filter to modeled players only
    team_a_modeled = [p for p in team_a_roster if p in player_models]
    team_b_modeled = [p for p in team_b_roster if p in player_models]

    print(f"\n{team_a_name}: {len(team_a_modeled)} modeled players")
    print(f"{team_b_name}: {len(team_b_modeled)} modeled players")

    # Simulate all games for both teams
    print(f"\nSimulating 500 weeks (3 games/player per week)...")
    print("This will create detailed logs of every individual game...")

    print(f"\n{team_a_name}...")
    team_a_games, team_a_weekly = simulate_team_week_with_player_logs(
        team_a_modeled, player_models, n_simulations=500, n_games=3
    )

    print(f"{team_b_name}...")
    team_b_games, team_b_weekly = simulate_team_week_with_player_logs(
        team_b_modeled, player_models, n_simulations=500, n_games=3
    )

    # Save all individual games
    print("\nSaving individual game logs...")
    team_a_games.to_csv(f'{output_dir}/team_a_all_player_games.csv', index=False)
    team_b_games.to_csv(f'{output_dir}/team_b_all_player_games.csv', index=False)
    print(f"Saved: {output_dir}/team_a_all_player_games.csv ({len(team_a_games)} games)")
    print(f"Saved: {output_dir}/team_b_all_player_games.csv ({len(team_b_games)} games)")

    # Create player summary statistics
    print("\nCalculating player summary statistics...")
    team_a_summary = create_player_summary_stats(team_a_games)
    team_b_summary = create_player_summary_stats(team_b_games)

    team_a_summary['team'] = team_a_name
    team_b_summary['team'] = team_b_name

    combined_summary = pd.concat([team_a_summary, team_b_summary])
    combined_summary.to_csv(f'{output_dir}/player_summary_stats.csv', index=False)
    print(f"Saved: {output_dir}/player_summary_stats.csv")

    # Create visualizations
    print("\nCreating player comparison visualizations...")
    create_player_comparison_viz(team_a_summary, team_b_summary,
                                 team_a_name, team_b_name, output_dir)

    # Print detailed player stats
    print("\n" + "="*80)
    print(f"{team_a_name.upper()} - PLAYER STATISTICS (Average per Week)")
    print("="*80)
    print(f"{'Player':<25} {'PTS':>6} {'REB':>6} {'AST':>6} {'FG%':>6} {'FT%':>6} {'3PM':>6}")
    print("-"*80)

    for _, player in team_a_summary.sort_values('PTS_weekly_mean', ascending=False).iterrows():
        print(f"{player['player_name']:<25} "
              f"{player['PTS_weekly_mean']:>6.1f} "
              f"{player['REB_weekly_mean']:>6.1f} "
              f"{player['AST_weekly_mean']:>6.1f} "
              f"{player['FG_PCT_mean']:>6.3f} "
              f"{player['FT_PCT_mean']:>6.3f} "
              f"{player['FG3M_weekly_mean']:>6.1f}")

    print(f"\n{'TEAM TOTAL':<25} "
          f"{team_a_summary['PTS_weekly_mean'].sum():>6.1f} "
          f"{team_a_summary['REB_weekly_mean'].sum():>6.1f} "
          f"{team_a_summary['AST_weekly_mean'].sum():>6.1f} "
          f"{'':>6} {'':>6} "
          f"{team_a_summary['FG3M_weekly_mean'].sum():>6.1f}")

    print("\n" + "="*80)
    print(f"{team_b_name.upper()} - PLAYER STATISTICS (Average per Week)")
    print("="*80)
    print(f"{'Player':<25} {'PTS':>6} {'REB':>6} {'AST':>6} {'FG%':>6} {'FT%':>6} {'3PM':>6}")
    print("-"*80)

    for _, player in team_b_summary.sort_values('PTS_weekly_mean', ascending=False).iterrows():
        print(f"{player['player_name']:<25} "
              f"{player['PTS_weekly_mean']:>6.1f} "
              f"{player['REB_weekly_mean']:>6.1f} "
              f"{player['AST_weekly_mean']:>6.1f} "
              f"{player['FG_PCT_mean']:>6.3f} "
              f"{player['FT_PCT_mean']:>6.3f} "
              f"{player['FG3M_weekly_mean']:>6.1f}")

    print(f"\n{'TEAM TOTAL':<25} "
          f"{team_b_summary['PTS_weekly_mean'].sum():>6.1f} "
          f"{team_b_summary['REB_weekly_mean'].sum():>6.1f} "
          f"{team_b_summary['AST_weekly_mean'].sum():>6.1f} "
          f"{'':>6} {'':>6} "
          f"{team_b_summary['FG3M_weekly_mean'].sum():>6.1f}")

    # Print variance analysis
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS (Standard Deviation of Weekly Stats)")
    print("="*80)

    print(f"\n{team_a_name}:")
    print(f"  PTS std: {team_a_weekly['PTS'].std():.1f}")
    print(f"  REB std: {team_a_weekly['REB'].std():.1f}")
    print(f"  AST std: {team_a_weekly['AST'].std():.1f}")
    print(f"  3PM std: {team_a_weekly['FG3M'].std():.1f}")

    print(f"\n{team_b_name}:")
    print(f"  PTS std: {team_b_weekly['PTS'].std():.1f}")
    print(f"  REB std: {team_b_weekly['REB'].std():.1f}")
    print(f"  AST std: {team_b_weekly['AST'].std():.1f}")
    print(f"  3PM std: {team_b_weekly['FG3M'].std():.1f}")

    print("\n" + "="*80)
    print("PLAYER-LEVEL DEBUGGING COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - team_a_all_player_games.csv ({len(team_a_games):,} individual games)")
    print(f"  - team_b_all_player_games.csv ({len(team_b_games):,} individual games)")
    print(f"  - player_summary_stats.csv (aggregated stats for all players)")
    print(f"  - player_level_comparison.png (visual comparison)")


if __name__ == "__main__":
    main()
