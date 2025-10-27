"""
Deep Dive Matchup Debugger

Analyzes a specific matchup to understand why win probabilities are so confident.
Creates detailed logs and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import json
import sys
sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import (
    load_data, fit_player_models, simulate_team_week,
    calculate_category_winner, FantasyProjectionModel
)


def analyze_roster_strength(roster_names: List[str], player_models: Dict,
                           team_name: str) -> pd.DataFrame:
    """Analyze the strength of each player on a roster."""
    players_data = []

    for player_name in roster_names:
        if player_name not in player_models:
            players_data.append({
                'player': player_name,
                'has_model': False,
                'fg_pct': None,
                'fga': None,
                'pts': None,
                'reb': None,
                'ast': None
            })
            continue

        model = player_models[player_name]

        player_info = {
            'player': player_name,
            'has_model': True
        }

        # Get shooting percentages
        if 'FG_PCT' in model.percentages:
            player_info['fg_pct'] = model.percentages['FG_PCT']['posterior_mean']
        else:
            player_info['fg_pct'] = None

        # Get key stats
        for stat in ['FGA', 'PTS', 'REB', 'AST']:
            if stat in model.distributions:
                player_info[stat.lower()] = model.distributions[stat]['posterior_mean']
            else:
                player_info[stat.lower()] = None

        players_data.append(player_info)

    df = pd.DataFrame(players_data)
    df['team'] = team_name
    return df


def simulate_with_logging(team_a_roster: List[str], team_b_roster: List[str],
                          player_models: Dict, n_simulations: int = 500):
    """Simulate matchup and log everything."""

    all_simulations = []
    category_results_log = []

    for sim_num in range(n_simulations):
        # Simulate weekly stats
        team_a_stats = simulate_team_week(team_a_roster, player_models, n_games=3)
        team_b_stats = simulate_team_week(team_b_roster, player_models, n_games=3)

        # Compare categories
        category_results = calculate_category_winner(team_a_stats, team_b_stats)

        # Count wins
        a_cats = sum(1 for v in category_results.values() if v == 'A')
        b_cats = sum(1 for v in category_results.values() if v == 'B')
        ties = sum(1 for v in category_results.values() if v == 'TIE')

        winner = 'A' if a_cats > b_cats else ('B' if b_cats > a_cats else 'TIE')

        # Log this simulation
        sim_record = {
            'sim_num': sim_num,
            'team_a_cats': a_cats,
            'team_b_cats': b_cats,
            'ties': ties,
            'winner': winner,
            **{f'team_a_{k}': v for k, v in team_a_stats.items()},
            **{f'team_b_{k}': v for k, v in team_b_stats.items()}
        }

        all_simulations.append(sim_record)

        # Log category results
        for cat, result in category_results.items():
            category_results_log.append({
                'sim_num': sim_num,
                'category': cat,
                'winner': result,
                'team_a_value': None,  # Will calculate below
                'team_b_value': None
            })

    return pd.DataFrame(all_simulations), pd.DataFrame(category_results_log)


def create_diagnostic_visualizations(simulations_df: pd.DataFrame,
                                     team_a_name: str, team_b_name: str,
                                     output_dir: str):
    """Create comprehensive diagnostic visualizations."""

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    categories = ['FG%', 'FT%', '3P%', 'FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD']

    for idx, cat in enumerate(categories):
        ax = axes[idx]

        if cat in ['FG%', 'FT%', '3P%']:
            # Calculate percentages
            if cat == 'FG%':
                team_a_vals = simulations_df['team_a_FGM'] / simulations_df['team_a_FGA']
                team_b_vals = simulations_df['team_b_FGM'] / simulations_df['team_b_FGA']
                xlabel = 'Field Goal %'
            elif cat == 'FT%':
                team_a_vals = simulations_df['team_a_FTM'] / simulations_df['team_a_FTA']
                team_b_vals = simulations_df['team_b_FTM'] / simulations_df['team_b_FTA']
                xlabel = 'Free Throw %'
            else:  # 3P%
                team_a_vals = simulations_df['team_a_FG3M'] / simulations_df['team_a_FG3A']
                team_b_vals = simulations_df['team_b_FG3M'] / simulations_df['team_b_FG3A']
                xlabel = '3-Point %'
        else:
            # Counting stats
            stat_map = {
                'FG3M': 'FG3M', 'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST',
                'STL': 'STL', 'BLK': 'BLK', 'TO': 'TOV', 'DD': 'DD'
            }
            stat_name = stat_map.get(cat, cat)
            team_a_vals = simulations_df[f'team_a_{stat_name}']
            team_b_vals = simulations_df[f'team_b_{stat_name}']
            xlabel = cat

        # Plot distributions
        ax.hist(team_a_vals, bins=20, alpha=0.5, color='blue', label=team_a_name, density=True)
        ax.hist(team_b_vals, bins=20, alpha=0.5, color='red', label=team_b_name, density=True)
        ax.axvline(team_a_vals.mean(), color='darkblue', linestyle='--', linewidth=2)
        ax.axvline(team_b_vals.mean(), color='darkred', linestyle='--', linewidth=2)

        # Calculate win rate
        if cat == 'TO':  # Lower is better for turnovers
            team_a_wins = (team_a_vals < team_b_vals).sum()
        else:
            team_a_wins = (team_a_vals > team_b_vals).sum()

        win_rate = team_a_wins / len(team_a_vals)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{cat}\n{team_a_name}: {win_rate:.1%} | {team_b_name}: {1-win_rate:.1%}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[-1].axis('off')

    plt.suptitle(f'Category-by-Category Analysis: {team_a_name} vs {team_b_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/category_distributions.png")


def create_matchup_summary(simulations_df: pd.DataFrame, roster_a_df: pd.DataFrame,
                          roster_b_df: pd.DataFrame, team_a_name: str,
                          team_b_name: str, output_dir: str):
    """Create summary report."""

    summary = {
        'matchup': f'{team_a_name} vs {team_b_name}',
        'n_simulations': len(simulations_df),
        'team_a_wins': (simulations_df['winner'] == 'A').sum(),
        'team_b_wins': (simulations_df['winner'] == 'B').sum(),
        'ties': (simulations_df['winner'] == 'TIE').sum(),
        'team_a_win_pct': (simulations_df['winner'] == 'A').mean(),
        'team_b_win_pct': (simulations_df['winner'] == 'B').mean(),

        # Roster info
        'team_a_roster_size': len(roster_a_df),
        'team_b_roster_size': len(roster_b_df),
        'team_a_modeled_players': roster_a_df['has_model'].sum(),
        'team_b_modeled_players': roster_b_df['has_model'].sum(),

        # Category wins distribution
        'team_a_avg_cats_won': simulations_df['team_a_cats'].mean(),
        'team_b_avg_cats_won': simulations_df['team_b_cats'].mean(),
        'team_a_cats_std': simulations_df['team_a_cats'].std(),
        'team_b_cats_std': simulations_df['team_b_cats'].std(),

        # Stat distributions
        'team_a_avg_pts': simulations_df['team_a_PTS'].mean(),
        'team_b_avg_pts': simulations_df['team_b_PTS'].mean(),
        'team_a_avg_reb': simulations_df['team_a_REB'].mean(),
        'team_b_avg_reb': simulations_df['team_b_REB'].mean(),
        'team_a_avg_ast': simulations_df['team_a_AST'].mean(),
        'team_b_avg_ast': simulations_df['team_b_AST'].mean(),
    }

    # Convert numpy types to Python types for JSON serialization
    summary_json = {}
    for key, value in summary.items():
        if isinstance(value, (np.integer, np.int64)):
            summary_json[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            summary_json[key] = float(value)
        else:
            summary_json[key] = value

    # Save as JSON
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved: {output_dir}/summary.json")

    return summary


def main():
    import os

    # Allow specifying matchup via command line or use default
    if len(sys.argv) > 1:
        target_team = sys.argv[1]
    else:
        target_team = 'Team Boricua Squad'

    print("="*80)
    print(f"MATCHUP DEBUGGER: {target_team}")
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

    print(f"\nAnalyzing: {team_a_name} (ID: {team_a_id}) vs {team_b_name} (ID: {team_b_id})")
    print(f"Output directory: {output_dir}")

    # Get rosters
    team_a_roster = roster[(roster['fantasy_team_id'] == team_a_id) &
                          (roster['currently_rostered'] == True)]['player_name'].tolist()
    team_b_roster = roster[(roster['fantasy_team_id'] == team_b_id) &
                          (roster['currently_rostered'] == True)]['player_name'].tolist()

    print(f"\n{team_a_name} roster: {len(team_a_roster)} players")
    print(f"{team_b_name} roster: {len(team_b_roster)} players")

    # Analyze rosters
    print("\nAnalyzing roster strengths...")
    roster_a_df = analyze_roster_strength(team_a_roster, player_models, team_a_name)
    roster_b_df = analyze_roster_strength(team_b_roster, player_models, team_b_name)

    # Save roster analysis
    combined_rosters = pd.concat([roster_a_df, roster_b_df])
    combined_rosters.to_csv(f'{output_dir}/roster_analysis.csv', index=False)
    print(f"Saved: {output_dir}/roster_analysis.csv")

    # Print roster summaries
    print(f"\n{team_a_name}:")
    print(f"  Modeled players: {roster_a_df['has_model'].sum()}/{len(roster_a_df)}")
    print(f"  Avg PTS: {roster_a_df['pts'].mean():.1f}")
    print(f"  Avg REB: {roster_a_df['reb'].mean():.1f}")
    print(f"  Avg AST: {roster_a_df['ast'].mean():.1f}")

    print(f"\n{team_b_name}:")
    print(f"  Modeled players: {roster_b_df['has_model'].sum()}/{len(roster_b_df)}")
    print(f"  Avg PTS: {roster_b_df['pts'].mean():.1f}")
    print(f"  Avg REB: {roster_b_df['reb'].mean():.1f}")
    print(f"  Avg AST: {roster_b_df['ast'].mean():.1f}")

    # Run simulations with logging
    print("\nRunning 500 simulations with detailed logging...")
    simulations_df, category_log_df = simulate_with_logging(
        team_a_roster, team_b_roster, player_models, n_simulations=500
    )

    # Save simulation logs
    simulations_df.to_csv(f'{output_dir}/all_simulations.csv', index=False)
    category_log_df.to_csv(f'{output_dir}/category_results.csv', index=False)
    print(f"Saved: {output_dir}/all_simulations.csv")
    print(f"Saved: {output_dir}/category_results.csv")

    # Create visualizations
    print("\nCreating diagnostic visualizations...")
    create_diagnostic_visualizations(simulations_df, team_a_name, team_b_name, output_dir)

    # Create summary
    print("\nGenerating summary...")
    summary = create_matchup_summary(simulations_df, roster_a_df, roster_b_df,
                                    team_a_name, team_b_name, output_dir)

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\nWin Probability:")
    print(f"  {team_a_name}: {summary['team_a_win_pct']:.1%}")
    print(f"  {team_b_name}: {summary['team_b_win_pct']:.1%}")

    print(f"\nCategory Performance:")
    print(f"  {team_a_name} avg cats won: {summary['team_a_avg_cats_won']:.2f} ± {summary['team_a_cats_std']:.2f}")
    print(f"  {team_b_name} avg cats won: {summary['team_b_avg_cats_won']:.2f} ± {summary['team_b_cats_std']:.2f}")

    print(f"\nRoster Composition:")
    print(f"  {team_a_name}: {summary['team_a_modeled_players']}/{summary['team_a_roster_size']} players modeled")
    print(f"  {team_b_name}: {summary['team_b_modeled_players']}/{summary['team_b_roster_size']} players modeled")

    # Identify dominant categories
    print(f"\nCategory Win Rates for {team_a_name}:")
    categories = ['FG%', 'FT%', '3P%', 'FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD']
    cat_wins = {}

    for cat in categories:
        if cat in ['FG%', 'FT%', '3P%']:
            if cat == 'FG%':
                team_a_vals = simulations_df['team_a_FGM'] / simulations_df['team_a_FGA']
                team_b_vals = simulations_df['team_b_FGM'] / simulations_df['team_b_FGA']
            elif cat == 'FT%':
                team_a_vals = simulations_df['team_a_FTM'] / simulations_df['team_a_FTA']
                team_b_vals = simulations_df['team_b_FTM'] / simulations_df['team_b_FTA']
            else:
                team_a_vals = simulations_df['team_a_FG3M'] / simulations_df['team_a_FG3A']
                team_b_vals = simulations_df['team_b_FG3M'] / simulations_df['team_b_FG3A']
        else:
            stat_map = {'FG3M': 'FG3M', 'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST',
                       'STL': 'STL', 'BLK': 'BLK', 'TO': 'TOV', 'DD': 'DD'}
            stat_name = stat_map.get(cat, cat)
            team_a_vals = simulations_df[f'team_a_{stat_name}']
            team_b_vals = simulations_df[f'team_b_{stat_name}']

        if cat == 'TO':
            win_rate = (team_a_vals < team_b_vals).mean()
        else:
            win_rate = (team_a_vals > team_b_vals).mean()

        cat_wins[cat] = win_rate
        print(f"  {cat}: {win_rate:.1%}")

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE! Check the debugging/ folder for detailed logs.")
    print("="*80)


if __name__ == "__main__":
    main()
