"""
Diagnostic: Investigate FT% systematic bias

This script examines why simulated FT% is 10-15 percentage points too low.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel, load_data, fit_player_models

def main():
    print("="*80)
    print("FT% DIAGNOSTIC")
    print("="*80)

    # Load data and fit models
    roster, matchups, historical, mapping, espn_projections = load_data()
    player_models = fit_player_models(roster, historical, mapping, espn_projections)

    print(f"\nAnalyzing {len(player_models)} player models...")

    # Extract FT% data
    ft_pct_data = []

    for player_name, model in player_models.items():
        if 'FT_PCT' in model.percentages:
            ft_info = model.percentages['FT_PCT']
            ft_pct_data.append({
                'player': player_name,
                'posterior_mean': ft_info['posterior_mean'],
                'obs_var': ft_info['obs_var'],
                'posterior_var': ft_info['posterior_var'],
                'total_var': ft_info['posterior_var'] + 0.8 * ft_info['obs_var']
            })

    df = pd.DataFrame(ft_pct_data)

    print("\n" + "="*80)
    print("FT% MODEL STATISTICS")
    print("="*80)
    print(f"\nNumber of players with FT% models: {len(df)}")
    print(f"\nPosterior Mean FT%:")
    print(f"  Mean: {df['posterior_mean'].mean():.3f} ({df['posterior_mean'].mean()*100:.1f}%)")
    print(f"  Median: {df['posterior_mean'].median():.3f} ({df['posterior_mean'].median()*100:.1f}%)")
    print(f"  Min: {df['posterior_mean'].min():.3f} ({df['posterior_mean'].min()*100:.1f}%)")
    print(f"  Max: {df['posterior_mean'].max():.3f} ({df['posterior_mean'].max()*100:.1f}%)")
    print(f"  Std: {df['posterior_mean'].std():.3f}")

    print(f"\nObservation Variance:")
    print(f"  Mean: {df['obs_var'].mean():.4f}")
    print(f"  Median: {df['obs_var'].median():.4f}")

    print(f"\nTotal Variance (posterior_var + 0.8*obs_var):")
    print(f"  Mean: {df['total_var'].mean():.4f}")
    print(f"  Median: {df['total_var'].median():.4f}")

    # Show top FT shooters
    print("\n" + "="*80)
    print("TOP 10 FT SHOOTERS (by model)")
    print("="*80)
    top_ft = df.nlargest(10, 'posterior_mean')
    for idx, row in top_ft.iterrows():
        print(f"  {row['player']:30s} {row['posterior_mean']:.3f} ({row['posterior_mean']*100:.1f}%)")

    # Show bottom FT shooters
    print("\n" + "="*80)
    print("BOTTOM 10 FT SHOOTERS (by model)")
    print("="*80)
    bottom_ft = df.nsmallest(10, 'posterior_mean')
    for idx, row in bottom_ft.iterrows():
        print(f"  {row['player']:30s} {row['posterior_mean']:.3f} ({row['posterior_mean']*100:.1f}%)")

    # Simulate weighted average (how actual simulations work)
    print("\n" + "="*80)
    print("SIMULATED TEAM FT% (weighted by FTA)")
    print("="*80)

    # Get FTA data
    fta_data = []
    for player_name, model in player_models.items():
        if 'FTA' in model.distributions:
            fta_data.append({
                'player': player_name,
                'fta_mean': model.distributions['FTA']['posterior_mean']
            })

    fta_df = pd.DataFrame(fta_data)
    combined = df.merge(fta_df, on='player')

    # Calculate weighted average FT%
    combined['weighted_ft_pct'] = combined['posterior_mean'] * combined['fta_mean']

    total_fta = combined['fta_mean'].sum()
    total_weighted_ft = combined['weighted_ft_pct'].sum()

    if total_fta > 0:
        avg_team_ft_pct = total_weighted_ft / total_fta
        print(f"\nWeighted average FT% across all players: {avg_team_ft_pct:.3f} ({avg_team_ft_pct*100:.1f}%)")
        print(f"  (weighted by FTA per game)")

    # Also show simple average for comparison
    simple_avg = df['posterior_mean'].mean()
    print(f"\nSimple average FT%: {simple_avg:.3f} ({simple_avg*100:.1f}%)")
    print(f"\nDifference (weighted - simple): {(avg_team_ft_pct - simple_avg)*100:.1f} percentage points")

    # Now check actual Week 6 FT%
    print("\n" + "="*80)
    print("ACTUAL WEEK 6 FT%")
    print("="*80)

    # Load box scores
    box_scores = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/box_scores_latest.csv')
    week6 = box_scores[box_scores['week'] == 6]

    # Parse stat_0 to get FTM/FTA
    import ast

    team_ft_data = []

    for team_id in week6['team_id'].unique():
        team_data = week6[week6['team_id'] == team_id]
        team_name = team_data['team_name'].iloc[0]

        total_ftm = 0
        total_fta = 0

        for _, row in team_data.iterrows():
            try:
                stats = ast.literal_eval(row['stat_0'])
                ftm = stats.get('total', {}).get('FTM', 0)
                fta = stats.get('total', {}).get('FTA', 0)
                total_ftm += ftm
                total_fta += fta
            except:
                continue

        if total_fta > 0:
            ft_pct = total_ftm / total_fta
            team_ft_data.append({
                'team': team_name,
                'FTM': total_ftm,
                'FTA': total_fta,
                'FT%': ft_pct
            })

    actual_df = pd.DataFrame(team_ft_data)
    print(f"\nActual Week 6 team FT%:")
    for _, row in actual_df.iterrows():
        print(f"  {row['team']:30s} {row['FT%']:.3f} ({row['FT%']*100:.1f}%)  [{row['FTM']}/{row['FTA']}]")

    print(f"\nMean actual FT%: {actual_df['FT%'].mean():.3f} ({actual_df['FT%'].mean()*100:.1f}%)")
    print(f"Median actual FT%: {actual_df['FT%'].median():.3f} ({actual_df['FT%'].median()*100:.1f}%)")

    # Gap analysis
    print("\n" + "="*80)
    print("GAP ANALYSIS")
    print("="*80)

    gap = actual_df['FT%'].mean() - avg_team_ft_pct
    print(f"\nActual mean FT%: {actual_df['FT%'].mean():.3f} ({actual_df['FT%'].mean()*100:.1f}%)")
    print(f"Simulated mean FT%: {avg_team_ft_pct:.3f} ({avg_team_ft_pct*100:.1f}%)")
    print(f"GAP: {gap:.3f} ({gap*100:.1f} percentage points)")
    print(f"\nThis explains why we're seeing 1.5-2.7σ deviations in FT%!")

    # Check if this is historical data vs current season
    print("\n" + "="*80)
    print("HISTORICAL FT% CHECK")
    print("="*80)

    # Get all FT% from historical data
    hist_ft_pct = historical[historical['FT_PCT'].notna() &
                             (historical['FT_PCT'] > 0) &
                             (historical['FT_PCT'] <= 1)]['FT_PCT']

    print(f"\nHistorical FT% (all players, all games):")
    print(f"  Mean: {hist_ft_pct.mean():.3f} ({hist_ft_pct.mean()*100:.1f}%)")
    print(f"  Median: {hist_ft_pct.median():.3f} ({hist_ft_pct.median()*100:.1f}%)")

    # Check 2024-25 season specifically
    from datetime import datetime
    from weekly_projection_system import parse_date

    historical['parsed_date'] = historical['GAME_DATE'].apply(parse_date)
    season_2024 = historical[historical['parsed_date'] >= datetime(2024, 10, 1)]

    if len(season_2024) > 0:
        season_ft_pct = season_2024[season_2024['FT_PCT'].notna() &
                                    (season_2024['FT_PCT'] > 0) &
                                    (season_2024['FT_PCT'] <= 1)]['FT_PCT']

        print(f"\n2024-25 Season FT% (historical data):")
        print(f"  Mean: {season_ft_pct.mean():.3f} ({season_ft_pct.mean()*100:.1f}%)")
        print(f"  Median: {season_ft_pct.median():.3f} ({season_ft_pct.median()*100:.1f}%)")
        print(f"  N games: {len(season_ft_pct)}")

if __name__ == '__main__':
    main()
