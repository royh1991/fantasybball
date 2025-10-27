"""
Variance Calibration Debugger

Compares simulated distributions to actual Week 6 results to diagnose
variance underestimation.

Shows how many results fall within 1σ and 2σ of simulated mean.
Well-calibrated model should have:
- ~68% of results within 1σ
- ~95% of results within 2σ
- ~5% of results beyond 2σ
"""

import pandas as pd
import numpy as np
import ast
import sys
from pathlib import Path

sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel


def parse_box_score_stats(stat_str: str) -> dict:
    """Parse the stat_0 column."""
    try:
        stat_dict = ast.literal_eval(stat_str)
        return stat_dict.get('total', {})
    except:
        return {}


def get_actual_team_totals(box_scores: pd.DataFrame, matchup_name: str, team_side: str) -> dict:
    """Extract actual team totals from box scores."""
    matchup_data = box_scores[
        (box_scores['matchup'] == matchup_name) &
        (box_scores['team_side'] == team_side)
    ].copy()

    matchup_data['parsed_stats'] = matchup_data['stat_0'].apply(parse_box_score_stats)

    totals = {
        'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
        'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
    }

    # Map box score keys to our keys
    key_map = {
        'FGM': 'FGM', 'FGA': 'FGA',
        'FTM': 'FTM', 'FTA': 'FTA',
        'FG3M': '3PM', 'FG3A': '3PA',  # Box scores use 3PM/3PA
        'PTS': 'PTS', 'REB': 'REB',
        'AST': 'AST', 'STL': 'STL',
        'BLK': 'BLK', 'TOV': 'TO'  # Box scores use TO not TOV
    }

    for _, row in matchup_data.iterrows():
        stats = row['parsed_stats']
        for our_key, box_key in key_map.items():
            if box_key in stats:
                totals[our_key] += stats[box_key]

        # Calculate DD from individual stats
        pts = stats.get('PTS', 0)
        reb = stats.get('REB', 0)
        ast = stats.get('AST', 0)
        stl = stats.get('STL', 0)
        blk = stats.get('BLK', 0)
        dd_count = sum([pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10])
        totals['DD'] += 1 if dd_count >= 2 else 0

    return totals


def get_matchup_players_from_box_scores(box_scores: pd.DataFrame, matchup_name: str):
    """Extract players and game counts from box_scores."""
    matchup_data = box_scores[box_scores['matchup'] == matchup_name].copy()
    matchup_data['parsed_stats'] = matchup_data['stat_0'].apply(parse_box_score_stats)
    matchup_data['games_played'] = matchup_data['parsed_stats'].apply(lambda x: x.get('GP', 0))

    home_team_name = matchup_data[matchup_data['team_side'] == 'home']['team_name'].iloc[0]
    away_team_name = matchup_data[matchup_data['team_side'] == 'away']['team_name'].iloc[0]

    home_players = {}
    for _, row in matchup_data[matchup_data['team_side'] == 'home'].iterrows():
        player_name = row['player_name']
        games = row['games_played']
        if games > 0:
            home_players[player_name] = int(games)

    away_players = {}
    for _, row in matchup_data[matchup_data['team_side'] == 'away'].iterrows():
        player_name = row['player_name']
        games = row['games_played']
        if games > 0:
            away_players[player_name] = int(games)

    return home_players, away_players, home_team_name, away_team_name


def simulate_matchup(home_players: dict, away_players: dict,
                    player_models: dict, mapping: pd.DataFrame,
                    n_simulations: int = 500):
    """Simulate matchup and return distributions."""
    all_model_names = set(player_models.keys())

    team_a_results = []
    team_b_results = []

    for sim_num in range(n_simulations):
        # Simulate Team A (home)
        team_a_totals = {
            'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
        }

        for player_name, n_games in home_players.items():
            # Try to find model
            nba_name = player_name
            match = mapping[mapping['espn_name'].str.lower() == player_name.lower()]
            if len(match) > 0:
                nba_name = match.iloc[0]['nba_api_name']

            if nba_name not in player_models:
                if player_name not in player_models:
                    continue
                nba_name = player_name

            model = player_models[nba_name]

            for _ in range(n_games):
                game = model.simulate_game()
                for stat in team_a_totals.keys():
                    team_a_totals[stat] += game.get(stat, 0)

        # Simulate Team B (away)
        team_b_totals = {
            'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
        }

        for player_name, n_games in away_players.items():
            nba_name = player_name
            match = mapping[mapping['espn_name'].str.lower() == player_name.lower()]
            if len(match) > 0:
                nba_name = match.iloc[0]['nba_api_name']

            if nba_name not in player_models:
                if player_name not in player_models:
                    continue
                nba_name = player_name

            model = player_models[nba_name]

            for _ in range(n_games):
                game = model.simulate_game()
                for stat in team_b_totals.keys():
                    team_b_totals[stat] += game.get(stat, 0)

        team_a_results.append(team_a_totals)
        team_b_results.append(team_b_totals)

    return team_a_results, team_b_results


def calculate_distributions(results_list):
    """Calculate mean and std for each stat."""
    stats = {}
    categories = ['FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A',
                  'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'DD']

    for cat in categories:
        values = [r[cat] for r in results_list]
        stats[cat] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    # Calculate percentage categories
    fg_pcts = [r['FGM'] / r['FGA'] if r['FGA'] > 0 else 0 for r in results_list]
    ft_pcts = [r['FTM'] / r['FTA'] if r['FTA'] > 0 else 0 for r in results_list]
    fg3_pcts = [r['FG3M'] / r['FG3A'] if r['FG3A'] > 0 else 0 for r in results_list]

    stats['FG%'] = {'mean': np.mean(fg_pcts), 'std': np.std(fg_pcts)}
    stats['FT%'] = {'mean': np.mean(ft_pcts), 'std': np.std(ft_pcts)}
    stats['3P%'] = {'mean': np.mean(fg3_pcts), 'std': np.std(fg3_pcts)}

    return stats


def main():
    print("="*80)
    print("VARIANCE CALIBRATION DEBUGGER")
    print("="*80)
    print("\nComparing simulated distributions to actual Week 6 results...")
    print()

    # Load data
    print("Loading data...")
    box_scores = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/box_scores_latest.csv')
    historical = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    mapping = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/mappings/player_mapping_latest.csv')
    espn_proj = pd.read_csv('/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv')

    week6_data = box_scores[box_scores['week'] == 6]
    matchups = week6_data['matchup'].unique()

    # Fit player models
    print("\nFitting player models...")
    player_models = {}
    unique_players = historical['PLAYER_NAME'].unique()

    for player_name in unique_players:
        model = FantasyProjectionModel(evolution_rate=0.5)
        success = model.fit_player(historical, player_name)
        if success:
            player_models[player_name] = model

    # Add ESPN projections as fallback
    espn_proj = espn_proj.rename(columns={'PLAYER': 'player_name'})
    for _, row in espn_proj.iterrows():
        espn_name = row['player_name']
        if espn_name not in player_models:
            model = FantasyProjectionModel(evolution_rate=0.5)
            success = model.fit_from_espn_projection(row)
            if success:
                player_models[espn_name] = model

    print(f"  Fitted {len(player_models)} player models")

    # Analyze each matchup
    all_z_scores = []
    all_comparisons = []

    for matchup_name in matchups:
        print(f"\n{'='*80}")
        print(f"{matchup_name}")
        print(f"{'='*80}")

        # Get actual results
        actual_home = get_actual_team_totals(week6_data, matchup_name, 'home')
        actual_away = get_actual_team_totals(week6_data, matchup_name, 'away')

        # Get players
        home_players, away_players, home_name, away_name = get_matchup_players_from_box_scores(
            week6_data, matchup_name
        )

        print(f"  {home_name}: {len(home_players)} players, {sum(home_players.values())} games")
        print(f"  {away_name}: {len(away_players)} players, {sum(away_players.values())} games")

        # Run simulations
        print(f"  Running 500 simulations...")
        home_sims, away_sims = simulate_matchup(
            home_players, away_players, player_models, mapping, n_simulations=500
        )

        # Calculate distributions
        home_stats = calculate_distributions(home_sims)
        away_stats = calculate_distributions(away_sims)

        # Compare categories (use actual stat names from distributions)
        display_categories = ['3PM', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD', 'FG%', 'FT%', '3P%']

        print(f"\n  {'Category':<8} | {'Team':<20} | {'Actual':<8} | {'Sim Mean±SD':<20} | {'Z-score':<10}")
        print(f"  {'-'*85}")

        for cat in display_categories:
            # Map display name to actual stat key
            if cat in ['FG%', 'FT%', '3P%']:
                sim_key = cat
                if cat == 'FG%':
                    actual_val_home = actual_home['FGM'] / actual_home['FGA'] if actual_home['FGA'] > 0 else 0
                    actual_val_away = actual_away['FGM'] / actual_away['FGA'] if actual_away['FGA'] > 0 else 0
                elif cat == 'FT%':
                    actual_val_home = actual_home['FTM'] / actual_home['FTA'] if actual_home['FTA'] > 0 else 0
                    actual_val_away = actual_away['FTM'] / actual_away['FTA'] if actual_away['FTA'] > 0 else 0
                elif cat == '3P%':
                    actual_val_home = actual_home['FG3M'] / actual_home['FG3A'] if actual_home['FG3A'] > 0 else 0
                    actual_val_away = actual_away['FG3M'] / actual_away['FG3A'] if actual_away['FG3A'] > 0 else 0
            else:
                # Map display categories to simulation keys
                if cat == '3PM':
                    sim_key = 'FG3M'
                    actual_val_home = actual_home['FG3M']
                    actual_val_away = actual_away['FG3M']
                elif cat == 'TO':
                    sim_key = 'TOV'
                    actual_val_home = actual_home['TOV']
                    actual_val_away = actual_away['TOV']
                else:
                    sim_key = cat
                    actual_val_home = actual_home[cat]
                    actual_val_away = actual_away[cat]

            sim_mean_home = home_stats[sim_key]['mean']
            sim_std_home = home_stats[sim_key]['std']
            sim_mean_away = away_stats[sim_key]['mean']
            sim_std_away = away_stats[sim_key]['std']

            z_home = (actual_val_home - sim_mean_home) / sim_std_home if sim_std_home > 0 else 0
            z_away = (actual_val_away - sim_mean_away) / sim_std_away if sim_std_away > 0 else 0

            abs_z_home = abs(z_home)
            abs_z_away = abs(z_away)

            flag_home = "🔴" if abs_z_home > 2 else ("🟡" if abs_z_home > 1 else "🟢")
            flag_away = "🔴" if abs_z_away > 2 else ("🟡" if abs_z_away > 1 else "🟢")

            all_z_scores.append(abs_z_home)
            all_z_scores.append(abs_z_away)

            all_comparisons.append({
                'matchup': matchup_name,
                'team': home_name,
                'category': cat,
                'actual': actual_val_home,
                'sim_mean': sim_mean_home,
                'sim_std': sim_std_home,
                'z_score': z_home,
                'abs_z': abs_z_home
            })

            all_comparisons.append({
                'matchup': matchup_name,
                'team': away_name,
                'category': cat,
                'actual': actual_val_away,
                'sim_mean': sim_mean_away,
                'sim_std': sim_std_away,
                'z_score': z_away,
                'abs_z': abs_z_away
            })

            if cat in ['FG%', 'FT%', '3P%']:
                print(f"  {cat:<8} | {home_name[:20]:<20} | {actual_val_home:.3f}    | "
                      f"{sim_mean_home:.3f} ± {sim_std_home:.3f}      | {z_home:+.2f}σ {flag_home}")
                print(f"           | {away_name[:20]:<20} | {actual_val_away:.3f}    | "
                      f"{sim_mean_away:.3f} ± {sim_std_away:.3f}      | {z_away:+.2f}σ {flag_away}")
            else:
                print(f"  {cat:<8} | {home_name[:20]:<20} | {actual_val_home:<8.0f} | "
                      f"{sim_mean_home:6.1f} ± {sim_std_home:5.1f}    | {z_home:+.2f}σ {flag_home}")
                print(f"           | {away_name[:20]:<20} | {actual_val_away:<8.0f} | "
                      f"{sim_mean_away:6.1f} ± {sim_std_away:5.1f}    | {z_away:+.2f}σ {flag_away}")

    # Overall statistics
    print(f"\n\n{'='*80}")
    print("CALIBRATION STATISTICS")
    print(f"{'='*80}")

    z_scores_array = np.array(all_z_scores)

    within_1sigma = (z_scores_array < 1.0).sum()
    within_2sigma = (z_scores_array < 2.0).sum()
    beyond_2sigma = (z_scores_array >= 2.0).sum()

    total = len(z_scores_array)

    print(f"\nTotal observations: {total}")
    print(f"\nDistribution of |z-scores|:")
    print(f"  Within 1σ:  {within_1sigma:3d} ({within_1sigma/total*100:5.1f}%)  [Expected: ~68%]")
    print(f"  Within 2σ:  {within_2sigma:3d} ({within_2sigma/total*100:5.1f}%)  [Expected: ~95%]")
    print(f"  Beyond 2σ:  {beyond_2sigma:3d} ({beyond_2sigma/total*100:5.1f}%)  [Expected: ~5%]")

    print(f"\nMean |z-score|: {z_scores_array.mean():.2f}σ")
    print(f"Median |z-score|: {np.median(z_scores_array):.2f}σ")
    print(f"Max |z-score|: {z_scores_array.max():.2f}σ")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")

    pct_within_2sigma = within_2sigma / total * 100
    pct_beyond_2sigma = beyond_2sigma / total * 100

    if pct_beyond_2sigma > 10:
        print(f"\n🔴 CRITICAL VARIANCE UNDERESTIMATION")
        print(f"   {pct_beyond_2sigma:.1f}% of results are >2σ away (expected ~5%)")
        print(f"   The model is significantly overconfident.")
        print(f"\n   RECOMMENDATION: Increase variance by ~{(z_scores_array.mean())**2:.1f}x")
    elif pct_beyond_2sigma > 7:
        print(f"\n🟡 MODERATE VARIANCE UNDERESTIMATION")
        print(f"   {pct_beyond_2sigma:.1f}% of results are >2σ away (expected ~5%)")
        print(f"   The model is somewhat overconfident.")
        print(f"\n   RECOMMENDATION: Increase variance by ~{(z_scores_array.mean())**2:.1f}x")
    else:
        print(f"\n🟢 GOOD CALIBRATION")
        print(f"   {pct_beyond_2sigma:.1f}% of results are >2σ away (close to expected 5%)")
        print(f"   Variance is reasonably well-calibrated.")

    # Save detailed results
    output_dir = Path('/Users/rhu/fantasybasketball2/fantasy_2026/debug_outputs')
    output_dir.mkdir(exist_ok=True)

    comparisons_df = pd.DataFrame(all_comparisons)
    comparisons_df.to_csv(output_dir / 'variance_calibration_detailed.csv', index=False)

    print(f"\n✅ Detailed results saved to: {output_dir}/variance_calibration_detailed.csv")


if __name__ == '__main__':
    main()
