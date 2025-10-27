"""
Fixed Matchup Simulation - Uses Correct Data Source

This script fixes the fundamental issue: using the correct Week 6 October 2025 data
instead of November 2024 data.

Data source: box_scores_latest.csv (has actual Week 6 game results)
Modeling: weekly_projection_system.py (known to work)
"""

import pandas as pd
import sys
import json
import ast
from pathlib import Path
from typing import Dict, Tuple

sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel


def parse_box_score_stats(stat_str: str) -> Dict:
    """Parse the stat_0 column which contains a dictionary."""
    try:
        stat_dict = ast.literal_eval(stat_str)
        return stat_dict.get('total', {})
    except:
        return {}


def load_data():
    """Load all required data."""
    print("Loading data...")

    # Load box scores (actual Week 6 game results)
    box_scores = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/box_scores_latest.csv')
    print(f"  Box scores: {len(box_scores)} records")

    # Load historical game logs for modeling
    historical = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    print(f"  Historical: {len(historical)} game logs")

    # Load player mapping
    mapping = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/mappings/player_mapping_latest.csv')
    mapping = mapping[['espn_name', 'nba_api_name']].drop_duplicates()
    mapping.columns = ['espn_name', 'nba_name']
    print(f"  Player mapping: {len(mapping)} players")

    # Load ESPN projections (fallback for players without historical data)
    espn_proj = pd.read_csv('/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv')
    espn_proj = espn_proj.rename(columns={'PLAYER': 'player_name'})
    print(f"  ESPN projections: {len(espn_proj)} players")

    return box_scores, historical, mapping, espn_proj


def fit_player_models(historical: pd.DataFrame, espn_proj: pd.DataFrame) -> Dict:
    """Fit models for all players in historical data."""
    print("\nFitting player models...")

    player_models = {}

    # Get unique players
    unique_players = historical['PLAYER_NAME'].unique()

    for player_name in unique_players:
        model = FantasyProjectionModel(evolution_rate=0.5)
        success = model.fit_player(historical, player_name)

        if success:
            player_models[player_name] = model

    # Add models from ESPN projections for players not in historical
    for _, row in espn_proj.iterrows():
        espn_name = row['player_name']
        if espn_name not in player_models:
            model = FantasyProjectionModel(evolution_rate=0.5)
            success = model.fit_from_espn_projection(row)
            if success:
                player_models[espn_name] = model

    print(f"  Fitted {len(player_models)} player models")
    return player_models


def get_matchup_players_from_box_scores(box_scores: pd.DataFrame, matchup_name: str) -> Tuple[Dict, Dict, str, str]:
    """Extract players and game counts from box_scores for a specific matchup."""
    matchup_data = box_scores[box_scores['matchup'] == matchup_name].copy()

    # Parse stats
    matchup_data['parsed_stats'] = matchup_data['stat_0'].apply(parse_box_score_stats)
    matchup_data['games_played'] = matchup_data['parsed_stats'].apply(lambda x: x.get('GP', 0))

    # Get team names
    home_team_name = matchup_data[matchup_data['team_side'] == 'home']['team_name'].iloc[0]
    away_team_name = matchup_data[matchup_data['team_side'] == 'away']['team_name'].iloc[0]

    # Get home team players
    home_players = {}
    for _, row in matchup_data[matchup_data['team_side'] == 'home'].iterrows():
        player_name = row['player_name']
        games = row['games_played']
        if games > 0:
            home_players[player_name] = int(games)

    # Get away team players
    away_players = {}
    for _, row in matchup_data[matchup_data['team_side'] == 'away'].iterrows():
        player_name = row['player_name']
        games = row['games_played']
        if games > 0:
            away_players[player_name] = int(games)

    return home_players, away_players, home_team_name, away_team_name


def map_player_name(espn_name: str, mapping: pd.DataFrame, all_model_names: set) -> str:
    """Map ESPN name to NBA API name."""
    # Try mapping file first
    match = mapping[mapping['espn_name'].str.lower() == espn_name.lower()]
    if len(match) > 0:
        return match.iloc[0]['nba_name']

    # Fallback: check if name exists directly in models
    if espn_name in all_model_names:
        return espn_name

    # Last resort: return original
    return espn_name


def simulate_matchup(home_players: Dict[str, int], away_players: Dict[str, int],
                    player_models: Dict, mapping: pd.DataFrame,
                    n_simulations: int = 500) -> Tuple[pd.DataFrame, Dict]:
    """Simulate matchup using player models."""

    results = []
    category_wins = {'A': {}, 'B': {}, 'TIE': {}}
    unmapped_players = set()
    all_model_names = set(player_models.keys())

    for sim_num in range(n_simulations):
        # Simulate Team A (home)
        team_a_totals = {
            'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
        }

        for player_name, n_games in home_players.items():
            # Map to NBA name
            nba_name = map_player_name(player_name, mapping, all_model_names)

            if nba_name not in player_models:
                unmapped_players.add(player_name)
                continue

            model = player_models[nba_name]

            # Simulate n_games for this player
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
            nba_name = map_player_name(player_name, mapping, all_model_names)

            if nba_name not in player_models:
                unmapped_players.add(player_name)
                continue

            model = player_models[nba_name]

            for _ in range(n_games):
                game = model.simulate_game()
                for stat in team_b_totals.keys():
                    team_b_totals[stat] += game.get(stat, 0)

        # Calculate category winners
        categories = calculate_category_winner(team_a_totals, team_b_totals)

        # Count wins
        a_cats = sum(1 for v in categories.values() if v == 'A')
        b_cats = sum(1 for v in categories.values() if v == 'B')
        ties = sum(1 for v in categories.values() if v == 'TIE')

        winner = 'A' if a_cats > b_cats else ('B' if b_cats > a_cats else 'TIE')

        # Track results
        results.append({
            'sim_num': sim_num,
            'team_a_cats': a_cats,
            'team_b_cats': b_cats,
            'ties': ties,
            'winner': winner,
            **{f'team_a_{k}': v for k, v in team_a_totals.items()},
            **{f'team_b_{k}': v for k, v in team_b_totals.items()}
        })

        # Track category wins
        for cat, win in categories.items():
            category_wins[win][cat] = category_wins[win].get(cat, 0) + 1

    if unmapped_players:
        print(f"    WARNING: {len(unmapped_players)} players could not be mapped")
        print(f"             {list(unmapped_players)[:10]}")

    return pd.DataFrame(results), category_wins


def calculate_category_winner(team_a: Dict, team_b: Dict) -> Dict[str, str]:
    """Determine which team wins each category."""
    categories = {}

    # Percentage categories
    for pct_stat, makes, attempts in [('FG%', 'FGM', 'FGA'), ('FT%', 'FTM', 'FTA'), ('3P%', 'FG3M', 'FG3A')]:
        a_pct = team_a[makes] / team_a[attempts] if team_a[attempts] > 0 else 0
        b_pct = team_b[makes] / team_b[attempts] if team_b[attempts] > 0 else 0

        if a_pct > b_pct:
            categories[pct_stat] = 'A'
        elif b_pct > a_pct:
            categories[pct_stat] = 'B'
        else:
            categories[pct_stat] = 'TIE'

    # Counting categories (higher is better)
    for stat in ['FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'DD']:
        if team_a[stat] > team_b[stat]:
            categories[stat] = 'A'
        elif team_b[stat] > team_a[stat]:
            categories[stat] = 'B'
        else:
            categories[stat] = 'TIE'

    # Turnovers (lower is better)
    if team_a['TOV'] < team_b['TOV']:
        categories['TO'] = 'A'
    elif team_b['TOV'] < team_a['TOV']:
        categories['TO'] = 'B'
    else:
        categories['TO'] = 'TIE'

    return categories


def main():
    """Main execution."""
    print("="*80)
    print("FIXED MATCHUP SIMULATION - USING CORRECT DATA")
    print("="*80)
    print("Data source: box_scores_latest.csv (Week 6, October 2025)")
    print("="*80)

    # Load data
    box_scores, historical, mapping, espn_proj = load_data()

    # Fit player models
    player_models = fit_player_models(historical, espn_proj)

    # Filter to Week 6
    week6_data = box_scores[box_scores['week'] == 6]
    matchups = week6_data['matchup'].unique()

    print(f"\nFound {len(matchups)} Week 6 matchups:")
    for m in matchups:
        print(f"  - {m}")

    # Create output directory
    output_dir = Path('/Users/rhu/fantasybasketball2/fantasy_2026/fixed_simulations')
    output_dir.mkdir(exist_ok=True)

    all_summaries = []

    # Simulate each matchup
    for matchup_name in matchups:
        print(f"\n{'-'*80}")
        print(f"{matchup_name}")
        print(f"{'-'*80}")

        # Get actual players who played
        home_players, away_players, home_name, away_name = get_matchup_players_from_box_scores(
            week6_data, matchup_name
        )

        print(f"  {home_name}: {len(home_players)} players, {sum(home_players.values())} games")
        print(f"  {away_name}: {len(away_players)} players, {sum(away_players.values())} games")

        if len(home_players) == 0 or len(away_players) == 0:
            print(f"  ⚠️  Skipping - no players found")
            continue

        # Run simulations
        print(f"  Simulating 500 matchups...")
        results_df, _ = simulate_matchup(
            home_players, away_players, player_models, mapping, n_simulations=500
        )

        # Calculate summary
        team_a_wins = int((results_df['winner'] == 'A').sum())
        team_b_wins = int((results_df['winner'] == 'B').sum())
        ties = int((results_df['winner'] == 'TIE').sum())

        summary = {
            'matchup': matchup_name,
            'team_a_name': home_name,
            'team_b_name': away_name,
            'n_simulations': len(results_df),
            'team_a_wins': team_a_wins,
            'team_b_wins': team_b_wins,
            'ties': ties,
            'team_a_win_pct': float(team_a_wins / len(results_df)),
            'team_b_win_pct': float(team_b_wins / len(results_df)),
            'team_a_avg_cats_won': float(results_df['team_a_cats'].mean()),
            'team_b_avg_cats_won': float(results_df['team_b_cats'].mean()),
            'team_a_players': len(home_players),
            'team_b_players': len(away_players),
            'team_a_total_games': sum(home_players.values()),
            'team_b_total_games': sum(away_players.values())
        }

        all_summaries.append(summary)

        # Save results
        safe_name = matchup_name.replace(' ', '_').replace('vs', 'vs')
        matchup_dir = output_dir / safe_name
        matchup_dir.mkdir(exist_ok=True)

        results_df.to_csv(matchup_dir / 'all_simulations.csv', index=False)
        with open(matchup_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✅ Results: {home_name} {summary['team_a_win_pct']:.1%} | {away_name} {summary['team_b_win_pct']:.1%}")

    # Save overall summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / 'all_matchups_summary.csv', index=False)

    print(f"\n{'='*80}")
    print(f"SIMULATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Simulated {len(all_summaries)} matchups")
    print(f"Output: {output_dir}/")

    # Now run comparison with actual results
    print(f"\n{' COMPARING WITH ACTUAL RESULTS ':-^80}")

    # Load analyze script and run comparison
    import subprocess
    subprocess.run([
        'python', 'analyze_actual_vs_simulated.py'
    ], cwd='/Users/rhu/fantasybasketball2/fantasy_2026')


if __name__ == '__main__':
    main()
