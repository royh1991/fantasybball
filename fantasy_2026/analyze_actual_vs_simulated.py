"""
Analyze actual Week 6 results vs simulated predictions

This script:
1. Parses box_scores_latest.csv to get actual game results
2. Compares with simulation results from matchup_simulations/
3. Identifies discrepancies and provides detailed analysis
"""

import pandas as pd
import json
import ast
from pathlib import Path
from typing import Dict, Tuple

def parse_stat_dict(stat_str: str) -> Dict:
    """Parse the stat_0 column which contains a dictionary as a string."""
    try:
        stat_dict = ast.literal_eval(stat_str)
        return stat_dict.get('total', {})
    except:
        return {}

def calculate_category_values(team_stats: Dict) -> Dict:
    """Calculate 11-category values from team totals."""
    fgm = team_stats.get('FGM', 0)
    fga = team_stats.get('FGA', 1)
    ftm = team_stats.get('FTM', 0)
    fta = team_stats.get('FTA', 1)
    fg3m = team_stats.get('3PM', 0)
    fg3a = team_stats.get('3PA', 1)

    # Calculate percentages
    fg_pct = fgm / fga if fga > 0 else 0
    ft_pct = ftm / fta if fta > 0 else 0
    fg3_pct = fg3m / fg3a if fg3a > 0 else 0

    return {
        'FG%': fg_pct,
        'FT%': ft_pct,
        '3P%': fg3_pct,
        '3PM': fg3m,
        'PTS': team_stats.get('PTS', 0),
        'REB': team_stats.get('REB', 0),
        'AST': team_stats.get('AST', 0),
        'STL': team_stats.get('STL', 0),
        'BLK': team_stats.get('BLK', 0),
        'TO': team_stats.get('TO', 0),
        'DD': team_stats.get('DD', 0)
    }

def compare_categories(team_a: Dict, team_b: Dict) -> Tuple[str, int, int, int]:
    """Compare two teams across all 11 categories."""
    categories_better_higher = ['FG%', 'FT%', '3P%', '3PM', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'DD']
    categories_better_lower = ['TO']

    team_a_wins = 0
    team_b_wins = 0
    ties = 0

    for cat in categories_better_higher:
        if team_a[cat] > team_b[cat]:
            team_a_wins += 1
        elif team_b[cat] > team_a[cat]:
            team_b_wins += 1
        else:
            ties += 1

    for cat in categories_better_lower:
        if team_a[cat] < team_b[cat]:
            team_a_wins += 1
        elif team_b[cat] < team_a[cat]:
            team_b_wins += 1
        else:
            ties += 1

    if team_a_wins > team_b_wins:
        winner = 'team_a'
    elif team_b_wins > team_a_wins:
        winner = 'team_b'
    else:
        winner = 'tie'

    return winner, team_a_wins, team_b_wins, ties

def analyze_matchup(box_scores_df: pd.DataFrame, matchup_name: str) -> Dict:
    """Analyze a specific matchup from box scores."""
    matchup_data = box_scores_df[box_scores_df['matchup'] == matchup_name].copy()

    if len(matchup_data) == 0:
        return None

    # Get team names
    teams = matchup_data[['team_id', 'team_name', 'team_side']].drop_duplicates()
    team_a = teams[teams['team_side'] == 'home'].iloc[0] if len(teams[teams['team_side'] == 'home']) > 0 else None
    team_b = teams[teams['team_side'] == 'away'].iloc[0] if len(teams[teams['team_side'] == 'away']) > 0 else None

    if team_a is None or team_b is None:
        return None

    # Parse stats for each player
    matchup_data['parsed_stats'] = matchup_data['stat_0'].apply(parse_stat_dict)

    # Aggregate team totals
    team_a_stats = {}
    team_b_stats = {}

    stat_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD', 'FGM', 'FGA', 'FTM', 'FTA', '3PM', '3PA']

    for stat in stat_columns:
        team_a_stats[stat] = matchup_data[matchup_data['team_side'] == 'home']['parsed_stats'].apply(
            lambda x: x.get(stat, 0)).sum()
        team_b_stats[stat] = matchup_data[matchup_data['team_side'] == 'away']['parsed_stats'].apply(
            lambda x: x.get(stat, 0)).sum()

    # Calculate category values
    team_a_cats = calculate_category_values(team_a_stats)
    team_b_cats = calculate_category_values(team_b_stats)

    # Compare categories
    winner, team_a_wins, team_b_wins, ties = compare_categories(team_a_cats, team_b_cats)

    return {
        'matchup': matchup_name,
        'team_a_name': team_a['team_name'],
        'team_b_name': team_b['team_name'],
        'team_a_categories': team_a_cats,
        'team_b_categories': team_b_cats,
        'team_a_wins': team_a_wins,
        'team_b_wins': team_b_wins,
        'ties': ties,
        'winner': winner,
        'score': f"{team_a_wins}-{team_b_wins}-{ties}",
        'team_a_players': len(matchup_data[matchup_data['team_side'] == 'home']),
        'team_b_players': len(matchup_data[matchup_data['team_side'] == 'away'])
    }

def load_simulation_results(matchup_folder: Path) -> Dict:
    """Load simulation results for a matchup."""
    summary_file = matchup_folder / 'summary.json'
    if not summary_file.exists():
        return None

    with open(summary_file, 'r') as f:
        return json.load(f)

def main():
    # Load actual results
    print("Loading actual Week 6 results from box_scores_latest.csv...")
    box_scores_path = Path('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/box_scores_latest.csv')
    box_scores_df = pd.read_csv(box_scores_path)

    # Filter to Week 6
    week6_df = box_scores_df[box_scores_df['week'] == 6].copy()

    # Get all unique matchups
    matchups = week6_df['matchup'].unique()
    print(f"\nFound {len(matchups)} matchups in Week 6:")
    for m in matchups:
        print(f"  - {m}")

    # Analyze each matchup
    results_comparison = []

    for matchup_name in matchups:
        print(f"\n{'='*80}")
        print(f"Analyzing: {matchup_name}")
        print('='*80)

        # Get actual results
        actual = analyze_matchup(week6_df, matchup_name)
        if actual is None:
            print("  ⚠️  Could not parse actual results")
            continue

        print(f"\n📊 ACTUAL RESULTS:")
        print(f"  {actual['team_a_name']}: {actual['team_a_wins']} categories")
        print(f"  {actual['team_b_name']}: {actual['team_b_wins']} categories")
        print(f"  Score: {actual['score']}")
        print(f"  Winner: {actual['team_a_name'] if actual['winner'] == 'team_a' else actual['team_b_name']}")

        print(f"\n  Category Breakdown:")
        print(f"  {'Category':<8} {actual['team_a_name']:<25} {actual['team_b_name']:<25} Winner")
        print(f"  {'-'*85}")
        for cat in ['FG%', 'FT%', '3P%', '3PM', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD']:
            a_val = actual['team_a_categories'][cat]
            b_val = actual['team_b_categories'][cat]

            if cat in ['FG%', 'FT%', '3P%']:
                a_str = f"{a_val:.3f}"
                b_str = f"{b_val:.3f}"
            else:
                a_str = f"{a_val:.0f}"
                b_str = f"{b_val:.0f}"

            if cat == 'TO':
                winner = actual['team_a_name'] if a_val < b_val else (actual['team_b_name'] if b_val < a_val else 'TIE')
            else:
                winner = actual['team_a_name'] if a_val > b_val else (actual['team_b_name'] if b_val > a_val else 'TIE')

            print(f"  {cat:<8} {a_str:<25} {b_str:<25} {winner}")

        # Try to find corresponding simulation results
        sim_folder_name = matchup_name.replace(' vs ', '_vs_').replace(' ', '_')
        sim_folder = Path('/Users/rhu/fantasybasketball2/fantasy_2026/fixed_simulations') / sim_folder_name

        if sim_folder.exists():
            sim_results = load_simulation_results(sim_folder)
            if sim_results:
                print(f"\n🎲 SIMULATED RESULTS:")
                print(f"  {actual['team_a_name']}: {sim_results['team_a_win_pct']*100:.1f}% win probability")
                print(f"  {actual['team_b_name']}: {sim_results['team_b_win_pct']*100:.1f}% win probability")
                print(f"  Avg categories won: {sim_results['team_a_avg_cats_won']:.2f} vs {sim_results['team_b_avg_cats_won']:.2f}")

                # Compare prediction vs actual
                predicted_winner = 'team_a' if sim_results['team_a_win_pct'] > sim_results['team_b_win_pct'] else 'team_b'
                actual_winner = actual['winner']

                if predicted_winner == actual_winner:
                    print(f"\n  ✅ CORRECT PREDICTION")
                else:
                    print(f"\n  ❌ INCORRECT PREDICTION")
                    print(f"     Predicted: {actual['team_a_name'] if predicted_winner == 'team_a' else actual['team_b_name']} ({sim_results['team_a_win_pct' if predicted_winner == 'team_a' else 'team_b_win_pct']*100:.1f}%)")
                    print(f"     Actual: {actual['team_a_name'] if actual_winner == 'team_a' else actual['team_b_name']}")

                results_comparison.append({
                    'matchup': matchup_name,
                    'actual_winner': actual['team_a_name'] if actual['winner'] == 'team_a' else actual['team_b_name'],
                    'predicted_winner': actual['team_a_name'] if predicted_winner == 'team_a' else actual['team_b_name'],
                    'correct': predicted_winner == actual_winner,
                    'actual_score': actual['score'],
                    'team_a_win_prob': sim_results['team_a_win_pct'],
                    'team_b_win_prob': sim_results['team_b_win_pct']
                })
        else:
            print(f"\n  ⚠️  No simulation results found at: {sim_folder}")

    # Summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print('='*80)

    if len(results_comparison) > 0:
        correct = sum(1 for r in results_comparison if r['correct'])
        total = len(results_comparison)
        accuracy = correct / total * 100

        print(f"\nPrediction Accuracy: {correct}/{total} ({accuracy:.1f}%)")

        print(f"\nDetailed Results:")
        for r in results_comparison:
            status = "✅" if r['correct'] else "❌"
            print(f"  {status} {r['matchup']}")
            print(f"     Predicted: {r['predicted_winner']}")
            print(f"     Actual: {r['actual_winner']} ({r['actual_score']})")
    else:
        print("No simulation results found for comparison")

if __name__ == '__main__':
    main()
