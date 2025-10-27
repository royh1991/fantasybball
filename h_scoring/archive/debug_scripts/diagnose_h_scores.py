"""
Diagnostic script to investigate H-scoring issues.

This script will:
1. Load player data and calculate G-scores
2. Show X-scores for key players
3. Trace through H-score calculation for specific players
4. Identify why Franz Wagner ranks higher than expected
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer import HScoreOptimizer


def load_data():
    """Load the latest data files."""
    data_dir = 'data'

    # Find latest files
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        raise FileNotFoundError("No data files found!")

    weekly_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print(f"Loading data from:")
    print(f"  - {weekly_file}")
    print(f"  - {variance_file}")

    league_data = pd.read_csv(weekly_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    return league_data, player_variances


def analyze_player_stats(league_data, player_variances, player_names):
    """Analyze stats for specific players."""
    print("\n" + "=" * 80)
    print("PLAYER STATISTICS ANALYSIS")
    print("=" * 80)

    for player_name in player_names:
        print(f"\n{player_name}:")
        print("-" * 40)

        # Get player data
        player_df = league_data[league_data['PLAYER_NAME'] == player_name]

        if player_df.empty:
            print(f"  No data found for {player_name}")
            continue

        # Calculate averages
        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'DD', 'FG_PCT', 'FT_PCT', 'FG3_PCT']

        print("  Season Averages (per week):")
        for cat in categories:
            if cat in player_df.columns:
                mean_val = player_df[cat].mean()
                if cat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                    print(f"    {cat:8s}: {mean_val:.3f}")
                else:
                    print(f"    {cat:8s}: {mean_val:.1f}")

        # Show variance data
        if player_name in player_variances:
            print("\n  Per-Game Variance:")
            for cat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']:
                if cat in player_variances[player_name]:
                    var_data = player_variances[player_name][cat]
                    std = var_data.get('std_per_game', 0)
                    mean = var_data.get('mean_per_game', 0)
                    cv = std / mean if mean > 0 else 0
                    print(f"    {cat:8s}: σ={std:.2f}, CV={cv:.2%}")


def compare_scores(league_data, player_variances, player_names):
    """Compare G-scores and X-scores for players."""
    print("\n" + "=" * 80)
    print("G-SCORES AND X-SCORES COMPARISON")
    print("=" * 80)

    # Initialize scoring system
    scoring = PlayerScoring(league_data, player_variances)

    # Calculate for each player
    results = []
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'DD', 'FG_PCT', 'FT_PCT', 'FG3_PCT']

    for player_name in player_names:
        g_scores = scoring.calculate_all_g_scores(player_name)

        # Calculate X-scores
        x_scores = {}
        for cat in categories:
            x_scores[cat] = scoring.calculate_x_score(player_name, cat)

        results.append({
            'Player': player_name,
            'G_Total': g_scores['TOTAL'],
            'X_Sum': sum(x_scores.values()),
            **{f'G_{cat}': g_scores.get(cat, 0) for cat in ['PTS', 'REB', 'AST', 'FG3M']},
            **{f'X_{cat}': x_scores.get(cat, 0) for cat in ['PTS', 'REB', 'AST', 'FG3M']}
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    return scoring


def trace_h_score_calculation(scoring, league_data, player_variances,
                             candidate_name, my_team=[], opponent_teams=[]):
    """Trace through H-score calculation for a specific player."""
    print("\n" + "=" * 80)
    print(f"H-SCORE CALCULATION TRACE: {candidate_name}")
    print("=" * 80)

    # Setup covariance calculator - fix the parameter order
    cov_calc = CovarianceCalculator(scoring, league_data['PLAYER_NAME'].unique().tolist())
    setup_params = cov_calc.calculate_all()

    # Initialize optimizer
    optimizer = HScoreOptimizer(setup_params, scoring)

    # Calculate X-scores for candidate
    categories = setup_params['categories']
    candidate_x = np.array([
        scoring.calculate_x_score(candidate_name, cat)
        for cat in categories
    ])

    print(f"\nCandidate X-scores:")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {candidate_x[i]:8.3f}")

    # Current team X-scores (empty for first pick)
    current_team_x = np.zeros(len(categories))
    if my_team:
        for player in my_team:
            player_x = np.array([
                scoring.calculate_x_score(player, cat)
                for cat in categories
            ])
            current_team_x += player_x

    print(f"\nCurrent team X-scores sum:")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {current_team_x[i]:8.3f}")

    # Opponent X-scores (average)
    opponent_x = np.zeros(len(categories))
    if opponent_teams:
        for opp_team in opponent_teams:
            for player in opp_team:
                player_x = np.array([
                    scoring.calculate_x_score(player, cat)
                    for cat in categories
                ])
                opponent_x += player_x
        opponent_x /= len(opponent_teams)

    # Initial weights
    initial_weights = optimizer.baseline_weights.copy()
    print(f"\nInitial weights:")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {initial_weights[i]:.4f}")

    # Calculate X_delta
    n_remaining = 12  # For first pick
    x_delta = optimizer.calculate_x_delta(initial_weights, n_remaining)

    print(f"\nX_delta (expected future adjustment):")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {x_delta[i]:8.3f}")

    # Team projection
    team_projection = current_team_x + candidate_x + x_delta

    print(f"\nTeam projection (current + candidate + future):")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {team_projection[i]:8.3f}")

    # Calculate initial objective
    initial_obj = optimizer.calculate_objective(
        initial_weights, candidate_x, current_team_x,
        opponent_x, n_remaining
    )

    print(f"\nInitial objective value: {initial_obj:.4f}")

    # Optimize
    optimal_weights, h_score = optimizer.optimize_weights(
        candidate_x, current_team_x, opponent_x,
        n_remaining, initial_weights, max_iterations=20
    )

    print(f"\nOptimal weights after optimization:")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {optimal_weights[i]:.4f}")

    print(f"\nFinal H-score: {h_score:.4f}")

    return h_score, optimal_weights


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("H-SCORING DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    # Load data
    league_data, player_variances = load_data()

    # Players to analyze
    test_players = [
        'Franz Wagner',
        'Anthony Edwards',
        'Anthony Davis',
        'Karl-Anthony Towns',
        'Nikola Jokic',
        'Victor Wembanyama',
        'Luka Doncic',
        'Joel Embiid',
        'Donovan Mitchell',
        'Zion Williamson',
        'Pascal Siakam',
        'Paolo Banchero',
        'Jalen Williams',
        'Ja Morant'
    ]

    # Filter to available players
    available_players = []
    for player in test_players:
        if player in league_data['PLAYER_NAME'].values:
            available_players.append(player)
        else:
            print(f"Warning: {player} not found in data")

    # Analyze player stats
    analyze_player_stats(league_data, player_variances, available_players[:5])

    # Compare scores
    scoring = compare_scores(league_data, player_variances, available_players)

    # Trace H-score calculation for Franz Wagner vs Anthony Edwards
    print("\n" + "=" * 80)
    print("DETAILED H-SCORE COMPARISON")
    print("=" * 80)

    if 'Franz Wagner' in available_players:
        print("\n>>> Franz Wagner:")
        franz_h, franz_w = trace_h_score_calculation(
            scoring, league_data, player_variances, 'Franz Wagner'
        )

    if 'Anthony Edwards' in available_players:
        print("\n>>> Anthony Edwards:")
        ant_h, ant_w = trace_h_score_calculation(
            scoring, league_data, player_variances, 'Anthony Edwards'
        )

    # Check for NaN issues
    print("\n" + "=" * 80)
    print("CHECKING FOR NaN/INF ISSUES")
    print("=" * 80)

    # Setup covariance
    cov_calc = CovarianceCalculator(scoring, league_data)
    setup_params = cov_calc.calculate_all()

    # Check covariance matrix
    cov_matrix = setup_params['covariance_matrix']
    inv_cov = setup_params['inverse_covariance']

    print(f"\nCovariance matrix shape: {cov_matrix.shape}")
    print(f"Contains NaN: {np.any(np.isnan(cov_matrix))}")
    print(f"Contains Inf: {np.any(np.isinf(cov_matrix))}")
    print(f"Matrix condition number: {np.linalg.cond(cov_matrix):.2e}")

    print(f"\nInverse covariance shape: {inv_cov.shape}")
    print(f"Contains NaN: {np.any(np.isnan(inv_cov))}")
    print(f"Contains Inf: {np.any(np.isinf(inv_cov))}")

    # Check v_vector
    v_vector = setup_params['v_vector']
    print(f"\nV-vector:")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {v_vector[i]:.4f}")

    # Check baseline weights
    baseline_weights = setup_params['baseline_weights']
    print(f"\nBaseline weights:")
    for i, cat in enumerate(setup_params['categories']):
        print(f"  {cat:8s}: {baseline_weights[i]:.4f}")


if __name__ == "__main__":
    main()