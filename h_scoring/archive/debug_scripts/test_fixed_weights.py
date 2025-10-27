"""
Test evaluating players with FIXED baseline weights (no optimization).

This tests the hypothesis that on pick #1, all candidates should be
evaluated with the same baseline weights, not custom-optimized weights.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def evaluate_with_fixed_weights(optimizer, player_name, weights, picks_made=0):
    """Evaluate player using fixed weights (no optimization)."""

    categories = optimizer.categories

    # Get player X-scores
    candidate_x = np.array([
        optimizer.scoring.calculate_x_score(player_name, cat)
        for cat in categories
    ])

    # Empty team (pick #1)
    current_team_x = np.zeros(len(categories))

    # Opponent (pick #1 = 0 picks made)
    opponent_x = optimizer._calculate_average_opponent_x([], player_name, picks_made)

    # Calculate X_delta with fixed weights
    n_remaining = 13 - picks_made - 1
    x_delta = optimizer._compute_xdelta_simplified(
        jC=weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=picks_made
    )

    # Team projection
    team_projection = current_team_x + candidate_x + x_delta

    # Calculate win probabilities
    category_variances = optimizer._calculate_category_variances()

    win_probs = []
    for i, cat in enumerate(categories):
        diff = team_projection[i] - opponent_x[i]
        variance = category_variances[cat]
        sigma = np.sqrt(variance)

        if sigma > 0:
            z = diff / sigma
            z = np.clip(z, -10, 10)
            p_win = norm.cdf(z)
        else:
            p_win = 0.5

        win_probs.append(p_win)

    # H-score = sum of win probabilities
    h_score = sum(win_probs)

    return h_score, win_probs


def main():
    """Compare KD vs Sabonis with fixed vs optimized weights."""

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
    cov_calc = CovarianceCalculator(league_data, scoring)
    setup_params = cov_calc.get_setup_params()

    categories = setup_params['categories']
    baseline_weights = setup_params['baseline_weights']

    # Create optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    print("=" * 80)
    print("FIXED WEIGHTS vs OPTIMIZED WEIGHTS COMPARISON")
    print("=" * 80)

    print("\nBaseline weights (DD=22.6%, FT_PCT=3.1%):")
    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {baseline_weights[i]*100:5.1f}%")

    # Test top candidates
    candidates = [
        "Kevin Durant",
        "Domantas Sabonis",
        "Anthony Davis",
        "Stephen Curry",
        "James Harden"
    ]

    print("\n" + "=" * 80)
    print("METHOD 1: FIXED BASELINE WEIGHTS (FAIR COMPARISON)")
    print("=" * 80)
    print("\nEvaluating all candidates with SAME baseline weights...")

    fixed_results = []
    for player in candidates:
        h_score, win_probs = evaluate_with_fixed_weights(optimizer, player, baseline_weights, picks_made=0)
        g_score = scoring.calculate_all_g_scores(player)['TOTAL']

        fixed_results.append({
            'player': player,
            'h_score_fixed': h_score,
            'g_score': g_score
        })

        print(f"  {player:<25} H-score: {h_score:.4f}  G-score: {g_score:.2f}")

    # Sort by H-score
    fixed_results.sort(key=lambda x: x['h_score_fixed'], reverse=True)

    print("\nRanking with FIXED baseline weights:")
    for i, result in enumerate(fixed_results, 1):
        print(f"  {i}. {result['player']:<25} H-score: {result['h_score_fixed']:.4f}")

    print("\n" + "=" * 80)
    print("METHOD 2: OPTIMIZED WEIGHTS (CURRENT IMPLEMENTATION)")
    print("=" * 80)
    print("\nEvaluating candidates with CUSTOM-OPTIMIZED weights...")

    optimized_results = []
    for player in candidates:
        h_score, optimal_weights = optimizer.evaluate_player(
            player,
            my_team=[],
            opponent_teams=[],
            picks_made=0,
            total_picks=13,
            last_weights=None,
            format='each_category'
        )
        g_score = scoring.calculate_all_g_scores(player)['TOTAL']

        optimized_results.append({
            'player': player,
            'h_score_optimized': h_score,
            'g_score': g_score,
            'optimal_weights': optimal_weights
        })

        print(f"  {player:<25} H-score: {h_score:.4f}  G-score: {g_score:.2f}")

    # Sort by H-score
    optimized_results.sort(key=lambda x: x['h_score_optimized'], reverse=True)

    print("\nRanking with OPTIMIZED weights:")
    for i, result in enumerate(optimized_results, 1):
        print(f"  {i}. {result['player']:<25} H-score: {result['h_score_optimized']:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    # Create comparison table
    comparison = []
    for fixed, optimized in zip(fixed_results, optimized_results):
        comparison.append({
            'player': fixed['player'],
            'g_score': fixed['g_score'],
            'h_fixed': fixed['h_score_fixed'],
            'h_optimized': optimized['h_score_optimized'],
            'h_difference': optimized['h_score_optimized'] - fixed['h_score_fixed']
        })

    # Sort by G-score for comparison
    comparison.sort(key=lambda x: x['g_score'], reverse=True)

    print(f"\n{'Player':<25} {'G-score':<10} {'H-fixed':<10} {'H-optim':<10} {'Benefit':<10}")
    print("-" * 80)
    for row in comparison:
        benefit = row['h_difference']
        print(f"{row['player']:<25} {row['g_score']:<10.2f} {row['h_fixed']:<10.4f} {row['h_optimized']:<10.4f} {benefit:+.4f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    print("\nWith FIXED baseline weights (fair comparison):")
    print(f"  Top pick: {fixed_results[0]['player']}")

    print("\nWith OPTIMIZED weights (current implementation):")
    print(f"  Top pick: {optimized_results[0]['player']}")

    if fixed_results[0]['player'] != optimized_results[0]['player']:
        print("\n⚠️  DIFFERENT RECOMMENDATIONS!")
        print("\nThe current implementation allows the optimizer to 'cheat' by")
        print("customizing weights to each candidate on pick #1, which creates")
        print("an unfair comparison.")
        print("\nFIX: On pick #1 (no team context), evaluate ALL candidates with")
        print("the SAME baseline weights. Only optimize weights after drafting players.")
    else:
        print("\n✓ Same recommendation with both methods")


if __name__ == "__main__":
    main()
