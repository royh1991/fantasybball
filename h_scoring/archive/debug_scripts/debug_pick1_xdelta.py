"""
Debug X_delta calculation on pick #1 to see if it's biasing the optimizer.

On pick #1 with no team context, X_delta should not be creating strong
biases toward certain categories.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def debug_pick1_xdelta():
    """Debug what X_delta looks like on pick #1."""

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
    print("X_DELTA ANALYSIS ON PICK #1")
    print("=" * 80)

    # Evaluate Jokic on pick #1
    candidate = "Nikola Jokić"

    # Get candidate X-scores
    candidate_x = np.array([
        scoring.calculate_x_score(candidate, cat)
        for cat in categories
    ])

    print(f"\nCandidate: {candidate}")
    print("\nCandidate X-scores:")
    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {candidate_x[i]:6.2f}")

    # Calculate X_delta with baseline weights
    n_remaining = 13 - 0 - 1  # 12 remaining picks

    print(f"\n" + "-" * 80)
    print(f"X_delta calculation (n_remaining = {n_remaining}):")
    print("-" * 80)

    # Call the internal X_delta calculation
    x_delta = optimizer._compute_xdelta_simplified(
        jC=baseline_weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=0  # picks_made = 0
    )

    print("\nX_delta values:")
    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {x_delta[i]:6.2f}")

    # Calculate norm of X_delta vs candidate
    x_delta_norm = np.linalg.norm(x_delta)
    candidate_norm = np.linalg.norm(candidate_x)

    print(f"\nMagnitudes:")
    print(f"  ||candidate_x|| = {candidate_norm:.2f}")
    print(f"  ||x_delta||     = {x_delta_norm:.2f}")
    print(f"  Ratio:           {x_delta_norm / candidate_norm:.2%}")

    # Check team projection
    current_team_x = np.zeros(len(categories))  # Empty team
    team_projection = current_team_x + candidate_x + x_delta

    print(f"\n" + "-" * 80)
    print("Team projection (current + candidate + x_delta):")
    print("-" * 80)

    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {current_team_x[i]:6.2f} + {candidate_x[i]:6.2f} + {x_delta[i]:6.2f} = {team_projection[i]:6.2f}")

    # Get opponent X-scores
    opponent_x = optimizer._calculate_average_opponent_x([], candidate, picks_made=0)

    print(f"\n" + "-" * 80)
    print("Opponent X-scores (picks_made=0):")
    print("-" * 80)

    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {opponent_x[i]:6.2f}")

    opponent_norm = np.linalg.norm(opponent_x)
    print(f"\n  ||opponent_x|| = {opponent_norm:.2f}")

    if opponent_norm < 0.01:
        print("\n✓ Opponent is effectively zero on pick #1 (as expected)")
    else:
        print(f"\n⚠️  Opponent has non-zero X-scores on pick #1!")

    # Calculate win probabilities with baseline weights
    print(f"\n" + "=" * 80)
    print("WIN PROBABILITIES WITH BASELINE WEIGHTS")
    print("=" * 80)

    from scipy.stats import norm as normal_dist

    # Use category-specific variances
    category_variances = optimizer._calculate_category_variances()

    win_probs = []
    for i, cat in enumerate(categories):
        diff = team_projection[i] - opponent_x[i]
        variance = category_variances[cat]
        sigma = np.sqrt(variance)

        if sigma > 0:
            z = diff / sigma
            z = np.clip(z, -10, 10)  # Prevent overflow
            p_win = normal_dist.cdf(z)
        else:
            p_win = 0.5

        win_probs.append(p_win)
        print(f"  {cat:<12} P(win) = {p_win:.3f}")

    total_win_prob = sum(win_probs)
    print(f"\n  Total: {total_win_prob:.3f} expected categories won")

    # Now check what happens if we increase FT_PCT weight and decrease DD weight
    print(f"\n" + "=" * 80)
    print("TESTING MODIFIED WEIGHTS")
    print("=" * 80)

    # Create modified weights similar to what optimizer chose
    modified_weights = baseline_weights.copy()
    ft_idx = categories.index('FT_PCT')
    dd_idx = categories.index('DD')

    # Shift weight from DD to FT_PCT
    shift = 0.12  # 12%
    modified_weights[dd_idx] -= shift
    modified_weights[ft_idx] += shift
    modified_weights = modified_weights / modified_weights.sum()  # Renormalize

    print("\nModified weights (DD down, FT_PCT up):")
    for i, cat in enumerate(categories):
        if cat in ['DD', 'FT_PCT']:
            print(f"  {cat:<12} {baseline_weights[i]*100:5.1f}% → {modified_weights[i]*100:5.1f}%")

    # Calculate X_delta with modified weights
    x_delta_mod = optimizer._compute_xdelta_simplified(
        jC=modified_weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=0
    )

    team_projection_mod = current_team_x + candidate_x + x_delta_mod

    # Calculate win probabilities
    win_probs_mod = []
    for i, cat in enumerate(categories):
        diff = team_projection_mod[i] - opponent_x[i]
        variance = category_variances[cat]
        sigma = np.sqrt(variance)

        if sigma > 0:
            z = diff / sigma
            z = np.clip(z, -10, 10)
            p_win = normal_dist.cdf(z)
        else:
            p_win = 0.5

        win_probs_mod.append(p_win)

    total_win_prob_mod = sum(win_probs_mod)

    print(f"\nWin probability comparison:")
    print(f"  Baseline weights: {total_win_prob:.3f}")
    print(f"  Modified weights: {total_win_prob_mod:.3f}")
    print(f"  Improvement:      {total_win_prob_mod - total_win_prob:+.3f}")

    if total_win_prob_mod > total_win_prob:
        print("\n⚠️  Modified weights give higher win probability!")
        print("This explains why gradient descent moves away from baseline.")
        print("\nThe issue is that X_delta calculation is sensitive to weight changes")
        print("even on pick #1, creating an artificial bias.")
    else:
        print("\n✓ Baseline weights are optimal")


if __name__ == "__main__":
    debug_pick1_xdelta()
