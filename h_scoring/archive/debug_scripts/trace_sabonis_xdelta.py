"""
Trace the EXACT X_delta calculation for Sabonis on pick #1.
Show every step in detail.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def main():
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
    print("EXACT X_DELTA CALCULATION FOR SABONIS")
    print("=" * 80)

    # Parameters
    N = 13  # total picks
    K = 0   # picks made (pick #1)

    print(f"\nInputs:")
    print(f"  N (total picks) = {N}")
    print(f"  K (picks made) = {K}")
    print(f"  N - K - 1 (remaining) = {N - K - 1}")
    print(f"  gamma = {optimizer.gamma}")
    print(f"  omega = {optimizer.omega}")

    # Call the actual X_delta function
    x_delta = optimizer._compute_xdelta_simplified(
        jC=baseline_weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=N,
        K=K
    )

    print("\n" + "=" * 80)
    print("FULL X_DELTA VECTOR")
    print("=" * 80)

    for i, cat in enumerate(categories):
        print(f"{cat:<12} {x_delta[i]:>10.4f}")

    dd_idx = categories.index('DD')
    print(f"\nDD X_delta = {x_delta[dd_idx]:.4f}")

    # Now show Sabonis evaluation
    print("\n" + "=" * 80)
    print("SABONIS EVALUATION")
    print("=" * 80)

    # Sabonis X-scores
    sabonis_x = np.array([
        scoring.calculate_x_score("Domantas Sabonis", cat)
        for cat in categories
    ])

    print("\nSabonis X-scores:")
    for i, cat in enumerate(categories):
        print(f"{cat:<12} {sabonis_x[i]:>10.4f}")

    # Current team (empty)
    current_team_x = np.zeros(len(categories))

    # Team projection
    team_projection = current_team_x + sabonis_x + x_delta

    print("\n" + "=" * 80)
    print("TEAM PROJECTION")
    print("=" * 80)

    print(f"\n{'Category':<12} {'Current':<10} {'Sabonis':<10} {'X_delta':<10} {'Total':<10}")
    print("-" * 60)
    for i, cat in enumerate(categories):
        print(f"{cat:<12} {current_team_x[i]:>9.4f} {sabonis_x[i]:>9.4f} {x_delta[i]:>9.4f} {team_projection[i]:>9.4f}")

    print("\n" + "=" * 80)
    print("KEY QUESTION")
    print("=" * 80)

    print(f"\nSabonis DD X-score: {sabonis_x[dd_idx]:.4f}")
    print(f"X_delta DD: {x_delta[dd_idx]:.4f}")
    print(f"Total DD projection: {team_projection[dd_idx]:.4f}")

    print("\nIs this X_delta CORRECT for Sabonis?")
    print("\nThink about it:")
    print("  - Sabonis already provides 7.50 DD")
    print("  - X_delta adds another 5.57 DD")
    print("  - Total: 13.07 DD")
    print("")
    print("  - Does it make sense that your FUTURE picks would add 5.57 DD")
    print("    when you've already drafted the BEST DD player (7.50)?")
    print("")
    print("  - Shouldn't X_delta for Sabonis be LOWER in DD?")
    print("    (Because you already have elite DD, focus on other categories)")

    print("\n" + "=" * 80)
    print("CALCULATION DETAILS")
    print("=" * 80)

    print("\nX_delta is calculated using:")
    print("  - jC (category weights) = BASELINE weights")
    print(f"  - DD baseline weight = {baseline_weights[dd_idx]*100:.1f}%")
    print("")
    print("The calculation does NOT know:")
    print("  - Which candidate you're evaluating (Sabonis vs KD)")
    print("  - What the candidate's X-scores are")
    print("  - What your current team composition is")
    print("")
    print("It ONLY knows:")
    print("  - Baseline weights (DD = 22.6%)")
    print("  - Number of remaining picks (12)")
    print("  - Generic parameters (gamma, omega)")

    print("\n" + "=" * 80)
    print("IS THIS THE BUG?")
    print("=" * 80)

    print("\nShould X_delta be calculated DIFFERENTLY for each candidate?")
    print("")
    print("Option 1 (current):")
    print("  - Calculate X_delta ONCE using baseline weights")
    print("  - Apply SAME X_delta to all candidates")
    print("  - KD gets: -0.16 + 5.57 = 5.41 DD")
    print("  - Sabonis gets: 7.50 + 5.57 = 13.07 DD")
    print("")
    print("Option 2 (alternative):")
    print("  - Calculate X_delta based on candidate's impact on weights")
    print("  - If candidate is strong in DD, X_delta DD should be lower")
    print("  - If candidate is weak in DD, X_delta DD should be higher")
    print("  - This would create candidate-specific X_delta")


if __name__ == "__main__":
    main()
