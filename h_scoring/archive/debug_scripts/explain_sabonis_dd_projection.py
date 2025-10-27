"""
Explain how Sabonis gets DD projection of 13.07

Breakdown:
Team projection = current_team_x + candidate_x + x_delta
                = 0 (empty) + 7.50 (Sabonis) + 5.57 (future picks)
                = 13.07
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def main():
    """Explain Sabonis DD projection calculation."""

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
    print("SABONIS DD PROJECTION BREAKDOWN")
    print("=" * 80)

    # Step 1: Get Sabonis's raw DD stats
    print("\nSTEP 1: SABONIS RAW DD STATS")
    print("-" * 80)

    sabonis_data = league_data[league_data['PLAYER_NAME'] == 'Domantas Sabonis']

    if len(sabonis_data) == 0:
        print("ERROR: Sabonis not found in data")
        return

    # Get DD stats
    dd_values = sabonis_data['DD'].values
    dd_mean = dd_values.mean()
    dd_total = dd_values.sum()
    weeks = len(dd_values)

    print(f"Sabonis data ({weeks} weeks):")
    print(f"  DD per week (mean): {dd_mean:.2f}")
    print(f"  DD total: {dd_total:.1f}")
    print(f"  Sample weeks: {dd_values[:5]}")

    # Step 2: Calculate league average DD
    print("\n" + "=" * 80)
    print("STEP 2: LEAGUE AVERAGE DD")
    print("-" * 80)

    # Get league stats - use the scoring system's calculated values
    # The X-score calculation uses: (player_mean - league_mean) / sigma_within
    # We can back-calculate league_mean and sigma_within from the X-score

    # Simpler: just use values from the formula
    all_dd = league_data.groupby('PLAYER_NAME')['DD'].mean()
    league_mean = all_dd.mean()
    league_std = all_dd.std()

    # For sigma_within, we'll approximate it
    # X-score removes between-player variance, so sigma_within ≈ average player std
    player_stds = []
    for player in league_data['PLAYER_NAME'].unique():
        player_data = league_data[league_data['PLAYER_NAME'] == player]['DD']
        if len(player_data) > 1:
            player_stds.append(player_data.std())

    league_std_within = np.mean(player_stds) if player_stds else league_std

    print(f"League DD statistics:")
    print(f"  League mean (per week): {league_mean:.4f}")
    print(f"  Within-player std dev: {league_std_within:.4f}")

    # Step 3: Calculate Sabonis X-score
    print("\n" + "=" * 80)
    print("STEP 3: SABONIS DD X-SCORE")
    print("-" * 80)

    sabonis_dd_xscore = scoring.calculate_x_score('Domantas Sabonis', 'DD')

    print(f"X-score formula: X = (player_mean - league_mean) / sigma_within")
    print(f"")
    print(f"Sabonis DD X-score:")
    print(f"  X = ({dd_mean:.4f} - {league_mean:.4f}) / {league_std_within:.4f}")
    print(f"  X = {dd_mean - league_mean:.4f} / {league_std_within:.4f}")
    print(f"  X = {sabonis_dd_xscore:.4f}")

    print(f"\nInterpretation:")
    print(f"  Sabonis averages {dd_mean:.2f} DD per week")
    print(f"  League average is {league_mean:.2f} DD per week")
    print(f"  Sabonis is {(dd_mean - league_mean) / league_std_within:.2f} standard deviations above average")

    # Step 4: Calculate X_delta for DD
    print("\n" + "=" * 80)
    print("STEP 4: X_DELTA (FUTURE PICKS ADJUSTMENT)")
    print("-" * 80)

    # Calculate X_delta with baseline weights
    x_delta = optimizer._compute_xdelta_simplified(
        jC=baseline_weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=0  # picks_made = 0
    )

    dd_idx = categories.index('DD')
    dd_xdelta = x_delta[dd_idx]

    print(f"X_delta for DD: {dd_xdelta:.4f}")
    print(f"\nWhat is X_delta?")
    print(f"  X_delta estimates the expected stats from your REMAINING draft picks.")
    print(f"  With 12 picks remaining (13 - 1 for Sabonis = 12), the optimizer")
    print(f"  predicts you'll accumulate {dd_xdelta:.2f} DD X-score from future picks.")

    print(f"\nWhy is DD X_delta so high ({dd_xdelta:.2f})?")
    dd_weight = baseline_weights[dd_idx] * 100
    print(f"  1. DD has the HIGHEST baseline weight ({dd_weight:.1f}%)")
    print(f"  2. This signals DD is very scarce/valuable")
    print(f"  3. X_delta assumes you'll prioritize drafting DD-heavy players later")
    print(f"  4. So it projects {dd_xdelta:.2f} DD from those 12 future picks")

    # Show formula breakdown
    print(f"\nX_delta calculation involves:")
    print(f"  - Remaining picks: 12")
    print(f"  - Category weights (jC): DD weight = {dd_weight:.1f}%")
    print(f"  - Covariance matrix: How DD correlates with other stats")
    print(f"  - Multiplier: 0.25 × 12 = 3.0")
    print(f"  Result: {dd_xdelta:.2f}")

    # Step 5: Total team projection
    print("\n" + "=" * 80)
    print("STEP 5: TOTAL TEAM DD PROJECTION")
    print("-" * 80)

    current_team_dd = 0  # Empty team
    total_projection = current_team_dd + sabonis_dd_xscore + dd_xdelta

    print(f"Team DD projection = current_team + candidate + x_delta")
    print(f"                   = {current_team_dd:.2f} + {sabonis_dd_xscore:.2f} + {dd_xdelta:.2f}")
    print(f"                   = {total_projection:.2f}")

    print(f"\nBreakdown:")
    print(f"  {current_team_dd:>6.2f}  Current team DD (empty team on pick #1)")
    print(f"  {sabonis_dd_xscore:>6.2f}  Sabonis's DD X-score")
    print(f"+ {dd_xdelta:>6.2f}  Expected DD from 12 future picks")
    print(f"  ------")
    print(f"  {total_projection:>6.2f}  Total projected DD X-score")

    # Step 6: Win probability
    print("\n" + "=" * 80)
    print("STEP 6: DD WIN PROBABILITY")
    print("-" * 80)

    from scipy.stats import norm

    # Opponent DD (empty on pick #1)
    opponent_dd = 0

    # Category variance
    category_variances = optimizer._calculate_category_variances()
    dd_variance = category_variances['DD']
    dd_std = np.sqrt(dd_variance)

    # Calculate z-score
    diff = total_projection - opponent_dd
    z_score = diff / dd_std
    p_win = norm.cdf(z_score)

    print(f"Win probability formula:")
    print(f"  z = (your_team - opponent_team) / sqrt(variance)")
    print(f"  P(win) = Φ(z)  [normal CDF]")
    print(f"")
    print(f"For DD:")
    print(f"  Your team:  {total_projection:.2f}")
    print(f"  Opponent:   {opponent_dd:.2f}")
    print(f"  Difference: {diff:.2f}")
    print(f"  Variance:   {dd_variance:.2f}")
    print(f"  Std dev:    {dd_std:.2f}")
    print(f"  z-score:    {diff:.2f} / {dd_std:.2f} = {z_score:.4f}")
    print(f"  P(win DD):  {p_win:.3f} = {p_win*100:.1f}%")

    # Step 7: Contribution to H-score
    print("\n" + "=" * 80)
    print("STEP 7: CONTRIBUTION TO H-SCORE")
    print("-" * 80)

    dd_contribution = p_win

    print(f"DD contributes {p_win:.3f} to total H-score")
    print(f"(H-score = sum of all 11 category win probabilities)")
    print(f"")
    print(f"With {dd_weight:.1f}% weight on DD (highest of all categories),")
    print(f"Sabonis gets {p_win*100:.1f}% win probability in DD category.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nSabonis DD projection of {total_projection:.2f} comes from:")
    print(f"  1. Sabonis's elite DD production: {sabonis_dd_xscore:.2f} X-score")
    print(f"     (averages {dd_mean:.2f} DD/week vs league {league_mean:.2f})")
    print(f"")
    print(f"  2. Expected future picks: {dd_xdelta:.2f} X-score")
    print(f"     (12 remaining picks × DD weight {dd_weight:.1f}% × covariance adjustments)")
    print(f"")
    print(f"This gives {p_win*100:.1f}% probability to win DD category,")
    print(f"contributing {p_win:.3f} to Sabonis's total H-score.")


if __name__ == "__main__":
    main()
