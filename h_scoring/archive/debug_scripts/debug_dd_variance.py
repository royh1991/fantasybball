"""
Debug the DD variance calculation.

DD variance = 130.75 seems too high, diluting Sabonis's advantage.
Let's trace where this comes from.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def main():
    """Debug DD variance calculation."""

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
    cov_matrix = setup_params['covariance_matrix']

    # Create optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    print("=" * 80)
    print("DD VARIANCE INVESTIGATION")
    print("=" * 80)

    dd_idx = categories.index('DD')

    # Get diagonal of covariance matrix for DD
    dd_cov_diagonal = cov_matrix[dd_idx, dd_idx]

    print(f"\n1. Covariance matrix diagonal for DD: {dd_cov_diagonal:.4f}")

    # Calculate category variance
    base_variance = dd_cov_diagonal
    scaled_variance = base_variance * 13 * 2  # roster_size * 2 teams
    capped_variance = min(400.0, max(20.0, scaled_variance))

    print(f"2. Base variance (cov diagonal): {base_variance:.4f}")
    print(f"3. Scaled variance (* 26): {scaled_variance:.4f}")
    print(f"4. Capped variance [20, 400]: {capped_variance:.4f}")

    # Check what the optimizer actually uses
    category_variances = optimizer._calculate_category_variances()
    dd_variance_used = category_variances['DD']

    print(f"5. Variance used in H-score: {dd_variance_used:.4f}")

    # Now let's calculate what the DD variance SHOULD be
    print("\n" + "=" * 80)
    print("WHAT SHOULD DD VARIANCE BE?")
    print("=" * 80)

    # Get all player DD X-scores
    all_dd_x = []
    for player in league_data['PLAYER_NAME'].unique():
        try:
            dd_x = scoring.calculate_x_score(player, 'DD')
            if not np.isnan(dd_x) and not np.isinf(dd_x):
                all_dd_x.append(dd_x)
        except:
            pass

    all_dd_x = np.array(all_dd_x)

    print(f"\nDD X-scores across {len(all_dd_x)} players:")
    print(f"  Mean: {np.mean(all_dd_x):.4f}")
    print(f"  Std:  {np.std(all_dd_x):.4f}")
    print(f"  Variance: {np.var(all_dd_x):.4f}")
    print(f"  Min: {np.min(all_dd_x):.4f}")
    print(f"  Max: {np.max(all_dd_x):.4f}")

    # This variance represents variation across players
    # But for win probability, we need variance of TEAM TOTALS

    print("\n" + "=" * 80)
    print("SIMULATION: TEAM DD VARIANCE")
    print("=" * 80)

    # Simulate team DD totals by randomly sampling 13 players
    np.random.seed(42)
    n_simulations = 10000
    team_totals = []

    for _ in range(n_simulations):
        team = np.random.choice(all_dd_x, size=13, replace=False)
        team_total = team.sum()
        team_totals.append(team_total)

    team_totals = np.array(team_totals)

    print(f"\nSimulated team DD totals ({n_simulations} teams):")
    print(f"  Mean: {np.mean(team_totals):.2f}")
    print(f"  Std:  {np.std(team_totals):.2f}")
    print(f"  Variance: {np.var(team_totals):.2f}")

    # The variance of difference between two teams
    diff_variance = 2 * np.var(team_totals)
    print(f"\nVariance of difference between two teams:")
    print(f"  Var(Team1 - Team2) = 2 * Var(Team) = {diff_variance:.2f}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\nCurrent implementation uses: {dd_variance_used:.2f}")
    print(f"Simulation suggests:         {diff_variance:.2f}")
    print(f"Ratio (current/simulated):   {dd_variance_used / diff_variance:.2f}x")

    if dd_variance_used > diff_variance * 1.5:
        print("\n⚠️  Current variance is TOO HIGH!")
        print("This is diluting the impact of DD in win probability calculations.")
    elif dd_variance_used < diff_variance * 0.67:
        print("\n⚠️  Current variance is TOO LOW!")
        print("This is overstating the impact of DD.")
    else:
        print("\n✓ Variance is in reasonable range")

    # Show impact on win probability
    print("\n" + "=" * 80)
    print("IMPACT ON WIN PROBABILITY")
    print("=" * 80)

    from scipy.stats import norm

    sabonis_dd_proj = 13.07

    print(f"\nSabonis DD team projection: {sabonis_dd_proj:.2f}")
    print(f"\n{'Variance':<15} {'Std Dev':<10} {'z-score':<10} {'P(win)':<10}")
    print("-" * 50)

    for var in [50, 100, dd_variance_used, 150, 200]:
        std = np.sqrt(var)
        z = sabonis_dd_proj / std
        p_win = norm.cdf(z)
        print(f"{var:<15.1f} {std:<10.2f} {z:<10.2f} {p_win:<10.3f}")

    print("\n" + "=" * 80)
    print("RECOMMENDED FIX")
    print("=" * 80)

    print("\nThe variance calculation in _calculate_category_variances():")
    print("  scaled_variance = base_variance * roster_size * 2")
    print("\nThis assumes:")
    print("  1. Var(X1 + X2 + ... + X13) = 13 * Var(X)")
    print("  2. This holds IF X-scores are independent")
    print("\nBut DD X-scores are NOT independent!")
    print("  - Centers tend to get DDs (correlated)")
    print("  - Guards rarely get DDs (correlated)")
    print("\nA team with 3 centers will have MUCH higher DD variance than")
    print("a team with 5 guards.")
    print("\nBetter approach:")
    print("  Use simulation (like above) to estimate team variance")
    print("  OR: Use a variance multiplier < 26 to account for correlation")

    # Calculate what multiplier would match simulation
    simulated_team_var = np.var(team_totals)
    needed_multiplier = simulated_team_var / base_variance

    print(f"\nCurrent multiplier: 26 (13 players * 2 teams)")
    print(f"Simulated suggests: {needed_multiplier:.1f}")


if __name__ == "__main__":
    main()
