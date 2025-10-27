"""
Debug gradient descent to see why it's moving away from optimal baseline weights.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


# Monkey-patch the optimizer to add logging
original_optimize_weights = HScoreOptimizerPaperFaithful.optimize_weights


def logged_optimize_weights(self, candidate_x, current_team_x, opponent_x,
                            n_remaining, initial_weights=None,
                            max_iterations=100, learning_rate=0.01,
                            format='each_category'):
    """Logged version of optimize_weights."""

    # Initialize weights
    if initial_weights is None:
        weights = self.baseline_weights.copy()
    else:
        weights = initial_weights.copy()

    # Normalize
    weights = weights / weights.sum()

    # Adam optimizer parameters
    m = np.zeros(self.n_cats)
    v = np.zeros(self.n_cats)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Calculate initial objective value
    initial_value = self.calculate_objective(
        weights, candidate_x, current_team_x,
        opponent_x, n_remaining, format
    )

    print(f"\n{'Iter':<6} {'Objective':<12} {'Change':<12} {'Best':<12} DD%     FT_PCT%")
    print("-" * 70)

    best_value = initial_value
    best_weights = weights.copy()

    dd_idx = self.categories.index('DD')
    ft_idx = self.categories.index('FT_PCT')

    print(f"{'Init':<6} {initial_value:<12.6f} {'':<12} {best_value:<12.6f} {weights[dd_idx]*100:5.1f}   {weights[ft_idx]*100:5.1f}")

    for iteration in range(max_iterations):
        # Calculate gradient
        gradient = self.calculate_gradient(
            weights, candidate_x, current_team_x,
            opponent_x, n_remaining, format
        )

        # Check for NaN in gradient
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            print(f"{'':6} NaN in gradient - stopping")
            break

        # Adam update
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** (iteration + 1))
        v_hat = v / (1 - beta2 ** (iteration + 1))

        # Update weights
        weights = weights + learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Ensure positive weights
        weights = np.maximum(weights, 1e-4)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        # Calculate current objective value
        current_value = self.calculate_objective(
            weights, candidate_x, current_team_x,
            opponent_x, n_remaining, format
        )

        # Check for NaN/inf in objective
        if np.isnan(current_value) or np.isinf(current_value):
            print(f"{'':6} NaN in objective - stopping")
            break

        # Track best
        improved = ""
        if current_value > best_value:
            best_value = current_value
            best_weights = weights.copy()
            improved = "*"

        change = current_value - initial_value

        # Print every 10 iterations or if best
        if iteration % 10 == 0 or improved or iteration < 5:
            print(f"{iteration:<6} {current_value:<12.6f} {change:+12.6f} {best_value:<12.6f} {weights[dd_idx]*100:5.1f}   {weights[ft_idx]*100:5.1f} {improved}")

        # Check convergence
        if iteration > 10 and abs(current_value - best_value) < 1e-6:
            print(f"{'':6} Converged at iteration {iteration}")
            break

    print("-" * 70)
    print(f"{'Final':<6} {'':<12} {'':<12} {best_value:<12.6f} {best_weights[dd_idx]*100:5.1f}   {best_weights[ft_idx]*100:5.1f}")

    return best_weights, best_value


def main():
    """Run logged optimization for Jokic on pick #1."""

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

    # Create optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    # Monkey-patch
    optimizer.optimize_weights = lambda *args, **kwargs: logged_optimize_weights(optimizer, *args, **kwargs)

    print("=" * 80)
    print("GRADIENT DESCENT DEBUGGING - Jokić on Pick #1")
    print("=" * 80)

    # Evaluate Jokic
    h_score, optimal_weights = optimizer.evaluate_player(
        "Nikola Jokić",
        my_team=[],
        opponent_teams=[],
        picks_made=0,
        total_picks=13,
        last_weights=None,
        format='each_category'
    )

    print(f"\nFinal H-score: {h_score:.4f}")

    categories = setup_params['categories']
    baseline_weights = setup_params['baseline_weights']

    print("\nWeight comparison:")
    dd_idx = categories.index('DD')
    ft_idx = categories.index('FT_PCT')

    print(f"  DD:      {baseline_weights[dd_idx]*100:.1f}% → {optimal_weights[dd_idx]*100:.1f}%")
    print(f"  FT_PCT:  {baseline_weights[ft_idx]*100:.1f}% → {optimal_weights[ft_idx]*100:.1f}%")


if __name__ == "__main__":
    main()
