"""
Compare Kevin Durant vs Domantas Sabonis on pick #1.

KD: H-score 5.93, G-score 4.94, ADP 22.7
Sabonis: H-score 5.84, G-score 7.78, ADP 10.1

Why does the optimizer prefer KD despite much lower G-score?
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def compare_kd_sabonis():
    """Compare KD vs Sabonis on pick #1."""

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
    print("KEVIN DURANT vs DOMANTAS SABONIS - PICK #1 COMPARISON")
    print("=" * 80)

    # Get their stats
    kd_g = scoring.calculate_all_g_scores("Kevin Durant")
    sabonis_g = scoring.calculate_all_g_scores("Domantas Sabonis")

    kd_x = scoring.calculate_all_x_scores("Kevin Durant")
    sabonis_x = scoring.calculate_all_x_scores("Domantas Sabonis")

    print("\n" + "-" * 80)
    print("G-SCORES (variance-adjusted value)")
    print("-" * 80)
    print(f"{'Category':<12} {'KD G-score':<12} {'Sabonis G-score':<15} {'Difference':<12}")
    print("-" * 80)

    for cat in categories:
        kd_val = kd_g.get(cat, 0)
        sabonis_val = sabonis_g.get(cat, 0)
        diff = sabonis_val - kd_val
        winner = "Sabonis" if diff > 0 else "KD"
        print(f"{cat:<12} {kd_val:>11.2f} {sabonis_val:>14.2f} {diff:>11.2f} ({winner})")

    print("-" * 80)
    print(f"{'TOTAL':<12} {kd_g['TOTAL']:>11.2f} {sabonis_g['TOTAL']:>14.2f} {sabonis_g['TOTAL'] - kd_g['TOTAL']:>11.2f}")

    print("\n" + "-" * 80)
    print("X-SCORES (optimization basis)")
    print("-" * 80)
    print(f"{'Category':<12} {'KD X-score':<12} {'Sabonis X-score':<15} {'Difference':<12}")
    print("-" * 80)

    for cat in categories:
        kd_val = kd_x.get(cat, 0)
        sabonis_val = sabonis_x.get(cat, 0)
        diff = sabonis_val - kd_val
        winner = "Sabonis" if diff > 0 else "KD"
        print(f"{cat:<12} {kd_val:>11.2f} {sabonis_val:>14.2f} {diff:>11.2f} ({winner})")

    # Evaluate both players
    print("\n" + "=" * 80)
    print("H-SCORE EVALUATION (PICK #1, EMPTY TEAM)")
    print("=" * 80)

    h_kd, weights_kd = optimizer.evaluate_player(
        "Kevin Durant",
        my_team=[],
        opponent_teams=[],
        picks_made=0,
        total_picks=13,
        last_weights=None,
        format='each_category'
    )

    h_sabonis, weights_sabonis = optimizer.evaluate_player(
        "Domantas Sabonis",
        my_team=[],
        opponent_teams=[],
        picks_made=0,
        total_picks=13,
        last_weights=None,
        format='each_category'
    )

    print(f"\nKevin Durant H-score:    {h_kd:.4f}")
    print(f"Domantas Sabonis H-score: {h_sabonis:.4f}")
    print(f"Difference:               {h_kd - h_sabonis:+.4f} (KD {'wins' if h_kd > h_sabonis else 'loses'})")

    # Compare optimal weights
    print("\n" + "-" * 80)
    print("OPTIMAL WEIGHTS CHOSEN BY OPTIMIZER")
    print("-" * 80)
    print(f"{'Category':<12} {'Baseline%':<12} {'KD Optimal%':<14} {'Sabonis Opt%':<14}")
    print("-" * 80)

    for i, cat in enumerate(categories):
        baseline_pct = baseline_weights[i] * 100
        kd_pct = weights_kd[i] * 100
        sabonis_pct = weights_sabonis[i] * 100
        print(f"{cat:<12} {baseline_pct:>11.1f} {kd_pct:>13.1f} {sabonis_pct:>13.1f}")

    # Check if weights are similar
    weight_diff = np.abs(weights_kd - weights_sabonis).mean()
    print(f"\nAverage absolute weight difference: {weight_diff*100:.2f}%")

    if weight_diff < 0.02:
        print("✓ Weights are very similar (as expected on pick #1)")
    else:
        print("⚠️  Weights differ significantly - optimizer is biasing toward player strengths")

    # Calculate weighted contributions
    print("\n" + "=" * 80)
    print("WEIGHTED CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print("\nUsing each player's optimal weights:")
    print(f"{'Category':<12} {'KD contrib':<12} {'Sabonis contrib':<15} {'Difference':<12}")
    print("-" * 80)

    kd_total = 0
    sabonis_total = 0

    for i, cat in enumerate(categories):
        kd_contrib = weights_kd[i] * kd_x.get(cat, 0)
        sabonis_contrib = weights_sabonis[i] * sabonis_x.get(cat, 0)
        diff = kd_contrib - sabonis_contrib

        kd_total += kd_contrib
        sabonis_total += sabonis_contrib

        winner = "KD" if diff > 0 else "Sabonis"
        print(f"{cat:<12} {kd_contrib:>11.3f} {sabonis_contrib:>14.3f} {diff:>11.3f} ({winner})")

    print("-" * 80)
    print(f"{'TOTAL':<12} {kd_total:>11.3f} {sabonis_total:>14.3f} {kd_total - sabonis_total:>11.3f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print(f"\n1. G-score (static ranking):")
    print(f"   Sabonis: {sabonis_g['TOTAL']:.2f}")
    print(f"   KD: {kd_g['TOTAL']:.2f}")
    print(f"   → Sabonis is {sabonis_g['TOTAL'] - kd_g['TOTAL']:.2f} points better")

    print(f"\n2. H-score (optimized for team context):")
    print(f"   KD: {h_kd:.4f}")
    print(f"   Sabonis: {h_sabonis:.4f}")
    print(f"   → KD is {h_kd - h_sabonis:.4f} points better")

    print(f"\n3. The question:")
    print(f"   Why does H-scoring prefer KD on pick #1 when:")
    print(f"   - There's no team context yet")
    print(f"   - Sabonis has much higher G-score (7.78 vs 4.94)")
    print(f"   - Sabonis has more appropriate ADP (10.1 vs 22.7)")

    print(f"\n4. Possible explanations:")
    print(f"   a) X_delta calculation favors KD's stat profile")
    print(f"   b) Covariance matrix interactions favor KD")
    print(f"   c) Category variance scaling favors KD")
    print(f"   d) This is a bug in the H-scoring implementation")


if __name__ == "__main__":
    compare_kd_sabonis()
