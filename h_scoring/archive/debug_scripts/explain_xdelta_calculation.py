"""
Explain the X_delta calculation step by step.

X_delta is supposed to estimate the expected stats from your REMAINING draft picks.
This is one of the most complex parts of the H-scoring algorithm.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def explain_xdelta_step_by_step(optimizer, weights, N, K):
    """
    Explain X_delta calculation step by step.

    This replicates the _compute_xdelta_simplified function but with detailed logging.
    """

    categories = optimizer.categories
    jC = weights
    v = optimizer.v_vector
    Sigma = optimizer.cov_matrix_original
    gamma = optimizer.gamma
    omega = optimizer.omega

    m = len(jC)

    print("=" * 80)
    print("X_DELTA CALCULATION WALKTHROUGH")
    print("=" * 80)

    print("\nINPUTS:")
    print(f"  N (total picks): {N}")
    print(f"  K (picks made): {K}")
    print(f"  Remaining picks: {N - K - 1}")
    print(f"  gamma (generic value penalty): {gamma}")
    print(f"  omega (category strength weighting): {omega}")

    print("\n" + "=" * 80)
    print("STEP 1: NORMALIZE WEIGHTS")
    print("=" * 80)

    jC_sum = np.sum(jC)
    print(f"\nSum of input weights: {jC_sum:.6f}")

    if abs(jC_sum) < 1e-12:
        jC = np.ones_like(jC) / float(len(jC))
        print("⚠️  Sum too small, using uniform weights")
    else:
        jC = jC / float(jC_sum)
        print("✓ Normalized weights to sum to 1.0")

    print("\nNormalized category weights:")
    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {jC[i]:.6f} ({jC[i]*100:.2f}%)")

    print("\n" + "=" * 80)
    print("STEP 2: RIDGE REGULARIZATION ON SIGMA")
    print("=" * 80)

    trace = np.trace(Sigma)
    ridge_frac = optimizer.ridge_frac
    ridge = max(ridge_frac * (trace + 1e-12), 1e-12)

    print(f"\nCovariance matrix Sigma ({m}x{m}):")
    print(f"  Trace(Sigma): {trace:.6f}")
    print(f"  Ridge fraction: {ridge_frac}")
    print(f"  Ridge value: {ridge:.10f}")

    Sigma_reg = Sigma + ridge * np.eye(m)

    print(f"\nSigma_reg = Sigma + {ridge:.10f} × I")
    print("(Adds small value to diagonal for numerical stability)")

    print("\n" + "=" * 80)
    print("STEP 3: PROJECT jC ONTO v (in Sigma-metric)")
    print("=" * 80)

    print("\nWhat is v-vector?")
    print("  v converts X-scores to G-scores: G = v^T × X")
    print("  v[i] = sigma_within[i] / sqrt(sigma_within[i]^2 + sigma_between[i]^2)")

    print("\nv-vector values:")
    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {v[i]:.6f}")

    # Calculate projection coefficient
    denom_vSv = float(v.T @ Sigma_reg @ v)
    proj_coeff = float((v.T @ Sigma_reg @ jC) / denom_vSv)

    print(f"\nProjection calculation:")
    print(f"  v^T @ Sigma_reg @ v = {denom_vSv:.6f}")
    print(f"  v^T @ Sigma_reg @ jC = {v.T @ Sigma_reg @ jC:.6f}")
    print(f"  Projection coefficient = {proj_coeff:.6f}")

    jC_parallel = v * proj_coeff
    jC_perp = jC - jC_parallel

    print(f"\nDecompose jC into parallel and perpendicular components:")
    print(f"  jC_parallel = {proj_coeff:.6f} × v")
    print(f"  jC_perp = jC - jC_parallel")

    print(f"\n||jC_parallel|| = {np.linalg.norm(jC_parallel):.6f}")
    print(f"||jC_perp||     = {np.linalg.norm(jC_perp):.6f}")

    print("\n" + "=" * 80)
    print("STEP 4: COMPUTE SIGMA (uncertainty measure)")
    print("=" * 80)

    sigma2 = float(max(jC_perp.T @ Sigma_reg @ jC_perp, 0.0))
    sigma = np.sqrt(sigma2) if sigma2 > 0 else 0.0

    print(f"\nσ² = jC_perp^T @ Sigma_reg @ jC_perp")
    print(f"   = {sigma2:.6f}")
    print(f"σ  = {sigma:.6f}")

    print("\nWhat does σ represent?")
    print("  σ measures the 'uncertainty' or 'spread' of the weight vector jC")
    print("  in the Sigma-metric (accounting for category correlations)")

    print("\n" + "=" * 80)
    print("STEP 5: BUILD CONSTRAINT MATRIX U AND TARGET VECTOR b")
    print("=" * 80)

    U = np.vstack([v, jC])
    b = np.array([-gamma * sigma, omega * sigma], dtype=float)

    print(f"\nU = [v^T]  (2 × {m} matrix)")
    print(f"    [jC^T]")

    print(f"\nb = [-γ × σ ]")
    print(f"    [ω × σ  ]")
    print(f"  = [{b[0]:.6f}]")
    print(f"    [{b[1]:.6f}]")

    print("\nConstraint interpretation:")
    print(f"  1. v^T @ X_delta = -γσ = {b[0]:.6f}")
    print(f"     (Total G-score from future picks should be negative × γσ)")
    print(f"     This is the 'generic value penalty' - assume future picks")
    print(f"     will have slightly below-average total value")

    print(f"\n  2. jC^T @ X_delta = ωσ = {b[1]:.6f}")
    print(f"     (Weighted score should be positive × ωσ)")
    print(f"     This says future picks will align with your weight priorities")

    print("\n" + "=" * 80)
    print("STEP 6: SOLVE FOR X_DELTA")
    print("=" * 80)

    # Form M = U @ Sigma_reg @ U^T
    M = U @ Sigma_reg @ U.T

    print(f"\nM = U @ Sigma_reg @ U^T  (2 × 2 matrix)")
    print(f"M = [[{M[0,0]:8.4f}, {M[0,1]:8.4f}],")
    print(f"     [{M[1,0]:8.4f}, {M[1,1]:8.4f}]]")

    cond_M = np.linalg.cond(M)
    print(f"\nCondition number of M: {cond_M:.2e}")

    if cond_M > 1e12:
        jitter = 1e-8 * (np.trace(M) + 1e-12)
        M = M + jitter * np.eye(2)
        print(f"⚠️  High condition number, adding jitter: {jitter:.2e}")

    # Solve M @ z = b
    z = np.linalg.solve(M, b)

    print(f"\nSolve: M @ z = b")
    print(f"z = [{z[0]:.6f}, {z[1]:.6f}]")

    # X_delta (per unit) = Sigma_reg @ U^T @ z
    xdelta_unit = Sigma_reg @ U.T @ z

    print(f"\nX_delta (unit) = Sigma_reg @ U^T @ z")
    print(f"||X_delta (unit)|| = {np.linalg.norm(xdelta_unit):.6f}")

    print("\nX_delta (unit) by category:")
    for i, cat in enumerate(categories):
        print(f"  {cat:<12} {xdelta_unit[i]:8.4f}")

    print("\n" + "=" * 80)
    print("STEP 7: SCALE BY REMAINING PICKS")
    print("=" * 80)

    remaining_picks = float(max(0, N - K - 1))
    effective_multiplier = remaining_picks * 0.25  # Paper uses 0.25 multiplier

    print(f"\nRemaining picks: {remaining_picks:.0f}")
    print(f"Multiplier per pick: 0.25")
    print(f"Effective multiplier: {effective_multiplier:.2f}")

    x_delta = effective_multiplier * xdelta_unit

    print(f"\nX_delta (final) = {effective_multiplier:.2f} × X_delta (unit)")

    print("\n" + "=" * 80)
    print("FINAL X_DELTA VALUES")
    print("=" * 80)

    print(f"\n{'Category':<12} {'Unit':<10} {'× {:.2f}'.format(effective_multiplier):<10} {'= Final':<10}")
    print("-" * 50)
    for i, cat in enumerate(categories):
        print(f"{cat:<12} {xdelta_unit[i]:>9.4f} {'':<10} {x_delta[i]:>9.4f}")

    print(f"\n||X_delta|| = {np.linalg.norm(x_delta):.4f}")

    return x_delta, {
        'jC': jC,
        'v': v,
        'sigma': sigma,
        'z': z,
        'xdelta_unit': xdelta_unit,
        'effective_multiplier': effective_multiplier
    }


def main():
    """Run X_delta explanation."""

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

    baseline_weights = setup_params['baseline_weights']

    # Create optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    # Run explanation
    x_delta, details = explain_xdelta_step_by_step(
        optimizer,
        weights=baseline_weights,
        N=13,  # total picks
        K=0    # picks made (pick #1)
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    categories = setup_params['categories']
    dd_idx = categories.index('DD')

    print(f"\nFor DD specifically:")
    print(f"  Baseline weight: {baseline_weights[dd_idx]*100:.1f}% (HIGHEST)")
    print(f"  X_delta (unit): {details['xdelta_unit'][dd_idx]:.4f}")
    print(f"  Remaining picks: 12")
    print(f"  Effective multiplier: {details['effective_multiplier']:.2f}")
    print(f"  X_delta (final): {x_delta[dd_idx]:.4f}")

    print(f"\nWhat this means:")
    print(f"  The algorithm predicts your 12 remaining draft picks will")
    print(f"  accumulate {x_delta[dd_idx]:.2f} DD X-score.")

    print(f"\nIs {x_delta[dd_idx]:.2f} reasonable?")
    print(f"  If you drafted 12 average players:")
    print(f"    12 × 0 (average X-score) = 0")
    print(f"  If you drafted 12 elite DD players (like Sabonis):")
    print(f"    12 × 7.50 = 90.0")
    print(f"  X_delta predicts: {x_delta[dd_idx]:.2f}")

    avg_per_pick = x_delta[dd_idx] / 12
    print(f"\n  That's {avg_per_pick:.2f} DD X-score per future pick")
    print(f"  Which is {'reasonable' if avg_per_pick < 1.0 else 'HIGH - might be overestimating'}")

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)

    print("\nX_delta is calculated using a CONSTRAINT OPTIMIZATION:")
    print(f"  1. Total value (v^T @ X_delta) = {details['z'][0] * details['effective_multiplier']:.2f}")
    print(f"     (negative = slightly below average picks)")
    print(f"  2. Aligned value (jC^T @ X_delta) = {details['z'][1] * details['effective_multiplier']:.2f}")
    print(f"     (positive = picks aligned with your category priorities)")

    print("\nThe algorithm assumes you'll draft players who:")
    print("  - Are slightly below league average overall (generic value penalty γ)")
    print("  - Strongly match your category weight priorities (ω = 0.7)")

    print("\nBecause DD has 22.6% weight (highest), the optimizer assumes")
    print("you'll prioritize DD-heavy players in future picks, giving high X_delta.")


if __name__ == "__main__":
    main()
