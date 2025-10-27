#!/usr/bin/env python
"""
Exact implementation of X_delta from the paper with numerical stability fixes.
"""

import numpy as np


def compute_xdelta_exact(jC, v, Sigma, gamma, omega, N, K, muC_P=None,
                         eps_sigma_rel=1e-8, eps_Uinv=1e-8, ridge_frac=1e-8):
    """
    Exact X_delta calculation from the paper with numerical stability.

    Parameters:
    -----------
    jC, v : 1D arrays (length m) -- ensure both sum to 1 (paper's convention)
    Sigma : (m,m) covariance matrix (must be symmetric)
    gamma, omega : scalars (paper used omega=0.7, gamma=0.25 as starting values)
    N, K : integers; picks remaining multiplier will be (N - K - 1)
    muC_P: optional vector to add at the end (positional adjustment μ_C P)

    Returns:
    --------
    x_delta : (length m) expected future pick adjustment
    diagnostics : dict with intermediate values for debugging
    """

    # --- basic sanity / forcing shapes ---
    jC = np.asarray(jC, dtype=float).reshape(-1)
    v  = np.asarray(v,  dtype=float).reshape(-1)
    m  = jC.size
    assert v.size == m
    Sigma = np.asarray(Sigma, dtype=float)
    assert Sigma.shape == (m, m)

    # 1) normalize jC and v to sum to 1 (paper does this after each gradient step)
    def normalized(x):
        s = x.sum()
        if abs(s) < 1e-10:
            # If sum is near zero, return uniform weights
            return np.ones_like(x) / len(x)
        return x / s

    jC = normalized(jC)
    v  = normalized(v)

    # 2) regularize Sigma if badly conditioned
    trace = np.trace(Sigma)
    ridge = max(ridge_frac * trace, 1e-12)
    Sigma_reg = Sigma + ridge * np.eye(m)

    # 3) compute the projection used in sigma^2:
    #    proj_coeff = (v^T Sigma jC) / (v^T Sigma v)
    denom_vSv = float(v.T @ Sigma_reg @ v)
    if denom_vSv <= 0:
        # If denominator is bad, return zero adjustment
        return np.zeros(m), {"error": "v^T Sigma v <= 0"}

    proj_coeff = float((v.T @ Sigma_reg @ jC) / denom_vSv)
    jC_perp = jC - v * proj_coeff   # (jC - projection of jC onto v in Sigma-metric)

    # 4) sigma^2 = jC_perp^T Sigma jC_perp  (see paper). Protect against negative rounding.
    sigma2 = float(jC_perp.T @ Sigma_reg @ jC_perp)
    sigma = float(np.sqrt(max(sigma2, eps_sigma_rel * (trace + 1e-12))))

    # 5) Build U and b (constraints vector). U should be 2 x m (rows [v; jC])
    U = np.vstack([v, jC])            # shape (2, m)
    # paper defines constraints v^T x = -gamma*sigma, jC^T x = omega*sigma
    b = np.array([-gamma * sigma, omega * sigma], dtype=float)  # shape (2,)

    # 6) Compute middle matrix M = U Sigma U^T (2x2) and invert robustly
    M = U @ Sigma_reg @ U.T          # shape (2,2)

    # regularize M for inversion if condition number high / near-singular
    cond_M = np.linalg.cond(M)
    if cond_M > 1e12:
        # add small jitter proportional to trace
        jitter = eps_Uinv * (np.trace(M) + 1e-12)
        M = M + jitter * np.eye(2)

    # solve rather than invert directly
    try:
        z = np.linalg.solve(M, b)    # shape (2,)
    except np.linalg.LinAlgError:
        # fallback: pseudo-inverse
        z = np.linalg.pinv(M) @ b

    # 7) final x_delta (before multiplying by remaining picks):
    xdelta_unit = Sigma_reg @ U.T @ z   # shape (m,)

    # 8) Scale by remaining picks
    remaining_picks = N - K - 1
    x_delta = remaining_picks * xdelta_unit

    # 9) Add positional adjustment if provided
    if muC_P is not None:
        x_delta = x_delta + muC_P

    # Collect diagnostics
    diagnostics = {
        "sigma": sigma,
        "sigma2": sigma2,
        "proj_coeff": proj_coeff,
        "cond_M": cond_M,
        "z": z,
        "remaining_picks": remaining_picks,
        "xdelta_unit_norm": np.linalg.norm(xdelta_unit),
        "xdelta_norm": np.linalg.norm(x_delta),
        "trace_Sigma": trace,
        "ridge": ridge
    }

    return x_delta, diagnostics


def test_exact_xdelta():
    """Test the exact X_delta implementation."""
    import pandas as pd
    import json
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from modules.scoring import PlayerScoring
    from modules.covariance import CovarianceCalculator

    print("=" * 80)
    print("TESTING EXACT X_DELTA IMPLEMENTATION")
    print("=" * 80)

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    weekly_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    league_data = pd.read_csv(weekly_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Initialize
    scoring = PlayerScoring(league_data, player_variances)
    cov_calc = CovarianceCalculator(league_data, scoring)
    setup_params = cov_calc.get_setup_params()

    # Get parameters
    jC = setup_params['baseline_weights']
    v = setup_params['v_vector']
    Sigma = setup_params['covariance_matrix']
    gamma = 0.25
    omega = 0.7
    N = 13  # roster size
    K = 0   # picks made (first pick)

    print(f"\nParameters:")
    print(f"  Categories: {len(jC)}")
    print(f"  N = {N}, K = {K}")
    print(f"  gamma = {gamma}, omega = {omega}")
    print(f"  Remaining picks: {N - K - 1}")

    # Compute exact X_delta
    x_delta, diagnostics = compute_xdelta_exact(jC, v, Sigma, gamma, omega, N, K)

    print(f"\nDiagnostics:")
    for key, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, norm={np.linalg.norm(value):.4f}")
        else:
            print(f"  {key}: {value:.6f}")

    print(f"\nX_delta values:")
    categories = setup_params['categories']
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {x_delta[i]:10.4f}")
    print(f"  SUM: {np.sum(x_delta):.4f}")
    print(f"  NORM: {np.linalg.norm(x_delta):.4f}")

    # Compare with original implementation
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL IMPLEMENTATION")
    print("=" * 80)

    from modules.h_optimizer import HScoreOptimizer
    optimizer = HScoreOptimizer(setup_params, scoring)

    # Calculate using original method
    x_delta_original = optimizer.calculate_x_delta(jC, N - K - 1)

    print(f"\nOriginal X_delta values:")
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {x_delta_original[i]:10.4f}")
    print(f"  SUM: {np.sum(x_delta_original):.4f}")
    print(f"  NORM: {np.linalg.norm(x_delta_original):.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"Exact implementation norm: {np.linalg.norm(x_delta):.4f}")
    print(f"Original implementation norm: {np.linalg.norm(x_delta_original):.4f}")
    print(f"Ratio: {np.linalg.norm(x_delta_original) / np.linalg.norm(x_delta):.2f}x")

    if np.linalg.norm(x_delta_original) > 100 * np.linalg.norm(x_delta):
        print("\n⚠️ Original implementation produces values 100x larger!")
        print("This explains why player quality is being ignored.")


if __name__ == "__main__":
    test_exact_xdelta()