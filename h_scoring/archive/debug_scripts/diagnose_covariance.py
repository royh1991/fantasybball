#!/usr/bin/env python
"""
Diagnose covariance matrix scale issues.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator


def diagnose_covariance():
    """Check if covariance matrix is properly scaled."""
    print("=" * 80)
    print("COVARIANCE MATRIX DIAGNOSTICS")
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

    Sigma = setup_params['covariance_matrix']
    categories = setup_params['categories']

    print("\n1. COVARIANCE MATRIX PROPERTIES")
    print("-" * 40)
    print(f"Shape: {Sigma.shape}")
    print(f"Diagonal values (should be O(0.1-2) for X-scores):")
    diag = np.diag(Sigma)
    for i, cat in enumerate(categories):
        print(f"  {cat:8s}: {diag[i]:10.4f}")

    print(f"\nDiagonal stats:")
    print(f"  Min: {np.min(diag):.4f}")
    print(f"  Max: {np.max(diag):.4f}")
    print(f"  Mean: {np.mean(diag):.4f}")
    print(f"  Std: {np.std(diag):.4f}")

    # Check eigenvalues
    eigenvalues = np.linalg.eigvals(Sigma)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    print(f"\n2. EIGENVALUE ANALYSIS")
    print("-" * 40)
    print(f"Top 5 eigenvalues:")
    for i in range(min(5, len(eigenvalues))):
        print(f"  λ_{i+1}: {eigenvalues[i]:.4f}")

    print(f"\nEigenvalue stats:")
    print(f"  Max: {np.max(eigenvalues):.4f}")
    print(f"  Min: {np.min(eigenvalues):.4f}")
    print(f"  Condition number: {np.max(eigenvalues)/np.min(eigenvalues):.2e}")

    # Check X-score distributions
    print(f"\n3. X-SCORE DISTRIBUTIONS")
    print("-" * 40)
    print("Checking X-score ranges for sample players...")

    test_players = ['Nikola Jokić', 'Anthony Edwards', 'Franz Wagner']
    for player in test_players:
        if player in league_data['PLAYER_NAME'].values:
            x_scores = [scoring.calculate_x_score(player, cat) for cat in categories]
            print(f"\n{player}:")
            print(f"  X-score range: [{min(x_scores):.2f}, {max(x_scores):.2f}]")
            print(f"  X-score norm: {np.linalg.norm(x_scores):.2f}")

    # Test with identity matrix
    print(f"\n4. IDENTITY MATRIX TEST")
    print("-" * 40)
    print("Testing X_delta with identity covariance...")

    from modules.h_optimizer import HScoreOptimizer

    # Create optimizer with identity covariance
    setup_params_test = setup_params.copy()
    setup_params_test['covariance_matrix'] = np.eye(len(categories))

    optimizer_identity = HScoreOptimizer(setup_params_test, scoring)

    # Test X_delta calculation
    weights = optimizer_identity.baseline_weights.copy()
    N, K = 13, 0  # First pick

    x_delta_identity, diag = optimizer_identity._compute_xdelta_exact(
        jC=weights,
        v=optimizer_identity.v_vector,
        Sigma=np.eye(len(categories)),
        gamma=0.25,
        omega=0.7,
        N=N,
        K=K
    )

    print(f"With identity Σ:")
    print(f"  norm(xdelta_unit): {diag['norm_xdelta_unit']:.4f}")
    print(f"  norm(xdelta): {diag['norm_xdelta']:.4f}")
    print(f"  sigma: {diag['sigma']:.4f}")
    print(f"  multiplier: {diag['multiplier']:.0f}")

    # Compare with actual covariance
    x_delta_actual, diag_actual = optimizer_identity._compute_xdelta_exact(
        jC=weights,
        v=optimizer_identity.v_vector,
        Sigma=Sigma,
        gamma=0.25,
        omega=0.7,
        N=N,
        K=K
    )

    print(f"\nWith actual Σ:")
    print(f"  norm(xdelta_unit): {diag_actual['norm_xdelta_unit']:.4f}")
    print(f"  norm(xdelta): {diag_actual['norm_xdelta']:.4f}")
    print(f"  sigma: {diag_actual['sigma']:.4f}")
    print(f"  multiplier: {diag_actual['multiplier']:.0f}")

    print(f"\n5. DIAGNOSIS")
    print("-" * 40)

    if np.max(diag) > 10:
        print("⚠️ Covariance diagonal values are too large (>10)")
        print("   This suggests X-scores are not properly normalized")
    elif np.max(diag) < 0.01:
        print("⚠️ Covariance diagonal values are too small (<0.01)")
        print("   This suggests over-normalization")
    else:
        print("✓ Covariance diagonal values are in reasonable range")

    if np.max(eigenvalues) > 100:
        print("⚠️ Maximum eigenvalue is very large (>100)")
        print("   This will amplify X_delta calculations")
    else:
        print("✓ Eigenvalues are in reasonable range")

    if diag_actual['norm_xdelta'] > 20:
        print("⚠️ X_delta norm is too large (>20)")
        print("   This will dominate player contributions")
    else:
        print("✓ X_delta norm is reasonable")


if __name__ == "__main__":
    diagnose_covariance()