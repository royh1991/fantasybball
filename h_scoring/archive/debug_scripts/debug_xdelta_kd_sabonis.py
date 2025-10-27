"""
Deep dive into X_delta calculation for KD vs Sabonis.

Why does X_delta favor KD over Sabonis on pick #1?
"""

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def analyze_xdelta_impact(optimizer, player_name, weights):
    """Analyze X_delta impact for a player."""

    categories = optimizer.categories

    # Get player X-scores
    candidate_x = np.array([
        optimizer.scoring.calculate_x_score(player_name, cat)
        for cat in categories
    ])

    # Empty team (pick #1)
    current_team_x = np.zeros(len(categories))

    # Opponent (pick #1 = 0 picks made)
    opponent_x = optimizer._calculate_average_opponent_x([], player_name, picks_made=0)

    # Calculate X_delta
    n_remaining = 13 - 0 - 1  # 12 remaining picks
    x_delta = optimizer._compute_xdelta_simplified(
        jC=weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=0
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

    h_score = sum(win_probs)

    return {
        'candidate_x': candidate_x,
        'x_delta': x_delta,
        'team_projection': team_projection,
        'opponent_x': opponent_x,
        'category_variances': category_variances,
        'win_probs': win_probs,
        'h_score': h_score
    }


def main():
    """Compare X_delta impact for KD vs Sabonis."""

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
    print("X_DELTA DEEP DIVE: KD vs SABONIS")
    print("=" * 80)

    # Analyze both players
    kd_analysis = analyze_xdelta_impact(optimizer, "Kevin Durant", baseline_weights)
    sabonis_analysis = analyze_xdelta_impact(optimizer, "Domantas Sabonis", baseline_weights)

    print("\n" + "=" * 80)
    print("STEP 1: CANDIDATE X-SCORES")
    print("=" * 80)
    print(f"\n{'Category':<12} {'Baseline%':<12} {'KD X-score':<12} {'Sabonis X':<12} {'Diff (KD-Sab)':<15}")
    print("-" * 80)

    for i, cat in enumerate(categories):
        baseline_pct = baseline_weights[i] * 100
        kd_x = kd_analysis['candidate_x'][i]
        sabonis_x = sabonis_analysis['candidate_x'][i]
        diff = kd_x - sabonis_x
        winner = "KD" if diff > 0 else "Sabonis"

        print(f"{cat:<12} {baseline_pct:>11.1f} {kd_x:>11.2f} {sabonis_x:>11.2f} {diff:>14.2f} ({winner})")

    print("\n" + "=" * 80)
    print("STEP 2: X_DELTA (Future picks adjustment)")
    print("=" * 80)
    print(f"\n{'Category':<12} {'KD X_delta':<12} {'Sabonis Δ':<12} {'Diff (KD-Sab)':<15}")
    print("-" * 80)

    for i, cat in enumerate(categories):
        kd_delta = kd_analysis['x_delta'][i]
        sabonis_delta = sabonis_analysis['x_delta'][i]
        diff = kd_delta - sabonis_delta
        winner = "KD" if diff > 0 else "Sabonis"

        print(f"{cat:<12} {kd_delta:>11.2f} {sabonis_delta:>11.2f} {diff:>14.2f} ({winner})")

    # Highlight key differences
    kd_delta_norm = np.linalg.norm(kd_analysis['x_delta'])
    sabonis_delta_norm = np.linalg.norm(sabonis_analysis['x_delta'])

    print(f"\n||X_delta|| KD:      {kd_delta_norm:.2f}")
    print(f"||X_delta|| Sabonis: {sabonis_delta_norm:.2f}")

    print("\n⚠️  KEY QUESTION: Why are X_delta values different for KD vs Sabonis?")
    print("   X_delta should depend on BASELINE WEIGHTS (which are the same),")
    print("   not on the specific candidate being evaluated!")

    print("\n" + "=" * 80)
    print("STEP 3: TEAM PROJECTION (current + candidate + x_delta)")
    print("=" * 80)
    print(f"\n{'Category':<12} {'KD Proj':<12} {'Sabonis Proj':<12} {'Diff (KD-Sab)':<15}")
    print("-" * 80)

    for i, cat in enumerate(categories):
        kd_proj = kd_analysis['team_projection'][i]
        sabonis_proj = sabonis_analysis['team_projection'][i]
        diff = kd_proj - sabonis_proj
        winner = "KD" if diff > 0 else "Sabonis"

        print(f"{cat:<12} {kd_proj:>11.2f} {sabonis_proj:>11.2f} {diff:>14.2f} ({winner})")

    print("\n" + "=" * 80)
    print("STEP 4: CATEGORY VARIANCES")
    print("=" * 80)
    print(f"\n{'Category':<12} {'Variance':<12} {'Std Dev':<12}")
    print("-" * 80)

    for i, cat in enumerate(categories):
        variance = kd_analysis['category_variances'][cat]
        std_dev = np.sqrt(variance)
        print(f"{cat:<12} {variance:>11.2f} {std_dev:>11.2f}")

    dd_variance = kd_analysis['category_variances']['DD']
    print(f"\n⚠️  DD variance = {dd_variance:.2f}")
    print(f"   High variance means DD differences have LESS impact on win probability")
    print(f"   (differences get divided by σ = {np.sqrt(dd_variance):.2f})")

    print("\n" + "=" * 80)
    print("STEP 5: WIN PROBABILITIES")
    print("=" * 80)
    print(f"\n{'Category':<12} {'Weight%':<10} {'KD P(win)':<12} {'Sab P(win)':<12} {'Diff':<12}")
    print("-" * 80)

    kd_total = 0
    sabonis_total = 0

    for i, cat in enumerate(categories):
        weight_pct = baseline_weights[i] * 100
        kd_prob = kd_analysis['win_probs'][i]
        sabonis_prob = sabonis_analysis['win_probs'][i]
        diff = kd_prob - sabonis_prob
        winner = "KD" if diff > 0 else "Sabonis"

        kd_total += kd_prob
        sabonis_total += sabonis_prob

        print(f"{cat:<12} {weight_pct:>9.1f} {kd_prob:>11.3f} {sabonis_prob:>11.3f} {diff:>11.3f} ({winner})")

    print("-" * 80)
    print(f"{'TOTAL H-score':<12} {'':<10} {kd_total:>11.3f} {sabonis_total:>11.3f} {kd_total - sabonis_total:>11.3f}")

    print("\n" + "=" * 80)
    print("WEIGHTED WIN PROBABILITY CONTRIBUTION")
    print("=" * 80)
    print(f"\nHow much does each category contribute to the H-score gap?")
    print(f"\n{'Category':<12} {'Weight%':<10} {'KD contrib':<12} {'Sab contrib':<12} {'Gap':<12}")
    print("-" * 80)

    contributions = []
    for i, cat in enumerate(categories):
        weight = baseline_weights[i]
        kd_contrib = weight * kd_analysis['win_probs'][i]
        sabonis_contrib = weight * sabonis_analysis['win_probs'][i]
        gap = kd_contrib - sabonis_contrib

        contributions.append({
            'category': cat,
            'weight': weight,
            'gap': gap
        })

        print(f"{cat:<12} {weight*100:>9.1f} {kd_contrib:>11.4f} {sabonis_contrib:>11.4f} {gap:>11.4f}")

    # Sort by absolute gap
    contributions.sort(key=lambda x: abs(x['gap']), reverse=True)

    print("\n" + "=" * 80)
    print("TOP CATEGORIES DRIVING THE GAP (sorted by impact)")
    print("=" * 80)

    print(f"\n{'Category':<12} {'Weight%':<10} {'Gap':<12} {'Favors':<12}")
    print("-" * 80)
    for item in contributions[:5]:
        cat = item['category']
        weight_pct = item['weight'] * 100
        gap = item['gap']
        favors = "KD" if gap > 0 else "Sabonis"
        print(f"{cat:<12} {weight_pct:>9.1f} {gap:>11.4f} {favors:<12}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Find the smoking gun
    dd_idx = categories.index('DD')
    fg3m_idx = categories.index('FG3M')

    dd_gap = contributions[dd_idx]['gap']
    fg3m_gap = contributions[fg3m_idx]['gap']

    print(f"\n1. DD (Double-Doubles):")
    print(f"   Weight: {baseline_weights[dd_idx]*100:.1f}% (HIGHEST)")
    print(f"   Sabonis X-score: {sabonis_analysis['candidate_x'][dd_idx]:.2f}")
    print(f"   KD X-score: {kd_analysis['candidate_x'][dd_idx]:.2f}")
    print(f"   Sabonis advantage: {sabonis_analysis['candidate_x'][dd_idx] - kd_analysis['candidate_x'][dd_idx]:.2f}")
    print(f"   BUT contribution gap: {dd_gap:+.4f} (favors {'KD' if dd_gap > 0 else 'Sabonis'})")
    print(f"   DD variance: {kd_analysis['category_variances']['DD']:.2f}")
    print(f"   → Sabonis's DD advantage is being DILUTED by high variance!")

    print(f"\n2. Why is KD winning despite lower G-score?")
    top_kd_cats = [x for x in contributions if x['gap'] > 0][:3]
    print(f"   KD's advantages:")
    for item in top_kd_cats:
        cat = item['category']
        idx = categories.index(cat)
        print(f"   - {cat}: weight {item['weight']*100:.1f}%, contributes {item['gap']:+.4f}")

    print(f"\n3. Is this a bug?")
    if dd_gap < 0 and abs(dd_gap) < 0.02:
        print(f"   ⚠️  YES! DD has 22.6% weight but only contributes {abs(dd_gap):.4f} to Sabonis.")
        print(f"   Despite Sabonis having a MASSIVE DD advantage (7.50 vs -0.16 X-score),")
        print(f"   the high DD variance ({kd_analysis['category_variances']['DD']:.2f}) is")
        print(f"   diluting this advantage in the win probability calculation.")
        print(f"\n   The category variance calculation appears to be wrong!")
    else:
        print(f"   Need more investigation...")


if __name__ == "__main__":
    main()
