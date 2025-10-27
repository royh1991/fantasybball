"""
Debug the first pick (pick #6) to see if optimizer is incorrectly
flipping baseline weights before there's any team context.

The user is skeptical that:
- Baseline weights: DD=22.6%, FT_PCT=3.1%
- Optimal weights on FIRST pick: FT_PCT high, DD low

This would be a BUG if true, because there's no team context yet
to justify deviating from baseline weights.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def debug_first_pick():
    """Debug what happens on the very first pick."""

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print("=" * 80)
    print("DEBUGGING FIRST PICK (PICK #6 IN ROUND 1)")
    print("=" * 80)

    # Load
    league_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Initialize
    scoring = PlayerScoring(league_data, player_variances, roster_size=13)
    cov_calc = CovarianceCalculator(league_data, scoring)
    setup_params = cov_calc.get_setup_params()

    # Use actual category order from setup_params (DON'T hardcode!)
    categories = setup_params['categories']

    print("\n" + "=" * 80)
    print("BASELINE WEIGHTS (from CV calculation)")
    print("=" * 80)

    baseline_weights = setup_params['baseline_weights']

    baseline_df = []
    for i, cat in enumerate(categories):
        baseline_df.append({
            'category': cat,
            'baseline_weight': baseline_weights[i],
            'baseline_pct': baseline_weights[i] * 100
        })

    baseline_df = pd.DataFrame(baseline_df)
    baseline_df = baseline_df.sort_values('baseline_weight', ascending=False)

    print("\nBaseline weights (sorted):")
    print(baseline_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("SIMULATE PICKS 1-5 (OTHER TEAMS)")
    print("=" * 80)

    # Simulate first 5 picks (before your pick)
    # In a snake draft from position 6, picks 1-5 go to other teams
    adp_rankings = [
        "Nikola Jokic",      # Pick 1
        "Luka Doncic",       # Pick 2
        "Giannis Antetokounmpo",  # Pick 3
        "Shai Gilgeous-Alexander",  # Pick 4
        "Anthony Davis"      # Pick 5
    ]

    already_drafted = adp_rankings[:5]
    print(f"\nPicks 1-5 (already drafted):")
    for i, player in enumerate(already_drafted, 1):
        print(f"  {i}. {player}")

    print("\n" + "=" * 80)
    print("YOUR FIRST PICK (PICK #6)")
    print("=" * 80)
    print("\nTeam context:")
    print("  - my_team: [] (empty - no players drafted yet)")
    print("  - opponent_teams: [] (using average opponent model)")
    print("  - picks_made: 0")
    print("  - last_weights: None (no previous weights)")

    # Create optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    print("\n" + "-" * 80)
    print("Evaluating top candidates for pick #6")
    print("-" * 80)

    # Get top players by G-score (excluding already drafted)
    g_rankings = scoring.rank_players_by_g_score(top_n=50)
    available = [p for p in g_rankings['PLAYER_NAME'] if p not in already_drafted]

    print(f"\nTop 10 available players by G-score:")
    for i, p in enumerate(available[:10], 1):
        g_score = g_rankings[g_rankings['PLAYER_NAME'] == p]['TOTAL_G_SCORE'].values[0]
        print(f"  {i}. {p:<30} G-score: {g_score:.2f}")

    top_candidates = available[:5]

    results = []

    for candidate in top_candidates:
        print(f"\nCandidate: {candidate}")

        # Evaluate with EMPTY team (first pick)
        h_score, optimal_weights = optimizer.evaluate_player(
            candidate,
            my_team=[],  # EMPTY TEAM
            opponent_teams=[],
            picks_made=0,  # FIRST PICK
            total_picks=13,
            last_weights=None,  # NO PREVIOUS WEIGHTS
            format='each_category'
        )

        print(f"  H-score: {h_score:.4f}")

        # Store results
        results.append({
            'player': candidate,
            'h_score': h_score,
            'optimal_weights': optimal_weights
        })

    print("\n" + "=" * 80)
    print("WEIGHT ANALYSIS FOR TOP PICK")
    print("=" * 80)

    # Analyze the top pick's weights
    top_pick = results[0]
    top_player = top_pick['player']
    top_weights = top_pick['optimal_weights']

    print(f"\nTop recommended player: {top_player}")
    print(f"H-score: {top_pick['h_score']:.4f}")

    print("\n" + "-" * 80)
    print("BASELINE vs OPTIMAL WEIGHTS")
    print("-" * 80)

    comparison = []
    for i, cat in enumerate(categories):
        baseline_pct = baseline_weights[i] * 100
        optimal_pct = top_weights[i] * 100
        diff_pct = optimal_pct - baseline_pct

        comparison.append({
            'category': cat,
            'baseline_pct': baseline_pct,
            'optimal_pct': optimal_pct,
            'diff_pct': diff_pct
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('diff_pct', ascending=False)

    print("\nComparison (sorted by difference):")
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Calculate how much weights changed
    weight_changes = np.abs(baseline_weights - top_weights)
    avg_change = weight_changes.mean()
    max_change = weight_changes.max()

    print(f"\nAverage absolute weight change: {avg_change*100:.1f}%")
    print(f"Maximum absolute weight change: {max_change*100:.1f}%")

    # Highlight specific categories user mentioned
    ft_idx = categories.index('FT_PCT')
    dd_idx = categories.index('DD')

    ft_baseline = baseline_weights[ft_idx] * 100
    ft_optimal = top_weights[ft_idx] * 100
    ft_change = ft_optimal - ft_baseline

    dd_baseline = baseline_weights[dd_idx] * 100
    dd_optimal = top_weights[dd_idx] * 100
    dd_change = dd_optimal - dd_baseline

    print(f"\nFT_PCT:")
    print(f"  Baseline: {ft_baseline:.1f}%")
    print(f"  Optimal:  {ft_optimal:.1f}%")
    print(f"  Change:   {ft_change:+.1f}%")

    print(f"\nDD:")
    print(f"  Baseline: {dd_baseline:.1f}%")
    print(f"  Optimal:  {dd_optimal:.1f}%")
    print(f"  Change:   {dd_change:+.1f}%")

    print("\n" + "=" * 80)
    print("EXPECTED BEHAVIOR vs ACTUAL BEHAVIOR")
    print("=" * 80)

    print("\n✓ EXPECTED (correct behavior):")
    print("  - On first pick, optimal weights should be CLOSE to baseline weights")
    print("  - Small perturbations OK (within ~5% per category)")
    print("  - No dramatic flips (DD high → low, FT_PCT low → high)")
    print("  - Weights should only diverge significantly AFTER drafting players")

    print("\n✗ BUG SYMPTOMS (if weights are very different):")
    print("  - Large changes (>10% per category) on first pick")
    print("  - Complete inversion of baseline weights")
    print("  - DD drops from 22.6% to <10%")
    print("  - FT_PCT rises from 3.1% to >15%")

    # Determine if this is a bug
    if avg_change > 0.05:  # More than 5% average change
        print("\n" + "=" * 80)
        print("⚠️  POTENTIAL BUG DETECTED")
        print("=" * 80)
        print("\nWeights are changing significantly on the FIRST pick!")
        print("This suggests the optimizer is not respecting baseline weights.")
        print("\nPossible causes:")
        print("1. _perturb_weights_toward_player() is too aggressive")
        print("2. Gradient descent is diverging from initial weights")
        print("3. X_delta calculation is biasing weights even with empty team")
        print("4. Initial weights are being set incorrectly")
    else:
        print("\n" + "=" * 80)
        print("✓ WEIGHTS LOOK REASONABLE")
        print("=" * 80)
        print("\nWeights stayed close to baseline on first pick.")
        print("This is expected behavior - optimizer should only deviate")
        print("from baseline after you have team context.")

    # Show all candidates' weights to see if they're all similar
    print("\n" + "=" * 80)
    print("WEIGHT CONSISTENCY CHECK")
    print("=" * 80)
    print("\nDo all candidates get similar weights on pick #1?")
    print("(They should, since there's no team context yet)")

    for result in results:
        player = result['player']
        weights = result['optimal_weights']
        h_score = result['h_score']

        ft_pct_weight = weights[ft_idx] * 100
        dd_weight = weights[dd_idx] * 100

        print(f"\n{player}:")
        print(f"  H-score: {h_score:.4f}")
        print(f"  FT_PCT weight: {ft_pct_weight:.1f}%")
        print(f"  DD weight: {dd_weight:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    debug_first_pick()
