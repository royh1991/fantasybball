"""
Test what happens to X_delta across multiple picks.

Does the algorithm actually draft DD players after passing on Sabonis?
Or does it keep promising future DD picks while never actually taking them?
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def simulate_picks(optimizer, scoring, scenario_name, first_picks):
    """
    Simulate multiple picks and track X_delta evolution.

    Parameters:
    -----------
    optimizer : HScoreOptimizerPaperFaithful
    scoring : PlayerScoring
    scenario_name : str
    first_picks : list
        Players to draft in order
    """

    categories = optimizer.categories
    baseline_weights = optimizer.baseline_weights
    dd_idx = categories.index('DD')

    print("=" * 80)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 80)

    my_team = []

    for pick_num, player_name in enumerate(first_picks, 1):
        print(f"\n{'='*80}")
        print(f"PICK #{pick_num}: Drafting {player_name}")
        print(f"{'='*80}")

        # Calculate X_delta BEFORE drafting this player
        n_remaining = 13 - (pick_num - 1) - 1

        # Use baseline weights for X_delta calculation
        x_delta = optimizer._compute_xdelta_simplified(
            jC=baseline_weights,
            v=optimizer.v_vector,
            Sigma=optimizer.cov_matrix_original,
            gamma=optimizer.gamma,
            omega=optimizer.omega,
            N=13,
            K=pick_num - 1
        )

        dd_xdelta = x_delta[dd_idx]

        # Get player's DD X-score
        player_dd = scoring.calculate_x_score(player_name, 'DD')

        # Current team DD
        team_dd = sum([scoring.calculate_x_score(p, 'DD') for p in my_team])

        # Total projection
        total_dd = team_dd + player_dd + dd_xdelta

        print(f"\nDD breakdown:")
        print(f"  Current team DD:     {team_dd:>8.2f}")
        print(f"  {player_name:<20} {player_dd:>8.2f}")
        print(f"  X_delta ({n_remaining} picks):   {dd_xdelta:>8.2f}")
        print(f"  Total projection:    {total_dd:>8.2f}")

        # Draft the player
        my_team.append(player_name)

        # Show cumulative team DD
        new_team_dd = sum([scoring.calculate_x_score(p, 'DD') for p in my_team])
        print(f"\nAfter drafting {player_name}:")
        print(f"  Team DD (cumulative): {new_team_dd:.2f}")

    return my_team


def compare_scenarios(optimizer, scoring):
    """Compare two scenarios: drafting Sabonis vs drafting KD first."""

    categories = optimizer.categories
    dd_idx = categories.index('DD')

    print("\n" + "=" * 80)
    print("COMPARING TWO DRAFT STRATEGIES")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STRATEGY A: Draft KD (balanced player) on pick #1")
    print("=" * 80)

    # Simulate drafting KD first
    kd_team = simulate_picks(
        optimizer,
        scoring,
        "Draft KD First",
        ["Kevin Durant"]
    )

    print("\n" + "=" * 80)
    print("STRATEGY B: Draft Sabonis (DD specialist) on pick #1")
    print("=" * 80)

    # Simulate drafting Sabonis first
    sabonis_team = simulate_picks(
        optimizer,
        scoring,
        "Draft Sabonis First",
        ["Domantas Sabonis"]
    )

    print("\n" + "=" * 80)
    print("THE PARADOX")
    print("=" * 80)

    print("\nOn PICK #1 (before drafting anyone):")
    print("  X_delta promises: 5.57 DD from future picks")
    print("")
    print("  This makes the algorithm think:")
    print("    'I'll get DD later, so Sabonis (7.50 DD) isn't that special'")
    print("    'KD (balanced) is better value'")

    print("\nBut the TRUTH is:")
    print("  - X_delta is calculated assuming HIGH DD weight (22.6%)")
    print("  - That weight is only high BECAUSE DD is scarce!")
    print("  - If you pass on Sabonis (elite DD), DD stays scarce")
    print("  - So the algorithm SHOULD draft DD players next")

    print("\nLet's check: What are the top H-score picks on pick #2?")

    # After drafting KD, what does the algorithm recommend next?
    print("\n" + "-" * 80)
    print("After drafting KD, top 5 H-score recommendations:")
    print("-" * 80)

    candidates = [
        "Domantas Sabonis",
        "Nikola Jokic",
        "Anthony Davis",
        "Karl-Anthony Towns",
        "Bam Adebayo",
        "Rudy Gobert",
        "Stephen Curry",
        "James Harden"
    ]

    results = []
    for candidate in candidates:
        h_score, _ = optimizer.evaluate_player(
            candidate,
            my_team=["Kevin Durant"],
            opponent_teams=[],
            picks_made=1,
            total_picks=13,
            last_weights=None,
            format='each_category'
        )

        dd_x = scoring.calculate_x_score(candidate, 'DD')
        g_score = scoring.calculate_all_g_scores(candidate)['TOTAL']

        results.append({
            'player': candidate,
            'h_score': h_score,
            'g_score': g_score,
            'dd_x': dd_x
        })

    results.sort(key=lambda x: x['h_score'], reverse=True)

    print(f"\n{'Rank':<6} {'Player':<25} {'H-score':<10} {'G-score':<10} {'DD X-score':<12}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r['player']:<25} {r['h_score']:<10.4f} {r['g_score']:<10.2f} {r['dd_x']:<12.2f}")

    # Count how many DD specialists in top 5
    top5 = results[:5]
    dd_specialists = [r for r in top5 if r['dd_x'] > 2.0]

    print(f"\n{'='*80}")
    print("ANSWER TO YOUR QUESTION")
    print("=" * 80)

    if len(dd_specialists) >= 2:
        print(f"\n✓ YES! After passing on Sabonis and taking KD:")
        print(f"  The algorithm recommends {len(dd_specialists)} DD specialists in top 5")
        print(f"  It's trying to fulfill the X_delta promise of future DD picks")
        print("\nSo the logic is:")
        print("  Pick #1: Pass on Sabonis (assuming you'll get DD later)")
        print("  Pick #2: NOW draft DD players")
    else:
        print(f"\n✗ NO! After passing on Sabonis and taking KD:")
        print(f"  The algorithm only recommends {len(dd_specialists)} DD specialists in top 5")
        print(f"  It's NOT actually fulfilling its X_delta promise!")
        print("\nThis is the BUG:")
        print("  Pick #1: 'I'll get 5.57 DD later, so I don't need Sabonis'")
        print("  Pick #2: 'I'll get DD later, so I don't need these DD guys either'")
        print("  Pick #3: 'I'll get DD later...'")
        print("  ...")
        print("  Pick #13: 'Oops, I never drafted DD players!'")


def main():
    """Run the multi-pick simulation."""

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

    compare_scenarios(optimizer, scoring)


if __name__ == "__main__":
    main()
