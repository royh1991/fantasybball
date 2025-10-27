"""
Test the projection-aware scoring system.

Shows impact on:
1. Cade Cunningham (young improving player)
2. Injury-prone veterans
3. Comparison with original system
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.scoring_with_projections import ProjectionAwareScoringSystem
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_final import HScoreOptimizerFinal


def main():
    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])
    projections_file = '../data/fantasy_basketball_clean2.csv'

    league_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    print("=" * 100)
    print("PROJECTION-AWARE SCORING SYSTEM TEST")
    print("=" * 100)

    # Initialize both systems
    print("\nInitializing scoring systems...")
    scoring_original = PlayerScoring(league_data, player_variances, roster_size=13)
    scoring_projection = ProjectionAwareScoringSystem(
        league_data, player_variances, roster_size=13,
        projections_file=projections_file,
        projection_weight=0.5,  # Base weight (adjusted per player)
        injury_penalty_strength=1.0  # Moderate injury penalty
    )

    # Test players
    test_players = [
        "Cade Cunningham",      # Young improving player (should benefit from projections)
        "Joel Embiid",          # Injury-prone superstar
        "Kawhi Leonard",        # Injury-prone vet
        "Paul George",          # Injury-prone vet
        "Domantas Sabonis",     # Durable, established
        "Karl-Anthony Towns",   # Established
        "LeBron James",         # Old but durable
    ]

    print("\n" + "=" * 100)
    print("PLAYER PROFILES")
    print("=" * 100)

    for player in test_players:
        info = scoring_projection.get_player_info(player)

        print(f"\n{player}:")
        print(f"  Experience: {info['experience_seasons']} seasons in dataset")
        print(f"  Projection weight: {info['projection_weight']*100:.0f}% (history: {(1-info['projection_weight'])*100:.0f}%)")

        injury_risk = info['injury_risk']
        if injury_risk:
            gp = injury_risk['gp_projected']
            avail = injury_risk['availability_pct']
            risk = injury_risk['risk_factor']
            print(f"  GP projected: {gp:.0f}/82 ({avail*100:.1f}% availability)")
            print(f"  Injury risk factor: {risk:.2f}x {'⚠️ HIGH RISK' if risk < 0.85 else '✓ Low risk' if risk > 0.95 else ''}")

    # Compare X-scores
    print("\n" + "=" * 100)
    print("X-SCORE COMPARISON: ORIGINAL vs PROJECTION-AWARE")
    print("=" * 100)

    categories = scoring_original.all_cats

    for player in test_players:
        print(f"\n{player}:")
        print(f"  {'Category':<12} {'Original':<10} {'Proj-Aware':<12} {'Change':<10} {'Impact':<15}")
        print("-" * 70)

        total_change = 0
        significant_changes = []

        for cat in categories:
            x_orig = scoring_original.calculate_x_score(player, cat)
            x_proj = scoring_projection.calculate_x_score(player, cat)
            change = x_proj - x_orig

            total_change += abs(change)

            marker = ""
            if abs(change) > 0.3:
                significant_changes.append((cat, change))
                marker = " ← BIG CHANGE"

            print(f"  {cat:<12} {x_orig:>8.2f} {x_proj:>10.2f} {change:>+9.2f} {marker}")

        print(f"\n  Total absolute change: {total_change:.2f}")

        if significant_changes:
            print(f"  Biggest changes:")
            for cat, change in sorted(significant_changes, key=lambda x: abs(x[1]), reverse=True)[:3]:
                print(f"    {cat}: {change:+.2f}")

    # Compare H-scores
    print("\n" + "=" * 100)
    print("H-SCORE COMPARISON")
    print("=" * 100)

    # Original system
    cov_calc_orig = CovarianceCalculator(league_data, scoring_original)
    setup_params_orig = cov_calc_orig.get_setup_params()
    optimizer_orig = HScoreOptimizerFinal(setup_params_orig, scoring_original, omega=0.7, gamma=0.25)

    # Projection-aware system
    cov_calc_proj = CovarianceCalculator(league_data, scoring_projection)
    setup_params_proj = cov_calc_proj.get_setup_params()
    optimizer_proj = HScoreOptimizerFinal(setup_params_proj, scoring_projection, omega=0.7, gamma=0.25)

    my_team = []
    opponent_teams = [[]]

    results = []

    for player in test_players:
        try:
            # Original H-score
            h_orig, _ = optimizer_orig.evaluate_player(
                player, my_team, opponent_teams, picks_made=0, total_picks=13
            )

            # Projection-aware H-score
            h_proj, _ = optimizer_proj.evaluate_player(
                player, my_team, opponent_teams, picks_made=0, total_picks=13
            )

            # G-scores
            g_orig = sum(scoring_original.calculate_all_g_scores(player).values())

            results.append({
                'player': player,
                'h_orig': h_orig,
                'h_proj': h_proj,
                'h_change': h_proj - h_orig,
                'g_orig': g_orig
            })
        except Exception as e:
            print(f"Error for {player}: {e}")

    # Sort by projection-aware H-score
    results.sort(key=lambda x: x['h_proj'], reverse=True)

    print(f"\n{'Rank':<6} {'Player':<25} {'H-orig':<10} {'H-proj':<10} {'Change':<10} {'G-score':<10}")
    print("-" * 85)

    for i, r in enumerate(results, 1):
        change = r['h_change']
        marker = ""
        if 'Cade' in r['player']:
            marker = " ← Cade (young improving)"
        elif abs(change) > 0.01:
            marker = " ← Big change" if change > 0 else " ← Big drop"

        print(f"{i:<6} {r['player']:<25} {r['h_orig']:>8.4f} {r['h_proj']:>8.4f} {change:>+9.4f} {r['g_orig']:>9.2f}{marker}")

    # Focus on Cade
    print("\n" + "=" * 100)
    print("CADE CUNNINGHAM DEEP DIVE")
    print("=" * 100)

    cade_info = scoring_projection.get_player_info("Cade Cunningham")

    print(f"\nExperience: {cade_info['experience_seasons']} seasons → {cade_info['projection_weight']*100:.0f}% projection weight")

    cade_injury = cade_info['injury_risk']
    if cade_injury:
        print(f"Injury risk: {cade_injury['gp_projected']:.0f} GP → {cade_injury['risk_factor']:.2f}x adjustment")

    print("\nCategory-by-category impact:")
    print(f"  {'Category':<12} {'Historical':<12} {'Blended':<12} {'Change':<10} {'X-score':<10}")
    print("-" * 70)

    for cat in categories:
        cat_info = cade_info['categories'].get(cat, {})
        hist = cat_info.get('historical_mean')
        blend = cat_info.get('blended_mean')
        x = cat_info.get('x_score')
        diff = cat_info.get('difference')

        if hist is not None and blend is not None:
            marker = " ← Improved" if diff > 0 else " ← Declined" if diff < 0 else ""
            print(f"  {cat:<12} {hist:>10.2f} {blend:>10.2f} {diff:>+9.2f} {x:>9.2f}{marker}")

    cade_result = next((r for r in results if 'Cade' in r['player']), None)
    if cade_result:
        print(f"\nImpact on draft position:")
        orig_rank = next((i for i, r in enumerate(sorted(results, key=lambda x: x['h_orig'], reverse=True), 1)
                         if 'Cade' in r['player']), "?")
        proj_rank = next((i for i, r in enumerate(results, 1) if 'Cade' in r['player']), "?")

        print(f"  Original system: #{orig_rank}")
        print(f"  Projection-aware: #{proj_rank}")
        print(f"  H-score change: {cade_result['h_change']:+.4f}")


if __name__ == "__main__":
    main()
