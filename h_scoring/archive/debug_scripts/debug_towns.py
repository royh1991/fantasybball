"""
Investigate why the H-scoring algorithm doesn't like Karl-Anthony Towns.

KAT is typically a high-value fantasy player due to his unique skill set:
- Elite scoring
- Good rebounding
- 3-point shooting (rare for bigs)
- Good percentages
- Some blocks

Let's see what the algorithm sees.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_final import HScoreOptimizerFinal


def main():
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
    optimizer = HScoreOptimizerFinal(setup_params, scoring, omega=0.7, gamma=0.25)

    print("=" * 80)
    print("KARL-ANTHONY TOWNS ANALYSIS")
    print("=" * 80)

    # Get KAT's scores
    kat_g_scores = scoring.calculate_all_g_scores("Karl-Anthony Towns")
    print(f"\nG-Score Total: {kat_g_scores['TOTAL']:.2f}")

    # Get ADP
    try:
        adp_df = pd.read_csv('../data/fantasy_basketball_clean2.csv')
        kat_adp = adp_df[adp_df['Player'] == 'Karl-Anthony Towns']['ADP'].values[0]
        print(f"ADP: {kat_adp:.1f}")
    except:
        kat_adp = None
        print("ADP: Not found")

    # Get X-scores
    kat_x = np.array([
        scoring.calculate_x_score("Karl-Anthony Towns", cat)
        for cat in categories
    ])

    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)

    print(f"\n{'Category':<12} {'G-score':<10} {'X-score':<10} {'Baseline Wt':<14} {'Comment':<30}")
    print("-" * 80)

    for i, cat in enumerate(categories):
        g_score = kat_g_scores.get(cat, 0.0)
        x_score = kat_x[i]
        weight = baseline_weights[i]

        comment = ""
        if x_score > 3.0:
            comment = "Elite"
        elif x_score > 1.5:
            comment = "Good"
        elif x_score < -1.5:
            comment = "Weak"
        elif x_score < -3.0:
            comment = "Very weak"

        print(f"{cat:<12} {g_score:>8.2f} {x_score:>9.2f} {weight*100:>12.1f}% {comment:<30}")

    # Evaluate on pick #1
    print("\n" + "=" * 80)
    print("H-SCORE EVALUATION (Pick #1)")
    print("=" * 80)

    my_team = []
    opponent_teams = [[]]

    kat_h, kat_weights = optimizer.evaluate_player(
        "Karl-Anthony Towns", my_team, opponent_teams, picks_made=0, total_picks=13
    )

    print(f"\nH-score: {kat_h:.4f}")

    # Compare to other elite bigs
    print("\n" + "=" * 80)
    print("COMPARISON TO OTHER ELITE BIGS")
    print("=" * 80)

    comparison_players = [
        "Karl-Anthony Towns",
        "Domantas Sabonis",
        "Anthony Davis",
        "Joel Embiid",
        "Nikola Jokic"
    ]

    print(f"\n{'Player':<25} {'G-score':<10} {'H-score':<10} {'ADP':<8} {'DD':<8} {'BLK':<8} {'FG3M':<8}")
    print("-" * 80)

    for player in comparison_players:
        try:
            g_total = scoring.calculate_all_g_scores(player)['TOTAL']
            h_score, _ = optimizer.evaluate_player(
                player, my_team, opponent_teams, picks_made=0, total_picks=13
            )

            dd_x = scoring.calculate_x_score(player, 'DD')
            blk_x = scoring.calculate_x_score(player, 'BLK')
            fg3m_x = scoring.calculate_x_score(player, 'FG3M')

            try:
                player_adp = adp_df[adp_df['Player'] == player]['ADP'].values[0]
                adp_str = f"{player_adp:.1f}"
            except:
                adp_str = "?"

            marker = ""
            if player == "Karl-Anthony Towns":
                marker = " ← KAT"

            print(f"{player:<25} {g_total:>8.2f} {h_score:>9.4f} {adp_str:>6} {dd_x:>7.2f} {blk_x:>7.2f} {fg3m_x:>7.2f}{marker}")
        except Exception as e:
            print(f"{player:<25} ERROR: {e}")

    # Calculate X_delta for KAT
    print("\n" + "=" * 80)
    print("X_DELTA ANALYSIS")
    print("=" * 80)

    current_team_x = np.zeros(len(categories))
    opponent_x = np.zeros(len(categories))

    x_delta_kat = optimizer.calculate_x_delta(
        weights=baseline_weights,
        n_remaining=12,
        candidate_x=kat_x,
        current_team_x=current_team_x
    )

    # Team projection
    team_proj_kat = current_team_x + kat_x + x_delta_kat

    # Win probabilities
    win_probs_kat = optimizer.calculate_win_probabilities(team_proj_kat, opponent_x)

    print(f"\n{'Category':<12} {'KAT X':<10} {'X_delta':<10} {'Total':<10} {'Win Prob':<10} {'Weight':<10} {'Contrib':<10}")
    print("-" * 85)

    total_contrib = 0
    for i, cat in enumerate(categories):
        contrib = baseline_weights[i] * win_probs_kat[i]
        total_contrib += contrib

        marker = ""
        if contrib > 0.10:
            marker = " ← High value"
        elif contrib < 0.02:
            marker = " ← Low value"

        print(f"{cat:<12} {kat_x[i]:>8.2f} {x_delta_kat[i]:>9.2f} {team_proj_kat[i]:>9.2f} {win_probs_kat[i]:>9.1%} {baseline_weights[i]*100:>8.1f}% {contrib:>9.4f}{marker}")

    print("-" * 85)
    print(f"{'TOTAL':<12} {'':>8} {'':>9} {'':>9} {np.sum(win_probs_kat):>9.2f} {'100.0%':>8} {total_contrib:>9.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Find KAT's weaknesses
    weak_cats = []
    for i, cat in enumerate(categories):
        if kat_x[i] < -1.0:
            weak_cats.append((cat, kat_x[i], baseline_weights[i]))

    if weak_cats:
        print("\nKAT's weaknesses in high-weight categories:")
        for cat, x_score, weight in sorted(weak_cats, key=lambda x: x[2], reverse=True):
            print(f"  {cat:<12} X-score: {x_score:>6.2f} (weight: {weight*100:.1f}%)")

    # Find KAT's strengths
    strong_cats = []
    for i, cat in enumerate(categories):
        if kat_x[i] > 1.5:
            strong_cats.append((cat, kat_x[i], baseline_weights[i]))

    if strong_cats:
        print("\nKAT's strengths:")
        for cat, x_score, weight in sorted(strong_cats, key=lambda x: x[1], reverse=True):
            contrib = baseline_weights[categories.index(cat)] * win_probs_kat[categories.index(cat)]
            print(f"  {cat:<12} X-score: {x_score:>6.2f} (weight: {weight*100:.1f}%, contrib: {contrib:.4f})")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    dd_idx = categories.index('DD')
    blk_idx = categories.index('BLK')

    print(f"\n1. DD Performance:")
    print(f"   - KAT DD X-score: {kat_x[dd_idx]:.2f}")
    print(f"   - DD weight: {baseline_weights[dd_idx]*100:.1f}% (highest)")
    print(f"   - DD win prob: {win_probs_kat[dd_idx]:.1%}")
    print(f"   - DD contribution: {baseline_weights[dd_idx] * win_probs_kat[dd_idx]:.4f}")

    sabonis_dd = scoring.calculate_x_score("Domantas Sabonis", 'DD')
    print(f"\n   Compare to Sabonis DD: {sabonis_dd:.2f}")
    print(f"   Difference: {sabonis_dd - kat_x[dd_idx]:.2f}")

    print(f"\n2. BLK Performance:")
    print(f"   - KAT BLK X-score: {kat_x[blk_idx]:.2f}")
    print(f"   - BLK weight: {baseline_weights[blk_idx]*100:.1f}% (2nd highest)")
    print(f"   - BLK win prob: {win_probs_kat[blk_idx]:.1%}")
    print(f"   - BLK contribution: {baseline_weights[blk_idx] * win_probs_kat[blk_idx]:.4f}")

    print("\n3. Overall:")
    print(f"   - Total weighted H-score: {total_contrib:.4f}")
    print(f"   - Total categories expected to win: {np.sum(win_probs_kat):.2f} / 11")


if __name__ == "__main__":
    main()
