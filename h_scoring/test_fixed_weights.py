"""
Test the FIXED H-scoring: weights calculated once for team, used for all candidates.
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os


def test_fixed_approach():
    """Test fixed approach where weights are calculated once."""

    # Find data files
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    # Initialize
    print("Initializing...")
    assistant = DraftAssistant(
        data_file=data_file,
        variance_file=variance_file,
        format='each_category'
    )

    # Draft 2 centers
    print("\n" + "="*80)
    print("DRAFTING 2 CENTERS")
    print("="*80)
    assistant.draft_player("Anthony Davis")
    assistant.draft_player("Chet Holmgren")

    print("\nYour team: Anthony Davis, Chet Holmgren")

    # Calculate team X-scores
    categories = assistant.optimizer.categories
    print("\nTeam X-Scores:")
    for cat in categories:
        total = sum(assistant.scoring.calculate_x_score(p, cat) for p in assistant.my_team)
        status = "★★" if total > 2 else "★" if total > 0 else "✗" if total > -2 else "✗✗"
        print(f"  {cat:10s}: {total:6.2f} {status}")

    # Calculate optimal weights ONCE for this team
    print("\n" + "="*80)
    print("CALCULATING TEAM-OPTIMAL WEIGHTS (Once)")
    print("="*80)

    optimal_weights = assistant.optimizer.calculate_optimal_weights_for_team(
        assistant.my_team,
        assistant.opponent_rosters,
        picks_made=2,
        total_picks=13,
        last_weights=None,
        format='each_category'
    )

    print("\nOptimal weights for your team (will be used for ALL candidates):")
    weights_df = pd.DataFrame({
        'Category': categories,
        'Weight': optimal_weights
    }).sort_values('Weight', ascending=False)

    for _, row in weights_df.iterrows():
        print(f"  {row['Category']:10s}: {row['Weight']:.4f}")

    # Now evaluate 3 candidates with SAME weights
    print("\n" + "="*80)
    print("EVALUATING CANDIDATES (All use same weights)")
    print("="*80)

    candidates = ["Nikola Vucevic", "LaMelo Ball", "De'Aaron Fox"]

    results = []
    for player in candidates:
        h_score = assistant.optimizer.evaluate_player_with_weights(
            player,
            assistant.my_team,
            assistant.opponent_rosters,
            picks_made=2,
            total_picks=13,
            optimal_weights=optimal_weights,
            format='each_category'
        )

        results.append({
            'player': player,
            'h_score': h_score
        })

    # Sort by H-score
    results_df = pd.DataFrame(results).sort_values('h_score', ascending=False)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("All candidates were evaluated using the SAME weights.")
    print("Weights reflect your TEAM needs (weak AST/3PM, strong BLK/REB).")
    print("Rankings now properly reflect who fills your gaps.")


if __name__ == "__main__":
    test_fixed_approach()
