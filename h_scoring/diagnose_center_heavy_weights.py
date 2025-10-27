"""
Diagnose what weights the H-scoring optimizer is calculating after 2 centers.

This script will show:
1. What weights the optimizer is finding
2. What those weights mean (which categories are prioritized)
3. Why it's recommending centers vs guards
4. Whether the optimizer is working correctly
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os
import json


def analyze_optimizer_weights():
    """Analyze what the optimizer does after 2 center picks."""

    # Find data files
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    # Initialize
    print("Initializing H-scoring system...")
    assistant = DraftAssistant(
        data_file=data_file,
        variance_file=variance_file,
        format='each_category'
    )

    # Simulate the 2-center draft
    print("\n" + "="*80)
    print("SIMULATING 2-CENTER DRAFT")
    print("="*80)

    centers = ["Anthony Davis", "Chet Holmgren"]

    for i, player in enumerate(centers, 1):
        assistant.draft_player(player)
        print(f"Pick {i}: {player}")

    # Get baseline weights (starting point)
    print("\n" + "="*80)
    print("BASELINE WEIGHTS (Starting Point)")
    print("="*80)
    print("\nThese are the 'scarcity-based' weights the optimizer starts from.")
    print("They represent category importance without any team context.\n")

    baseline = assistant.optimizer.baseline_weights
    categories = assistant.optimizer.categories

    baseline_df = pd.DataFrame({
        'Category': categories,
        'Weight': baseline
    }).sort_values('Weight', ascending=False)

    for _, row in baseline_df.iterrows():
        print(f"  {row['Category']:10s}: {row['Weight']:.4f}")

    # Analyze 3 candidate players in detail
    print("\n\n" + "="*80)
    print("DETAILED WEIGHT ANALYSIS FOR 3 CANDIDATES")
    print("="*80)

    candidates = [
        ("Nikola Vucevic", "Center (H-score ranked #1)"),
        ("LaMelo Ball", "Guard (H-score ranked #9)"),
        ("De'Aaron Fox", "Guard (H-score ranked #15)")
    ]

    for player_name, description in candidates:
        print("\n" + "-"*80)
        print(f"EVALUATING: {player_name} - {description}")
        print("-"*80)

        # Get player's X-scores
        print(f"\n{player_name}'s X-Scores (statistical contributions):")
        player_x = {}
        for cat in categories:
            x = assistant.scoring.calculate_x_score(player_name, cat)
            player_x[cat] = x
            if abs(x) > 0.1:
                print(f"  {cat:10s}: {x:6.2f}")

        # Get optimal weights for this player
        print(f"\nCalculating optimal weights for drafting {player_name}...")

        # Call the optimizer
        h_score, optimal_weights = assistant.optimizer.evaluate_player(
            player_name,
            assistant.my_team,
            assistant.opponent_rosters,
            picks_made=2,
            total_picks=13,
            last_weights=None,
            format='each_category'
        )

        print(f"\n✓ H-Score: {h_score:.4f}")

        # Show optimal weights
        print(f"\nOptimal weights found by optimizer:")
        print("(These show what categories the optimizer is prioritizing)\n")

        weights_df = pd.DataFrame({
            'Category': categories,
            'Weight': optimal_weights,
            'vs_Baseline': optimal_weights - baseline
        }).sort_values('Weight', ascending=False)

        for _, row in weights_df.iterrows():
            change = row['vs_Baseline']
            change_str = f"({change:+.4f})" if abs(change) > 0.001 else ""
            print(f"  {row['Category']:10s}: {row['Weight']:.4f} {change_str}")

        # Calculate current team stats with this player
        print(f"\nTeam X-Scores if we draft {player_name}:")
        team_x = {}
        for cat in categories:
            total = 0.0
            for teammate in assistant.my_team:
                total += assistant.scoring.calculate_x_score(teammate, cat)
            total += player_x[cat]
            team_x[cat] = total

        sorted_team = sorted(team_x.items(), key=lambda x: x[1], reverse=True)
        for cat, score in sorted_team:
            strength = "★★★" if score > 4 else "★★" if score > 2 else "★" if score > 0 else "✗" if score > -2 else "✗✗"
            print(f"  {cat:10s}: {score:6.2f}  {strength}")

        # Calculate win probabilities
        print(f"\nWin probabilities by category:")

        # Build team vector
        team_x_vector = np.array([team_x[cat] for cat in categories])

        # Get opponent average
        opponent_x = assistant.optimizer.avg_gscore_profile

        # Calculate win probs
        win_probs = assistant.optimizer.calculate_win_probabilities(team_x_vector, opponent_x)

        total_expected_wins = sum(win_probs)

        for i, cat in enumerate(categories):
            prob = win_probs[i]
            prob_pct = prob * 100
            status = "✓" if prob > 0.6 else "~" if prob > 0.4 else "✗"
            print(f"  {cat:10s}: {prob_pct:5.1f}% {status}")

        print(f"\nExpected categories won: {total_expected_wins:.2f} / 11")
        print(f"(Need 6 to win matchup)")

    # Compare the weight differences
    print("\n\n" + "="*80)
    print("WEIGHT COMPARISON ANALYSIS")
    print("="*80)

    print("\nKey Questions:")
    print("-" * 80)

    print("\n1. Are the optimal weights DIFFERENT for each player?")
    print("   → If yes: optimizer is adapting to each player's fit")
    print("   → If no: optimizer is stuck/not working")

    print("\n2. After 2 centers (weak AST/3PM), are AST/3PM weights HIGH?")
    print("   → If yes: optimizer recognizes team needs")
    print("   → If no: optimizer not adapting to weaknesses")

    print("\n3. After 2 centers (strong BLK/REB), are BLK/REB weights LOW?")
    print("   → If yes: optimizer recognizes diminishing returns")
    print("   → If no: optimizer keeps stacking strengths")

    print("\n4. Why is Vucevic (center) ranked higher than LaMelo (guard)?")
    print("   → Check if Vucevic's weights favor his strengths")
    print("   → Check if LaMelo's AST contribution is properly valued")

    print("\n5. Is regularization preventing weight changes?")
    print("   → At pick 2/13, regularization strength = {:.4f}".format(
        2.0 * (1.0 - 2/13) ** 2
    ))
    print("   → This pulls weights back toward baseline")
    print("   → May prevent adapting to extreme team compositions")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nReview the weights above to understand what the optimizer is doing.")
    print("Look for:")
    print("  - Are weights shifting away from baseline?")
    print("  - Do weak categories (AST, 3PM) get high weights?")
    print("  - Do strong categories (BLK, REB) get low weights?")
    print("  - Or are all weights staying close to baseline?")


if __name__ == "__main__":
    analyze_optimizer_weights()
