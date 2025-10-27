"""
Example usage of H-scoring draft assistant.

This script demonstrates how to use the H-scoring system for a draft.
"""

from draft_assistant import DraftAssistant
import os


def example_full_draft():
    """Example of running a full draft with H-scoring."""

    print("=" * 70)
    print("H-SCORING EXAMPLE: FULL DRAFT SIMULATION")
    print("=" * 70)

    # Initialize assistant
    assistant = DraftAssistant(format='each_category')

    # Check if data exists, otherwise collect it
    data_dir = 'data'
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("\nCollecting NBA data (this may take 5-10 minutes)...")
        assistant.collect_data(
            seasons=['2023-24'],
            max_players=150  # Top 150 players
        )
    else:
        # Load existing data
        data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
        if data_files:
            latest_data = data_files[-1]
            data_file = os.path.join(data_dir, latest_data)
            variance_file = data_file.replace('league_weekly_data', 'player_variances').replace('.csv', '.json')
            assistant.load_data(data_file, variance_file)

    # Show G-score rankings for reference
    print("\n" + "=" * 70)
    print("TOP 30 PLAYERS BY G-SCORE (Static Rankings)")
    print("=" * 70)
    g_rankings = assistant.get_player_rankings(method='g_score', top_n=30)
    print(g_rankings[['RANK', 'PLAYER_NAME', 'TOTAL_G_SCORE']].to_string(index=False))

    # Simulate draft picks
    print("\n" + "=" * 70)
    print("SIMULATING DRAFT")
    print("=" * 70)

    # Example: You have the 5th pick in a 12-team league
    draft_position = 5
    num_teams = 12

    # Simulate some picks being made
    print(f"\nYou are drafting at position {draft_position} in a {num_teams}-team league")

    # Round 1 - Get recommendations for your first pick
    print("\n" + "-" * 70)
    print("ROUND 1")
    print("-" * 70)

    recommendations = assistant.recommend_pick(top_n=10)

    # Draft your first pick
    first_pick = recommendations.iloc[0]['PLAYER_NAME']
    print(f"\nDrafting: {first_pick}")
    assistant.draft_player(first_pick)

    # Simulate opponents drafting (for demonstration)
    # In real usage, you'd update this with actual draft picks
    opponent_rosters = [
        [],  # Team 1
        [],  # Team 2
        # ... etc
    ]

    # Round 2 - Snake draft, so you pick again
    print("\n" + "-" * 70)
    print("ROUND 2")
    print("-" * 70)

    recommendations = assistant.recommend_pick(top_n=10)
    second_pick = recommendations.iloc[0]['PLAYER_NAME']
    print(f"\nDrafting: {second_pick}")
    assistant.draft_player(second_pick)

    # Continue for a few more rounds...
    for round_num in range(3, 6):
        print("\n" + "-" * 70)
        print(f"ROUND {round_num}")
        print("-" * 70)

        recommendations = assistant.recommend_pick(top_n=10)
        pick = recommendations.iloc[0]['PLAYER_NAME']
        print(f"\nDrafting: {pick}")
        assistant.draft_player(pick)

    # Show final team
    assistant.show_team_summary()

    # Export results
    assistant.export_results()

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


def example_single_pick_analysis():
    """Example of analyzing a single pick decision."""

    print("=" * 70)
    print("H-SCORING EXAMPLE: SINGLE PICK ANALYSIS")
    print("=" * 70)

    # Initialize with existing data
    assistant = DraftAssistant()

    # Load data
    data_dir = 'data'
    data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    if not data_files:
        print("No data found. Run example_full_draft() first to collect data.")
        return

    latest_data = data_files[-1]
    data_file = os.path.join(data_dir, latest_data)
    variance_file = data_file.replace('league_weekly_data', 'player_variances').replace('.csv', '.json')
    assistant.load_data(data_file, variance_file)

    # Simulate having already drafted a few players
    my_team = ['Nikola Jokic', 'Anthony Edwards', 'Bam Adebayo']
    for player in my_team:
        assistant.draft_player(player)

    print("\nCurrent team:")
    assistant.show_team_summary()

    # Compare H-score vs G-score recommendations
    print("\n" + "=" * 70)
    print("NEXT PICK RECOMMENDATIONS")
    print("=" * 70)

    recommendations = assistant.recommend_pick(top_n=15)

    print("\nNotice how H-scores differ from G-scores based on your team composition!")
    print("H-scoring adapts to complement your existing players.")


def example_compare_strategies():
    """Example comparing different draft strategies."""

    print("=" * 70)
    print("H-SCORING EXAMPLE: STRATEGY COMPARISON")
    print("=" * 70)

    # Initialize
    assistant = DraftAssistant()

    # Load data
    data_dir = 'data'
    data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    if not data_files:
        print("No data found. Run example_full_draft() first to collect data.")
        return

    latest_data = data_files[-1]
    data_file = os.path.join(data_dir, latest_data)
    variance_file = data_file.replace('league_weekly_data', 'player_variances').replace('.csv', '.json')
    assistant.load_data(data_file, variance_file)

    # Scenario 1: Draft a guard-heavy team
    print("\n" + "-" * 70)
    print("SCENARIO 1: Guard-Heavy Team")
    print("-" * 70)

    guard_team = ['Stephen Curry', 'Damian Lillard', 'Trae Young']
    for player in guard_team:
        assistant.draft_player(player)

    print("\nRecommendations after drafting guards:")
    recommendations = assistant.recommend_pick(top_n=10)

    # Show how H-scoring adapts
    print("\nH-scoring will naturally recommend complementary players!")
    print("It discovers that you need bigs/defensive stats automatically.")

    # Export for analysis
    assistant.export_results('draft_guard_heavy.json')


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == 'full':
            example_full_draft()
        elif example == 'single':
            example_single_pick_analysis()
        elif example == 'compare':
            example_compare_strategies()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_usage.py [full|single|compare]")
    else:
        print("Usage: python example_usage.py [full|single|compare]")
        print("\nExamples:")
        print("  full    - Run a full draft simulation")
        print("  single  - Analyze a single pick decision")
        print("  compare - Compare different draft strategies")
        print("\nRunning 'full' example by default...\n")
        example_full_draft()