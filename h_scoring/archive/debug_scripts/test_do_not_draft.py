"""
Test that do_not_draft functionality works correctly.
"""

import os
from draft_assistant import DraftAssistant

def test_do_not_draft():
    """Test that do_not_draft players are excluded from recommendations."""

    # Find most recent data files
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print("=" * 80)
    print("TESTING DO_NOT_DRAFT FUNCTIONALITY")
    print("=" * 80)

    # Initialize draft assistant
    assistant = DraftAssistant(
        data_file=data_file,
        variance_file=variance_file,
        format='each_category'
    )

    print("\n" + "=" * 80)
    print("DO_NOT_DRAFT LIST:")
    print("=" * 80)
    if assistant.do_not_draft:
        for player in sorted(assistant.do_not_draft):
            print(f"  - {player}")
    else:
        print("  (No players in do_not_draft list)")

    # Get recommendations (should exclude do_not_draft players)
    print("\n" + "=" * 80)
    print("GETTING TOP 20 RECOMMENDATIONS")
    print("=" * 80)

    recommendations = assistant.recommend_pick(top_n=20)

    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)

    # Check if any do_not_draft players appear in recommendations
    recommended_players = set(recommendations['PLAYER_NAME'].tolist())
    excluded_players_found = recommended_players & assistant.do_not_draft

    if excluded_players_found:
        print("❌ FAILED: The following do_not_draft players appeared in recommendations:")
        for player in excluded_players_found:
            print(f"  - {player}")
    else:
        print("✓ PASSED: No do_not_draft players appeared in recommendations")

    # Show which do_not_draft players exist in the dataset
    all_players = set(assistant.league_data['PLAYER_NAME'].unique())
    do_not_draft_in_data = assistant.do_not_draft & all_players

    print(f"\nDo_not_draft players in dataset: {len(do_not_draft_in_data)}")
    for player in sorted(do_not_draft_in_data):
        print(f"  - {player}")

    return len(excluded_players_found) == 0


if __name__ == "__main__":
    success = test_do_not_draft()
    exit(0 if success else 1)
