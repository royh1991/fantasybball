"""
Collect full NBA dataset for H-scoring.

This script collects data for the top 200 players.
Takes approximately 5-10 minutes to run.
"""

from modules.data_collector import NBADataCollector
import sys


def collect_full_data(max_players=200, seasons=['2023-24'], resume=False, checkpoint_file=None, target_players_file=None):
    """Collect full dataset."""

    print("=" * 70)
    print("COLLECTING FULL NBA DATASET" + (" (RESUME MODE)" if resume else ""))
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  - Seasons: {', '.join(seasons)}")
    print(f"  - Max players: {max_players}")
    print(f"  - Min weeks required: 20")
    print(f"  - Resume: {resume}")
    if target_players_file:
        print(f"  - Target players: {target_players_file}")
    if not resume:
        print(f"\nEstimated time: {max_players * 0.6 / 60:.1f} minutes")
    print("\nStarting collection...\n")

    # Initialize collector
    collector = NBADataCollector(
        seasons=seasons,
        data_dir='data',
        checkpoint_file=checkpoint_file,
        name_mapping_file='player_name_mappings.json'
    )

    # Collect data
    league_weekly_data, league_game_data, player_variances = collector.collect_league_data(
        min_weeks=20,
        max_players=max_players,
        resume=resume,
        target_players_file=target_players_file
    )

    # Save data
    game_file, weekly_file, variance_file = collector.save_data(
        league_weekly_data, league_game_data, player_variances
    )

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Successfully collected data for {len(player_variances)} players")
    print(f"✓ Total game-level observations: {len(league_game_data)}")
    print(f"✓ Total weekly observations: {len(league_weekly_data)}")
    print(f"\n📁 Files saved:")
    print(f"   - {game_file} (game-level)")
    print(f"   - {weekly_file} (weekly aggregated)")
    print(f"   - {variance_file} (per-game variances)")

    return game_file, weekly_file, variance_file


if __name__ == "__main__":
    # Parse command line arguments
    max_players = 600  # Default to 600 players
    seasons = ['2022-23', '2023-24', '2024-25']  # NBA API format: 'YYYY-YY'
    resume = False
    checkpoint_file = None
    target_players_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    if len(sys.argv) > 1:
        if sys.argv[1] == '--resume':
            resume = True
            # Use most recent checkpoint file
            import glob
            checkpoints = sorted(glob.glob('data/.temp/checkpoint_*.json'))
            if checkpoints:
                checkpoint_file = checkpoints[-1]
                print(f"Using checkpoint: {checkpoint_file}\n")
            else:
                print("No checkpoint file found. Starting fresh.\n")
                resume = False
        else:
            max_players = int(sys.argv[1])

    if len(sys.argv) > 2 and not resume:
        seasons = sys.argv[2].split(',')

    try:
        game_file, weekly_file, variance_file = collect_full_data(
            max_players, seasons, resume=resume, checkpoint_file=checkpoint_file,
            target_players_file=target_players_file
        )

        print("\n" + "=" * 70)
        print("READY FOR DRAFT")
        print("=" * 70)
        print("\nYou can now run the draft assistant:")
        print(f"  python draft_assistant.py")
        print("\nOr use the data programmatically:")
        print(f"""
from draft_assistant import DraftAssistant

assistant = DraftAssistant(
    data_file='{weekly_file}',  # Use weekly data for H-scoring
    variance_file='{variance_file}'
)
recommendations = assistant.recommend_pick(top_n=10)

# Game-level data is available at: {game_file}
""")

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("Partial data may be saved. Run again to continue.")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)