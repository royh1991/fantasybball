"""
Supplemental data collection for rookies after they start playing.

This script:
1. Checks which rookies from the mapping file are now in the NBA API
2. Collects their game/weekly data
3. Appends to existing dataset files
"""

from modules.data_collector import NBADataCollector
import pandas as pd
import json
import os
import sys


def collect_rookie_supplement(existing_weekly_file, existing_game_file, existing_variance_file):
    """
    Collect data for rookies and append to existing files.

    Parameters:
    -----------
    existing_weekly_file : str
        Path to existing weekly data CSV
    existing_game_file : str
        Path to existing game data CSV
    existing_variance_file : str
        Path to existing variance JSON
    """
    print("=" * 70)
    print("COLLECTING ROOKIE SUPPLEMENT DATA")
    print("=" * 70)

    # Load rookie names to check
    with open('player_name_mappings.json', 'r') as f:
        mapping_data = json.load(f)
        rookie_names = mapping_data.get('rookies_not_in_api', [])

    print(f"\nChecking {len(rookie_names)} rookies from mapping file...")

    # Initialize collector
    collector = NBADataCollector(
        seasons=['2024-25'],  # Current season only for rookies
        data_dir='data',
        name_mapping_file='player_name_mappings.json'
    )

    # Get all active players
    from nba_api.stats.static import players
    player_list = players.get_players()
    all_players_df = pd.DataFrame(player_list)

    # Check which rookies are now in API
    all_players_df['full_name_normalized'] = all_players_df['full_name'].apply(collector._normalize_name)
    rookie_names_normalized = [collector._normalize_name(name) for name in rookie_names]

    found_rookies = all_players_df[
        all_players_df['full_name_normalized'].isin(rookie_names_normalized) &
        all_players_df['is_active']
    ]

    if found_rookies.empty:
        print("\n❌ No rookies found in NBA API yet.")
        print("They may not have played their first NBA game yet.")
        return

    print(f"\n✓ Found {len(found_rookies)} rookie(s) now in NBA API:")
    for _, player in found_rookies.iterrows():
        print(f"  - {player['full_name']}")

    # Collect data for found rookies
    print("\nCollecting data for rookies...\n")

    all_weekly_data = []
    all_game_data = []
    player_variances = {}

    for idx, player in found_rookies.iterrows():
        player_id = player['id']
        player_name = player['full_name']

        print(f"[{idx+1}/{len(found_rookies)}] {player_name}")

        player_weekly_data = []
        player_game_data = []

        # Fetch for current season only
        games = collector.fetch_player_gamelogs(player_id, player_name, '2024-25')

        if games is not None:
            processed = collector.process_gamelogs(games)

            if processed is not None:
                processed_with_season = collector.add_nba_season_week(processed)

                if processed_with_season is not None:
                    player_game_data.append(processed_with_season)

                    weekly = collector.aggregate_to_weekly(processed_with_season)

                    if weekly is not None:
                        player_weekly_data.append(weekly)

        # Combine all seasons for this player
        if player_weekly_data and player_game_data:
            player_combined_weekly = pd.concat(player_weekly_data, ignore_index=True)
            player_combined_games = pd.concat(player_game_data, ignore_index=True)

            # Check if enough data (min 5 weeks for rookies - lower threshold)
            if len(player_combined_weekly) >= 5:
                all_weekly_data.append(player_combined_weekly)
                all_game_data.append(player_combined_games)

                # Calculate variance
                variances = collector.calculate_player_variance(
                    player_combined_games,
                    player_combined_weekly
                )
                player_variances[player_name] = variances

                print(f"  ✓ Collected {len(player_combined_weekly)} weeks, {len(player_combined_games)} games")
            else:
                print(f"  ✗ Only {len(player_combined_weekly)} weeks (need 5+)")
        else:
            print(f"  ✗ No data yet")

    if not all_weekly_data:
        print("\n❌ No rookies had sufficient data yet.")
        return

    # Combine rookie data
    rookie_weekly_data = pd.concat(all_weekly_data, ignore_index=True)
    rookie_game_data = pd.concat(all_game_data, ignore_index=True)

    print(f"\n✓ Collected data for {len(player_variances)} rookie(s)")

    # Load existing data
    print("\nLoading existing data files...")
    existing_weekly = pd.read_csv(existing_weekly_file)
    existing_game = pd.read_csv(existing_game_file)

    with open(existing_variance_file, 'r') as f:
        existing_variances = json.load(f)

    print(f"  - Existing weekly records: {len(existing_weekly)}")
    print(f"  - Existing game records: {len(existing_game)}")
    print(f"  - Existing players: {len(existing_variances)}")

    # Append rookie data
    print("\nAppending rookie data...")
    combined_weekly = pd.concat([existing_weekly, rookie_weekly_data], ignore_index=True)
    combined_game = pd.concat([existing_game, rookie_game_data], ignore_index=True)
    combined_variances = {**existing_variances, **player_variances}

    # Save updated files (create new files with timestamp)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    new_weekly_file = f'data/league_weekly_data_{timestamp}.csv'
    new_game_file = f'data/league_game_data_{timestamp}.csv'
    new_variance_file = f'data/player_variances_{timestamp}.json'

    combined_weekly.to_csv(new_weekly_file, index=False)
    combined_game.to_csv(new_game_file, index=False)

    with open(new_variance_file, 'w') as f:
        json.dump(combined_variances, f, indent=2)

    print("\n" + "=" * 70)
    print("SUPPLEMENT COMPLETE")
    print("=" * 70)
    print(f"\n✓ New weekly records: {len(combined_weekly)} (+{len(rookie_weekly_data)})")
    print(f"✓ New game records: {len(combined_game)} (+{len(rookie_game_data)})")
    print(f"✓ New player count: {len(combined_variances)} (+{len(player_variances)})")

    print(f"\n📁 Updated files saved:")
    print(f"   - {new_weekly_file}")
    print(f"   - {new_game_file}")
    print(f"   - {new_variance_file}")

    return new_game_file, new_weekly_file, new_variance_file


if __name__ == "__main__":
    # Find most recent data files
    import glob

    weekly_files = sorted(glob.glob('data/league_weekly_data_*.csv'))
    game_files = sorted(glob.glob('data/league_game_data_*.csv'))
    variance_files = sorted(glob.glob('data/player_variances_*.json'))

    if not weekly_files or not game_files or not variance_files:
        print("❌ Error: No existing data files found!")
        print("\nPlease run collect_full_data.py first to create the base dataset.")
        sys.exit(1)

    # Use most recent files
    existing_weekly = weekly_files[-1]
    existing_game = game_files[-1]
    existing_variance = variance_files[-1]

    print(f"\nUsing existing files:")
    print(f"  - {existing_weekly}")
    print(f"  - {existing_game}")
    print(f"  - {existing_variance}\n")

    try:
        collect_rookie_supplement(existing_weekly, existing_game, existing_variance)

        print("\n" + "=" * 70)
        print("READY TO USE")
        print("=" * 70)
        print("\nYou can now use the updated dataset files for drafting or analysis.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
