#!/usr/bin/env python3
"""
Create a canonical player mapping file.

This script generates a mapping between:
- ESPN player names (from roster snapshots)
- NBA API player names (from historical game logs)
- Normalized names (for matching)

The output file allows for easy joins between roster and game log data.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import unicodedata


def normalize_name(name):
    """
    Normalize name by removing accents and special characters.

    E.g., "Luka Dončić" -> "luka doncic"
    """
    # Normalize unicode (decompose accented characters)
    nfd = unicodedata.normalize('NFD', name)
    # Remove combining characters (accents)
    without_accents = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    # Lowercase and strip
    return without_accents.lower().strip()


def load_roster_players():
    """Load unique players from latest roster snapshot."""
    roster_dir = Path(__file__).parent.parent / "data" / "roster_snapshots"
    roster_file = roster_dir / "roster_latest.csv"

    if not roster_file.exists():
        print("✗ No roster snapshot found. Run step 1 first.")
        return None

    roster_df = pd.read_csv(roster_file)

    # Get unique players with their ESPN info
    roster_players = roster_df[['player_name', 'player_id_espn', 'position']].drop_duplicates()
    roster_players = roster_players.rename(columns={'player_name': 'espn_name'})

    print(f"✓ Loaded {len(roster_players)} unique players from roster")
    return roster_players


def load_gamelog_players():
    """Load unique players from historical game logs."""
    gamelogs_dir = Path(__file__).parent.parent / "data" / "historical_gamelogs"
    gamelogs_file = gamelogs_dir / "historical_gamelogs_latest.csv"

    if not gamelogs_file.exists():
        print("✗ No historical game logs found. Run step 2 first.")
        return None

    # Read just the PLAYER_NAME column to get unique names
    gamelogs_df = pd.read_csv(gamelogs_file, usecols=['PLAYER_NAME', 'Player_ID'])

    # Get unique players
    gamelog_players = gamelogs_df[['PLAYER_NAME', 'Player_ID']].drop_duplicates()
    gamelog_players = gamelog_players.rename(columns={
        'PLAYER_NAME': 'nba_api_name',
        'Player_ID': 'nba_api_id'
    })

    print(f"✓ Loaded {len(gamelog_players)} unique players from game logs")
    return gamelog_players


def create_player_mapping(roster_players, gamelog_players):
    """
    Create player mapping by matching on normalized names.

    Returns:
        DataFrame with columns:
        - espn_name
        - player_id_espn
        - nba_api_name
        - nba_api_id
        - normalized_name
        - position
        - matched (boolean)
    """
    print("\nCreating player mapping...")

    # Add normalized names
    roster_players['normalized_name'] = roster_players['espn_name'].apply(normalize_name)
    gamelog_players['normalized_name'] = gamelog_players['nba_api_name'].apply(normalize_name)

    # Merge on normalized name
    mapping = pd.merge(
        roster_players,
        gamelog_players,
        on='normalized_name',
        how='left',
        indicator=True
    )

    # Add matched flag
    mapping['matched'] = mapping['_merge'] == 'both'
    mapping = mapping.drop('_merge', axis=1)

    # Reorder columns
    mapping = mapping[[
        'espn_name',
        'player_id_espn',
        'nba_api_name',
        'nba_api_id',
        'normalized_name',
        'position',
        'matched'
    ]]

    # Sort by matched status (unmatched first for visibility) then by name
    mapping = mapping.sort_values(['matched', 'espn_name'])

    # Report
    matched_count = mapping['matched'].sum()
    unmatched_count = len(mapping) - matched_count

    print(f"\n✓ Matched {matched_count} players")
    if unmatched_count > 0:
        print(f"✗ {unmatched_count} players not matched:")
        unmatched = mapping[~mapping['matched']]
        for _, row in unmatched.iterrows():
            print(f"    - {row['espn_name']} (ESPN ID: {row['player_id_espn']})")

    return mapping


def save_player_mapping(mapping_df, output_dir):
    """Save player mapping to CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"player_mapping_{timestamp}.csv"
    filepath = output_path / filename

    # Save timestamped version
    mapping_df.to_csv(filepath, index=False)
    print(f"\n✓ Saved player mapping to: {filepath}")

    # Also save as "latest"
    latest_filepath = output_path / "player_mapping_latest.csv"
    mapping_df.to_csv(latest_filepath, index=False)
    print(f"✓ Saved as latest: {latest_filepath}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total players: {len(mapping_df)}")
    print(f"  Matched: {mapping_df['matched'].sum()}")
    print(f"  Unmatched: {(~mapping_df['matched']).sum()}")

    # Show example mappings
    print(f"\nExample mappings (first 10):")
    print(mapping_df.head(10).to_string(index=False))


def main():
    """Main execution."""
    print("="*60)
    print("PLAYER MAPPING CREATION")
    print("="*60)

    # Load roster players
    roster_players = load_roster_players()
    if roster_players is None:
        return 1

    # Load game log players
    gamelog_players = load_gamelog_players()
    if gamelog_players is None:
        return 1

    # Create mapping
    mapping = create_player_mapping(roster_players, gamelog_players)

    # Save mapping
    output_dir = Path(__file__).parent.parent / "data" / "mappings"
    save_player_mapping(mapping, output_dir)

    print("\n" + "="*60)
    print("✓ PLAYER MAPPING COMPLETE")
    print("="*60)

    print("\nUsage:")
    print("  Use this mapping to join roster data with game log data:")
    print("")
    print("  import pandas as pd")
    print("  mapping = pd.read_csv('data/mappings/player_mapping_latest.csv')")
    print("  roster = pd.read_csv('data/roster_snapshots/roster_latest.csv')")
    print("  gamelogs = pd.read_csv('data/historical_gamelogs/historical_gamelogs_latest.csv')")
    print("")
    print("  # Join roster to mapping")
    print("  roster_mapped = roster.merge(mapping, left_on='player_name', right_on='espn_name')")
    print("")
    print("  # Join to game logs")
    print("  full_data = roster_mapped.merge(gamelogs, left_on='nba_api_name', right_on='PLAYER_NAME')")

    return 0


if __name__ == '__main__':
    sys.exit(main())
