#!/usr/bin/env python3
"""
Get all players who have played in the 2025-26 season.
This creates a definitive list to use for mapping.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "espn-api"))

from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players as static_players
import pandas as pd
import time

def get_all_2025_26_players():
    """
    Get all players who have played games in 2025-26 season.
    Uses league player stats which gives us all active players.
    """
    print("Fetching all players with 2025-26 season stats from NBA API...")
    print("This may take 10-20 seconds...\n")

    try:
        time.sleep(1)  # Rate limiting

        # Get all player stats from 2025-26 season
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            season_type_all_star="Regular Season"
        )

        stats_df = player_stats.get_data_frames()[0]

        if stats_df.empty:
            print("No player stats found for 2025-26 season!")
            return None

        print(f"✓ Found {len(stats_df)} players with stats")

        # Get unique players
        unique_players = stats_df['PLAYER_NAME'].unique()
        print(f"✓ {len(unique_players)} unique players have played in 2025-26\n")

        # Sort alphabetically
        unique_players = sorted(unique_players)

        # Save to file
        output_file = Path(__file__).parent / "data" / "2025_26_active_players.txt"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            for player in unique_players:
                f.write(f"{player}\n")

        print(f"✓ Saved to: {output_file}\n")

        # Print first 50
        print("First 50 players:")
        for i, player in enumerate(unique_players[:50], 1):
            print(f"  {i:3d}. {player}")

        if len(unique_players) > 50:
            print(f"  ... and {len(unique_players) - 50} more")

        return unique_players

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_roster_against_2025_26(roster_file):
    """Check which roster players are NOT in the 2025-26 player list."""
    print("\n" + "="*60)
    print("CHECKING ROSTER AGAINST 2025-26 SEASON PLAYERS")
    print("="*60 + "\n")

    # Get 2025-26 players
    active_players_2025_26 = get_all_2025_26_players()

    if not active_players_2025_26:
        return

    # Normalize names for comparison
    def normalize(name):
        import unicodedata
        nfd = unicodedata.normalize('NFD', name)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn').lower().strip()

    active_normalized = {normalize(p): p for p in active_players_2025_26}

    # Load roster
    import pandas as pd
    roster_df = pd.read_csv(roster_file)
    roster_players = roster_df['player_name'].unique()

    print(f"\n✓ Loaded {len(roster_players)} players from roster\n")

    # Find missing players
    missing = []
    found = []

    for roster_player in roster_players:
        normalized = normalize(roster_player)
        if normalized in active_normalized:
            found.append((roster_player, active_normalized[normalized]))
        else:
            missing.append(roster_player)

    print(f"✓ Found {len(found)} players in 2025-26 data")
    print(f"✗ Missing {len(missing)} players from 2025-26 data\n")

    if missing:
        print("Players NOT found in 2025-26 season:")
        for i, player in enumerate(missing, 1):
            print(f"  {i:3d}. {player}")

        # Try to find close matches
        print("\n" + "="*60)
        print("SEARCHING FOR CLOSE MATCHES")
        print("="*60 + "\n")

        from fuzzywuzzy import fuzz

        for roster_player in missing:
            print(f"{roster_player}:")

            # Find top 5 fuzzy matches
            scores = []
            for nba_player in active_players_2025_26:
                score = fuzz.ratio(normalize(roster_player), normalize(nba_player))
                scores.append((score, nba_player))

            scores.sort(reverse=True)
            top_matches = scores[:5]

            for score, match in top_matches:
                if score > 50:  # Only show reasonable matches
                    print(f"  {score}% - {match}")
            print()


if __name__ == '__main__':
    roster_file = Path(__file__).parent / "data" / "roster_snapshots" / "roster_latest.csv"

    if roster_file.exists():
        check_roster_against_2025_26(roster_file)
    else:
        print("Roster file not found, just getting 2025-26 players...")
        get_all_2025_26_players()
