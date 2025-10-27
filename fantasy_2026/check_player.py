#!/usr/bin/env python3
"""
Quick script to search for players in NBA API and check 2025-26 season availability.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "espn-api"))

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import time

def search_player(search_term):
    """Search for a player by name."""
    all_players = players.get_players()

    print(f"\nSearching for '{search_term}'...")
    matches = [p for p in all_players if search_term.lower() in p['full_name'].lower()]

    if matches:
        print(f"Found {len(matches)} match(es):")
        for p in matches:
            print(f"  - {p['full_name']} (ID: {p['id']})")
        return matches
    else:
        print("No matches found")
        return []


def check_season_data(player_id, player_name, season="2025-26"):
    """Check if a player has data for a specific season."""
    print(f"\nChecking {season} data for {player_name} (ID: {player_id})...")

    try:
        time.sleep(0.6)  # Rate limiting
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season
        )
        df = gamelog.get_data_frames()[0]

        if not df.empty:
            print(f"✓ Found {len(df)} games in {season}")
            print("\nFirst few games:")
            print(df[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']].head())
            return True
        else:
            print(f"✗ No data found for {season}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def get_all_2025_26_players():
    """Try to get all players with 2025-26 data (this might take a while!)."""
    print("\nWARNING: This will be slow - checking recent players for 2025-26 data...")
    print("Press Ctrl+C to cancel\n")

    all_players = players.get_players()

    # Get only active players (filter by checking if they're recent)
    # We can check a subset - say players from recent years
    active_2025_26 = []

    print("Sampling 50 recent players to check for 2025-26 data...")
    for i, p in enumerate(all_players[:50]):  # Just check first 50
        try:
            time.sleep(0.6)
            gamelog = playergamelog.PlayerGameLog(
                player_id=p['id'],
                season="2025-26"
            )
            df = gamelog.get_data_frames()[0]

            if not df.empty:
                active_2025_26.append(p['full_name'])
                print(f"  ✓ {p['full_name']} - {len(df)} games")
        except:
            pass

    print(f"\nFound {len(active_2025_26)} players with 2025-26 data (in sample)")
    return active_2025_26


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Search for specific player
        search_term = ' '.join(sys.argv[1:])
        matches = search_player(search_term)

        if matches:
            # Check 2025-26 data for first match
            player = matches[0]
            check_season_data(player['id'], player['full_name'])
    else:
        # Default: search for VJ Edgecombe
        print("Usage: python check_player.py <player name>")
        print("\nExample searches:")

        # Try VJ variations
        for name in ["VJ Edgecombe", "Victor Edgecombe", "Edgecombe"]:
            matches = search_player(name)
            if matches:
                check_season_data(matches[0]['id'], matches[0]['full_name'])
                break
