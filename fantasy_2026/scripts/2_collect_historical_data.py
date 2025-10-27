#!/usr/bin/env python3
"""
Collect historical game logs for all rostered players.

Uses NBA API to fetch 3 seasons of historical data for each player,
with intelligent name matching and deduplication to avoid re-fetching
players we already have data for.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml
import time
import unicodedata
from fuzzywuzzy import fuzz
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats
from nba_api.stats.static import players
import json


def load_config():
    """Load league configuration."""
    config_path = Path(__file__).parent.parent / "config" / "league_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_latest_roster():
    """Load the most recent roster snapshot."""
    config = load_config()
    roster_dir = Path(__file__).parent.parent / config['output']['roster_snapshots']
    latest_file = roster_dir / "roster_latest.csv"

    if not latest_file.exists():
        print("✗ No roster snapshot found. Run 1_extract_rosters.py first.")
        return None

    df = pd.read_csv(latest_file)
    print(f"✓ Loaded roster with {len(df)} players from {df['extraction_date'].iloc[0]}")
    return df


def load_existing_game_logs():
    """Load existing historical game logs to avoid re-fetching."""
    parent_dir = Path(__file__).parent.parent.parent

    # Try multiple possible locations (OWN file first, then parent directory)
    possible_paths = [
        Path(__file__).parent.parent / "data" / "historical_gamelogs" / "historical_gamelogs_latest.csv",  # Check own output first!
        parent_dir / "data" / "static" / "player_stats_historical" / "active_players_historical_game_logs.csv",
        parent_dir / "data" / "static" / "active_players_historical_game_logs.csv",
    ]

    for historical_file in possible_paths:
        if historical_file.exists():
            print(f"✓ Loading existing game logs from {historical_file}")
            df = pd.read_csv(historical_file)

            # Standardize column names (handle both PLAYER_NAME and player_name)
            if 'player_name' in df.columns and 'PLAYER_NAME' not in df.columns:
                df['PLAYER_NAME'] = df['player_name']

            if 'PLAYER_NAME' in df.columns:
                print(f"  Found {len(df)} existing game logs for {df['PLAYER_NAME'].nunique()} players")
            else:
                print(f"  Found {len(df)} existing game logs")

            return df

    print("  No existing game logs found")
    return pd.DataFrame()


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


def fuzzy_match_player(espn_name, nba_players_list, threshold=85):
    """
    Match ESPN player name to NBA API player using fuzzy matching.

    Args:
        espn_name: Player name from ESPN
        nba_players_list: List of NBA player dictionaries
        threshold: Fuzzy match threshold (0-100)

    Returns:
        Best matching NBA player dict or None
    """
    best_match = None
    best_score = 0

    # Normalize the ESPN name
    espn_normalized = normalize_name(espn_name)

    for nba_player in nba_players_list:
        nba_name = nba_player['full_name']
        nba_normalized = normalize_name(nba_name)

        # Direct match (normalized)
        if espn_normalized == nba_normalized:
            return nba_player

        # Fuzzy match on normalized names
        score = fuzz.ratio(espn_normalized, nba_normalized)

        if score > best_score:
            best_score = score
            best_match = nba_player

    if best_score >= threshold:
        return best_match

    return None


def get_nba_player_id(player_name, nba_players_list):
    """
    Get NBA API player ID from player name.

    Args:
        player_name: ESPN player name
        nba_players_list: List of all NBA players

    Returns:
        Player ID and matched name, or (None, None)
    """
    player_normalized = normalize_name(player_name)

    # Try exact match first (normalized)
    for p in nba_players_list:
        if normalize_name(p['full_name']) == player_normalized:
            return p['id'], p['full_name']

    # Try fuzzy match
    match = fuzzy_match_player(player_name, nba_players_list, threshold=85)
    if match:
        return match['id'], match['full_name']

    return None, None


def fetch_player_game_log(player_id, player_name, season, rate_limit_ms=600):
    """
    Fetch game log for a player in a specific season.

    Args:
        player_id: NBA API player ID
        player_name: Player name (for logging)
        season: Season string (e.g., '2023-24')
        rate_limit_ms: Milliseconds to wait between requests

    Returns:
        DataFrame with game logs or empty DataFrame
    """
    try:
        time.sleep(rate_limit_ms / 1000)  # Rate limiting

        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season
        )
        df = gamelog.get_data_frames()[0]

        if not df.empty:
            df['PLAYER_NAME'] = player_name
            df['season'] = season

        return df

    except Exception as e:
        print(f"      ✗ Error fetching {season}: {e}")
        return pd.DataFrame()


def collect_historical_data_for_players(roster_df, existing_logs, seasons, rate_limit_ms=600):
    """
    Collect historical game logs for all players in roster.

    Args:
        roster_df: DataFrame with roster data
        existing_logs: DataFrame with existing game logs
        seasons: List of season strings to fetch
        rate_limit_ms: Rate limit for API calls

    Returns:
        DataFrame with all collected game logs
    """
    # Get all NBA players for matching
    print("\nFetching NBA player list...")
    nba_players_list = players.get_players()
    print(f"✓ Loaded {len(nba_players_list)} NBA players from static database")

    # Also get current season players to catch brand new rookies not in static DB
    try:
        print("  Fetching 2025-26 season players for new rookies...")
        time.sleep(1)
        current_season_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            season_type_all_star="Regular Season"
        )
        current_season_df = current_season_stats.get_data_frames()[0]

        if not current_season_df.empty:
            # Add any players not already in the static list
            existing_names = {p['full_name'].lower() for p in nba_players_list}
            new_rookies = 0

            for _, row in current_season_df.iterrows():
                player_name = row['PLAYER_NAME']
                player_id = row['PLAYER_ID']

                if player_name.lower() not in existing_names:
                    # Add this new player to our list
                    nba_players_list.append({
                        'id': player_id,
                        'full_name': player_name,
                        'first_name': player_name.split()[0] if ' ' in player_name else player_name,
                        'last_name': player_name.split()[-1] if ' ' in player_name else ''
                    })
                    new_rookies += 1
                    existing_names.add(player_name.lower())

            if new_rookies > 0:
                print(f"  ✓ Added {new_rookies} new rookies from 2025-26 season")
            print(f"✓ Total players available: {len(nba_players_list)}")
    except Exception as e:
        print(f"  ⚠ Could not fetch current season players: {e}")
        print(f"  Continuing with {len(nba_players_list)} players from static database")

    # Track which players we have ANY data for, and which specific seasons
    existing_players_normalized = set()
    existing_player_seasons = {}

    if not existing_logs.empty:
        for name in existing_logs['PLAYER_NAME'].unique():
            normalized_name = normalize_name(name)
            existing_players_normalized.add(normalized_name)

            # Try to get seasons if the column exists and has data
            if 'season' in existing_logs.columns:
                player_data = existing_logs[existing_logs['PLAYER_NAME'] == name]
                existing_seasons = set(player_data['season'].dropna().unique())
                if existing_seasons:  # Only store if we have season data
                    existing_player_seasons[normalized_name] = existing_seasons

    # Collect new data
    all_game_logs = []
    player_names = roster_df['player_name'].unique()

    print(f"\nProcessing {len(player_names)} unique players across {len(seasons)} seasons...")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Rate limit: {rate_limit_ms}ms between requests\n")

    for i, player_name in enumerate(player_names, 1):
        print(f"[{i}/{len(player_names)}] {player_name}")

        # Match to NBA API
        player_id, matched_name = get_nba_player_id(player_name, nba_players_list)

        if not player_id:
            print(f"  ✗ Could not find NBA player ID (might be rookie or name mismatch)")
            continue

        if matched_name != player_name:
            print(f"  → Matched to: {matched_name}")

        # Check which seasons we need to fetch for this player
        player_normalized = normalize_name(player_name)

        # Determine what to fetch based on what we have
        if player_normalized in existing_players_normalized:
            # Player exists in our data - assume we have historical, just get current season
            current_season = seasons[0]  # 2025-26 (first in list)

            # Check if we already have the current season
            if player_normalized in existing_player_seasons:
                existing_seasons = existing_player_seasons[player_normalized]
                if current_season in existing_seasons:
                    print(f"  ✓ Already have all seasons (including {current_season}), skipping")
                    continue

            seasons_to_fetch = [current_season]
            print(f"  → Already have historical data, fetching current season: {current_season}")
        else:
            # Brand new player - fetch all seasons
            seasons_to_fetch = seasons
            print(f"  → New player, fetching all {len(seasons)} seasons")

        # Fetch data for each season we don't have
        player_game_logs = []
        for season in seasons_to_fetch:
            print(f"    Fetching {season}...", end=" ")

            df = fetch_player_game_log(player_id, matched_name, season, rate_limit_ms)

            if not df.empty:
                print(f"✓ {len(df)} games")
                player_game_logs.append(df)
            else:
                print("(no data)")

        if player_game_logs:
            combined = pd.concat(player_game_logs, ignore_index=True)
            all_game_logs.append(combined)
            print(f"  ✓ Total: {len(combined)} games across {len(player_game_logs)} seasons")
        else:
            print(f"  ✗ No historical data found")

    if all_game_logs:
        return pd.concat(all_game_logs, ignore_index=True)
    else:
        return pd.DataFrame()


def save_historical_data(new_data, existing_data, output_dir):
    """
    Save historical game log data, combining with existing data.

    Args:
        new_data: Newly fetched game logs
        existing_data: Existing game logs
        output_dir: Directory to save output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Combine new and existing data
    if not existing_data.empty and not new_data.empty:
        combined = pd.concat([existing_data, new_data], ignore_index=True)

        # Remove duplicates (same player, same game)
        if 'GAME_ID' in combined.columns and 'PLAYER_NAME' in combined.columns:
            combined = combined.drop_duplicates(subset=['PLAYER_NAME', 'GAME_ID'], keep='first')

        print(f"\n✓ Combined {len(existing_data)} existing + {len(new_data)} new = {len(combined)} total game logs")

    elif not new_data.empty:
        combined = new_data
        print(f"\n✓ {len(combined)} new game logs")
    else:
        print("\n✗ No new data to save")
        return

    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"historical_gamelogs_{timestamp}.csv"
    filepath = output_path / filename

    combined.to_csv(filepath, index=False)
    print(f"✓ Saved to: {filepath}")

    # Also save as "latest"
    latest_filepath = output_path / "historical_gamelogs_latest.csv"
    combined.to_csv(latest_filepath, index=False)
    print(f"✓ Saved as latest: {latest_filepath}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total game logs: {len(combined)}")
    print(f"  Unique players: {combined['PLAYER_NAME'].nunique()}")

    # Handle season column (may have NaN values from old data)
    if 'season' in combined.columns:
        seasons = combined['season'].dropna().unique()
        if len(seasons) > 0:
            print(f"  Seasons covered: {sorted(seasons)}")
        else:
            print(f"  Seasons covered: Not available")
    else:
        print(f"  Seasons covered: Not available")


def main():
    """Main execution."""
    print("="*60)
    print("HISTORICAL GAME LOG COLLECTION")
    print("="*60)

    # Load configuration
    config = load_config()
    seasons = config['data_collection']['nba_seasons']
    rate_limit_ms = config['data_collection']['rate_limit_ms']
    output_dir = Path(__file__).parent.parent / config['output']['historical_gamelogs']

    # Load roster
    roster_df = load_latest_roster()
    if roster_df is None:
        return 1

    # Load existing historical data
    existing_logs = load_existing_game_logs()

    # Collect new data
    new_data = collect_historical_data_for_players(
        roster_df,
        existing_logs,
        seasons,
        rate_limit_ms
    )

    if new_data.empty:
        print("\n✓ No new players to fetch (all players already have historical data)")
        return 0

    # Save combined data
    save_historical_data(new_data, existing_logs, output_dir)

    print("\n" + "="*60)
    print("✓ HISTORICAL DATA COLLECTION COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
