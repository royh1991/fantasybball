"""
Collect data for players missing from the main dataset.

Handles:
1. Name mismatches (Nicolas Claxton → Nic Claxton, Jimmy Butler → Jimmy Butler III)
2. 2024 rookies with limited data
3. 2025 draft prospects (not in NBA yet - will use projections only)
"""

import os
import json
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
import time
from datetime import datetime


# Name mappings for known mismatches
NAME_MAPPINGS = {
    'Nicolas Claxton': 'Nic Claxton',
    'Jimmy Butler': 'Jimmy Butler',  # Try without III first
    'Kel\'el Ware': 'Kel\'el Ware',
    'Alexandre Sarr': 'Alexandre Sarr',
    'Jared McCain': 'Jared McCain',
    'Adem Bona': 'Adem Bona',
}

# 2025 prospects (not in NBA yet - skip data collection)
FUTURE_PROSPECTS = [
    'Cooper Flagg',
    'Dylan Harper',
    'Ace Bailey',
    'VJ Edgecombe',
]


def find_player_id(player_name):
    """Find player ID with fuzzy matching for name variations."""
    all_players = players.get_players()

    # Try exact match first
    for player in all_players:
        if player['full_name'].lower() == player_name.lower():
            return player['id']

    # Try without III, Jr, etc
    name_clean = player_name.replace(' III', '').replace(' Jr.', '').replace(' Sr.', '').strip()
    for player in all_players:
        player_clean = player['full_name'].replace(' III', '').replace(' Jr.', '').replace(' Sr.', '').strip()
        if player_clean.lower() == name_clean.lower():
            print(f"  Found match: '{player_name}' → '{player['full_name']}'")
            return player['id']

    # Try last name match (for rookies who might have different first name format)
    last_name = player_name.split()[-1].lower()
    matches = [p for p in all_players if p['full_name'].split()[-1].lower() == last_name]

    if len(matches) == 1:
        print(f"  Found match by last name: '{player_name}' → '{matches[0]['full_name']}'")
        return matches[0]['id']
    elif len(matches) > 1:
        print(f"  Multiple matches for '{player_name}': {[m['full_name'] for m in matches]}")
        return matches[0]['id']  # Take first match

    return None


def collect_player_data(player_name, seasons=['2022-23', '2023-24', '2024-25']):
    """
    Collect game log data for a player across multiple seasons.

    Returns:
    --------
    tuple: (game_data_df, variances_dict) or (None, None) if not found
    """
    print(f"\nCollecting data for {player_name}...")

    # Check if future prospect
    if player_name in FUTURE_PROSPECTS:
        print(f"  → {player_name} is a 2025 draft prospect (not in NBA yet) - skipping")
        return None, None

    # Map name if needed
    search_name = NAME_MAPPINGS.get(player_name, player_name)

    # Find player ID
    player_id = find_player_id(search_name)

    if player_id is None:
        print(f"  → Player not found in NBA API")
        return None, None

    all_games = []

    for season in seasons:
        try:
            print(f"  Fetching {season}...")

            # Fetch game logs
            time.sleep(0.6)  # Rate limiting
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )

            games_df = gamelog.get_data_frames()[0]

            if len(games_df) == 0:
                print(f"    No games in {season}")
                continue

            print(f"    Found {len(games_df)} games")

            # Add season identifier
            games_df['NBA_SEASON'] = season
            all_games.append(games_df)

        except Exception as e:
            print(f"    Error fetching {season}: {e}")
            continue

    if len(all_games) == 0:
        print(f"  → No data found for any season")
        return None, None

    # Combine all games
    combined_df = pd.concat(all_games, ignore_index=True)

    # Calculate per-game variances
    variances = {}

    stat_columns = {
        'PTS': 'PTS',
        'REB': 'REB',
        'AST': 'AST',
        'STL': 'STL',
        'BLK': 'BLK',
        'TOV': 'TOV',
        'FG3M': 'FG3M',
        'FG_PCT': 'FG_PCT',
        'FT_PCT': 'FT_PCT',
        'FG3_PCT': 'FG3_PCT',
    }

    for our_name, nba_name in stat_columns.items():
        if nba_name in combined_df.columns:
            values = combined_df[nba_name].dropna()
            if len(values) > 0:
                variances[our_name] = {
                    'mean_per_game': float(values.mean()),
                    'std_per_game': float(values.std()),
                    'var_per_game': float(values.var()),
                    'cv_per_game': float(values.std() / values.mean()) if values.mean() != 0 else 0,
                    'games': len(values)
                }

    # Calculate DD (double-doubles)
    if 'PTS' in combined_df.columns and 'REB' in combined_df.columns and 'AST' in combined_df.columns:
        dd_count = ((combined_df[['PTS', 'REB', 'AST', 'STL', 'BLK']] >= 10).sum(axis=1) >= 2).astype(int)
        variances['DD'] = {
            'mean_per_game': float(dd_count.mean()),
            'std_per_game': float(dd_count.std()),
            'var_per_game': float(dd_count.var()),
            'cv_per_game': 0,
            'games': len(dd_count)
        }

    print(f"  ✓ Collected {len(combined_df)} games across {len(all_games)} seasons")

    return combined_df, variances


def aggregate_to_weekly(game_df):
    """Aggregate game logs to weekly data."""
    if game_df is None or len(game_df) == 0:
        return None

    # Add week identifier (approximate - group by date ranges)
    game_df['GAME_DATE'] = pd.to_datetime(game_df['GAME_DATE'])
    game_df['SEASON_WEEK'] = game_df['GAME_DATE'].dt.isocalendar().week

    # Create week ID
    game_df['SEASON_WEEK_ID'] = game_df['NBA_SEASON'] + '_W' + game_df['SEASON_WEEK'].astype(str)

    # Aggregate by week
    weekly_stats = game_df.groupby(['SEASON_WEEK_ID', 'NBA_SEASON']).agg({
        'PTS': 'sum',
        'REB': 'sum',
        'AST': 'sum',
        'STL': 'sum',
        'BLK': 'sum',
        'TOV': 'sum',
        'FG3M': 'sum',
        'FG3A': 'sum',
        'FGM': 'sum',
        'FGA': 'sum',
        'FTM': 'sum',
        'FTA': 'sum',
        'GAME_DATE': 'count'  # Count games played in week
    }).reset_index()

    # Rename games count
    weekly_stats.rename(columns={'GAME_DATE': 'GAMES_PLAYED'}, inplace=True)

    # Calculate percentages
    weekly_stats['FG_PCT'] = weekly_stats['FGM'] / weekly_stats['FGA']
    weekly_stats['FT_PCT'] = weekly_stats['FTM'] / weekly_stats['FTA']
    weekly_stats['FG3_PCT'] = weekly_stats['FG3M'] / weekly_stats['FG3A']

    # Calculate DDs per week (approximate)
    # This is rough - would need game-by-game for exact count
    # For now, estimate based on averages
    weekly_stats['DD'] = 0  # Placeholder - would need game-by-game data

    return weekly_stats


def main():
    """Collect data for all missing players."""
    print("=" * 80)
    print("COLLECTING MISSING PLAYER DATA")
    print("=" * 80)

    # Missing players from the warnings
    missing_players = [
        'Nicolas Claxton',
        'Jimmy Butler',
        'Alexandre Sarr',
        'Kel\'el Ware',
        'Jared McCain',
        'Adem Bona',
        'Cooper Flagg',
        'Dylan Harper',
    ]

    # Load existing data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if len(weekly_files) == 0 or len(variance_files) == 0:
        print("ERROR: No existing data files found!")
        print("Please run collect_full_data.py first")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print(f"\nLoading existing data:")
    print(f"  Weekly data: {data_file}")
    print(f"  Variances: {variance_file}")

    league_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    print(f"\nExisting data: {league_data['PLAYER_NAME'].nunique()} players, {len(league_data)} weeks")

    # Collect new data
    new_weekly_data = []
    new_variances = {}

    for player_name in missing_players:
        # Check if already exists
        if player_name in player_variances or player_name in league_data['PLAYER_NAME'].values:
            print(f"\n{player_name} - already in dataset, skipping")
            continue

        game_df, variances = collect_player_data(player_name)

        if game_df is not None:
            # Aggregate to weekly
            weekly_df = aggregate_to_weekly(game_df)

            if weekly_df is not None and len(weekly_df) > 0:
                # Add player info
                player_id = find_player_id(NAME_MAPPINGS.get(player_name, player_name))
                weekly_df.insert(0, 'PLAYER_ID', player_id)
                weekly_df.insert(1, 'PLAYER_NAME', player_name)

                new_weekly_data.append(weekly_df)
                new_variances[player_name] = variances

                print(f"  ✓ Added {len(weekly_df)} weeks")

    # Combine with existing data
    if len(new_weekly_data) > 0:
        print("\n" + "=" * 80)
        print("SAVING COMBINED DATA")
        print("=" * 80)

        # Combine weekly data
        all_weekly = pd.concat([league_data] + new_weekly_data, ignore_index=True)

        # Combine variances
        all_variances = {**player_variances, **new_variances}

        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        new_weekly_file = os.path.join(data_dir, f'league_weekly_data_{timestamp}.csv')
        new_variance_file = os.path.join(data_dir, f'player_variances_{timestamp}.json')

        all_weekly.to_csv(new_weekly_file, index=False)
        with open(new_variance_file, 'w') as f:
            json.dump(all_variances, f, indent=2)

        print(f"\n✓ Saved updated data:")
        print(f"  {new_weekly_file}")
        print(f"  {new_variance_file}")
        print(f"\nNew totals:")
        print(f"  Players: {all_weekly['PLAYER_NAME'].nunique()} (+{len(new_variances)})")
        print(f"  Weeks: {len(all_weekly)} (+{sum(len(df) for df in new_weekly_data)})")

        # Show what was added
        print("\n" + "=" * 80)
        print("PLAYERS ADDED")
        print("=" * 80)
        for player_name, var_data in new_variances.items():
            games = var_data.get('PTS', {}).get('games', 0)
            print(f"  ✓ {player_name}: {games} games")

    else:
        print("\n" + "=" * 80)
        print("NO NEW DATA COLLECTED")
        print("=" * 80)
        print("\nEither all players already exist or none were found in NBA API.")

    # Report on players that couldn't be found
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    found_players = set(new_variances.keys())
    not_found = [p for p in missing_players if p not in found_players and p not in FUTURE_PROSPECTS]

    if len(not_found) > 0:
        print("\nPlayers NOT found (likely name mismatches):")
        for player in not_found:
            print(f"  ✗ {player}")
        print("\nThese players will need manual name mappings in NAME_MAPPINGS dict")

    future = [p for p in missing_players if p in FUTURE_PROSPECTS]
    if len(future) > 0:
        print("\n2025 prospects (not in NBA yet):")
        for player in future:
            print(f"  → {player}")
        print("\nThese players will use projections only (no historical data)")


if __name__ == "__main__":
    main()
