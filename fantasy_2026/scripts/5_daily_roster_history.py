#!/usr/bin/env python3
"""
Extract daily roster history - which players were on which teams each day.

Uses ESPN transactions to reconstruct the roster composition for each day
of the season. This allows you to answer:
- "Who was on Team X on October 25th?"
- "Which team owned Player Y on a specific date?"
- "Show me all the games Player Z played while on my roster"
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yaml

# Add espn-api to path
parent_dir = Path(__file__).parent.parent.parent
espn_api_path = parent_dir / "espn-api"
sys.path.insert(0, str(espn_api_path))

from espn_api.basketball import League


def load_config():
    """Load league configuration."""
    config_path = Path(__file__).parent.parent / "config" / "league_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_initial_rosters(league):
    """
    Get the initial roster (draft results) from the league.

    Returns:
        dict: {team_id: set(player_names)}
    """
    print("\nGetting initial rosters (draft results)...")

    # Start with current rosters and work backwards
    rosters = {}
    for team in league.teams:
        rosters[team.team_id] = {
            'team_name': team.team_name,
            'players': set()
        }

    return rosters


def extract_all_transactions(league):
    """
    Extract all transactions from the season.

    Returns:
        DataFrame with all transactions, sorted by date
    """
    print("\nExtracting all transactions from the season...")

    all_transactions = []
    current_week = league.current_week

    for week in range(1, current_week + 1):
        try:
            transactions = league.transactions(scoring_period=week)

            for trans in transactions:
                trans_date = datetime.fromtimestamp(trans.date/1000) if trans.date else None

                for item in trans.items:
                    all_transactions.append({
                        'date': trans_date,
                        'team_id': trans.team.team_id,
                        'team_name': trans.team.team_name,
                        'action': item.type,  # 'ADD' or 'DROP'
                        'player_name': item.player,
                    })
        except Exception as e:
            print(f"  Error fetching week {week}: {e}")

    df = pd.DataFrame(all_transactions)

    if not df.empty:
        df = df.sort_values('date')
        print(f"✓ Found {len(df)} transactions from {df['date'].min()} to {df['date'].max()}")
    else:
        print("✗ No transactions found")

    return df


def get_current_rosters(league):
    """
    Get current roster for each team as the starting point.

    Returns:
        dict: {team_id: {'team_name': name, 'players': set(player_names)}}
    """
    print("\nGetting current rosters as baseline...")

    rosters = {}
    for team in league.teams:
        players = {player.name for player in team.roster}
        rosters[team.team_id] = {
            'team_name': team.team_name,
            'players': players
        }
        print(f"  {team.team_name}: {len(players)} players")

    return rosters


def build_daily_rosters(current_rosters, transactions_df, start_date, end_date):
    """
    Build roster snapshots for each day by applying transactions in reverse.

    Args:
        current_rosters: Current roster state (today)
        transactions_df: All transactions sorted by date
        start_date: First date to reconstruct
        end_date: Last date (today)

    Returns:
        DataFrame with daily roster composition
    """
    print(f"\nReconstructing daily rosters from {start_date.date()} to {end_date.date()}...")

    # Start with current rosters
    rosters = {team_id: data['players'].copy() for team_id, data in current_rosters.items()}
    team_names = {team_id: data['team_name'] for team_id, data in current_rosters.items()}

    # Generate all dates from start to end
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Apply transactions in REVERSE to go back in time
    transactions_reverse = transactions_df.sort_values('date', ascending=False)

    # Build roster history
    daily_rosters = []

    for current_date in reversed(all_dates):
        # Apply all transactions that happened AFTER this date (reverse them)
        trans_after = transactions_reverse[transactions_reverse['date'] > current_date]

        # Reset to current rosters
        day_rosters = {team_id: current_rosters[team_id]['players'].copy()
                      for team_id in current_rosters.keys()}

        # Reverse each transaction
        for _, trans in trans_after.iterrows():
            team_id = trans['team_id']
            player = trans['player_name']
            action = trans['action']

            if team_id in day_rosters:
                if action == 'ADD':
                    # Reverse an ADD = remove the player
                    day_rosters[team_id].discard(player)
                elif action == 'DROP':
                    # Reverse a DROP = add the player back
                    day_rosters[team_id].add(player)

        # Record roster for this date
        for team_id, players in day_rosters.items():
            for player in players:
                daily_rosters.append({
                    'date': current_date,
                    'team_id': team_id,
                    'team_name': team_names.get(team_id, f'Team {team_id}'),
                    'player_name': player,
                })

    df = pd.DataFrame(daily_rosters)
    df = df.sort_values(['date', 'team_name', 'player_name'])

    print(f"✓ Generated {len(df)} daily roster records across {len(all_dates)} days")

    return df


def save_daily_rosters(df, output_dir):
    """Save daily roster history to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save timestamped version
    filename = f"daily_rosters_{timestamp}.csv"
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    print(f"\n✓ Saved daily rosters to: {filepath}")

    # Save as latest
    latest_filepath = output_path / "daily_rosters_latest.csv"
    df.to_csv(latest_filepath, index=False)
    print(f"✓ Saved as latest: {latest_filepath}")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("DAILY ROSTER SUMMARY")
    print("="*60)

    if not df.empty:
        print(f"\nTotal Records: {len(df)}")
        print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Days Covered: {df['date'].nunique()}")
        print(f"Unique Players: {df['player_name'].nunique()}")
        print(f"Teams: {df['team_name'].nunique()}")

        print("\nAverage Roster Size by Team:")
        avg_roster = df.groupby(['date', 'team_name']).size().groupby('team_name').mean()
        for team, size in avg_roster.items():
            print(f"  {team}: {size:.1f} players/day")

        # Show example for a specific date
        latest_date = df['date'].max()
        latest = df[df['date'] == latest_date]
        print(f"\nExample - Rosters on {latest_date.date()}:")
        for team in latest['team_name'].unique():
            team_players = latest[latest['team_name'] == team]['player_name'].tolist()
            print(f"  {team}: {len(team_players)} players")


def main():
    """Main execution."""
    print("="*60)
    print("DAILY ROSTER HISTORY EXTRACTION")
    print("="*60)

    # Load configuration
    config = load_config()
    league_id = config['league']['id']
    season = config['league']['season']
    output_dir = Path(__file__).parent.parent / "data" / "ownership_history"

    # Get ESPN credentials
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("\n✗ ERROR: ESPN credentials not found in environment variables")
        print("  Please set ESPN_S2 and ESPN_SWID")
        return 1

    # Connect to league
    print(f"\nConnecting to ESPN league {league_id}, season {season}...")

    try:
        league = League(
            league_id=league_id,
            year=season,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"✓ Connected successfully")
        print(f"  Current Week: {league.current_week}")
    except Exception as e:
        print(f"✗ Failed to connect to league: {e}")
        return 1

    # Get current rosters (our endpoint)
    current_rosters = get_current_rosters(league)

    # Get all transactions
    transactions_df = extract_all_transactions(league)

    if transactions_df.empty:
        print("\n⚠ No transactions found. Using current rosters only.")
        # Just export current roster as single day
        today = datetime.now()
        records = []
        for team_id, data in current_rosters.items():
            for player in data['players']:
                records.append({
                    'date': today,
                    'team_id': team_id,
                    'team_name': data['team_name'],
                    'player_name': player,
                })
        daily_rosters_df = pd.DataFrame(records)
    else:
        # Determine date range
        earliest_trans = transactions_df['date'].min()
        today = datetime.now()

        # Start from the day before first transaction (assume rosters were stable before that)
        start_date = earliest_trans - timedelta(days=1)

        # Build daily rosters
        daily_rosters_df = build_daily_rosters(
            current_rosters,
            transactions_df,
            start_date,
            today
        )

    # Save data
    save_daily_rosters(daily_rosters_df, output_dir)

    # Print summary
    print_summary(daily_rosters_df)

    print("\n" + "="*60)
    print("✓ DAILY ROSTER HISTORY COMPLETE")
    print("="*60)

    print("\nUsage Examples:")
    print("\n1. Who was on Team X on a specific date:")
    print("   df = pd.read_csv('data/ownership_history/daily_rosters_latest.csv')")
    print("   df['date'] = pd.to_datetime(df['date'])")
    print("   df[(df['team_name'] == 'My Team') & (df['date'] == '2025-10-25')]")

    print("\n2. Which team owned Player Y on a date:")
    print("   df[(df['player_name'] == 'Luka Doncic') & (df['date'] == '2025-10-25')]")

    print("\n3. Track a player's ownership over time:")
    print("   df[df['player_name'] == 'LeBron James'][['date', 'team_name']]")

    print("\n4. Join with game logs to get 'games while rostered':")
    print("   gamelogs = pd.read_csv('data/historical_gamelogs/historical_gamelogs_latest.csv')")
    print("   gamelogs['GAME_DATE'] = pd.to_datetime(gamelogs['GAME_DATE'])")
    print("   merged = gamelogs.merge(df, left_on=['PLAYER_NAME', 'GAME_DATE'], right_on=['player_name', 'date'])")
    print("   # Now you have: games + who owned the player that day!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
