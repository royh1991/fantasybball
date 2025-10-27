#!/usr/bin/env python3
"""
Extract current fantasy league rosters from ESPN API.

Creates a snapshot of all team rosters with:
- Team information
- Player names, positions, pro teams
- Current roster status (active/benched/IR)
- Extraction timestamp
"""

import os
import sys
from pathlib import Path
from datetime import datetime
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


def extract_rosters(league_id, season, espn_s2, swid):
    """
    Extract all rosters from ESPN Fantasy league.

    Args:
        league_id: ESPN league ID
        season: Season year
        espn_s2: ESPN authentication cookie
        swid: ESPN SWID cookie

    Returns:
        DataFrame with all players across all teams
    """
    print(f"Connecting to ESPN league {league_id}, season {season}...")

    try:
        league = League(
            league_id=league_id,
            year=season,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"✓ Connected successfully")
        print(f"  League: {league.settings.name if hasattr(league.settings, 'name') else 'N/A'}")
        print(f"  Teams: {len(league.teams)}")
        print(f"  Current Week: {league.current_week}")

    except Exception as e:
        print(f"✗ Failed to connect to league: {e}")
        return None

    # Extract roster data
    all_players = []
    extraction_time = datetime.now()

    for team in league.teams:
        print(f"\nProcessing: {team.team_name} ({team.team_abbrev})")

        for player in team.roster:
            player_data = {
                # Extraction metadata
                'extraction_date': extraction_time.strftime('%Y-%m-%d'),
                'extraction_timestamp': extraction_time.isoformat(),
                'season': season,

                # Team information
                'fantasy_team_id': team.team_id,
                'fantasy_team_name': team.team_name,
                'fantasy_team_abbrev': team.team_abbrev,
                'team_owner': team.owner if hasattr(team, 'owner') else None,

                # Player information
                'player_name': player.name,
                'player_id_espn': player.playerId,
                'position': player.position,
                'eligible_positions': ','.join(player.eligibleSlots) if hasattr(player, 'eligibleSlots') else player.position,

                # NBA team
                'pro_team': player.proTeam,

                # Fantasy status
                'currently_rostered': True,  # They're on the roster right now
                'injury_status': player.injuryStatus if hasattr(player, 'injuryStatus') else 'ACTIVE',
                'acquisition_type': ','.join(player.acquisitionType) if player.acquisitionType else 'UNKNOWN',

                # Stats (if available)
                'season_avg_pts': player.avg_points if hasattr(player, 'avg_points') else None,
                'season_total_pts': player.total_points if hasattr(player, 'total_points') else None,
            }

            # Add nine-cat stats if available
            if hasattr(player, 'nine_cat_averages') and player.nine_cat_averages:
                for stat, value in player.nine_cat_averages.items():
                    player_data[f'nine_cat_{stat.lower()}'] = value

            all_players.append(player_data)

        print(f"  → {len(team.roster)} players")

    print(f"\nTotal players extracted: {len(all_players)}")

    return pd.DataFrame(all_players)


def save_roster_snapshot(df, output_dir):
    """
    Save roster snapshot to CSV with timestamp.

    Args:
        df: DataFrame with roster data
        output_dir: Directory to save snapshots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"roster_snapshot_{timestamp}.csv"
    filepath = output_path / filename

    df.to_csv(filepath, index=False)
    print(f"\n✓ Saved roster snapshot: {filepath}")

    # Also save as "latest" for easy access
    latest_filepath = output_path / "roster_latest.csv"
    df.to_csv(latest_filepath, index=False)
    print(f"✓ Saved as latest: {latest_filepath}")

    return filepath


def print_summary(df):
    """Print summary statistics of extracted rosters."""
    print("\n" + "="*60)
    print("ROSTER EXTRACTION SUMMARY")
    print("="*60)

    print(f"\nTotal Players: {len(df)}")
    print(f"Total Teams: {df['fantasy_team_name'].nunique()}")
    print(f"Extraction Date: {df['extraction_date'].iloc[0]}")

    print(f"\nPlayers by Team:")
    team_counts = df.groupby('fantasy_team_name').size().sort_values(ascending=False)
    for team, count in team_counts.items():
        print(f"  {team:30s}: {count:2d} players")

    print(f"\nPlayers by Pro Team:")
    pro_team_counts = df['pro_team'].value_counts().head(10)
    for team, count in pro_team_counts.items():
        print(f"  {team}: {count}")

    print(f"\nInjury Status:")
    injury_counts = df['injury_status'].value_counts()
    for status, count in injury_counts.items():
        print(f"  {status}: {count}")

    print(f"\nPosition Distribution:")
    pos_counts = df['position'].value_counts()
    for pos, count in pos_counts.items():
        print(f"  {pos}: {count}")


def main():
    """Main execution."""
    print("="*60)
    print("ESPN FANTASY ROSTER EXTRACTION")
    print("="*60)

    # Load configuration
    config = load_config()
    league_id = config['league']['id']
    season = config['league']['season']
    output_dir = Path(__file__).parent.parent / config['output']['roster_snapshots']

    # Get ESPN credentials from environment
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("\n✗ ERROR: ESPN credentials not found in environment variables")
        print("  Please set ESPN_S2 and ESPN_SWID")
        return 1

    # Extract rosters
    df = extract_rosters(league_id, season, espn_s2, swid)

    if df is None or len(df) == 0:
        print("\n✗ Failed to extract rosters")
        return 1

    # Save snapshot
    filepath = save_roster_snapshot(df, output_dir)

    # Print summary
    print_summary(df)

    print("\n" + "="*60)
    print("✓ ROSTER EXTRACTION COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
