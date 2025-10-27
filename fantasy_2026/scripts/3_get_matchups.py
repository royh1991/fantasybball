#!/usr/bin/env python3
"""
Get current week matchup data from ESPN Fantasy API.

Extracts:
- Current week's matchups
- Team lineups (who's playing vs who)
- Projected and actual scores
- Box scores if games have started
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


def get_current_week_matchups(league):
    """
    Get current week's matchups.

    Returns:
        List of matchup dictionaries
    """
    print(f"\n{'='*60}")
    print(f"CURRENT WEEK {league.current_week} MATCHUPS")
    print(f"{'='*60}\n")

    matchups = league.scoreboard()
    matchup_data = []

    for i, matchup in enumerate(matchups, 1):
        print(f"Matchup {i}:")
        print(f"  {matchup.home_team.team_name} vs {matchup.away_team.team_name}")

        matchup_info = {
            'week': league.current_week,
            'matchup_period': league.currentMatchupPeriod,
            'matchup_num': i,
            'home_team_id': matchup.home_team.team_id,
            'home_team_name': matchup.home_team.team_name,
            'home_team_abbrev': matchup.home_team.team_abbrev,
            'away_team_id': matchup.away_team.team_id,
            'away_team_name': matchup.away_team.team_name,
            'away_team_abbrev': matchup.away_team.team_abbrev,
        }

        # Check if it's a category or points league
        if hasattr(matchup, 'home_score'):
            # Points league
            matchup_info['scoring_type'] = 'points'
            matchup_info['home_score'] = matchup.home_score
            matchup_info['away_score'] = matchup.away_score
            matchup_info['home_projected'] = getattr(matchup, 'home_projected', None)
            matchup_info['away_projected'] = getattr(matchup, 'away_projected', None)

            print(f"  Score: {matchup.home_score} - {matchup.away_score}")
            if matchup_info['home_projected']:
                print(f"  Projected: {matchup_info['home_projected']} - {matchup_info['away_projected']}")

        elif hasattr(matchup, 'home_wins'):
            # Category league
            matchup_info['scoring_type'] = 'category'
            matchup_info['home_wins'] = matchup.home_wins
            matchup_info['home_losses'] = matchup.home_losses
            matchup_info['home_ties'] = matchup.home_ties
            matchup_info['away_wins'] = getattr(matchup, 'away_wins', None)
            matchup_info['away_losses'] = getattr(matchup, 'away_losses', None)
            matchup_info['away_ties'] = getattr(matchup, 'away_ties', None)

            print(f"  Category Score: {matchup_info['home_wins']}W-{matchup_info['home_ties']}T-{matchup_info['home_losses']}L")

        matchup_data.append(matchup_info)
        print()

    return matchup_data


def get_box_scores(league):
    """
    Get detailed box scores for current week.

    Returns:
        List of player performance data
    """
    print(f"\n{'='*60}")
    print(f"CURRENT WEEK BOX SCORES")
    print(f"{'='*60}\n")

    box_scores = league.box_scores()

    if not box_scores:
        print("No box scores available yet (week hasn't started)")
        return []

    all_player_stats = []

    for box in box_scores:
        matchup_info = f"{box.home_team.team_name} vs {box.away_team.team_name}"
        print(f"\n{matchup_info}")
        print("-" * len(matchup_info))

        # Process home team lineup
        for player in box.home_lineup:
            player_data = {
                'week': league.current_week,
                'matchup': matchup_info,
                'team_id': box.home_team.team_id,
                'team_name': box.home_team.team_name,
                'team_side': 'home',
                'player_name': player.name,
                'player_id': player.playerId,
                'position': player.position,
                'slot_position': player.slot_position,
                'pro_team': player.proTeam,
            }

            # Add stats if available
            if hasattr(player, 'points'):
                player_data['points'] = player.points
            if hasattr(player, 'projected_points'):
                player_data['projected_points'] = player.projected_points

            # Add nine-cat stats if available
            if hasattr(player, 'stats') and player.stats:
                for stat_name, stat_value in player.stats.items():
                    player_data[f'stat_{stat_name}'] = stat_value

            all_player_stats.append(player_data)

        # Process away team lineup
        for player in box.away_lineup:
            player_data = {
                'week': league.current_week,
                'matchup': matchup_info,
                'team_id': box.away_team.team_id,
                'team_name': box.away_team.team_name,
                'team_side': 'away',
                'player_name': player.name,
                'player_id': player.playerId,
                'position': player.position,
                'slot_position': player.slot_position,
                'pro_team': player.proTeam,
            }

            if hasattr(player, 'points'):
                player_data['points'] = player.points
            if hasattr(player, 'projected_points'):
                player_data['projected_points'] = player.projected_points

            if hasattr(player, 'stats') and player.stats:
                for stat_name, stat_value in player.stats.items():
                    player_data[f'stat_{stat_name}'] = stat_value

            all_player_stats.append(player_data)

        # Show summary
        if hasattr(box, 'home_score'):
            print(f"  Score: {box.home_score} - {box.away_score}")
            print(f"  Projected: {box.home_projected} - {box.away_projected}")

    return all_player_stats


def save_matchup_data(matchups, box_scores, output_dir):
    """Save matchup data to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save matchups
    if matchups:
        matchups_df = pd.DataFrame(matchups)
        matchups_file = output_path / f"matchups_{timestamp}.csv"
        matchups_df.to_csv(matchups_file, index=False)
        print(f"\n✓ Saved matchups to: {matchups_file}")

        # Also save as latest
        latest_matchups = output_path / "matchups_latest.csv"
        matchups_df.to_csv(latest_matchups, index=False)
        print(f"✓ Saved as latest: {latest_matchups}")

    # Save box scores
    if box_scores:
        box_scores_df = pd.DataFrame(box_scores)
        box_scores_file = output_path / f"box_scores_{timestamp}.csv"
        box_scores_df.to_csv(box_scores_file, index=False)
        print(f"\n✓ Saved box scores to: {box_scores_file}")

        # Also save as latest
        latest_box_scores = output_path / "box_scores_latest.csv"
        box_scores_df.to_csv(latest_box_scores, index=False)
        print(f"✓ Saved as latest: {latest_box_scores}")


def main():
    """Main execution."""
    print("="*60)
    print("ESPN FANTASY MATCHUP DATA EXTRACTION")
    print("="*60)

    # Load configuration
    config = load_config()
    league_id = config['league']['id']
    season = config['league']['season']
    output_dir = Path(__file__).parent.parent / "data" / "matchups"

    # Get ESPN credentials from environment
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
        print(f"  League: {config['league']['name']}")
        print(f"  Current Week: {league.current_week}")
        print(f"  Matchup Period: {league.currentMatchupPeriod}")

    except Exception as e:
        print(f"✗ Failed to connect to league: {e}")
        return 1

    # Get matchup data
    matchups = get_current_week_matchups(league)

    # Get box scores
    box_scores = get_box_scores(league)

    # Save data
    save_matchup_data(matchups, box_scores, output_dir)

    print("\n" + "="*60)
    print("✓ MATCHUP DATA EXTRACTION COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
