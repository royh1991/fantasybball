#!/usr/bin/env python3
"""
Test script for ESPN Fantasy Basketball API integration.

Tests:
1. League authentication and basic info
2. Team rosters and current matchups
3. Player data (Stephen Curry)
4. Current season game logs
"""

import os
import sys
from pathlib import Path

# Add espn-api to path
espn_api_path = Path(__file__).parent.parent / "espn-api"
sys.path.insert(0, str(espn_api_path))

from espn_api.basketball import League
import pandas as pd
from datetime import datetime


def test_league_connection():
    """Test 1: Connect to league and get basic info."""
    print("="*60)
    print("TEST 1: League Connection")
    print("="*60)

    # Get credentials from environment
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("ERROR: ESPN_S2 and ESPN_SWID environment variables required")
        print(f"ESPN_S2 present: {bool(espn_s2)}")
        print(f"ESPN_SWID present: {bool(swid)}")
        return None

    print(f"ESPN_S2: {espn_s2[:20]}...")
    print(f"ESPN_SWID: {swid}")

    try:
        # Initialize league
        league = League(
            league_id=40204,
            year=2026,
            espn_s2=espn_s2,
            swid=swid,
            debug=False
        )

        print(f"\n✓ Successfully connected to league!")
        print(f"  League ID: 40204")
        print(f"  Season: 2026 (2025-26 NBA season)")
        print(f"  Current Week: {league.current_week}")
        print(f"  Current Matchup Period: {league.currentMatchupPeriod}")
        print(f"  Scoring Period: {league.scoringPeriodId}")
        print(f"  Number of Teams: {len(league.teams)}")
        print(f"  League Settings: {league.settings}")

        return league

    except Exception as e:
        print(f"\n✗ Failed to connect to league")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_teams_and_matchups(league):
    """Test 2: Get teams and current week matchups."""
    print("\n" + "="*60)
    print("TEST 2: Teams and Current Matchups")
    print("="*60)

    if not league:
        print("Skipping - no league connection")
        return

    try:
        # Get all teams
        print(f"\nTeams in League:")
        for i, team in enumerate(league.teams, 1):
            print(f"  {i}. {team.team_name} ({team.team_abbrev})")
            print(f"     Record: {team.wins}W-{team.losses}L")
            print(f"     Standing: {team.standing}")
            print(f"     Division: {team.division_name}")
            print(f"     Roster Size: {len(team.roster)}")

        # Get current week matchups
        print(f"\nCurrent Week {league.current_week} Matchups:")
        matchups = league.scoreboard()

        for i, matchup in enumerate(matchups, 1):
            print(f"\n  Matchup {i}:")
            print(f"    {matchup.home_team.team_name} vs {matchup.away_team.team_name}")

            # Check if it's category or points league
            if hasattr(matchup, 'home_score'):
                print(f"    Score: {matchup.home_score} - {matchup.away_score}")
            elif hasattr(matchup, 'home_wins'):
                print(f"    Category Score: {matchup.home_wins}W-{matchup.home_ties}T-{matchup.home_losses}L")

        print("\n✓ Successfully retrieved teams and matchups")

    except Exception as e:
        print(f"\n✗ Failed to get teams/matchups")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_stephen_curry_data(league):
    """Test 3: Get Stephen Curry's player data."""
    print("\n" + "="*60)
    print("TEST 3: Stephen Curry Player Data")
    print("="*60)

    if not league:
        print("Skipping - no league connection")
        return

    try:
        # Try to find Stephen Curry
        print("\nSearching for Stephen Curry...")

        # Method 1: Search by name using player_info
        try:
            curry = league.player_info(name="Stephen Curry")
            print(f"\n✓ Found Stephen Curry via player_info()")
        except:
            # Method 2: Search through all rosters
            curry = None
            for team in league.teams:
                for player in team.roster:
                    if "curry" in player.name.lower() and "stephen" in player.name.lower():
                        curry = player
                        print(f"\n✓ Found Stephen Curry on {team.team_name}")
                        break
                if curry:
                    break

            # Method 3: Search free agents
            if not curry:
                print("\nSearching free agents...")
                free_agents = league.free_agents(size=500)
                for player in free_agents:
                    if "curry" in player.name.lower() and "stephen" in player.name.lower():
                        curry = player
                        print(f"\n✓ Found Stephen Curry as free agent")
                        break

        if not curry:
            print("\n✗ Could not find Stephen Curry")
            print("Trying to get any GSW player instead...")

            for team in league.teams:
                for player in team.roster:
                    if player.proTeam == "GSW":
                        curry = player
                        print(f"\n✓ Using {player.name} from Golden State Warriors instead")
                        break
                if curry:
                    break

        if curry:
            print(f"\nPlayer: {curry.name}")
            print(f"  Player ID: {curry.playerId}")
            print(f"  Position: {curry.position}")
            print(f"  Eligible Slots: {curry.eligibleSlots}")
            print(f"  Pro Team: {curry.proTeam}")
            print(f"  Injury Status: {curry.injuryStatus}")
            print(f"  Acquisition Type: {curry.acquisitionType}")

            # Season stats
            print(f"\nSeason Stats:")
            print(f"  Total Points: {curry.total_points}")
            print(f"  Average Points: {curry.avg_points}")
            print(f"  Projected Total: {curry.projected_total_points}")

            # Nine category averages
            print(f"\nNine Category Averages:")
            nine_cat = curry.nine_cat_averages
            if nine_cat:
                for stat, value in nine_cat.items():
                    print(f"  {stat}: {value}")
            else:
                print("  No nine-cat data available")

            # Stats breakdown
            print(f"\nStats by Period:")
            if hasattr(curry, 'stats') and curry.stats:
                for period, stats in curry.stats.items():
                    print(f"  Period {period}: {stats}")

            # Schedule
            print(f"\nPro Team Schedule:")
            if hasattr(curry, 'schedule') and curry.schedule:
                schedule_items = list(curry.schedule.items())[:5]  # First 5 games
                for date, opponent in schedule_items:
                    print(f"  {date}: vs {opponent}")

            print("\n✓ Successfully retrieved player data")
            return curry
        else:
            print("\n✗ Could not find any player to test with")

    except Exception as e:
        print(f"\n✗ Failed to get player data")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_box_scores(league):
    """Test 4: Get current week box scores."""
    print("\n" + "="*60)
    print("TEST 4: Current Week Box Scores")
    print("="*60)

    if not league:
        print("Skipping - no league connection")
        return

    try:
        print(f"\nGetting box scores for week {league.current_week}...")
        box_scores = league.box_scores()

        if not box_scores:
            print("No box scores available yet")
            return

        for i, box in enumerate(box_scores, 1):
            print(f"\nBox Score {i}:")
            print(f"  {box.home_team.team_name} vs {box.away_team.team_name}")

            # Check league type
            if hasattr(box, 'home_score'):
                # Points league
                print(f"  Score: {box.home_score} - {box.away_score}")
                print(f"  Projected: {box.home_projected} - {box.away_projected}")

                # Show some players
                print(f"\n  Home Team Starters:")
                for player in box.home_lineup[:5]:
                    print(f"    {player.name} ({player.slot_position}): {player.points} pts")
            else:
                # Category league
                print(f"  Category Record: {box.home_wins}W-{box.home_ties}T-{box.home_losses}L")

                if hasattr(box, 'home_stats'):
                    print(f"\n  Home Team Stats:")
                    for stat_name, stat_data in list(box.home_stats.items())[:5]:
                        if isinstance(stat_data, dict):
                            print(f"    {stat_name}: {stat_data.get('value', 'N/A')} ({stat_data.get('result', 'N/A')})")

        print("\n✓ Successfully retrieved box scores")

    except Exception as e:
        print(f"\n✗ Failed to get box scores")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_historical_data(league):
    """Test 5: Try to get historical game logs."""
    print("\n" + "="*60)
    print("TEST 5: Historical Game Logs (if available)")
    print("="*60)

    if not league:
        print("Skipping - no league connection")
        return

    print("\nNote: ESPN Fantasy API does not provide detailed historical")
    print("game-by-game logs. For that data, you should continue using:")
    print("  - NBA API (nba_api library) for official NBA game logs")
    print("  - Your existing data pipeline (2_collect_historical_gamelogs.py)")
    print("  - Existing CSV: ../data/static/active_players_historical_game_logs.csv")

    print("\nWhat ESPN API provides:")
    print("  ✓ Season totals and averages")
    print("  ✓ Last 7/15/30 day stats")
    print("  ✓ Projected stats")
    print("  ✓ Box scores for current week")
    print("  ✗ Historical game-by-game logs (use NBA API)")


def test_free_agents(league):
    """Test 6: Get top free agents."""
    print("\n" + "="*60)
    print("TEST 6: Top Free Agents")
    print("="*60)

    if not league:
        print("Skipping - no league connection")
        return

    try:
        print("\nGetting top 10 free agents...")
        free_agents = league.free_agents(size=10)

        print(f"\nTop Free Agents Available:")
        for i, player in enumerate(free_agents, 1):
            print(f"  {i}. {player.name} ({player.position}) - {player.proTeam}")
            print(f"     Avg Points: {player.avg_points:.1f}")
            if player.nine_cat_averages:
                pts = player.nine_cat_averages.get('PTS', 0)
                reb = player.nine_cat_averages.get('REB', 0)
                ast = player.nine_cat_averages.get('AST', 0)
                print(f"     Stats: {pts:.1f} PTS, {reb:.1f} REB, {ast:.1f} AST")

        print("\n✓ Successfully retrieved free agents")

    except Exception as e:
        print(f"\n✗ Failed to get free agents")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ESPN FANTASY BASKETBALL API TEST SUITE")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"League ID: 40204")
    print(f"Season: 2026 (2025-26 NBA season)")

    # Test 1: Connect to league
    league = test_league_connection()

    if not league:
        print("\n" + "="*60)
        print("TESTS FAILED - Could not connect to league")
        print("="*60)
        return 1

    # Test 2: Teams and matchups
    test_teams_and_matchups(league)

    # Test 3: Stephen Curry data
    test_stephen_curry_data(league)

    # Test 4: Box scores
    test_box_scores(league)

    # Test 5: Historical data explanation
    test_historical_data(league)

    # Test 6: Free agents
    test_free_agents(league)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
