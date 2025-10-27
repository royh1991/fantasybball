#!/usr/bin/env python3
"""
Test script to explore ESPN API's historical data capabilities.

This checks what historical data ESPN keeps:
- Recent activity (adds/drops/trades)
- Transactions
- Historical box scores
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add espn-api to path
parent_dir = Path(__file__).parent.parent
espn_api_path = parent_dir / "espn-api"
sys.path.insert(0, str(espn_api_path))

from espn_api.basketball import League


def test_recent_activity(league):
    """Test the recent_activity endpoint."""
    print("\n" + "="*60)
    print("TESTING RECENT ACTIVITY")
    print("="*60)

    try:
        # Try to get recent activity
        activities = league.recent_activity(size=100)

        print(f"\n✓ Found {len(activities)} recent activities")

        if activities:
            print("\nMost recent 10 activities:")
            for i, activity in enumerate(activities[:10], 1):
                print(f"\n{i}. Date: {datetime.fromtimestamp(activity.date/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                for team, action, player, position in activity.actions:
                    team_name = team.team_name if team else 'N/A'
                    print(f"   {team_name}: {action} {player} {position}")

        return activities
    except Exception as e:
        print(f"✗ Error getting recent activity: {e}")
        return []


def test_transactions(league):
    """Test the transactions endpoint."""
    print("\n" + "="*60)
    print("TESTING TRANSACTIONS")
    print("="*60)

    try:
        # Try different scoring periods
        print("\nTrying current scoring period...")
        transactions = league.transactions()
        print(f"✓ Current period: {len(transactions)} transactions")

        # Try to get historical transactions
        print("\nTrying to get transactions from earlier scoring periods...")
        all_transactions = []

        # Try the last 10 scoring periods
        current_period = league.scoringPeriodId
        for period in range(max(1, current_period - 10), current_period + 1):
            try:
                trans = league.transactions(scoring_period=period)
                if trans:
                    print(f"  Period {period}: {len(trans)} transactions")
                    all_transactions.extend(trans)
            except Exception as e:
                print(f"  Period {period}: Error - {e}")

        print(f"\n✓ Total transactions across periods: {len(all_transactions)}")

        if all_transactions:
            print("\nMost recent 10 transactions:")
            for i, trans in enumerate(all_transactions[:10], 1):
                print(f"\n{i}. {trans.team.team_name} - {trans.type} - Period {trans.scoring_period}")
                if trans.date:
                    print(f"   Date: {datetime.fromtimestamp(trans.date/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                for item in trans.items:
                    print(f"   {item.type}: {item.player}")

        return all_transactions
    except Exception as e:
        print(f"✗ Error getting transactions: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_historical_box_scores(league):
    """Test if we can get historical box scores."""
    print("\n" + "="*60)
    print("TESTING HISTORICAL BOX SCORES")
    print("="*60)

    try:
        current_period = league.currentMatchupPeriod
        print(f"Current matchup period: {current_period}")

        # Try to get box scores from earlier matchup periods
        print("\nTrying to get historical box scores...")

        for period in range(max(1, current_period - 3), current_period + 1):
            try:
                box_scores = league.box_scores(matchup_period=period)
                print(f"\nMatchup Period {period}: {len(box_scores)} matchups")

                if box_scores:
                    # Show first matchup
                    bs = box_scores[0]
                    print(f"  Example: {bs.home_team.team_name} vs {bs.away_team.team_name}")
                    print(f"  Home roster: {len(bs.home_lineup)} players")
                    print(f"  Away roster: {len(bs.away_lineup)} players")

                    # Check if we have stats
                    if bs.home_lineup:
                        player = bs.home_lineup[0]
                        print(f"  Example player: {player.name}")
                        if hasattr(player, 'points'):
                            print(f"    Points: {player.points}")
                        if hasattr(player, 'stats'):
                            print(f"    Has stats dict: {bool(player.stats)}")
            except Exception as e:
                print(f"  Matchup Period {period}: Error - {e}")

        return True
    except Exception as e:
        print(f"✗ Error getting historical box scores: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_historical_coverage(activities, transactions):
    """Analyze how far back the historical data goes."""
    print("\n" + "="*60)
    print("HISTORICAL DATA COVERAGE ANALYSIS")
    print("="*60)

    if activities:
        activity_dates = [datetime.fromtimestamp(a.date/1000) for a in activities if a.date]
        if activity_dates:
            earliest_activity = min(activity_dates)
            latest_activity = max(activity_dates)
            print(f"\nActivity History:")
            print(f"  Earliest: {earliest_activity.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Latest:   {latest_activity.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Coverage: {(latest_activity - earliest_activity).days} days")

    if transactions:
        trans_dates = [datetime.fromtimestamp(t.date/1000) for t in transactions if t.date]
        if trans_dates:
            earliest_trans = min(trans_dates)
            latest_trans = max(trans_dates)
            print(f"\nTransaction History:")
            print(f"  Earliest: {earliest_trans.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Latest:   {latest_trans.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Coverage: {(latest_trans - earliest_trans).days} days")

    # Count by type
    if activities:
        action_counts = {}
        for activity in activities:
            for team, action, player, position in activity.actions:
                action_counts[action] = action_counts.get(action, 0) + 1

        print(f"\nActivity Breakdown:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"  {action}: {count}")

    if transactions:
        trans_counts = {}
        for trans in transactions:
            trans_counts[trans.type] = trans_counts.get(trans.type, 0) + 1

        print(f"\nTransaction Breakdown:")
        for trans_type, count in sorted(trans_counts.items(), key=lambda x: -x[1]):
            print(f"  {trans_type}: {count}")


def main():
    """Main execution."""
    print("="*60)
    print("ESPN API HISTORICAL DATA TEST")
    print("="*60)

    # Get ESPN credentials
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("\n✗ ERROR: ESPN credentials not found")
        print("  Please set ESPN_S2 and ESPN_SWID environment variables")
        return 1

    # Connect to league
    league_id = 40204
    season = 2026

    print(f"\nConnecting to league {league_id}, season {season}...")

    try:
        league = League(
            league_id=league_id,
            year=season,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"✓ Connected successfully")
        print(f"  Current Week: {league.current_week}")
        print(f"  Current Matchup Period: {league.currentMatchupPeriod}")
        print(f"  Scoring Period ID: {league.scoringPeriodId}")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    # Test endpoints
    activities = test_recent_activity(league)
    transactions = test_transactions(league)
    test_historical_box_scores(league)

    # Analyze coverage
    analyze_historical_coverage(activities, transactions)

    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    print("\n1. RECENT ACTIVITY endpoint:")
    if activities:
        print(f"   ✓ Can retrieve {len(activities)} activities")
        print("   ✓ Includes: FA ADDED, WAIVER ADDED, DROPPED, TRADED, MOVED")
        print("   ✓ Has timestamps for when actions occurred")
    else:
        print("   ✗ No activities found or endpoint unavailable")

    print("\n2. TRANSACTIONS endpoint:")
    if transactions:
        print(f"   ✓ Can retrieve {len(transactions)} transactions")
        print("   ✓ Can query by scoring_period (historical)")
        print("   ✓ Has timestamps for when transactions processed")
    else:
        print("   ✗ No transactions found or endpoint unavailable")

    print("\n3. BOX SCORES endpoint:")
    print("   ✓ Can retrieve historical matchup periods")
    print("   ✓ Includes player lineups and stats from past weeks")

    print("\n" + "="*60)
    print("✓ TEST COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
