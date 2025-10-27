#!/usr/bin/env python3
"""
Main orchestration script for fantasy_2026 data collection pipeline.

This script runs the complete data collection workflow:
1. Extract current rosters from ESPN API
2. Collect historical game logs for all players (4 seasons: current + 3 historical)
3. Get current week matchup data and box scores
4. Create player mapping (links ESPN names to NBA API names)
5. Extract daily roster history (who owned which players on each date)

Usage:
    python run_all.py

    Or run individual steps:
    python run_all.py --step 1  # Only extract rosters
    python run_all.py --step 2  # Only collect historical data
    python run_all.py --step 3  # Only get matchup data
    python run_all.py --step 4  # Only create player mapping
    python run_all.py --step 5  # Only extract daily roster history
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess


def setup_environment():
    """Verify environment variables and dependencies."""
    print("=" * 60)
    print("ENVIRONMENT SETUP")
    print("=" * 60)

    # Check ESPN credentials
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("\n✗ ERROR: ESPN credentials not found in environment variables")
        print("  Please set ESPN_S2 and ESPN_SWID")
        print("\nExample:")
        print('  export ESPN_S2="your_espn_s2_cookie"')
        print('  export ESPN_SWID="{your_swid_cookie}"')
        return False

    print(f"✓ ESPN_S2: {espn_s2[:20]}...")
    print(f"✓ ESPN_SWID: {swid}")

    # Check required Python packages
    required_packages = [
        'pandas',
        'yaml',
        'nba_api',
        'fuzzywuzzy',
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} not found")

    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False

    # Check espn-api
    espn_api_path = Path(__file__).parent.parent / "espn-api"
    if not espn_api_path.exists():
        print(f"\n✗ espn-api directory not found at {espn_api_path}")
        print("  Clone it with: git clone https://github.com/cwendt94/espn-api")
        return False

    print(f"✓ espn-api found at {espn_api_path}")

    print("\n✓ All dependencies satisfied\n")
    return True


def run_script(script_name, description):
    """Run a Python script and capture output."""
    print("=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)

    script_path = Path(__file__).parent / "scripts" / script_name

    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False

    print(f"\nRunning: {script_path}\n")

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )

        print(f"\n✓ {description} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error running {description}")
        print(f"Error: {e}")
        return False


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Run fantasy_2026 data collection pipeline"
    )
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only a specific step (1=rosters, 2=historical data, 3=matchups, 4=player mapping, 5=daily rosters)'
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FANTASY 2026 DATA COLLECTION PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup and verify environment
    if not setup_environment():
        print("\n" + "=" * 60)
        print("✗ PIPELINE FAILED - Environment setup issues")
        print("=" * 60)
        return 1

    # Determine which steps to run
    steps_to_run = []

    if args.step is None:
        # Run all steps
        steps_to_run = [1, 2, 3, 4, 5]
    else:
        # Run only specified step
        steps_to_run = [args.step]

    # Execute steps
    success = True

    if 1 in steps_to_run:
        if not run_script("1_extract_rosters.py", "Extract ESPN Fantasy Rosters"):
            success = False
            if args.step is None:
                # If running all steps, stop on first failure
                print("\n✗ Stopping pipeline due to roster extraction failure")
                return 1

    if 2 in steps_to_run and success:
        if not run_script("2_collect_historical_data.py", "Collect Historical Game Logs"):
            success = False

    if 3 in steps_to_run and success:
        if not run_script("3_get_matchups.py", "Get Current Week Matchups"):
            success = False

    if 4 in steps_to_run and success:
        if not run_script("4_create_player_mapping.py", "Create Player Mapping"):
            success = False

    if 5 in steps_to_run and success:
        if not run_script("5_daily_roster_history.py", "Extract Daily Roster History"):
            success = False

    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("\nOutput Files:")

        # Check for roster snapshots
        roster_dir = Path(__file__).parent / "data" / "roster_snapshots"
        if roster_dir.exists():
            roster_latest = roster_dir / "roster_latest.csv"
            if roster_latest.exists():
                print(f"  Rosters: {roster_latest}")

        # Check for historical game logs
        gamelogs_dir = Path(__file__).parent / "data" / "historical_gamelogs"
        if gamelogs_dir.exists():
            gamelogs_latest = gamelogs_dir / "historical_gamelogs_latest.csv"
            if gamelogs_latest.exists():
                print(f"  Game Logs: {gamelogs_latest}")

        # Check for matchup data
        matchups_dir = Path(__file__).parent / "data" / "matchups"
        if matchups_dir.exists():
            matchups_latest = matchups_dir / "matchups_latest.csv"
            box_scores_latest = matchups_dir / "box_scores_latest.csv"
            if matchups_latest.exists():
                print(f"  Matchups: {matchups_latest}")
            if box_scores_latest.exists():
                print(f"  Box Scores: {box_scores_latest}")

        # Check for player mapping
        mappings_dir = Path(__file__).parent / "data" / "mappings"
        if mappings_dir.exists():
            mapping_latest = mappings_dir / "player_mapping_latest.csv"
            if mapping_latest.exists():
                print(f"  Player Mapping: {mapping_latest}")

        # Check for daily roster history
        ownership_dir = Path(__file__).parent / "data" / "ownership_history"
        if ownership_dir.exists():
            daily_rosters_latest = ownership_dir / "daily_rosters_latest.csv"
            if daily_rosters_latest.exists():
                print(f"  Daily Rosters: {daily_rosters_latest}")

        print("\nNext Steps:")
        print("  1. Review the roster snapshot to verify all teams are captured")
        print("  2. Check historical game logs for completeness")
        print("  3. Review current week matchups and box scores")
        print("  4. Use player mapping to join roster and game log data")
        print("  5. Use daily rosters to filter games by ownership")
        print("  6. Use the data for your fantasy basketball modeling")

        return 0
    else:
        print("✗ PIPELINE FAILED")
        print("=" * 60)
        print("\nCheck the error messages above for details")
        return 1


if __name__ == '__main__':
    sys.exit(main())
