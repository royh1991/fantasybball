#!/usr/bin/env python3
"""
Fantasy Basketball Modeling System - Main CLI Interface

Usage:
    python main.py fit --players <player_file>
    python main.py simulate --player <player_id> --date <date>
    python main.py project-week --week <week_num>
    python main.py compare --player1 <id1> --player2 <id2>
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.data_collector import DataCollector
from models.bayesian_model import BayesianPlayerModel, PlayerContext
from simulation.game_simulator import GameSimulator
from fantasy.scoring_system import ScoringSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argparse():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Fantasy Basketball Bayesian Modeling System'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Fit models command
    fit_parser = subparsers.add_parser('fit', help='Fit models for all players')
    fit_parser.add_argument('--min-games', type=int, default=10,
                           help='Minimum games required')
    fit_parser.add_argument('--output', type=str, default='fitted_models.json',
                           help='Output file for fitted models')

    # Simulate game command
    sim_parser = subparsers.add_parser('simulate', help='Simulate a game for a player')
    sim_parser.add_argument('--player', type=str, required=True,
                           help='Player ID or name')
    sim_parser.add_argument('--opponent', type=str, default='',
                           help='Opponent team')
    sim_parser.add_argument('--date', type=str, default='',
                           help='Game date (YYYY-MM-DD)')
    sim_parser.add_argument('--n-sims', type=int, default=1000,
                           help='Number of simulations')
    sim_parser.add_argument('--output', type=str, default='simulation_results.json',
                           help='Output file for results')

    # Project week command
    week_parser = subparsers.add_parser('project-week', help='Project weekly matchup')
    week_parser.add_argument('--week', type=int, required=True,
                            help='Week number')
    week_parser.add_argument('--roster', type=str,
                            help='JSON file with roster')
    week_parser.add_argument('--n-sims', type=int, default=1000,
                            help='Number of simulations')

    # Compare players command
    compare_parser = subparsers.add_parser('compare', help='Compare two players')
    compare_parser.add_argument('--player1', type=str, required=True,
                               help='First player ID')
    compare_parser.add_argument('--player2', type=str, required=True,
                               help='Second player ID')

    # Batch simulate command
    batch_parser = subparsers.add_parser('batch', help='Batch simulate multiple players')
    batch_parser.add_argument('--slate', type=str, required=True,
                             help='JSON file with slate of games')
    batch_parser.add_argument('--n-sims', type=int, default=1000,
                             help='Number of simulations per player')
    batch_parser.add_argument('--output', type=str, default='batch_results.csv',
                             help='Output CSV file')

    return parser


def fit_models(args):
    """Fit models for all players."""
    logger.info("Fitting Bayesian models for all players...")

    # Initialize data collector
    data_collector = DataCollector()

    # Load data
    logger.info("Loading historical data...")
    player_info, game_logs, projections = data_collector.prepare_modeling_data()

    # Filter by minimum games
    players_with_enough_data = game_logs.groupby('player_id').size()
    players_with_enough_data = players_with_enough_data[
        players_with_enough_data >= args.min_games
    ].index

    player_info = player_info[player_info['player_id'].isin(players_with_enough_data)]

    logger.info(f"Fitting models for {len(player_info)} players...")

    # Initialize simulator
    config_path = Path(__file__).parent / "config"
    simulator = GameSimulator(str(config_path))

    # Fit all players
    fitted_models = simulator.fit_all_players(player_info, game_logs, projections)

    logger.info(f"Successfully fitted {len(fitted_models)} models")
    logger.info(f"Results saved to {args.output}")

    return fitted_models


def simulate_game(args):
    """Simulate a single game for a player."""
    logger.info(f"Simulating game for player {args.player}")

    # Initialize simulator
    config_path = Path(__file__).parent / "config"
    simulator = GameSimulator(str(config_path))

    # Load data and fit model for this player
    data_collector = DataCollector()
    player_info, game_logs, projections = data_collector.prepare_modeling_data()

    # Find player
    player_id = args.player
    if player_id not in player_info['player_id'].values:
        # Try to find by name
        player_match = player_info[
            player_info['player_name'].str.lower() == player_id.lower()
        ]
        if len(player_match) > 0:
            player_id = player_match.iloc[0]['player_id']
        else:
            logger.error(f"Player {args.player} not found")
            return None

    # Get player info
    player = player_info[player_info['player_id'] == player_id].iloc[0]
    player_logs = game_logs[game_logs['player_id'] == player_id]

    # Fit model
    logger.info(f"Fitting model for {player['player_name']}...")
    simulator.bayesian_model.fit_player(
        player_id,
        player_logs,
        player['position'],
        None  # No ESPN projection for now
    )

    # Create context
    context = PlayerContext(
        player_id=player_id,
        position=player['position'],
        team=player.get('team', ''),
        opponent=args.opponent,
        is_home=True,
        days_rest=1
    )

    # Run simulation
    logger.info(f"Running {args.n_sims} simulations...")
    results = simulator.simulate_game(player_id, context, n_simulations=args.n_sims)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Simulation Results for {player['player_name']}")
    print(f"{'='*60}\n")

    print("Projected Stats (Mean):")
    for stat, values in results['projections'].items():
        if isinstance(values, dict) and 'mean' in values:
            print(f"  {stat.upper():8s}: {values['mean']:6.2f} ± {values.get('std', 0):5.2f}")

    print(f"\nFantasy Value:")
    if 'fantasy' in results and '11cat' in results['fantasy']:
        for stat, val in results['fantasy']['11cat'].items():
            if stat in ['fg3_pct', 'dd']:
                print(f"  {stat.upper():8s}: {val:6.2f}")

    # Save results
    simulator.save_results(results, args.output)
    logger.info(f"Full results saved to {args.output}")

    return results


def project_week(args):
    """Project weekly matchup."""
    logger.info(f"Projecting week {args.week}")

    # Load roster
    if not args.roster:
        logger.error("Roster file required for week projection")
        return None

    with open(args.roster, 'r') as f:
        roster = json.load(f)

    logger.info(f"Loaded roster with {len(roster)} players")

    # Initialize simulator
    config_path = Path(__file__).parent / "config"
    simulator = GameSimulator(str(config_path))

    # Load data and fit models
    data_collector = DataCollector()
    player_info, game_logs, projections = data_collector.prepare_modeling_data()

    simulator.fit_all_players(player_info, game_logs, projections)

    # Load matchup data for the week
    matchup_data = data_collector.load_matchup_data(week=args.week)

    logger.info(f"Simulating week {args.week} matchup...")

    # TODO: Implement full weekly simulation with roster optimization

    return None


def compare_players(args):
    """Compare two players."""
    logger.info(f"Comparing {args.player1} vs {args.player2}")

    # Initialize
    data_collector = DataCollector()
    player_info, game_logs, projections = data_collector.prepare_modeling_data()
    scoring_system = ScoringSystem()

    # Get player stats
    player1_logs = game_logs[game_logs['player_id'] == args.player1]
    player2_logs = game_logs[game_logs['player_id'] == args.player2]

    if len(player1_logs) == 0 or len(player2_logs) == 0:
        logger.error("One or both players not found")
        return None

    # Calculate per-game averages
    stats1 = player1_logs[['pts', 'reb', 'ast', 'stl', 'blk', 'tov']].mean().to_dict()
    stats2 = player2_logs[['pts', 'reb', 'ast', 'stl', 'blk', 'tov']].mean().to_dict()

    # Calculate z-scores
    league_stats = game_logs[['pts', 'reb', 'ast', 'stl', 'blk', 'tov']]

    comparison = scoring_system.compare_player_value(stats1, stats2, league_stats)

    # Print results
    print(f"\n{'='*60}")
    print(f"Player Comparison")
    print(f"{'='*60}\n")

    print(f"Player 1 ({args.player1}):")
    print(f"  Total Value: {comparison['player_a']['total_value']:.2f}")
    print(f"  Z-Scores: {comparison['player_a']['z_scores']}")

    print(f"\nPlayer 2 ({args.player2}):")
    print(f"  Total Value: {comparison['player_b']['total_value']:.2f}")
    print(f"  Z-Scores: {comparison['player_b']['z_scores']}")

    print(f"\nWinner: {comparison['winner']}")
    print(f"Value Difference: {comparison['difference']:.2f}\n")

    return comparison


def batch_simulate(args):
    """Batch simulate multiple players."""
    logger.info(f"Running batch simulation from {args.slate}")

    # Load slate
    with open(args.slate, 'r') as f:
        slate = json.load(f)

    logger.info(f"Loaded slate with {len(slate)} players")

    # Initialize simulator
    config_path = Path(__file__).parent / "config"
    simulator = GameSimulator(str(config_path))

    # Load data and fit models
    data_collector = DataCollector()
    player_info, game_logs, projections = data_collector.prepare_modeling_data()

    simulator.fit_all_players(player_info, game_logs, projections)

    # Run batch simulation
    logger.info(f"Simulating {len(slate)} players...")
    results_df = simulator.simulate_slate(slate, n_simulations=args.n_sims, parallel=True)

    # Save results
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch Simulation Results")
    print(f"{'='*60}\n")
    print(results_df[['player_id', 'pts_mean', 'reb_mean', 'ast_mean']].head(10))

    return results_df


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'fit':
            fit_models(args)
        elif args.command == 'simulate':
            simulate_game(args)
        elif args.command == 'project-week':
            project_week(args)
        elif args.command == 'compare':
            compare_players(args)
        elif args.command == 'batch':
            batch_simulate(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())