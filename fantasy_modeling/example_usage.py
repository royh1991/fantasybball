#!/usr/bin/env python3
"""
Example usage of the Fantasy Basketball Modeling System.

This script demonstrates the main features of the system.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.data_collector import DataCollector
from models.bayesian_model import BayesianPlayerModel, PlayerContext
from simulation.game_simulator import GameSimulator
from fantasy.scoring_system import ScoringSystem
import pandas as pd


def example_1_load_data():
    """Example 1: Load and explore your existing data."""
    print("="*60)
    print("Example 1: Loading Historical Data")
    print("="*60 + "\n")

    collector = DataCollector()

    # Load historical game logs
    game_logs = collector.load_historical_game_logs(min_games=10)
    print(f"Loaded {len(game_logs)} games for {game_logs['player_id'].nunique()} players\n")

    # Load ESPN projections
    projections = collector.load_espn_projections()
    print(f"Loaded ESPN projections for {len(projections)} players\n")

    # Show sample data
    if not game_logs.empty:
        print("Sample game log:")
        print(game_logs[['player_name', 'pts', 'reb', 'ast', 'fg_pct']].head())
    print()


def example_2_fit_single_player():
    """Example 2: Fit model for a single player."""
    print("="*60)
    print("Example 2: Fitting Model for Single Player")
    print("="*60 + "\n")

    # Load data
    collector = DataCollector()
    player_info, game_logs, projections = collector.prepare_modeling_data()

    # Pick a player with lots of data
    player_games = game_logs.groupby('player_id').size().sort_values(ascending=False)
    if len(player_games) == 0:
        print("No player data found!")
        return

    player_id = player_games.index[0]
    player_logs = game_logs[game_logs['player_id'] == player_id]

    print(f"Fitting model for player: {player_id}")
    print(f"Using {len(player_logs)} historical games\n")

    # Fit the model
    model = BayesianPlayerModel()
    fitted = model.fit_player(
        player_id=player_id,
        game_logs=player_logs,
        position='SF',  # You can get this from player_info
        espn_projection=None
    )

    print("Model fitted successfully!")
    print(f"Shooting models: {list(fitted['shooting'].keys())}")
    print(f"Counting models: {list(fitted['counting'].keys())}")
    print()


def example_3_simulate_game():
    """Example 3: Simulate a single game."""
    print("="*60)
    print("Example 3: Simulating a Single Game")
    print("="*60 + "\n")

    # Initialize simulator
    simulator = GameSimulator(config_path="config")

    # Load and fit data
    collector = DataCollector()
    player_info, game_logs, projections = collector.prepare_modeling_data()

    # Fit a few players
    top_players = game_logs.groupby('player_id').size().sort_values(ascending=False).head(5)

    for player_id in top_players.index[:1]:  # Just do one for the example
        player_logs = game_logs[game_logs['player_id'] == player_id]
        position = 'SF'  # Default

        print(f"Simulating game for: {player_id}")

        # Fit model
        simulator.bayesian_model.fit_player(player_id, player_logs, position)

        # Create context
        context = PlayerContext(
            player_id=player_id,
            position=position,
            team='',
            opponent='',
            is_home=True
        )

        # Simulate
        results = simulator.simulate_game(player_id, context, n_simulations=100)

        # Print results
        print("\nProjected Stats:")
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
            if stat in results['projections']:
                mean = results['projections'][stat]['mean']
                std = results['projections'][stat]['std']
                print(f"  {stat.upper():5s}: {mean:5.1f} ± {std:4.1f}")

        print()


def example_4_fantasy_scoring():
    """Example 4: Calculate fantasy scores."""
    print("="*60)
    print("Example 4: Fantasy Scoring System")
    print("="*60 + "\n")

    # Create scoring system for your 11-cat league
    scoring = ScoringSystem()

    print("Your league categories:")
    for cat in scoring.categories:
        print(f"  - {cat.name:8s} ({cat.type.value:10s}, {cat.better} is better)")

    # Example weekly totals
    team_a = {
        'pts': 850, 'reb': 420, 'ast': 230, 'stl': 80, 'blk': 52,
        'tov': 140, 'fg3m': 95,
        'fg_pct': 0.467, 'ft_pct': 0.812, 'fg3_pct': 0.361,
        'dd': 12
    }

    team_b = {
        'pts': 820, 'reb': 450, 'ast': 215, 'stl': 75, 'blk': 48,
        'tov': 135, 'fg3m': 88,
        'fg_pct': 0.479, 'ft_pct': 0.795, 'fg3_pct': 0.348,
        'dd': 14
    }

    # Compare matchup
    result = scoring.compare_matchup(team_a, team_b)

    print(f"\nMatchup Result: {result['winner']}")
    print(f"Score: {result['score']}")
    print("\nCategory Breakdown:")
    for cat, details in result['category_results'].items():
        winner = details['winner']
        print(f"  {cat:8s}: A={details['team_a']:6.1f} B={details['team_b']:6.1f} [{winner}]")

    print()


def example_5_compare_players():
    """Example 5: Compare two players."""
    print("="*60)
    print("Example 5: Player Comparison")
    print("="*60 + "\n")

    collector = DataCollector()
    player_info, game_logs, _ = collector.prepare_modeling_data()

    # Get two players with data
    players = game_logs.groupby('player_id').size().sort_values(ascending=False).head(2).index

    if len(players) < 2:
        print("Need at least 2 players with data")
        return

    player1_id = players[0]
    player2_id = players[1]

    # Calculate stats
    stats1 = game_logs[game_logs['player_id'] == player1_id][
        ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']
    ].mean().to_dict()

    stats2 = game_logs[game_logs['player_id'] == player2_id][
        ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']
    ].mean().to_dict()

    print(f"Player 1: {player1_id}")
    for stat, val in stats1.items():
        print(f"  {stat.upper():5s}: {val:5.1f}")

    print(f"\nPlayer 2: {player2_id}")
    for stat, val in stats2.items():
        print(f"  {stat.upper():5s}: {val:5.1f}")

    # Use scoring system to compare
    scoring = ScoringSystem()
    league_stats = game_logs[['pts', 'reb', 'ast', 'stl', 'blk', 'tov']]

    comparison = scoring.compare_player_value(stats1, stats2, league_stats)

    print(f"\n{comparison['winner']} is better by {abs(comparison['difference']):.2f} z-score units\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Fantasy Basketball Modeling System - Examples")
    print("="*60 + "\n")

    try:
        # Run all examples
        example_1_load_data()
        example_2_fit_single_player()
        example_3_simulate_game()
        example_4_fantasy_scoring()
        example_5_compare_players()

        print("="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: Make sure you have historical data in ../data/ directory")
        import traceback
        traceback.print_exc()