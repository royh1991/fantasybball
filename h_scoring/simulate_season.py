"""
Simulate fantasy basketball seasons with drafted teams.

Simulates 100 seasons where:
- Each team plays every other team twice (home/away)
- Weekly matchups use player means + variance sampling
- Each player has 3 games per week
- Outputs team rankings based on win percentage
"""

import pandas as pd
import numpy as np
from simulate_draft import DraftSimulator
import json
from datetime import datetime
import unicodedata
import argparse


def normalize_name(name):
    """Normalize player names by removing unicode characters."""
    # Convert to NFD (decomposed) form and remove combining characters
    nfd = unicodedata.normalize('NFD', name)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


class SeasonSimulator:
    """Simulate fantasy basketball seasons."""

    def __init__(self, drafted_teams, player_data, player_variances):
        """
        Initialize season simulator.

        Parameters:
        -----------
        drafted_teams : dict
            Dictionary of team rosters {team_id: [player_names]}
        player_data : DataFrame
            Weekly player statistics
        player_variances : dict
            Player variance data
        """
        self.teams = drafted_teams
        self.num_teams = len(drafted_teams)
        self.player_data = player_data
        self.player_variances = player_variances

        # Categories for scoring
        self.categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'DD', 'FG_PCT', 'FT_PCT', 'FG3_PCT']

        # Create name mapping for unicode characters
        self.name_mapping = self._create_name_mapping()

        # Calculate player means
        self.player_means = self._calculate_player_means()

    def _create_name_mapping(self):
        """Create mapping from normalized names to actual names in dataset."""
        name_mapping = {}
        for player_name in self.player_data['PLAYER_NAME'].unique():
            normalized = normalize_name(player_name)
            name_mapping[normalized] = player_name
            # Also map the original name to itself
            name_mapping[player_name] = player_name
        return name_mapping

    def _lookup_player(self, player_name):
        """Look up player with unicode-safe name matching."""
        # First try exact match
        if player_name in self.player_means:
            return player_name

        # Try normalized match
        normalized = normalize_name(player_name)
        if normalized in self.name_mapping:
            return self.name_mapping[normalized]

        # Not found
        return None

    def _calculate_player_means(self):
        """Calculate mean stats per game for each player."""
        player_means = {}

        for player_name in self.player_data['PLAYER_NAME'].unique():
            player_df = self.player_data[self.player_data['PLAYER_NAME'] == player_name]

            # Calculate per-game means (divide weekly totals by games played)
            means = {}
            for cat in self.categories:
                if cat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                    # Percentages are already averages
                    means[cat] = player_df[cat].mean()
                else:
                    # Counting stats: divide by games played
                    total = (player_df[cat] * player_df['GAMES_PLAYED']).sum()
                    games = player_df['GAMES_PLAYED'].sum()
                    means[cat] = total / games if games > 0 else 0

            player_means[player_name] = means

        return player_means

    def simulate_player_week(self, player_name, games_per_week=3):
        """
        Simulate one week of stats for a player.

        Parameters:
        -----------
        player_name : str
            Player name
        games_per_week : int
            Number of games to simulate

        Returns:
        --------
        dict : Weekly stats by category
        """
        # Look up player with unicode-safe matching
        actual_name = self._lookup_player(player_name)

        if actual_name is None or actual_name not in self.player_means:
            # Player not in database - print warning
            print(f"  WARNING: No data for {player_name}, using zeros")
            return {cat: 0 for cat in self.categories}

        means = self.player_means[actual_name]
        weekly_stats = {}

        for cat in self.categories:
            if cat not in self.player_variances.get(actual_name, {}):
                # No variance data, use mean
                if cat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                    weekly_stats[cat] = means[cat]
                else:
                    weekly_stats[cat] = means[cat] * games_per_week
                continue

            var_data = self.player_variances[actual_name][cat]
            mean_per_game = var_data['mean_per_game']
            std_per_game = var_data['std_per_game']

            if cat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                # Percentages: sample from normal, clip to [0, 1]
                weekly_stats[cat] = np.clip(
                    np.random.normal(mean_per_game, std_per_game),
                    0.0, 1.0
                )
            else:
                # Counting stats: sum of per-game samples
                game_samples = np.random.normal(mean_per_game, std_per_game, games_per_week)
                # Ensure non-negative (except TOV which can be any value)
                if cat != 'TOV':
                    game_samples = np.maximum(game_samples, 0)
                weekly_stats[cat] = game_samples.sum()

        return weekly_stats

    def simulate_matchup(self, team1_id, team2_id, games_per_week=3):
        """
        Simulate one weekly matchup between two teams.

        Parameters:
        -----------
        team1_id : int
            Team 1 ID
        team2_id : int
            Team 2 ID
        games_per_week : int
            Games per player per week

        Returns:
        --------
        int : Winner (team1_id, team2_id, or 0 for tie)
        """
        team1_roster = self.teams[team1_id]
        team2_roster = self.teams[team2_id]

        # Simulate weekly stats for each team
        team1_stats = {cat: 0 for cat in self.categories}
        team2_stats = {cat: 0 for cat in self.categories}

        # Accumulate counting stats, track percentage stats separately
        team1_makes = {'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0}
        team2_makes = {'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0}

        # Team 1
        for player in team1_roster:
            week_stats = self.simulate_player_week(player, games_per_week)
            for cat in self.categories:
                if cat not in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                    team1_stats[cat] += week_stats[cat]

            # Calculate makes/attempts for percentages
            # Approximate: FGM = FGA * FG_PCT
            if player in self.player_means:
                fga = self.player_means[player].get('FGA', 0) * games_per_week if 'FGA' in self.player_means[player] else 0
                fta = self.player_means[player].get('FTA', 0) * games_per_week if 'FTA' in self.player_means[player] else 0
                fg3a = self.player_means[player].get('FG3M', 0) * games_per_week / (week_stats.get('FG3_PCT', 0.35) + 0.001)

                team1_makes['FGA'] += fga
                team1_makes['FGM'] += fga * week_stats['FG_PCT']
                team1_makes['FTA'] += fta
                team1_makes['FTM'] += fta * week_stats['FT_PCT']
                team1_makes['FG3A'] += fg3a
                team1_makes['FG3M'] += fg3a * week_stats['FG3_PCT']

        # Team 2
        for player in team2_roster:
            week_stats = self.simulate_player_week(player, games_per_week)
            for cat in self.categories:
                if cat not in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                    team2_stats[cat] += week_stats[cat]

            # Calculate makes/attempts for percentages
            if player in self.player_means:
                fga = self.player_means[player].get('FGA', 0) * games_per_week if 'FGA' in self.player_means[player] else 0
                fta = self.player_means[player].get('FTA', 0) * games_per_week if 'FTA' in self.player_means[player] else 0
                fg3a = self.player_means[player].get('FG3M', 0) * games_per_week / (week_stats.get('FG3_PCT', 0.35) + 0.001)

                team2_makes['FGA'] += fga
                team2_makes['FGM'] += fga * week_stats['FG_PCT']
                team2_makes['FTA'] += fta
                team2_makes['FTM'] += fta * week_stats['FT_PCT']
                team2_makes['FG3A'] += fg3a
                team2_makes['FG3M'] += fg3a * week_stats['FG3_PCT']

        # Calculate team percentages
        team1_stats['FG_PCT'] = team1_makes['FGM'] / team1_makes['FGA'] if team1_makes['FGA'] > 0 else 0
        team1_stats['FT_PCT'] = team1_makes['FTM'] / team1_makes['FTA'] if team1_makes['FTA'] > 0 else 0
        team1_stats['FG3_PCT'] = team1_makes['FG3M'] / team1_makes['FG3A'] if team1_makes['FG3A'] > 0 else 0

        team2_stats['FG_PCT'] = team2_makes['FGM'] / team2_makes['FGA'] if team2_makes['FGA'] > 0 else 0
        team2_stats['FT_PCT'] = team2_makes['FTM'] / team2_makes['FTA'] if team2_makes['FTA'] > 0 else 0
        team2_stats['FG3_PCT'] = team2_makes['FG3M'] / team2_makes['FG3A'] if team2_makes['FG3A'] > 0 else 0

        # Count category wins
        team1_wins = 0
        team2_wins = 0

        for cat in self.categories:
            if cat == 'TOV':
                # Lower is better for turnovers
                if team1_stats[cat] < team2_stats[cat]:
                    team1_wins += 1
                elif team2_stats[cat] < team1_stats[cat]:
                    team2_wins += 1
            else:
                # Higher is better
                if team1_stats[cat] > team2_stats[cat]:
                    team1_wins += 1
                elif team2_stats[cat] > team1_stats[cat]:
                    team2_wins += 1

        # Determine winner
        if team1_wins > team2_wins:
            return team1_id
        elif team2_wins > team1_wins:
            return team2_id
        else:
            return 0  # Tie

    def simulate_season(self):
        """
        Simulate one full season.

        Each team plays every other team twice (home and away).

        Returns:
        --------
        dict : Season results {team_id: {'wins': int, 'losses': int, 'ties': int}}
        """
        results = {team_id: {'wins': 0, 'losses': 0, 'ties': 0} for team_id in self.teams.keys()}

        # Generate schedule: each team plays each other team twice
        for team1_id in self.teams.keys():
            for team2_id in self.teams.keys():
                if team1_id >= team2_id:
                    continue  # Skip self-matchups and duplicates

                # Play twice (home and away)
                for _ in range(2):
                    winner = self.simulate_matchup(team1_id, team2_id)

                    if winner == team1_id:
                        results[team1_id]['wins'] += 1
                        results[team2_id]['losses'] += 1
                    elif winner == team2_id:
                        results[team2_id]['wins'] += 1
                        results[team1_id]['losses'] += 1
                    else:
                        results[team1_id]['ties'] += 1
                        results[team2_id]['ties'] += 1

        return results

    def simulate_multiple_seasons(self, num_seasons=100):
        """
        Simulate multiple seasons.

        Parameters:
        -----------
        num_seasons : int
            Number of seasons to simulate

        Returns:
        --------
        DataFrame : Aggregated results across all seasons
        """
        print(f"\nSimulating {num_seasons} seasons...")
        print(f"Each team plays every other team twice per season")
        print(f"Total matchups per team per season: {(self.num_teams - 1) * 2}")

        all_results = {team_id: {'total_wins': 0, 'total_losses': 0, 'total_ties': 0}
                      for team_id in self.teams.keys()}

        for season_num in range(1, num_seasons + 1):
            if season_num % 10 == 0:
                print(f"  Completed {season_num}/{num_seasons} seasons...")

            season_results = self.simulate_season()

            for team_id, results in season_results.items():
                all_results[team_id]['total_wins'] += results['wins']
                all_results[team_id]['total_losses'] += results['losses']
                all_results[team_id]['total_ties'] += results['ties']

        # Calculate win percentages
        results_df = []
        for team_id, results in all_results.items():
            total_games = results['total_wins'] + results['total_losses'] + results['total_ties']
            win_pct = results['total_wins'] / total_games if total_games > 0 else 0

            results_df.append({
                'team_id': team_id,
                'wins': results['total_wins'],
                'losses': results['total_losses'],
                'ties': results['total_ties'],
                'win_pct': win_pct,
                'team_roster': ', '.join(self.teams[team_id][:3]) + '...'  # First 3 players
            })

        results_df = pd.DataFrame(results_df)
        results_df = results_df.sort_values('win_pct', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)

        return results_df


def main():
    """Run draft simulation followed by season simulations."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run H-scoring draft + season simulation')
    parser.add_argument(
        '-p', '--position',
        type=int,
        default=6,
        choices=range(1, 13),
        metavar='[1-12]',
        help='Your draft position (1-12). Default: 6'
    )
    parser.add_argument(
        '-n', '--num-seasons',
        type=int,
        default=100,
        metavar='N',
        help='Number of seasons to simulate. Default: 100'
    )
    parser.add_argument(
        '-t', '--num-teams',
        type=int,
        default=12,
        choices=range(2, 21),
        metavar='[2-20]',
        help='Number of teams in league. Default: 12'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DRAFT + SEASON SIMULATION")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  - Draft Position: {args.position}")
    print(f"  - Number of Teams: {args.num_teams}")
    print(f"  - Seasons to Simulate: {args.num_seasons}")

    # Step 1: Run draft
    print("\nSTEP 1: Running Draft Simulation...")
    print("-" * 80)

    import os
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])
    adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    print(f"Using data files:")
    print(f"  - {data_file}")
    print(f"  - {variance_file}")

    # Run draft
    simulator = DraftSimulator(
        adp_file=adp_file,
        data_file=data_file,
        variance_file=variance_file,
        your_position=args.position,
        num_teams=args.num_teams,
        roster_size=13
    )

    simulator.run_draft()

    # Get drafted teams
    drafted_teams = simulator.all_teams

    # Step 2: Load data for season simulation
    print("\n" + "=" * 80)
    print("STEP 2: Simulating Seasons...")
    print("-" * 80)

    player_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Run season simulations
    season_sim = SeasonSimulator(drafted_teams, player_data, player_variances)
    results = season_sim.simulate_multiple_seasons(num_seasons=args.num_seasons)

    # Display results
    print("\n" + "=" * 80)
    print(f"SEASON SIMULATION RESULTS ({args.num_seasons} seasons)")
    print("=" * 80)
    print(results[['rank', 'team_id', 'wins', 'losses', 'win_pct', 'team_roster']].to_string(index=False))

    # Highlight your team
    your_team_result = results[results['team_id'] == args.position].iloc[0]
    print(f"\n{'='*80}")
    print(f"YOUR TEAM (Team {args.position}) PERFORMANCE:")
    print(f"{'='*80}")
    print(f"  Rank: {your_team_result['rank']}/{args.num_teams}")
    print(f"  Wins: {your_team_result['wins']}")
    print(f"  Losses: {your_team_result['losses']}")
    print(f"  Win %: {your_team_result['win_pct']:.3f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'season_results_{timestamp}.csv'
    results.to_csv(output_file, index=False)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
