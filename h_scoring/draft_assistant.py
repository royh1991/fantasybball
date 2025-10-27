"""
H-Scoring Draft Assistant

Main script to run H-scoring draft recommendations.
Integrates data collection, scoring, covariance, and optimization.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from modules.data_collector import NBADataCollector
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


class DraftAssistant:
    """Main draft assistant using H-scoring."""

    def __init__(self, data_file=None, variance_file=None, format='each_category'):
        """
        Initialize draft assistant.

        Parameters:
        -----------
        data_file : str, optional
            Path to league data CSV
        variance_file : str, optional
            Path to variance JSON
        format : str
            'each_category' or 'most_categories'
        """
        self.format = format
        self.my_team = []
        self.opponent_rosters = []
        self.picks_made = 0
        self.last_optimal_weights = None
        self.weight_history = []

        # Load or collect data
        if data_file and variance_file:
            self.load_data(data_file, variance_file)
        else:
            print("No data files provided. Run collect_data() first.")

    def collect_data(self, seasons=['2023-24'], max_players=200):
        """
        Collect NBA data using nba_api.

        Parameters:
        -----------
        seasons : list of str
            Seasons to collect
        max_players : int
            Max players to collect
        """
        print("=" * 60)
        print("COLLECTING NBA DATA")
        print("=" * 60)

        collector = NBADataCollector(
            seasons=seasons,
            data_dir='data'
        )

        league_data, player_variances = collector.collect_league_data(
            min_weeks=20,
            max_players=max_players
        )

        data_file, variance_file = collector.save_data(league_data, player_variances)

        self.load_data(data_file, variance_file)

        return data_file, variance_file

    def load_data(self, data_file, variance_file):
        """Load existing data files."""
        print(f"\nLoading data from {data_file}")
        self.league_data = pd.read_csv(data_file)

        print(f"Loading variances from {variance_file}")
        with open(variance_file, 'r') as f:
            self.player_variances = json.load(f)

        # Load do_not_draft list
        do_not_draft_file = 'data/do_not_draft.csv'
        if os.path.exists(do_not_draft_file):
            print(f"Loading do_not_draft list from {do_not_draft_file}")
            df = pd.read_csv(do_not_draft_file)
            self.do_not_draft = set(df['do_not_draft'].tolist())
            print(f"  Loaded {len(self.do_not_draft)} players to exclude")
        else:
            self.do_not_draft = set()
            print("No do_not_draft.csv found - no players will be excluded")

        # Initialize scoring system
        print("Initializing scoring system...")
        self.scoring = PlayerScoring(
            self.league_data,
            self.player_variances,
            roster_size=13
        )

        # Initialize covariance calculator
        print("Calculating covariance matrix...")
        self.cov_calc = CovarianceCalculator(
            self.league_data,
            self.scoring
        )

        # Get setup parameters
        self.setup_params = self.cov_calc.get_setup_params()

        # Initialize H-score optimizer (paper-faithful version)
        print("Initializing paper-faithful H-score optimizer...")
        self.optimizer = HScoreOptimizerPaperFaithful(
            self.setup_params,
            self.scoring,
            omega=0.7,
            gamma=0.25
        )

        print("✓ Initialization complete!\n")

    def get_player_rankings(self, method='g_score', top_n=100):
        """
        Get player rankings.

        Parameters:
        -----------
        method : str
            'g_score' or 'h_score'
        top_n : int
            Number of top players

        Returns:
        --------
        DataFrame : Player rankings
        """
        if method == 'g_score':
            return self.scoring.rank_players_by_g_score(top_n=top_n)
        else:
            # H-score rankings
            return self._calculate_h_score_rankings(top_n=top_n)

    def _calculate_h_score_rankings(self, top_n=100):
        """Calculate H-score rankings for available players."""
        # Get top players by G-score as candidates
        g_rankings = self.scoring.rank_players_by_g_score(top_n=top_n * 2)

        h_scores = []

        print(f"Calculating H-scores for top {len(g_rankings)} players...")

        for idx, row in g_rankings.iterrows():
            player_name = row['PLAYER_NAME']

            # Skip if already drafted
            if player_name in self.my_team:
                continue

            # Calculate H-score
            h_score, optimal_weights = self.optimizer.evaluate_player(
                player_name,
                self.my_team,
                self.opponent_rosters,
                self.picks_made,
                total_picks=13,
                last_weights=self.last_optimal_weights,
                format=self.format
            )

            h_scores.append({
                'PLAYER_NAME': player_name,
                'H_SCORE': h_score,
                'G_SCORE': row['TOTAL_G_SCORE']
            })

            if len(h_scores) >= top_n:
                break

        h_rankings = pd.DataFrame(h_scores)
        h_rankings = h_rankings.sort_values('H_SCORE', ascending=False)
        h_rankings['H_RANK'] = range(1, len(h_rankings) + 1)

        return h_rankings

    def recommend_pick(self, available_players=None, top_n=20):
        """
        Recommend best pick using H-scoring.

        Parameters:
        -----------
        available_players : list of str, optional
            List of available player names (if None, uses all)
        top_n : int
            Number of recommendations to show

        Returns:
        --------
        DataFrame : Top recommendations with H-scores
        """
        print("\n" + "=" * 60)
        print(f"PICK #{self.picks_made + 1} RECOMMENDATIONS")
        print("=" * 60)

        if available_players is None:
            # Get all players
            available_players = self.league_data['PLAYER_NAME'].unique()

        # Filter out already drafted, do_not_draft, and players with no historical data
        available = [p for p in available_players
                    if p not in self.my_team
                    and p not in self.do_not_draft
                    and p in self.scoring.league_data['PLAYER_NAME'].values]

        if self.do_not_draft:
            excluded_in_pool = [p for p in available_players if p in self.do_not_draft]
            if excluded_in_pool:
                print(f"\nExcluding {len(excluded_in_pool)} do_not_draft players from consideration:")
                for player in excluded_in_pool[:5]:  # Show first 5
                    print(f"  - {player}")
                if len(excluded_in_pool) > 5:
                    print(f"  ... and {len(excluded_in_pool) - 5} more")

        # Calculate H-scores
        recommendations = []

        print(f"Evaluating {len(available[:100])} players...")

        # FIXED: Calculate optimal weights ONCE for current team state
        # Per Rosenof (2024), weights should be based on team composition,
        # not recalculated per candidate
        print("  Calculating optimal weights for current team...")
        optimal_weights = self.optimizer.calculate_optimal_weights_for_team(
            self.my_team,
            self.opponent_rosters,
            self.picks_made,
            total_picks=13,
            last_weights=self.last_optimal_weights,
            format=self.format
        )

        # Now evaluate ALL candidates using the SAME weights
        for player_name in available[:100]:  # Limit for speed
            h_score = self.optimizer.evaluate_player_with_weights(
                player_name,
                self.my_team,
                self.opponent_rosters,
                self.picks_made,
                total_picks=13,
                optimal_weights=optimal_weights,
                format=self.format
            )

            # Also get G-score for comparison
            g_scores = self.scoring.calculate_all_g_scores(player_name)

            recommendations.append({
                'PLAYER_NAME': player_name,
                'H_SCORE': h_score,
                'G_SCORE': g_scores['TOTAL'],
                'optimal_weights': optimal_weights
            })

        # Sort by H-score
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('H_SCORE', ascending=False)
        recommendations_df['RANK'] = range(1, len(recommendations_df) + 1)

        # Display top recommendations
        print("\nTop Recommendations:")
        print(recommendations_df[['RANK', 'PLAYER_NAME', 'H_SCORE', 'G_SCORE']].head(top_n).to_string(index=False))

        return recommendations_df.head(top_n)

    def draft_player(self, player_name):
        """
        Draft a player and update state.

        Parameters:
        -----------
        player_name : str
            Player to draft
        """
        if player_name in self.my_team:
            print(f"Warning: {player_name} already drafted!")
            return

        # Add to team
        self.my_team.append(player_name)
        self.picks_made += 1

        # Calculate optimal weights for this pick
        _, optimal_weights = self.optimizer.evaluate_player(
            player_name,
            self.my_team[:-1],  # Team before this pick
            self.opponent_rosters,
            self.picks_made - 1,
            total_picks=13,
            last_weights=self.last_optimal_weights,
            format=self.format
        )

        # Update state
        self.last_optimal_weights = optimal_weights
        self.weight_history.append(optimal_weights)

        print(f"\n✓ Drafted: {player_name}")
        print(f"Team size: {len(self.my_team)}/13")

        # Show emerging strategy after a few picks
        if self.picks_made >= 3:
            self._analyze_strategy()

    def _analyze_strategy(self):
        """Analyze emerging punt strategy."""
        if not self.weight_history:
            return

        # Average weights across recent picks
        avg_weights = np.mean(self.weight_history, axis=0)

        print("\nEmerging Strategy:")
        print("-" * 40)

        # Find punted categories (< 5% weight)
        punt_threshold = 0.05
        punted = []
        strong = []

        for idx, cat in enumerate(self.setup_params['categories']):
            weight = avg_weights[idx]
            if weight < punt_threshold:
                punted.append(cat)
            elif weight > 0.12:
                strong.append(cat)

        if punted:
            print(f"Punting: {', '.join(punted)}")
        if strong:
            print(f"Strong in: {', '.join(strong)}")

        print("-" * 40)

    def update_opponent_rosters(self, opponent_rosters):
        """
        Update opponent rosters after each round.

        Parameters:
        -----------
        opponent_rosters : list of lists
            List of opponent rosters
        """
        self.opponent_rosters = opponent_rosters

    def show_team_summary(self):
        """Display current team summary."""
        print("\n" + "=" * 60)
        print("MY TEAM")
        print("=" * 60)

        for idx, player in enumerate(self.my_team, 1):
            g_scores = self.scoring.calculate_all_g_scores(player)
            print(f"{idx}. {player} (G-Score: {g_scores['TOTAL']:.2f})")

        print("=" * 60)

    def export_results(self, output_file=None):
        """Export draft results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'draft_results_{timestamp}.json'

        results = {
            'my_team': self.my_team,
            'picks_made': self.picks_made,
            'weight_history': [w.tolist() for w in self.weight_history],
            'categories': self.setup_params['categories'],
            'format': self.format
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results exported to {output_file}")


def main():
    """Main function for interactive draft assistant."""
    print("\n" + "=" * 60)
    print("H-SCORING DRAFT ASSISTANT")
    print("=" * 60)

    # Initialize
    assistant = DraftAssistant()

    # Check if data exists
    data_dir = 'data'
    data_files = [f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')]

    if not data_files:
        print("\nNo data found. Collecting data...")
        assistant.collect_data(seasons=['2023-24'], max_players=150)
    else:
        # Load most recent data
        latest_data = sorted(data_files)[-1]
        data_file = os.path.join(data_dir, latest_data)
        variance_file = data_file.replace('league_weekly_data', 'player_variances').replace('.csv', '.json')

        assistant.load_data(data_file, variance_file)

    # Show initial rankings
    print("\nTop 20 Players by G-Score:")
    g_rankings = assistant.get_player_rankings(method='g_score', top_n=20)
    print(g_rankings[['RANK', 'PLAYER_NAME', 'TOTAL_G_SCORE']].to_string(index=False))

    # Interactive draft loop (example)
    print("\n" + "=" * 60)
    print("DRAFT MODE")
    print("Commands: 'rec' = recommendations, 'draft <name>' = draft player, 'team' = show team, 'quit' = exit")
    print("=" * 60)

    while assistant.picks_made < 13:
        command = input(f"\nPick #{assistant.picks_made + 1} > ").strip()

        if command == 'quit':
            break
        elif command == 'rec':
            assistant.recommend_pick(top_n=15)
        elif command.startswith('draft '):
            player_name = command[6:].strip()
            assistant.draft_player(player_name)
        elif command == 'team':
            assistant.show_team_summary()
        else:
            print("Unknown command. Use 'rec', 'draft <name>', 'team', or 'quit'")

    # Export results
    assistant.export_results()

    print("\nDraft complete!")


if __name__ == "__main__":
    main()