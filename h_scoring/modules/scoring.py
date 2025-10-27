"""
G-score and X-score calculation module for H-scoring system.

Implements the scoring systems described in Rosenof (2024):
- G-scores: Static player rankings that account for week-to-week variance
- X-scores: Dynamic scoring basis used in H-scoring optimization
"""

import pandas as pd
import numpy as np
from scipy import stats


class PlayerScoring:
    """Calculate G-scores and X-scores for fantasy basketball players."""

    def __init__(self, league_data, player_variances, roster_size=13):
        """
        Initialize scoring calculator.

        Parameters:
        -----------
        league_data : DataFrame
            Weekly player statistics
        player_variances : dict
            Player-specific variance by category
        roster_size : int
            League roster size (default 13)
        """
        self.league_data = league_data
        self.player_variances = player_variances
        self.roster_size = roster_size
        self.kappa = (2 * roster_size) / (2 * roster_size - 1)

        # 11-category leagues
        self.counting_cats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'DD']
        self.percentage_cats = ['FG_PCT', 'FT_PCT', 'FG3_PCT']
        self.all_cats = self.counting_cats + self.percentage_cats

        # Calculate league-wide statistics
        self._calculate_league_stats()

    def _calculate_league_stats(self):
        """Calculate league-wide mean and variance statistics."""
        self.league_stats = {}

        # Calculate season averages per player
        season_avgs = self.league_data.groupby('PLAYER_NAME').agg({
            **{cat: 'mean' for cat in self.counting_cats + self.percentage_cats}
        }).reset_index()

        for cat in self.all_cats:
            if cat not in season_avgs.columns:
                continue

            # Between-player variance (variance of season averages)
            player_means = season_avgs[cat].values
            sigma_between = np.var(player_means, ddof=1)

            # Within-player variance (average per-game variance)
            # This is more accurate: uses game-to-game consistency
            within_variances = []
            for player_name in self.player_variances:
                if cat in self.player_variances[player_name]:
                    # Use per-game variance (not weekly)
                    within_variances.append(
                        self.player_variances[player_name][cat]['var_per_game']
                    )

            sigma_within = np.mean(within_variances) if within_variances else 0

            # League mean
            mu_league = np.mean(player_means)

            self.league_stats[cat] = {
                'mu_league': mu_league,
                'sigma_between': sigma_between,
                'sigma_within': sigma_within,
                'sigma_between_sq': sigma_between,
                'sigma_within_sq': sigma_within
            }

    def calculate_g_score(self, player_name, category):
        """
        Calculate G-score for a player in a specific category.

        G-score formula:
        G = (mu_player - mu_league) / sqrt(sigma_between^2 + kappa * sigma_within^2)

        Parameters:
        -----------
        player_name : str
            Player name
        category : str
            Statistical category

        Returns:
        --------
        float : G-score
        """
        if category not in self.league_stats:
            return 0.0

        # Get player's season average
        player_data = self.league_data[
            self.league_data['PLAYER_NAME'] == player_name
        ]

        if player_data.empty:
            return 0.0

        mu_player = player_data[category].mean()

        # Get league stats
        mu_league = self.league_stats[category]['mu_league']
        sigma_between_sq = self.league_stats[category]['sigma_between_sq']

        # Get player-specific per-game variance
        if player_name in self.player_variances and category in self.player_variances[player_name]:
            sigma_within_sq = self.player_variances[player_name][category]['var_per_game']
        else:
            # Fallback to league average
            sigma_within_sq = self.league_stats[category]['sigma_within_sq']

        # Calculate G-score denominator
        denominator = np.sqrt(sigma_between_sq + self.kappa * sigma_within_sq)

        if denominator == 0:
            return 0.0

        # Turnovers are negative (lower is better)
        if category == 'TOV':
            g_score = (mu_league - mu_player) / denominator
        else:
            g_score = (mu_player - mu_league) / denominator

        return g_score

    def calculate_all_g_scores(self, player_name):
        """
        Calculate G-scores for all categories for a player.

        Returns:
        --------
        dict : G-scores by category
        """
        g_scores = {}
        for cat in self.all_cats:
            g_scores[cat] = self.calculate_g_score(player_name, cat)

        # Overall G-score (sum across categories)
        g_scores['TOTAL'] = sum(g_scores.values())

        return g_scores

    def calculate_x_score(self, player_name, category):
        """
        Calculate X-score for a player in a specific category.

        X-score is G-score with sigma_between and sigma_within set to 0,
        used for the H-scoring optimization framework.

        For counting stats: X = (mu_player - mu_league) / m_tau
        For percentage stats: X = (a_q/a_mu) * (r_q - r_mu) / r_tau

        Parameters:
        -----------
        player_name : str
            Player name
        category : str
            Statistical category

        Returns:
        --------
        float : X-score
        """
        if category not in self.league_stats:
            return 0.0

        player_data = self.league_data[
            self.league_data['PLAYER_NAME'] == player_name
        ]

        if player_data.empty:
            return 0.0

        mu_player = player_data[category].mean()
        mu_league = self.league_stats[category]['mu_league']

        # Check for NaN
        if np.isnan(mu_player) or np.isnan(mu_league):
            return 0.0

        # CRITICAL FIX: Use weekly variance to match weekly mean!
        # Previous bug: divided weekly_mean by sqrt(per_game_variance)
        # This created artificially inflated X-scores (e.g., Sabonis DD: 7.50 instead of ~2.5)

        # Calculate variance from the actual weekly data
        var_weekly = player_data[category].var()

        # Fallback to league average if player variance is invalid
        if np.isnan(var_weekly) or var_weekly <= 0 or len(player_data) < 3:
            sigma_within = np.sqrt(self.league_stats[category]['sigma_within_sq'])
        else:
            sigma_within = np.sqrt(var_weekly)

        # Ensure positive sigma_within
        if sigma_within == 0 or np.isnan(sigma_within):
            return 0.0

        # For percentage stats, weight by attempts
        if category in self.percentage_cats:
            # Get attempt column
            if category == 'FG_PCT':
                attempts_col = 'FGA'
            elif category == 'FT_PCT':
                attempts_col = 'FTA'
            elif category == 'FG3_PCT':
                attempts_col = 'FG3A'

            player_attempts = player_data[attempts_col].mean()
            league_attempts = self.league_data[attempts_col].mean()

            # Check for NaN or zero attempts
            if np.isnan(player_attempts) or np.isnan(league_attempts) or league_attempts == 0:
                return 0.0

            # Volume-weighted X-score for percentages
            x_score = (player_attempts / league_attempts) * (mu_player - mu_league) / sigma_within
        else:
            # Standard X-score for counting stats
            if category == 'TOV':
                x_score = (mu_league - mu_player) / sigma_within
            else:
                x_score = (mu_player - mu_league) / sigma_within

        # Final NaN check
        if np.isnan(x_score) or np.isinf(x_score):
            return 0.0

        return x_score

    def calculate_all_x_scores(self, player_name):
        """Calculate X-scores for all categories."""
        x_scores = {}
        for cat in self.all_cats:
            x_scores[cat] = self.calculate_x_score(player_name, cat)
        return x_scores

    def calculate_v_vector(self):
        """
        Calculate v vector that converts X-scores to G-scores.

        v = m_tau / sqrt(m_tau^2 + m_sigma^2) for counting stats
        v = r_tau / sqrt(r_tau^2 + r_sigma^2) for percentage stats

        Returns:
        --------
        numpy array : v vector (normalized to sum to 1)
        """
        v = np.zeros(len(self.all_cats))

        for idx, cat in enumerate(self.all_cats):
            if cat not in self.league_stats:
                continue

            sigma_within = np.sqrt(self.league_stats[cat]['sigma_within_sq'])
            sigma_between = np.sqrt(self.league_stats[cat]['sigma_between_sq'])

            denominator = np.sqrt(sigma_within**2 + sigma_between**2)

            if denominator > 0:
                v[idx] = sigma_within / denominator

        # Normalize to sum to 1
        if v.sum() > 0:
            v = v / v.sum()

        return v

    def rank_players_by_g_score(self, top_n=200):
        """
        Rank all players by total G-score.

        Parameters:
        -----------
        top_n : int
            Number of top players to return

        Returns:
        --------
        DataFrame with player rankings
        """
        unique_players = self.league_data['PLAYER_NAME'].unique()

        rankings = []
        for player in unique_players:
            g_scores = self.calculate_all_g_scores(player)
            rankings.append({
                'PLAYER_NAME': player,
                'TOTAL_G_SCORE': g_scores['TOTAL'],
                **{f'G_{cat}': g_scores[cat] for cat in self.all_cats}
            })

        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('TOTAL_G_SCORE', ascending=False)
        rankings_df['RANK'] = range(1, len(rankings_df) + 1)

        return rankings_df.head(top_n)

    def get_player_stats_summary(self, player_name):
        """Get comprehensive stats summary for a player."""
        player_data = self.league_data[
            self.league_data['PLAYER_NAME'] == player_name
        ]

        if player_data.empty:
            return None

        summary = {
            'player_name': player_name,
            'weeks_played': len(player_data),
            'season_averages': {},
            'g_scores': self.calculate_all_g_scores(player_name),
            'x_scores': self.calculate_all_x_scores(player_name)
        }

        for cat in self.all_cats:
            summary['season_averages'][cat] = player_data[cat].mean()

        return summary


if __name__ == "__main__":
    # Example usage
    import json

    # Load data
    league_data = pd.read_csv('../data/league_weekly_data_20240101_120000.csv')

    with open('../data/player_variances_20240101_120000.json', 'r') as f:
        player_variances = json.load(f)

    # Initialize scoring
    scoring = PlayerScoring(league_data, player_variances)

    # Rank players
    rankings = scoring.rank_players_by_g_score(top_n=50)
    print("\nTop 50 Players by G-Score:")
    print(rankings[['RANK', 'PLAYER_NAME', 'TOTAL_G_SCORE']].head(20))

    # Calculate v vector
    v = scoring.calculate_v_vector()
    print("\nV vector (X-score to G-score conversion):")
    for idx, cat in enumerate(scoring.all_cats):
        print(f"{cat}: {v[idx]:.4f}")