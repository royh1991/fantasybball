"""
Fantasy basketball scoring and category evaluation.

Handles different fantasy scoring formats including your 11-category league.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CategoryType(Enum):
    """Type of fantasy category."""
    COUNTING = "counting"
    PERCENTAGE = "percentage"


@dataclass
class FantasyCategory:
    """Definition of a fantasy category."""
    name: str
    type: CategoryType
    better: str  # 'higher' or 'lower'
    weight: float = 1.0


class ScoringSystem:
    """
    Fantasy scoring system for various league formats.

    Supports standard 9-cat, 11-cat (with 3P% and DD), and custom formats.
    """

    def __init__(self, categories: Optional[List[FantasyCategory]] = None):
        """
        Initialize scoring system.

        Args:
            categories: List of fantasy categories. If None, uses standard 11-cat
        """
        if categories is None:
            # Default to your 11-cat league
            self.categories = self._get_11cat_categories()
        else:
            self.categories = categories

    def _get_11cat_categories(self) -> List[FantasyCategory]:
        """Get categories for 11-cat league (standard + 3P% + DD)."""
        return [
            FantasyCategory("FG%", CategoryType.PERCENTAGE, "higher"),
            FantasyCategory("FT%", CategoryType.PERCENTAGE, "higher"),
            FantasyCategory("3P%", CategoryType.PERCENTAGE, "higher"),
            FantasyCategory("3PM", CategoryType.COUNTING, "higher"),
            FantasyCategory("PTS", CategoryType.COUNTING, "higher"),
            FantasyCategory("REB", CategoryType.COUNTING, "higher"),
            FantasyCategory("AST", CategoryType.COUNTING, "higher"),
            FantasyCategory("STL", CategoryType.COUNTING, "higher"),
            FantasyCategory("BLK", CategoryType.COUNTING, "higher"),
            FantasyCategory("TO", CategoryType.COUNTING, "lower"),
            FantasyCategory("DD", CategoryType.COUNTING, "higher"),
        ]

    def calculate_weekly_totals(self, game_simulations: List[pd.DataFrame]) -> Dict:
        """
        Calculate weekly totals from multiple game simulations.

        Args:
            game_simulations: List of DataFrames, one per game

        Returns:
            Dictionary with weekly totals for each category
        """
        # Combine all games
        all_games = pd.concat(game_simulations, ignore_index=True)

        weekly_totals = {}

        # Counting stats - sum across games
        counting_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg3m', 'dd']
        for stat in counting_stats:
            if stat in all_games.columns:
                weekly_totals[stat] = all_games[stat].sum()

        # Shooting percentages - weighted by attempts
        if 'fgm' in all_games.columns and 'fga' in all_games.columns:
            total_fgm = all_games['fgm'].sum()
            total_fga = all_games['fga'].sum()
            weekly_totals['fg_pct'] = total_fgm / total_fga if total_fga > 0 else 0

        if 'ftm' in all_games.columns and 'fta' in all_games.columns:
            total_ftm = all_games['ftm'].sum()
            total_fta = all_games['fta'].sum()
            weekly_totals['ft_pct'] = total_ftm / total_fta if total_fta > 0 else 0

        if 'fg3m' in all_games.columns and 'fg3a' in all_games.columns:
            total_fg3m = all_games['fg3m'].sum()
            total_fg3a = all_games['fg3a'].sum()
            weekly_totals['fg3_pct'] = total_fg3m / total_fg3a if total_fg3a > 0 else 0

        return weekly_totals

    def compare_matchup(self, team_a_stats: Dict, team_b_stats: Dict) -> Dict:
        """
        Compare two teams in a head-to-head matchup.

        Args:
            team_a_stats: Team A's weekly totals
            team_b_stats: Team B's weekly totals

        Returns:
            Dictionary with matchup results
        """
        results = {
            'team_a_wins': 0,
            'team_b_wins': 0,
            'ties': 0,
            'category_results': {}
        }

        # Map stat names to category names
        stat_to_category = {
            'fg_pct': 'FG%',
            'ft_pct': 'FT%',
            'fg3_pct': '3P%',
            'fg3m': '3PM',
            'pts': 'PTS',
            'reb': 'REB',
            'ast': 'AST',
            'stl': 'STL',
            'blk': 'BLK',
            'tov': 'TO',
            'dd': 'DD'
        }

        for stat, cat_name in stat_to_category.items():
            if stat not in team_a_stats or stat not in team_b_stats:
                continue

            # Find the category definition
            category = next((c for c in self.categories if c.name == cat_name), None)
            if not category:
                continue

            val_a = team_a_stats[stat]
            val_b = team_b_stats[stat]

            # Determine winner based on category type
            if category.better == 'higher':
                if val_a > val_b:
                    winner = 'A'
                    results['team_a_wins'] += 1
                elif val_b > val_a:
                    winner = 'B'
                    results['team_b_wins'] += 1
                else:
                    winner = 'tie'
                    results['ties'] += 1
            else:  # lower is better (turnovers)
                if val_a < val_b:
                    winner = 'A'
                    results['team_a_wins'] += 1
                elif val_b < val_a:
                    winner = 'B'
                    results['team_b_wins'] += 1
                else:
                    winner = 'tie'
                    results['ties'] += 1

            results['category_results'][cat_name] = {
                'team_a': val_a,
                'team_b': val_b,
                'winner': winner
            }

        # Overall winner
        if results['team_a_wins'] > results['team_b_wins']:
            results['winner'] = 'Team A'
        elif results['team_b_wins'] > results['team_a_wins']:
            results['winner'] = 'Team B'
        else:
            results['winner'] = 'Tie'

        results['score'] = f"{results['team_a_wins']}-{results['team_b_wins']}-{results['ties']}"

        return results

    def calculate_z_scores(self, player_stats: Dict, league_stats: pd.DataFrame) -> Dict:
        """
        Calculate z-scores for player stats relative to league.

        Args:
            player_stats: Player's per-game stats
            league_stats: DataFrame with league-wide per-game stats

        Returns:
            Dictionary of z-scores by category
        """
        z_scores = {}

        stat_columns = {
            'pts': 'PTS',
            'reb': 'REB',
            'ast': 'AST',
            'stl': 'STL',
            'blk': 'BLK',
            'tov': 'TO',
            'fg_pct': 'FG%',
            'ft_pct': 'FT%',
            'fg3_pct': '3P%',
            'fg3m': '3PM',
            'dd': 'DD'
        }

        for stat, cat_name in stat_columns.items():
            if stat not in player_stats or stat not in league_stats.columns:
                continue

            # Calculate z-score
            league_mean = league_stats[stat].mean()
            league_std = league_stats[stat].std()

            if league_std > 0:
                z_score = (player_stats[stat] - league_mean) / league_std

                # Flip sign for turnovers (lower is better)
                if stat == 'tov':
                    z_score = -z_score

                z_scores[cat_name] = z_score
            else:
                z_scores[cat_name] = 0

        return z_scores

    def calculate_total_value(self, z_scores: Dict) -> float:
        """
        Calculate total fantasy value from z-scores.

        Args:
            z_scores: Dictionary of z-scores by category

        Returns:
            Total value (sum of z-scores)
        """
        return sum(z_scores.values())

    def calculate_category_contributions(self, player_stats: Dict,
                                        games_played: int = 82) -> Dict:
        """
        Calculate player's contribution to each category over a season.

        Args:
            player_stats: Player's per-game stats
            games_played: Number of games to project

        Returns:
            Dictionary of season-long contributions
        """
        contributions = {}

        # Counting stats - multiply by games
        counting_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fg3m', 'dd']
        for stat in counting_stats:
            if stat in player_stats:
                contributions[stat] = player_stats[stat] * games_played

        # Shooting percentages - need attempts too
        if 'fgm' in player_stats and 'fga' in player_stats:
            contributions['fgm'] = player_stats['fgm'] * games_played
            contributions['fga'] = player_stats['fga'] * games_played
            contributions['fg_pct'] = player_stats.get('fg_pct', 0)

        if 'ftm' in player_stats and 'fta' in player_stats:
            contributions['ftm'] = player_stats['ftm'] * games_played
            contributions['fta'] = player_stats['fta'] * games_played
            contributions['ft_pct'] = player_stats.get('ft_pct', 0)

        if 'fg3m' in player_stats and 'fg3a' in player_stats:
            contributions['fg3m'] = player_stats['fg3m'] * games_played
            contributions['fg3a'] = player_stats['fg3a'] * games_played
            contributions['fg3_pct'] = player_stats.get('fg3_pct', 0)

        return contributions

    def compare_player_value(self, player_a: Dict, player_b: Dict,
                           league_stats: pd.DataFrame) -> Dict:
        """
        Compare the fantasy value of two players.

        Args:
            player_a: First player's stats
            player_b: Second player's stats
            league_stats: League-wide stats for context

        Returns:
            Comparison results
        """
        z_scores_a = self.calculate_z_scores(player_a, league_stats)
        z_scores_b = self.calculate_z_scores(player_b, league_stats)

        total_a = self.calculate_total_value(z_scores_a)
        total_b = self.calculate_total_value(z_scores_b)

        return {
            'player_a': {
                'z_scores': z_scores_a,
                'total_value': total_a
            },
            'player_b': {
                'z_scores': z_scores_b,
                'total_value': total_b
            },
            'difference': total_a - total_b,
            'winner': 'Player A' if total_a > total_b else 'Player B' if total_b > total_a else 'Tie'
        }

    def simulate_matchup_probability(self, team_a_simulations: List[Dict],
                                    team_b_simulations: List[Dict]) -> Dict:
        """
        Calculate probability of team A winning based on simulations.

        Args:
            team_a_simulations: List of simulated weekly totals for team A
            team_b_simulations: List of simulated weekly totals for team B

        Returns:
            Win probabilities and expected score
        """
        n_sims = len(team_a_simulations)
        wins = 0
        losses = 0
        ties = 0

        all_scores = []

        for sim_a, sim_b in zip(team_a_simulations, team_b_simulations):
            result = self.compare_matchup(sim_a, sim_b)

            if result['winner'] == 'Team A':
                wins += 1
            elif result['winner'] == 'Team B':
                losses += 1
            else:
                ties += 1

            all_scores.append({
                'a_wins': result['team_a_wins'],
                'b_wins': result['team_b_wins'],
                'ties': result['ties']
            })

        return {
            'win_probability': wins / n_sims,
            'loss_probability': losses / n_sims,
            'tie_probability': ties / n_sims,
            'expected_score': {
                'wins': np.mean([s['a_wins'] for s in all_scores]),
                'losses': np.mean([s['b_wins'] for s in all_scores]),
                'ties': np.mean([s['ties'] for s in all_scores])
            },
            'category_win_probabilities': self._calculate_category_win_probs(
                team_a_simulations, team_b_simulations
            )
        }

    def _calculate_category_win_probs(self, team_a_sims: List[Dict],
                                     team_b_sims: List[Dict]) -> Dict:
        """Calculate win probability for each category."""
        cat_wins = {cat.name: 0 for cat in self.categories}
        n_sims = len(team_a_sims)

        stat_to_category = {
            'fg_pct': 'FG%',
            'ft_pct': 'FT%',
            'fg3_pct': '3P%',
            'fg3m': '3PM',
            'pts': 'PTS',
            'reb': 'REB',
            'ast': 'AST',
            'stl': 'STL',
            'blk': 'BLK',
            'tov': 'TO',
            'dd': 'DD'
        }

        for sim_a, sim_b in zip(team_a_sims, team_b_sims):
            for stat, cat_name in stat_to_category.items():
                if stat not in sim_a or stat not in sim_b:
                    continue

                category = next((c for c in self.categories if c.name == cat_name), None)
                if not category:
                    continue

                val_a = sim_a[stat]
                val_b = sim_b[stat]

                # Check if team A wins this category
                if category.better == 'higher':
                    if val_a > val_b:
                        cat_wins[cat_name] += 1
                else:
                    if val_a < val_b:
                        cat_wins[cat_name] += 1

        # Convert counts to probabilities
        return {cat: wins / n_sims for cat, wins in cat_wins.items()}