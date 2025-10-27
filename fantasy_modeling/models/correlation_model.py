"""
Correlation modeling for basketball statistics.

Models the correlations between different basketball statistics
to ensure realistic stat combinations in simulations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CorrelationModel:
    """
    Models correlations between basketball statistics.

    Captures relationships like:
    - More assists typically mean more turnovers
    - Rebounds and blocks are correlated for big men
    - High usage correlates with points and shot attempts
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize correlation model.

        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.correlation_matrices = {}
        self.position_correlations = {}

    def fit_from_data(self, df: pd.DataFrame,
                     stat_columns: List[str],
                     position_col: Optional[str] = None) -> Dict:
        """
        Fit correlation matrices from historical data.

        Args:
            df: DataFrame with player game logs
            stat_columns: List of stat column names to correlate
            position_col: Optional column for position-specific correlations

        Returns:
            Dictionary of correlation matrices
        """
        # Filter to valid columns
        valid_columns = [col for col in stat_columns if col in df.columns]

        if len(valid_columns) < 2:
            logger.warning("Insufficient columns for correlation modeling")
            return {}

        # Overall correlation matrix
        overall_corr = df[valid_columns].corr()
        self.correlation_matrices['overall'] = overall_corr

        # Position-specific correlations if position column provided
        if position_col and position_col in df.columns:
            for position in df[position_col].unique():
                if pd.isna(position):
                    continue

                pos_df = df[df[position_col] == position]
                if len(pos_df) >= 20:  # Need minimum samples
                    pos_corr = pos_df[valid_columns].corr()
                    self.position_correlations[position] = pos_corr

        logger.info(f"Fitted correlations for {len(valid_columns)} stats")
        return self.correlation_matrices

    def get_correlation_matrix(self, position: Optional[str] = None) -> np.ndarray:
        """
        Get appropriate correlation matrix for position.

        Args:
            position: Player position (optional)

        Returns:
            Correlation matrix as numpy array
        """
        if position and position in self.position_correlations:
            return self.position_correlations[position].values
        elif 'overall' in self.correlation_matrices:
            return self.correlation_matrices['overall'].values
        else:
            # Return identity matrix if no correlations fitted
            return np.eye(5)  # Default 5 stats

    def apply_correlations(self, base_stats: Dict[str, float],
                         position: Optional[str] = None) -> Dict[str, float]:
        """
        Apply correlations to base statistics.

        Args:
            base_stats: Dictionary of base stat values
            position: Player position

        Returns:
            Adjusted statistics with correlations applied
        """
        stat_names = list(base_stats.keys())
        base_values = np.array([base_stats[s] for s in stat_names])

        # Get correlation matrix
        corr_matrix = self.get_correlation_matrix(position)

        # Ensure matrix dimensions match
        n_stats = len(stat_names)
        if corr_matrix.shape[0] != n_stats:
            # Resize or use identity
            corr_matrix = np.eye(n_stats)

        # Generate correlated noise
        mean = np.zeros(n_stats)
        cov = corr_matrix * 0.1  # Scale down correlation strength
        noise = np.random.multivariate_normal(mean, cov)

        # Apply noise to base values
        adjusted_values = base_values + noise * np.sqrt(base_values)

        # Ensure non-negative
        adjusted_values = np.maximum(adjusted_values, 0)

        return dict(zip(stat_names, adjusted_values))

    def get_stat_relationships(self) -> Dict[Tuple[str, str], float]:
        """
        Get key statistical relationships from fitted correlations.

        Returns:
            Dictionary of (stat1, stat2) -> correlation coefficient
        """
        relationships = {}

        if 'overall' not in self.correlation_matrices:
            return relationships

        corr = self.correlation_matrices['overall']

        # Extract strong correlations (> 0.3 absolute value)
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                stat1 = corr.columns[i]
                stat2 = corr.columns[j]
                corr_val = corr.iloc[i, j]

                if abs(corr_val) > 0.3:
                    relationships[(stat1, stat2)] = corr_val

        return relationships

    def simulate_correlated_stats(self, means: np.ndarray,
                                 covariance: np.ndarray,
                                 n_samples: int = 1) -> np.ndarray:
        """
        Simulate correlated statistics using multivariate normal.

        Args:
            means: Mean values for each stat
            covariance: Covariance matrix
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, n_stats)
        """
        # Ensure covariance is positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Sample from multivariate normal
        samples = np.random.multivariate_normal(means, covariance, n_samples)

        # Ensure non-negative values
        samples = np.maximum(samples, 0)

        return samples

    def get_position_tendencies(self, position: str) -> Dict[str, float]:
        """
        Get typical stat tendencies for a position.

        Args:
            position: Player position

        Returns:
            Dictionary of stat tendency multipliers
        """
        position_tendencies = {
            'PG': {
                'ast': 1.5,
                'tov': 1.3,
                'stl': 1.2,
                'reb': 0.7,
                'blk': 0.5
            },
            'SG': {
                'ast': 0.8,
                'tov': 0.9,
                'stl': 1.0,
                'reb': 0.8,
                'blk': 0.6
            },
            'SF': {
                'ast': 0.9,
                'tov': 0.9,
                'stl': 1.0,
                'reb': 1.0,
                'blk': 0.8
            },
            'PF': {
                'ast': 0.7,
                'tov': 0.8,
                'stl': 0.8,
                'reb': 1.3,
                'blk': 1.2
            },
            'C': {
                'ast': 0.6,
                'tov': 0.7,
                'stl': 0.6,
                'reb': 1.5,
                'blk': 1.5
            }
        }

        return position_tendencies.get(position, {})

    def adjust_for_pace(self, stats: Dict[str, float],
                       team_pace: float, opp_pace: float,
                       league_avg_pace: float = 100.0) -> Dict[str, float]:
        """
        Adjust statistics for game pace.

        Args:
            stats: Raw statistics
            team_pace: Team's pace factor
            opp_pace: Opponent's pace factor
            league_avg_pace: League average pace

        Returns:
            Pace-adjusted statistics
        """
        # Calculate expected game pace
        game_pace = (team_pace + opp_pace) / 2
        pace_factor = game_pace / league_avg_pace

        # Adjust counting stats by pace
        pace_affected_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov',
                              'fga', 'fta', 'fg3a']

        adjusted = stats.copy()
        for stat in pace_affected_stats:
            if stat in adjusted:
                adjusted[stat] *= pace_factor

        return adjusted

    def get_matchup_adjustments(self, position: str,
                               opp_def_rating: Dict[str, float]) -> Dict[str, float]:
        """
        Get adjustments based on opponent's defensive ratings.

        Args:
            position: Player's position
            opp_def_rating: Opponent's defensive ratings by stat

        Returns:
            Stat adjustment multipliers
        """
        adjustments = {}

        # Example: if opponent is weak defending the 3pt line
        if 'fg3_def' in opp_def_rating:
            # Higher rating = worse defense = positive adjustment
            adjustments['fg3a'] = 1 + (opp_def_rating['fg3_def'] - 100) / 200

        # If opponent is strong at protecting the rim
        if 'rim_protection' in opp_def_rating:
            # Lower rating = better defense = negative adjustment
            adjustments['fga_at_rim'] = 1 - (100 - opp_def_rating['rim_protection']) / 200

        return adjustments