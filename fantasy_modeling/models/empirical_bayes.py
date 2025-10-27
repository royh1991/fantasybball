"""
Empirical Bayes implementation for basketball shooting percentages.

This module implements the Empirical Bayes approach to estimate shooting
percentages, providing shrinkage towards population means to avoid overfitting
on small samples.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class EmpiricalBayes:
    """
    Empirical Bayes estimator for basketball shooting statistics.

    Uses Beta-Binomial framework to shrink observed shooting percentages
    towards league/position averages based on sample size.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Empirical Bayes estimator.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.priors = {}
        self.position_priors = {}

    def fit_beta_prior(self, makes: np.ndarray, attempts: np.ndarray,
                      method: str = 'mle') -> Tuple[float, float]:
        """
        Fit Beta distribution prior parameters using empirical data.

        Args:
            makes: Array of made shots
            attempts: Array of shot attempts
            method: Fitting method ('mle' or 'mom' for method of moments)

        Returns:
            Tuple of (alpha, beta) parameters for Beta distribution
        """
        # Remove invalid data
        mask = (attempts > 0) & (makes <= attempts)
        makes = makes[mask]
        attempts = attempts[mask]

        if len(makes) == 0:
            logger.warning("No valid data for Beta prior fitting")
            return 1.0, 1.0  # Uniform prior

        # Calculate observed rates
        rates = makes / attempts

        if method == 'mle':
            # Maximum likelihood estimation
            def neg_log_likelihood(params):
                alpha, beta = params
                if alpha <= 0 or beta <= 0:
                    return np.inf

                # Beta-binomial likelihood
                ll = 0
                for m, a in zip(makes, attempts):
                    ll += (
                        stats.betaln(m + alpha, a - m + beta) -
                        stats.betaln(alpha, beta) -
                        np.log(a + 1) -
                        stats.betaln(m + 1, a - m + 1)
                    )
                return -ll

            # Initial guess using method of moments
            mean_rate = np.mean(rates)
            var_rate = np.var(rates)

            # Avoid division by zero
            if var_rate == 0:
                var_rate = 0.001

            # Method of moments estimates
            common = mean_rate * (1 - mean_rate) / var_rate - 1
            alpha_init = mean_rate * common
            beta_init = (1 - mean_rate) * common

            # Ensure positive initial values
            alpha_init = max(alpha_init, 0.1)
            beta_init = max(beta_init, 0.1)

            # Optimize
            result = optimize.minimize(
                neg_log_likelihood,
                x0=[alpha_init, beta_init],
                bounds=[(0.01, 1000), (0.01, 1000)],
                method='L-BFGS-B'
            )

            if result.success:
                return result.x[0], result.x[1]
            else:
                logger.warning(f"MLE optimization failed: {result.message}")
                return alpha_init, beta_init

        else:  # method of moments
            mean_rate = np.mean(rates)
            var_rate = np.var(rates)

            if var_rate == 0:
                # No variance - use strong prior
                strength = 100
                alpha = mean_rate * strength
                beta = (1 - mean_rate) * strength
            else:
                # Standard method of moments
                common = mean_rate * (1 - mean_rate) / var_rate - 1
                alpha = max(mean_rate * common, 0.1)
                beta = max((1 - mean_rate) * common, 0.1)

            return alpha, beta

    def fit_priors_from_data(self, df: pd.DataFrame,
                           stat_columns: Dict[str, Tuple[str, str]],
                           position_col: Optional[str] = None) -> Dict:
        """
        Fit Beta priors for multiple shooting statistics.

        Args:
            df: DataFrame with player game logs
            stat_columns: Dict mapping stat names to (makes_col, attempts_col)
                        e.g., {'fg_pct': ('fgm', 'fga')}
            position_col: Column name for player positions

        Returns:
            Dictionary of fitted priors
        """
        priors = {}

        for stat_name, (makes_col, attempts_col) in stat_columns.items():
            if makes_col not in df.columns or attempts_col not in df.columns:
                logger.warning(f"Columns {makes_col}/{attempts_col} not found")
                continue

            # Fit overall prior
            makes = df[makes_col].values
            attempts = df[attempts_col].values
            alpha, beta = self.fit_beta_prior(makes, attempts)

            priors[stat_name] = {
                'alpha': alpha,
                'beta': beta,
                'n_samples': len(df)
            }

            # Fit position-specific priors if position column provided
            if position_col and position_col in df.columns:
                priors[stat_name]['position'] = {}

                for position in df[position_col].unique():
                    if pd.isna(position):
                        continue

                    pos_df = df[df[position_col] == position]
                    if len(pos_df) < 10:  # Need minimum samples
                        continue

                    pos_makes = pos_df[makes_col].values
                    pos_attempts = pos_df[attempts_col].values
                    pos_alpha, pos_beta = self.fit_beta_prior(
                        pos_makes, pos_attempts
                    )

                    priors[stat_name]['position'][position] = {
                        'alpha': pos_alpha,
                        'beta': pos_beta,
                        'n_samples': len(pos_df)
                    }

            logger.info(f"Fitted prior for {stat_name}: α={alpha:.2f}, β={beta:.2f}")

        self.priors = priors
        return priors

    def shrink_estimate(self, makes: int, attempts: int,
                       stat_name: str, position: Optional[str] = None,
                       custom_prior: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Apply Empirical Bayes shrinkage to observed shooting percentage.

        Args:
            makes: Number of made shots
            attempts: Number of attempts
            stat_name: Name of statistic (e.g., 'fg_pct')
            position: Player position for position-specific prior
            custom_prior: Optional custom (alpha, beta) prior

        Returns:
            Dictionary with posterior estimates
        """
        if attempts == 0:
            return {
                'observed': 0.0,
                'posterior_mean': 0.0,
                'posterior_alpha': 1.0,
                'posterior_beta': 1.0,
                'credible_interval': (0.0, 0.0),
                'shrinkage_weight': 1.0
            }

        # Get prior parameters
        if custom_prior:
            alpha_prior, beta_prior = custom_prior
        elif (stat_name in self.priors and
              position and
              position in self.priors[stat_name].get('position', {})):
            # Use position-specific prior
            prior_info = self.priors[stat_name]['position'][position]
            alpha_prior = prior_info['alpha']
            beta_prior = prior_info['beta']
        elif stat_name in self.priors:
            # Use overall prior
            alpha_prior = self.priors[stat_name]['alpha']
            beta_prior = self.priors[stat_name]['beta']
        else:
            # Default uninformative prior
            alpha_prior = 1.0
            beta_prior = 1.0

        # Posterior parameters (conjugate update)
        alpha_post = alpha_prior + makes
        beta_post = beta_prior + (attempts - makes)

        # Posterior mean (shrunk estimate)
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # Observed rate
        observed = makes / attempts if attempts > 0 else 0

        # Shrinkage weight (how much we trust the prior vs data)
        prior_strength = alpha_prior + beta_prior
        shrinkage_weight = prior_strength / (prior_strength + attempts)

        # 95% credible interval
        ci_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        ci_upper = stats.beta.ppf(0.975, alpha_post, beta_post)

        return {
            'observed': observed,
            'posterior_mean': posterior_mean,
            'posterior_alpha': alpha_post,
            'posterior_beta': beta_post,
            'credible_interval': (ci_lower, ci_upper),
            'shrinkage_weight': shrinkage_weight
        }

    def shrink_recent_games(self, games_df: pd.DataFrame,
                          makes_col: str, attempts_col: str,
                          stat_name: str, position: Optional[str] = None,
                          n_recent: int = 10, decay_factor: float = 0.9) -> Dict:
        """
        Apply shrinkage to recent games with decay weighting.

        Args:
            games_df: DataFrame of game logs (sorted by date)
            makes_col: Column name for makes
            attempts_col: Column name for attempts
            stat_name: Name of statistic
            position: Player position
            n_recent: Number of recent games to consider
            decay_factor: Exponential decay for older games

        Returns:
            Shrinkage estimates with weighted recent performance
        """
        # Take most recent games
        recent_games = games_df.tail(n_recent).copy()

        if len(recent_games) == 0:
            return self.shrink_estimate(0, 0, stat_name, position)

        # Apply decay weights (most recent = 1.0)
        n_games = len(recent_games)
        weights = decay_factor ** np.arange(n_games - 1, -1, -1)

        # Weight the makes and attempts
        weighted_makes = (recent_games[makes_col] * weights).sum()
        weighted_attempts = (recent_games[attempts_col] * weights).sum()

        # Round to integers (Beta-Binomial needs integers)
        weighted_makes = int(np.round(weighted_makes))
        weighted_attempts = int(np.round(weighted_attempts))

        result = self.shrink_estimate(
            weighted_makes, weighted_attempts, stat_name, position
        )

        # Add additional info
        result['n_games'] = n_games
        result['decay_factor'] = decay_factor
        result['effective_sample_size'] = weights.sum()

        return result

    def project_with_prior(self, espn_projection: float,
                          historical_stats: Dict,
                          stat_name: str,
                          blend_weight: float = 0.3) -> float:
        """
        Blend ESPN projection with Empirical Bayes estimate.

        Args:
            espn_projection: ESPN's projected percentage (0-1)
            historical_stats: Historical shrinkage estimates
            stat_name: Name of statistic
            blend_weight: Weight for ESPN projection (0-1)

        Returns:
            Blended projection
        """
        # Get Empirical Bayes estimate
        eb_estimate = historical_stats.get('posterior_mean', espn_projection)

        # Weighted average
        blended = (blend_weight * espn_projection +
                  (1 - blend_weight) * eb_estimate)

        # Ensure valid percentage
        return np.clip(blended, 0.0, 1.0)

    def sample_from_posterior(self, posterior_alpha: float,
                            posterior_beta: float,
                            n_samples: int = 1000) -> np.ndarray:
        """
        Sample shooting percentages from posterior distribution.

        Args:
            posterior_alpha: Alpha parameter of posterior Beta
            posterior_beta: Beta parameter of posterior Beta
            n_samples: Number of samples to draw

        Returns:
            Array of sampled shooting percentages
        """
        return np.random.beta(posterior_alpha, posterior_beta, n_samples)