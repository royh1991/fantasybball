"""
Bayesian Player Model for fantasy basketball projections.

This module implements the core Bayesian model for individual player
performance, combining shooting percentages (Beta-Binomial) with
counting statistics (Poisson/Negative Binomial).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from .empirical_bayes import EmpiricalBayes
from .distributions import (
    BetaBinomial, PoissonDistribution,
    NegativeBinomial, MultivariateNormalResiduals
)

logger = logging.getLogger(__name__)


@dataclass
class PlayerContext:
    """Context information for a player's game simulation."""
    player_id: str
    position: str
    team: str
    opponent: str
    is_home: bool
    days_rest: int = 1
    injury_status: Optional[str] = None
    projected_minutes: Optional[float] = None


@dataclass
class BoxScore:
    """Simulated box score for a player."""
    # Shooting stats
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int

    # Counting stats
    pts: int
    reb: int
    ast: int
    stl: int
    blk: int
    tov: int

    # Derived stats
    fg_pct: float
    ft_pct: float
    fg3_pct: float
    dd: int  # Double-double

    # Meta info
    minutes: float
    player_id: str


class BayesianPlayerModel:
    """
    Comprehensive Bayesian model for player performance projection.

    Combines multiple statistical distributions to model different
    aspects of player performance with proper uncertainty quantification.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Bayesian player model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.eb_estimator = EmpiricalBayes(config)

        # Distribution models for each player
        self.shooting_models = {}  # Player -> stat -> BetaBinomial
        self.counting_models = {}  # Player -> stat -> Poisson/NegBin
        self.correlation_models = {}  # Player -> MultivariateNormal

        # Cache for fitted models
        self.model_cache = {}

    def fit_player(self, player_id: str, game_logs: pd.DataFrame,
                  position: str, espn_projection: Optional[Dict] = None) -> Dict:
        """
        Fit all distribution models for a single player.

        Args:
            player_id: Player identifier
            game_logs: DataFrame with player's historical games
            position: Player's position
            espn_projection: Optional ESPN projections for priors

        Returns:
            Dictionary of fitted models
        """
        if len(game_logs) < 3:
            logger.warning(f"Insufficient data for {player_id}: {len(game_logs)} games")
            return self._get_default_models(position)

        models = {
            'shooting': {},
            'counting': {},
            'attempts': {},
            'correlation': None
        }

        # Fit shooting percentage models (Beta-Binomial)
        shooting_stats = {
            'fg_pct': ('fgm', 'fga'),
            'ft_pct': ('ftm', 'fta'),
            'fg3_pct': ('fg3m', 'fg3a')
        }

        for stat_name, (makes_col, attempts_col) in shooting_stats.items():
            if makes_col in game_logs.columns and attempts_col in game_logs.columns:
                # Get Empirical Bayes estimates
                eb_result = self.eb_estimator.shrink_recent_games(
                    game_logs, makes_col, attempts_col,
                    stat_name, position,
                    n_recent=10,
                    decay_factor=0.9
                )

                # Create Beta-Binomial model
                models['shooting'][stat_name] = BetaBinomial(
                    alpha=eb_result['posterior_alpha'],
                    beta=eb_result['posterior_beta']
                )

        # Fit shot attempt models (Poisson or NegBin)
        attempt_stats = ['fga', 'fg3a', 'fta']
        for stat in attempt_stats:
            if stat in game_logs.columns:
                values = game_logs[stat].values
                mean_val = np.mean(values)
                var_val = np.var(values)

                # Check for overdispersion
                cv = np.sqrt(var_val) / mean_val if mean_val > 0 else 0

                if cv > self.config.get('counting_stats', {}).get('negbin_cv_threshold', 1.5):
                    # Use Negative Binomial for overdispersed data
                    models['attempts'][stat] = NegativeBinomial(mean_val, var_val / mean_val)
                else:
                    # Use Poisson for regular dispersion
                    models['attempts'][stat] = PoissonDistribution(mean_val)

        # Fit counting stat models
        counting_stats = ['reb', 'ast', 'stl', 'blk', 'tov']
        for stat in counting_stats:
            if stat in game_logs.columns:
                values = game_logs[stat].values
                mean_val = np.mean(values)
                var_val = np.var(values)

                cv = np.sqrt(var_val) / mean_val if mean_val > 0 else 0

                if cv > self.config.get('counting_stats', {}).get('negbin_cv_threshold', 1.5):
                    models['counting'][stat] = NegativeBinomial(mean_val, var_val / mean_val)
                else:
                    models['counting'][stat] = PoissonDistribution(mean_val)

        # Fit correlation model for residuals
        models['correlation'] = self._fit_correlation_model(game_logs, counting_stats)

        # Blend with ESPN projections if available
        if espn_projection:
            models = self._blend_with_projections(models, espn_projection, position)

        # Cache the fitted models
        self.shooting_models[player_id] = models['shooting']
        self.counting_models[player_id] = models['counting']
        self.correlation_models[player_id] = models['correlation']
        self.model_cache[player_id] = models

        logger.info(f"Fitted models for {player_id} using {len(game_logs)} games")
        return models

    def _fit_correlation_model(self, game_logs: pd.DataFrame,
                              stats: List[str]) -> Optional[MultivariateNormalResiduals]:
        """
        Fit correlation model for counting stat residuals.

        Args:
            game_logs: Player game logs
            stats: List of stat column names

        Returns:
            Fitted multivariate normal model or None
        """
        valid_stats = [s for s in stats if s in game_logs.columns]
        if len(valid_stats) < 2:
            return None

        # Calculate residuals (actual - mean)
        residuals = []
        for stat in valid_stats:
            values = game_logs[stat].values
            mean_val = np.mean(values)
            residuals.append(values - mean_val)

        residuals = np.column_stack(residuals)

        # Fit multivariate normal to residuals
        if len(residuals) >= 10:  # Need sufficient samples
            mean = np.mean(residuals, axis=0)
            cov = np.cov(residuals, rowvar=False)

            # Add small regularization to diagonal
            cov += np.eye(len(valid_stats)) * 0.01

            return MultivariateNormalResiduals(mean, cov)

        return None

    def _blend_with_projections(self, models: Dict, projection: Dict,
                               position: str, weight: float = 0.3) -> Dict:
        """
        Blend fitted models with ESPN projections.

        Args:
            models: Fitted distribution models
            projection: ESPN projections
            position: Player position
            weight: Weight for ESPN projection (0-1)

        Returns:
            Blended models
        """
        # Adjust shooting models based on projected percentages
        if 'fg_pct' in projection and 'fg_pct' in models['shooting']:
            proj_fg = projection['fg_pct']
            current_model = models['shooting']['fg_pct']

            # Blend alpha/beta parameters
            proj_alpha = proj_fg * 100  # Virtual makes
            proj_beta = (1 - proj_fg) * 100  # Virtual misses

            new_alpha = (1 - weight) * current_model.alpha + weight * proj_alpha
            new_beta = (1 - weight) * current_model.beta + weight * proj_beta

            models['shooting']['fg_pct'] = BetaBinomial(new_alpha, new_beta)

        # Similar for other shooting stats...

        # Adjust counting stat models based on projected averages
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']:
            if stat in projection and stat in models['counting']:
                proj_mean = projection[stat]
                current_model = models['counting'][stat]

                if isinstance(current_model, PoissonDistribution):
                    new_lambda = (1 - weight) * current_model.lambda_param + weight * proj_mean
                    models['counting'][stat] = PoissonDistribution(new_lambda)
                elif isinstance(current_model, NegativeBinomial):
                    new_mean = (1 - weight) * current_model.mean + weight * proj_mean
                    models['counting'][stat] = NegativeBinomial(new_mean, current_model.dispersion)

        return models

    def simulate_game(self, context: PlayerContext,
                     n_simulations: int = 1) -> List[BoxScore]:
        """
        Simulate game performance for a player.

        Args:
            context: Game context information
            n_simulations: Number of simulations to run

        Returns:
            List of simulated box scores
        """
        player_id = context.player_id

        # Get fitted models for player
        if player_id not in self.model_cache:
            logger.warning(f"No fitted model for {player_id}, using defaults")
            models = self._get_default_models(context.position)
        else:
            models = self.model_cache[player_id]

        box_scores = []
        for _ in range(n_simulations):
            box_score = self._simulate_single_game(models, context)
            box_scores.append(box_score)

        return box_scores

    def _simulate_single_game(self, models: Dict, context: PlayerContext) -> BoxScore:
        """
        Simulate a single game using fitted models.

        Args:
            models: Fitted distribution models
            context: Game context

        Returns:
            Simulated box score
        """
        # Simulate minutes played
        minutes = self._simulate_minutes(context)

        # Adjust rates based on minutes
        minute_factor = minutes / 36.0  # Normalize to per-36

        # Simulate shot attempts
        fga = self._sample_with_minutes(models['attempts'].get('fga'), minute_factor)
        fg3a = self._sample_with_3pa_rate(fga, models, context.position)
        fta = self._sample_with_minutes(models['attempts'].get('fta'), minute_factor)

        # Simulate makes using Beta-Binomial
        fgm = self._sample_makes(fga, models['shooting'].get('fg_pct'))
        fg3m = min(self._sample_makes(fg3a, models['shooting'].get('fg3_pct')), fgm)
        ftm = self._sample_makes(fta, models['shooting'].get('ft_pct'))

        # Calculate points
        pts = 2 * (fgm - fg3m) + 3 * fg3m + ftm

        # Simulate counting stats
        base_counts = {}
        for stat in ['reb', 'ast', 'stl', 'blk', 'tov']:
            if stat in models['counting']:
                base_counts[stat] = self._sample_with_minutes(
                    models['counting'][stat], minute_factor
                )
            else:
                base_counts[stat] = 0

        # Apply correlations if available
        if models.get('correlation') and len(base_counts) > 1:
            correlated_counts = self._apply_correlations(base_counts, models['correlation'])
        else:
            correlated_counts = base_counts

        # Ensure non-negative values
        for stat in correlated_counts:
            correlated_counts[stat] = max(0, int(correlated_counts[stat]))

        # Check for double-double
        dd_stats = [pts, correlated_counts['reb'], correlated_counts['ast'],
                   correlated_counts['stl'], correlated_counts['blk']]
        dd = 1 if sum(s >= 10 for s in dd_stats) >= 2 else 0

        return BoxScore(
            fgm=fgm, fga=fga,
            fg3m=fg3m, fg3a=fg3a,
            ftm=ftm, fta=fta,
            pts=pts,
            reb=correlated_counts['reb'],
            ast=correlated_counts['ast'],
            stl=correlated_counts['stl'],
            blk=correlated_counts['blk'],
            tov=correlated_counts['tov'],
            fg_pct=fgm / fga if fga > 0 else 0,
            ft_pct=ftm / fta if fta > 0 else 0,
            fg3_pct=fg3m / fg3a if fg3a > 0 else 0,
            dd=dd,
            minutes=minutes,
            player_id=context.player_id
        )

    def _simulate_minutes(self, context: PlayerContext) -> float:
        """Simulate minutes played based on context."""
        if context.projected_minutes:
            base_minutes = context.projected_minutes
        else:
            # Default minutes by position
            position_minutes = {
                'PG': 32, 'SG': 30, 'SF': 30, 'PF': 28, 'C': 28
            }
            base_minutes = position_minutes.get(context.position, 28)

        # Add some variance
        minutes = np.random.normal(base_minutes, 3)

        # Injury adjustment
        if context.injury_status and 'questionable' in context.injury_status.lower():
            minutes *= 0.85

        # Ensure reasonable bounds
        return np.clip(minutes, 5, 48)

    def _sample_with_minutes(self, model: Any, minute_factor: float) -> int:
        """Sample from distribution adjusted for minutes."""
        if model is None:
            return 0

        if isinstance(model, PoissonDistribution):
            adjusted_lambda = model.lambda_param * minute_factor
            return np.random.poisson(adjusted_lambda)
        elif isinstance(model, NegativeBinomial):
            adjusted_mean = model.mean * minute_factor
            adjusted_model = NegativeBinomial(adjusted_mean, model.dispersion)
            return int(adjusted_model.sample())
        else:
            return 0

    def _sample_with_3pa_rate(self, fga: int, models: Dict, position: str) -> int:
        """Sample 3PA based on FGA and position tendencies."""
        if fga == 0:
            return 0

        # Position-based 3PA rates
        position_3pa_rates = {
            'PG': 0.38, 'SG': 0.42, 'SF': 0.35, 'PF': 0.25, 'C': 0.15
        }
        base_rate = position_3pa_rates.get(position, 0.30)

        # Add some variance
        rate = np.random.beta(base_rate * 100, (1 - base_rate) * 100)
        fg3a = np.random.binomial(fga, rate)

        return min(fg3a, fga)

    def _sample_makes(self, attempts: int, model: Optional[BetaBinomial]) -> int:
        """Sample makes from Beta-Binomial model."""
        if attempts == 0 or model is None:
            return 0

        makes = model.sample(attempts)
        return min(makes, attempts)

    def _apply_correlations(self, base_counts: Dict,
                          correlation_model: MultivariateNormalResiduals) -> Dict:
        """Apply correlations to counting stats."""
        # Sample correlated residuals
        residuals = correlation_model.sample()

        # Apply residuals to base counts
        stat_names = list(base_counts.keys())
        adjusted_counts = {}

        for i, stat in enumerate(stat_names):
            if i < len(residuals):
                adjusted_counts[stat] = base_counts[stat] + residuals[i]
            else:
                adjusted_counts[stat] = base_counts[stat]

        return adjusted_counts

    def _get_default_models(self, position: str) -> Dict:
        """Get default models for a position when no data available."""
        # Position-based defaults
        position_defaults = {
            'PG': {'pts': 15, 'reb': 4, 'ast': 6, 'fg_pct': 0.44},
            'SG': {'pts': 18, 'reb': 4, 'ast': 3, 'fg_pct': 0.45},
            'SF': {'pts': 16, 'reb': 6, 'ast': 3, 'fg_pct': 0.46},
            'PF': {'pts': 14, 'reb': 8, 'ast': 2, 'fg_pct': 0.48},
            'C': {'pts': 12, 'reb': 10, 'ast': 2, 'fg_pct': 0.52}
        }

        defaults = position_defaults.get(position, position_defaults['SF'])

        models = {
            'shooting': {
                'fg_pct': BetaBinomial(defaults['fg_pct'] * 100, (1 - defaults['fg_pct']) * 100),
                'ft_pct': BetaBinomial(75, 25),
                'fg3_pct': BetaBinomial(35, 65)
            },
            'counting': {
                'reb': PoissonDistribution(defaults['reb']),
                'ast': PoissonDistribution(defaults['ast']),
                'stl': PoissonDistribution(1),
                'blk': PoissonDistribution(0.5),
                'tov': PoissonDistribution(2)
            },
            'attempts': {
                'fga': PoissonDistribution(12),
                'fg3a': PoissonDistribution(4),
                'fta': PoissonDistribution(4)
            },
            'correlation': None
        }

        return models