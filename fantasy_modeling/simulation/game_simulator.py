"""
Game-level simulation engine for fantasy basketball.

Simulates individual games for players using fitted Bayesian models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from ..models.bayesian_model import BayesianPlayerModel, PlayerContext, BoxScore
from ..models.correlation_model import CorrelationModel

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """Extended context for game simulation."""
    date: str
    home_team: str
    away_team: str
    spread: Optional[float] = None
    total: Optional[float] = None
    pace_factor: Optional[float] = None
    injuries: Optional[Dict] = None
    weather: Optional[str] = None  # For outdoor venues


class GameSimulator:
    """
    Main simulation engine for projecting game-level performance.

    Coordinates between Bayesian models, correlation models,
    and context adjustments to produce realistic game simulations.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize game simulator.

        Args:
            config_path: Path to configuration files
        """
        self.config = self._load_config(config_path)
        self.bayesian_model = BayesianPlayerModel(self.config.get('model', {}))
        self.correlation_model = CorrelationModel(self.config.get('correlations', {}))

        # Cache for fitted models
        self.fitted_models = {}
        self.simulation_cache = {}

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from files."""
        config = {}

        if config_path:
            import yaml
            # Load model config
            try:
                with open(f"{config_path}/model_config.yaml", 'r') as f:
                    config['model'] = yaml.safe_load(f)
            except FileNotFoundError:
                logger.warning("Model config not found, using defaults")

            # Load league config
            try:
                with open(f"{config_path}/league_config.yaml", 'r') as f:
                    config['league'] = yaml.safe_load(f)
            except FileNotFoundError:
                logger.warning("League config not found, using defaults")

        return config

    def fit_all_players(self, player_data: pd.DataFrame,
                       game_logs: pd.DataFrame,
                       espn_projections: Optional[pd.DataFrame] = None) -> Dict:
        """
        Fit models for all players in the dataset.

        Args:
            player_data: DataFrame with player info (id, name, position, team)
            game_logs: DataFrame with historical game logs
            espn_projections: Optional ESPN projections

        Returns:
            Dictionary of fitted models by player
        """
        fitted = {}

        for _, player in player_data.iterrows():
            player_id = player['player_id']
            position = player.get('position', 'SF')

            # Get player's game logs
            player_logs = game_logs[game_logs['player_id'] == player_id].copy()

            if len(player_logs) < 3:
                logger.info(f"Skipping {player_id}: insufficient data ({len(player_logs)} games)")
                continue

            # Get ESPN projection if available
            espn_proj = None
            if espn_projections is not None and player_id in espn_projections['player_id'].values:
                proj_row = espn_projections[espn_projections['player_id'] == player_id].iloc[0]
                espn_proj = proj_row.to_dict()

            # Fit the model
            try:
                model = self.bayesian_model.fit_player(
                    player_id, player_logs, position, espn_proj
                )
                fitted[player_id] = model
                logger.info(f"Fitted model for {player_id}")
            except Exception as e:
                logger.error(f"Error fitting {player_id}: {e}")

        self.fitted_models = fitted
        logger.info(f"Fitted models for {len(fitted)} players")
        return fitted

    def simulate_game(self, player_id: str,
                     context: PlayerContext,
                     game_context: Optional[GameContext] = None,
                     n_simulations: int = 1000) -> Dict:
        """
        Simulate a game for a specific player.

        Args:
            player_id: Player identifier
            context: Player context for the game
            game_context: Additional game context
            n_simulations: Number of simulations to run

        Returns:
            Dictionary with simulation results and statistics
        """
        # Run simulations
        box_scores = self.bayesian_model.simulate_game(context, n_simulations)

        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(bs) for bs in box_scores])

        # Calculate statistics
        results = {
            'player_id': player_id,
            'n_simulations': n_simulations,
            'projections': {},
            'percentiles': {},
            'distributions': {}
        }

        # Get mean projections
        stat_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov',
                    'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'dd']

        for stat in stat_cols:
            if stat in df.columns:
                results['projections'][stat] = {
                    'mean': df[stat].mean(),
                    'median': df[stat].median(),
                    'std': df[stat].std(),
                    'min': df[stat].min(),
                    'max': df[stat].max()
                }

                # Calculate percentiles
                results['percentiles'][stat] = {
                    '10': df[stat].quantile(0.10),
                    '25': df[stat].quantile(0.25),
                    '50': df[stat].quantile(0.50),
                    '75': df[stat].quantile(0.75),
                    '90': df[stat].quantile(0.90)
                }

                # Store distribution for visualization
                results['distributions'][stat] = df[stat].tolist()

        # Calculate shooting percentages
        results['projections']['fg_pct'] = {
            'mean': (df['fgm'].sum() / df['fga'].sum()) if df['fga'].sum() > 0 else 0
        }
        results['projections']['ft_pct'] = {
            'mean': (df['ftm'].sum() / df['fta'].sum()) if df['fta'].sum() > 0 else 0
        }
        results['projections']['fg3_pct'] = {
            'mean': (df['fg3m'].sum() / df['fg3a'].sum()) if df['fg3a'].sum() > 0 else 0
        }

        # Calculate fantasy categories
        results['fantasy'] = self._calculate_fantasy_impact(df)

        return results

    def simulate_slate(self, slate: List[Dict],
                      n_simulations: int = 1000,
                      parallel: bool = True) -> pd.DataFrame:
        """
        Simulate an entire slate of games.

        Args:
            slate: List of dictionaries with player/game info
            n_simulations: Number of simulations per player
            parallel: Whether to run in parallel

        Returns:
            DataFrame with all simulation results
        """
        results = []

        if parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for game_info in slate:
                    future = executor.submit(
                        self._simulate_single_player,
                        game_info, n_simulations
                    )
                    futures.append(future)

                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for game_info in slate:
                result = self._simulate_single_player(game_info, n_simulations)
                if result:
                    results.append(result)

        return pd.DataFrame(results)

    def _simulate_single_player(self, game_info: Dict,
                               n_simulations: int) -> Optional[Dict]:
        """Simulate a single player for the slate."""
        try:
            player_id = game_info['player_id']
            context = PlayerContext(
                player_id=player_id,
                position=game_info.get('position', 'SF'),
                team=game_info.get('team', ''),
                opponent=game_info.get('opponent', ''),
                is_home=game_info.get('is_home', True)
            )

            result = self.simulate_game(player_id, context, n_simulations=n_simulations)

            # Add game info to result
            result.update(game_info)

            # Flatten projections for DataFrame
            for stat, values in result['projections'].items():
                if isinstance(values, dict):
                    result[f"{stat}_mean"] = values.get('mean', 0)
                    result[f"{stat}_std"] = values.get('std', 0)

            return result

        except Exception as e:
            logger.error(f"Error simulating {game_info.get('player_id')}: {e}")
            return None

    def _calculate_fantasy_impact(self, df: pd.DataFrame) -> Dict:
        """
        Calculate fantasy-specific metrics from simulations.

        Args:
            df: DataFrame of simulated box scores

        Returns:
            Fantasy impact metrics
        """
        fantasy = {}

        # Standard 9-cat impact
        fantasy['9cat'] = {
            'pts': df['pts'].mean(),
            'reb': df['reb'].mean(),
            'ast': df['ast'].mean(),
            'stl': df['stl'].mean(),
            'blk': df['blk'].mean(),
            'tov': df['tov'].mean(),
            'fgm': df['fgm'].mean(),
            'fga': df['fga'].mean(),
            'ftm': df['ftm'].mean(),
            'fta': df['fta'].mean(),
            'fg3m': df['fg3m'].mean()
        }

        # Your 11-cat league (adding 3P% and DD)
        fantasy['11cat'] = fantasy['9cat'].copy()
        fantasy['11cat']['fg3_pct'] = (df['fg3m'].sum() / df['fg3a'].sum()
                                       if df['fg3a'].sum() > 0 else 0)
        fantasy['11cat']['dd'] = df['dd'].mean()

        # DFS scoring (example DraftKings scoring)
        fantasy['dfs'] = (
            df['pts'] +
            df['reb'] * 1.25 +
            df['ast'] * 1.5 +
            df['stl'] * 2 +
            df['blk'] * 2 -
            df['tov'] * 0.5
        ).mean()

        # Consistency metrics
        fantasy['consistency'] = {
            'pts_cv': df['pts'].std() / df['pts'].mean() if df['pts'].mean() > 0 else 0,
            'usage_stability': 1 - (df['fga'].std() / df['fga'].mean()
                                   if df['fga'].mean() > 0 else 0)
        }

        # Ceiling/floor
        fantasy['ceiling'] = df['pts'].quantile(0.90)
        fantasy['floor'] = df['pts'].quantile(0.10)

        return fantasy

    def compare_to_projection(self, simulation_results: Dict,
                            projection: Dict) -> Dict:
        """
        Compare simulation results to a point projection.

        Args:
            simulation_results: Results from simulate_game
            projection: Point projection to compare against

        Returns:
            Comparison metrics
        """
        comparison = {}

        for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']:
            if stat in simulation_results['projections'] and stat in projection:
                sim_mean = simulation_results['projections'][stat]['mean']
                proj_val = projection[stat]

                comparison[stat] = {
                    'simulation': sim_mean,
                    'projection': proj_val,
                    'difference': sim_mean - proj_val,
                    'pct_difference': (sim_mean - proj_val) / proj_val * 100 if proj_val != 0 else 0
                }

                # What percentile is the projection?
                if stat in simulation_results['distributions']:
                    dist = simulation_results['distributions'][stat]
                    percentile = sum(v <= proj_val for v in dist) / len(dist) * 100
                    comparison[stat]['projection_percentile'] = percentile

        return comparison

    def save_results(self, results: Dict, filepath: str):
        """Save simulation results to file."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        results_clean = convert_types(results)

        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)

        logger.info(f"Saved simulation results to {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """Load simulation results from file."""
        with open(filepath, 'r') as f:
            return json.load(f)