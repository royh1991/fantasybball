"""
Unified data collector that orchestrates data fetching from multiple sources.

Integrates with existing data files and APIs to provide a consolidated
data interface for the modeling system.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Main data collection orchestrator.

    Coordinates between existing CSV files, ESPN API, and NBA API
    to provide unified data access.
    """

    def __init__(self, base_path: str = "../"):
        """
        Initialize data collector.

        Args:
            base_path: Base path to the fantasybasketball2 directory
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"

        # Verify paths exist
        if not self.data_path.exists():
            logger.warning(f"Data path {self.data_path} does not exist")

        # Load configuration for data locations
        self.data_locations = {
            'historical_logs': self.data_path / "static" / "active_players_historical_game_logs.csv",
            'fantasy_projections': self.data_path / "fantasy_basketball_clean2.csv",
            'player_stats_dir': self.data_path / "static" / "player_stats_historical",
            'daily_matchups_dir': self.data_path / "daily_matchups",
            'daily_gamelogs_dir': self.data_path / "daily_gamelogs",
            'intermediate_dir': self.data_path / "intermediate"
        }

        # Cache for loaded data
        self.cache = {}

    def load_historical_game_logs(self, season: Optional[str] = None,
                                 min_games: int = 10) -> pd.DataFrame:
        """
        Load historical game logs from existing CSV files.

        Args:
            season: Optional season filter (e.g., '2023-24')
            min_games: Minimum games required per player

        Returns:
            DataFrame with historical game logs
        """
        cache_key = f"historical_{season}_{min_games}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try to load the main historical file
        if self.data_locations['historical_logs'].exists():
            logger.info(f"Loading historical data from {self.data_locations['historical_logs']}")
            df = pd.read_csv(self.data_locations['historical_logs'])

            # Filter by season if specified
            if season and 'season' in df.columns:
                df = df[df['season'] == season]

            # Filter players with minimum games
            if min_games > 0:
                player_games = df.groupby('player_id').size()
                valid_players = player_games[player_games >= min_games].index
                df = df[df['player_id'].isin(valid_players)]

            # Standardize column names
            df = self._standardize_columns(df)

            self.cache[cache_key] = df
            logger.info(f"Loaded {len(df)} game logs for {df['player_id'].nunique()} players")
            return df

        else:
            # Fall back to individual player files
            return self._load_from_player_files(season, min_games)

    def _load_from_player_files(self, season: Optional[str] = None,
                               min_games: int = 10) -> pd.DataFrame:
        """Load data from individual player CSV files."""
        player_dir = self.data_locations['player_stats_dir']

        if not player_dir.exists():
            logger.warning(f"Player stats directory not found: {player_dir}")
            return pd.DataFrame()

        all_data = []
        for file in player_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                if len(df) >= min_games:
                    if season and 'season' in df.columns:
                        df = df[df['season'] == season]
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return self._standardize_columns(combined)

        return pd.DataFrame()

    def load_espn_projections(self) -> pd.DataFrame:
        """
        Load ESPN fantasy projections from existing CSV.

        Returns:
            DataFrame with ESPN projections
        """
        if 'espn_projections' in self.cache:
            return self.cache['espn_projections']

        proj_file = self.data_locations['fantasy_projections']

        if not proj_file.exists():
            logger.warning(f"ESPN projections file not found: {proj_file}")
            return pd.DataFrame()

        logger.info(f"Loading ESPN projections from {proj_file}")
        df = pd.read_csv(proj_file)

        # Map column names to our standard format
        column_mapping = {
            'Player': 'player_name',
            'POS': 'position',
            'Team': 'team',
            'GP': 'games_played',
            'MPG': 'minutes',
            'FG%': 'fg_pct',
            'FT%': 'ft_pct',
            '3P%': 'fg3_pct',
            '3PM': 'fg3m',
            'PTS': 'pts',
            'REB': 'reb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'TO': 'tov',
            'DD': 'dd'
        }

        df = df.rename(columns=column_mapping)

        # Convert percentages to decimals
        for col in ['fg_pct', 'ft_pct', 'fg3_pct']:
            if col in df.columns:
                df[col] = df[col] / 100 if df[col].max() > 1 else df[col]

        self.cache['espn_projections'] = df
        logger.info(f"Loaded projections for {len(df)} players")
        return df

    def load_current_season_data(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load current season game logs.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)

        Returns:
            DataFrame with current season games
        """
        gamelogs_dir = self.data_locations['daily_gamelogs_dir']

        if not gamelogs_dir.exists():
            logger.warning(f"Daily gamelogs directory not found: {gamelogs_dir}")
            return pd.DataFrame()

        all_games = []
        for file in sorted(gamelogs_dir.glob("*.csv")):
            # Extract date from filename (assuming format: gamelogs_YYYY-MM-DD.csv)
            try:
                file_date = file.stem.split('_')[-1]
                if start_date and file_date < start_date:
                    continue

                df = pd.read_csv(file)
                df['game_date'] = file_date
                all_games.append(df)

            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

        if all_games:
            combined = pd.concat(all_games, ignore_index=True)
            return self._standardize_columns(combined)

        return pd.DataFrame()

    def load_matchup_data(self, week: Optional[int] = None) -> pd.DataFrame:
        """
        Load fantasy matchup data for lineup optimization.

        Args:
            week: Optional week number filter

        Returns:
            DataFrame with matchup information
        """
        matchups_dir = self.data_locations['daily_matchups_dir']

        if not matchups_dir.exists():
            logger.warning(f"Daily matchups directory not found: {matchups_dir}")
            return pd.DataFrame()

        # Find the most recent or specific week file
        pattern = f"*week_{week}*" if week else "*.csv"
        files = list(matchups_dir.glob(pattern))

        if not files:
            logger.warning(f"No matchup files found for week {week}")
            return pd.DataFrame()

        # Use most recent file
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"Loading matchup data from {latest_file}")

        df = pd.read_csv(latest_file)
        return df

    def get_player_mapping(self) -> Dict[str, str]:
        """
        Get player name to ID mapping.

        Returns:
            Dictionary mapping player names to IDs
        """
        # Try to load from ESPN projections first
        projections = self.load_espn_projections()

        mapping = {}
        if not projections.empty and 'player_name' in projections.columns:
            # Create mapping from projections
            for _, row in projections.iterrows():
                name = row['player_name']
                # Generate ID from name (simplified)
                player_id = name.lower().replace(' ', '_')
                mapping[name] = player_id

        # Supplement with historical data if needed
        historical = self.load_historical_game_logs()
        if not historical.empty and 'player_name' in historical.columns:
            for _, row in historical[['player_name', 'player_id']].drop_duplicates().iterrows():
                if row['player_name'] not in mapping:
                    mapping[row['player_name']] = row['player_id']

        return mapping

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized columns
        """
        # Standard column mapping
        column_mapping = {
            # Common variations
            'PLAYER_NAME': 'player_name',
            'Player': 'player_name',
            'PLAYER_ID': 'player_id',
            'Player_ID': 'player_id',
            'GAME_DATE': 'game_date',
            'Game_Date': 'game_date',
            'TEAM_ABBREVIATION': 'team',
            'Team': 'team',
            'MIN': 'minutes',
            'MPG': 'minutes',
            'FGM': 'fgm',
            'FGA': 'fga',
            'FG3M': 'fg3m',
            'FG3A': 'fg3a',
            'FTM': 'ftm',
            'FTA': 'fta',
            'OREB': 'oreb',
            'DREB': 'dreb',
            'REB': 'reb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'TOV': 'tov',
            'TO': 'tov',
            'PTS': 'pts'
        }

        # Apply mapping
        df = df.rename(columns=column_mapping)

        # Ensure numeric columns are numeric
        numeric_cols = ['fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                       'pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'minutes']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate derived columns if missing
        if 'reb' not in df.columns and 'oreb' in df.columns and 'dreb' in df.columns:
            df['reb'] = df['oreb'] + df['dreb']

        # Add shooting percentages if missing
        if 'fg_pct' not in df.columns and 'fgm' in df.columns and 'fga' in df.columns:
            df['fg_pct'] = df.apply(lambda x: x['fgm'] / x['fga'] if x['fga'] > 0 else 0, axis=1)

        if 'ft_pct' not in df.columns and 'ftm' in df.columns and 'fta' in df.columns:
            df['ft_pct'] = df.apply(lambda x: x['ftm'] / x['fta'] if x['fta'] > 0 else 0, axis=1)

        if 'fg3_pct' not in df.columns and 'fg3m' in df.columns and 'fg3a' in df.columns:
            df['fg3_pct'] = df.apply(lambda x: x['fg3m'] / x['fg3a'] if x['fg3a'] > 0 else 0, axis=1)

        return df

    def prepare_modeling_data(self, player_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare all data needed for modeling.

        Args:
            player_ids: Optional list of player IDs to filter

        Returns:
            Tuple of (player_info, game_logs, projections) DataFrames
        """
        # Load all data sources
        historical = self.load_historical_game_logs()
        current = self.load_current_season_data()
        projections = self.load_espn_projections()

        # Combine historical and current game logs
        if not current.empty:
            game_logs = pd.concat([historical, current], ignore_index=True)
        else:
            game_logs = historical

        # Filter by player IDs if specified
        if player_ids:
            game_logs = game_logs[game_logs['player_id'].isin(player_ids)]
            projections = projections[projections['player_id'].isin(player_ids)]

        # Create player info DataFrame
        player_info = self._create_player_info(game_logs, projections)

        logger.info(f"Prepared data for {len(player_info)} players with {len(game_logs)} total games")

        return player_info, game_logs, projections

    def _create_player_info(self, game_logs: pd.DataFrame,
                          projections: pd.DataFrame) -> pd.DataFrame:
        """Create unified player information DataFrame."""
        # Get unique players from game logs
        if not game_logs.empty:
            players_from_logs = game_logs.groupby('player_id').agg({
                'player_name': 'first',
                'team': 'last',  # Most recent team
                'game_date': 'count'  # Number of games
            }).reset_index()
            players_from_logs = players_from_logs.rename(columns={'game_date': 'n_games'})
        else:
            players_from_logs = pd.DataFrame()

        # Get players from projections
        if not projections.empty and 'player_name' in projections.columns:
            players_from_proj = projections[['player_name', 'position', 'team']].copy()
            players_from_proj['player_id'] = players_from_proj['player_name'].str.lower().str.replace(' ', '_')
        else:
            players_from_proj = pd.DataFrame()

        # Merge the two sources
        if not players_from_logs.empty and not players_from_proj.empty:
            player_info = pd.merge(
                players_from_logs,
                players_from_proj[['player_id', 'position']],
                on='player_id',
                how='outer'
            )
        elif not players_from_logs.empty:
            player_info = players_from_logs
            player_info['position'] = 'SF'  # Default position
        elif not players_from_proj.empty:
            player_info = players_from_proj
            player_info['n_games'] = 0
        else:
            player_info = pd.DataFrame()

        return player_info