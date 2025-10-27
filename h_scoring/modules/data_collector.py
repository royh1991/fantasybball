"""
Data collection module for H-scoring system using nba_api.

This module fetches historical NBA player data and calculates weekly statistics
needed for G-score and variance calculations.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time
from datetime import datetime
import json
import os


class NBADataCollector:
    """Collects and processes NBA player data for H-scoring calculations."""

    def __init__(self, seasons=['2023-24'], data_dir='data', checkpoint_file=None, name_mapping_file=None):
        """
        Initialize the data collector.

        Parameters:
        -----------
        seasons : list of str
            NBA seasons to fetch (e.g., ['2022-23', '2023-24'])
        data_dir : str
            Directory to store collected data
        checkpoint_file : str
            Path to checkpoint file for resuming collection
        name_mapping_file : str
            Path to JSON file with name mappings
        """
        self.seasons = seasons
        self.data_dir = data_dir
        self.temp_dir = os.path.join(data_dir, '.temp')
        self.categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                          'FG3M', 'FGM', 'FGA', 'FTM', 'FTA']
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Load name mappings
        self.name_mappings = {}
        if name_mapping_file and os.path.exists(name_mapping_file):
            with open(name_mapping_file, 'r') as f:
                mapping_data = json.load(f)
                self.name_mappings = {k: v for k, v in mapping_data.get('mappings', {}).items() if v is not None}
            print(f"Loaded {len(self.name_mappings)} name mappings")

        # Checkpoint file for resume capability
        if checkpoint_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_file = os.path.join(self.temp_dir, f'checkpoint_{timestamp}.json')
        self.checkpoint_file = checkpoint_file

    def _normalize_name(self, name):
        """Normalize player name for matching."""
        import unicodedata
        # Remove accents
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
        # Lowercase and strip
        return name.strip().lower()

    def get_active_players(self, target_players_file=None):
        """
        Fetch list of active NBA players.

        Parameters:
        -----------
        target_players_file : str, optional
            Path to CSV with target players (must have 'PLAYER' column)

        Returns:
        --------
        DataFrame with player info, prioritized by target list if provided
        """
        player_list = players.get_players()
        all_players_df = pd.DataFrame(player_list)

        if target_players_file:
            # Load target players from ADP file
            target_df = pd.read_csv(target_players_file)
            target_names_original = target_df['PLAYER'].str.strip()

            # Apply name mappings first
            target_names_mapped = target_names_original.apply(
                lambda x: self.name_mappings.get(self._normalize_name(x), x)
            )
            target_names = set(target_names_mapped.apply(self._normalize_name))

            print(f"Loaded {len(target_names)} target players from ADP file")

            # Normalize NBA API names
            all_players_df['full_name_normalized'] = all_players_df['full_name'].apply(self._normalize_name)
            all_players_df['is_target'] = all_players_df['full_name_normalized'].isin(target_names)

            # Sort: target players first, then by is_active
            all_players_df = all_players_df.sort_values(['is_target', 'is_active'], ascending=[False, False])

            # Filter to active players (target or not, but prioritize targets)
            active_players = all_players_df[all_players_df['is_active']].copy()
            active_targets = active_players[active_players['is_target']]

            print(f"Found {len(active_targets)} active players matching ADP list")

            # Check for mismatches
            matched_names = set(active_targets['full_name_normalized'])
            unmatched = target_names - matched_names

            if unmatched:
                print(f"\n⚠️  {len(unmatched)} players from ADP not found in NBA API (may be rookies not yet in API):")
                for name in sorted(list(unmatched))[:15]:  # Show first 15
                    print(f"     - {name}")
                if len(unmatched) > 15:
                    print(f"     ... and {len(unmatched) - 15} more")
                print()

            return active_targets.drop(columns=['full_name_normalized', 'is_target']).reset_index(drop=True)

        else:
            # Original behavior: all active players
            active_players = all_players_df[all_players_df['is_active']]
            return active_players

    def fetch_player_gamelogs(self, player_id, player_name, season, max_retries=3):
        """
        Fetch game logs for a specific player and season.

        Parameters:
        -----------
        player_id : int
            NBA player ID
        player_name : str
            Player name
        season : str
            NBA season (e.g., '2023-24')
        max_retries : int
            Maximum number of retry attempts

        Returns:
        --------
        DataFrame with game logs
        """
        for attempt in range(max_retries):
            try:
                time.sleep(0.6)  # Rate limiting
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    timeout=60  # Increase timeout to 60 seconds
                )
                games = gamelog.get_data_frames()[0]

                if games.empty:
                    return None

                # Add player info
                games['PLAYER_NAME'] = player_name
                games['PLAYER_ID'] = player_id
                games['SEASON'] = season

                return games

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    print(f"  Retry {attempt + 1}/{max_retries} for {player_name} ({season}) after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Error fetching {player_name} ({season}): {str(e)}")
                    return None

    def process_gamelogs(self, games_df):
        """
        Process raw game logs into standardized format.

        Parameters:
        -----------
        games_df : DataFrame
            Raw game logs from nba_api

        Returns:
        --------
        DataFrame with processed statistics
        """
        if games_df is None or games_df.empty:
            return None

        # Convert date
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

        # Rename columns to match our categories
        column_mapping = {
            'PTS': 'PTS',
            'REB': 'REB',
            'AST': 'AST',
            'STL': 'STL',
            'BLK': 'BLK',
            'TOV': 'TOV',
            'FG3M': 'FG3M',
            'FG3A': 'FG3A',
            'FGM': 'FGM',
            'FGA': 'FGA',
            'FTM': 'FTM',
            'FTA': 'FTA',
            'MIN': 'MIN'
        }

        processed = games_df.copy()

        # Calculate percentages
        processed['FG_PCT'] = np.where(
            processed['FGA'] > 0,
            processed['FGM'] / processed['FGA'],
            0
        )
        processed['FT_PCT'] = np.where(
            processed['FTA'] > 0,
            processed['FTM'] / processed['FTA'],
            0
        )
        processed['FG3_PCT'] = np.where(
            processed['FG3A'] > 0,
            processed['FG3M'] / processed['FG3A'],
            0
        )

        # Calculate double-doubles
        stats_for_dd = processed[['PTS', 'REB', 'AST', 'STL', 'BLK']].copy()
        processed['DD'] = (stats_for_dd >= 10).sum(axis=1) >= 2
        processed['DD'] = processed['DD'].astype(int)

        return processed

    def add_nba_season_week(self, games_df):
        """
        Add NBA season and week identifiers to game data.

        NBA seasons run from October (start of season year) to April (end of season year+1).
        Week 1 of each season starts with opening week in October.

        Parameters:
        -----------
        games_df : DataFrame
            Game logs with GAME_DATE column

        Returns:
        --------
        DataFrame with NBA_SEASON and SEASON_WEEK_ID columns added
        """
        if games_df is None or games_df.empty:
            return None

        df = games_df.copy()

        # Determine NBA season (e.g., 2023-24 for games Oct 2023 - Apr 2024)
        # If month >= 10 (Oct-Dec), it's the start of that season
        # If month < 10 (Jan-Sep), it's the end of previous year's season
        df['YEAR'] = df['GAME_DATE'].dt.year
        df['MONTH'] = df['GAME_DATE'].dt.month

        df['SEASON_START_YEAR'] = df.apply(
            lambda row: row['YEAR'] if row['MONTH'] >= 10 else row['YEAR'] - 1,
            axis=1
        )
        df['SEASON_END_YEAR'] = df['SEASON_START_YEAR'] + 1
        df['NBA_SEASON'] = df['SEASON_START_YEAR'].astype(str) + '-' + df['SEASON_END_YEAR'].astype(str).str[-2:]

        # Calculate week within the season (starting from October)
        # Group by season and assign week numbers sequentially
        df = df.sort_values('GAME_DATE')

        def assign_season_week(group):
            # Get unique weeks in order
            group = group.copy()
            group['ISO_WEEK'] = group['GAME_DATE'].dt.isocalendar().week
            group['WEEK_START'] = group['GAME_DATE'] - pd.to_timedelta(group['GAME_DATE'].dt.dayofweek, unit='d')

            # Assign sequential week numbers within season
            unique_weeks = group['WEEK_START'].unique()
            week_mapping = {week: idx + 1 for idx, week in enumerate(sorted(unique_weeks))}
            group['SEASON_WEEK'] = group['WEEK_START'].map(week_mapping)

            return group

        # Suppress FutureWarning by using a different approach
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            df = df.groupby('NBA_SEASON', group_keys=False).apply(assign_season_week)

        # Create season week ID (e.g., "2023-24_W5")
        df['SEASON_WEEK_ID'] = df['NBA_SEASON'] + '_W' + df['SEASON_WEEK'].astype(str)

        # Clean up temporary columns
        df = df.drop(columns=['MONTH', 'SEASON_START_YEAR', 'SEASON_END_YEAR', 'ISO_WEEK', 'WEEK_START'])

        return df

    def aggregate_to_weekly(self, games_df):
        """
        Aggregate game logs to weekly statistics using NBA season weeks.

        Parameters:
        -----------
        games_df : DataFrame
            Processed game logs with SEASON_WEEK_ID

        Returns:
        --------
        DataFrame with weekly aggregated stats
        """
        if games_df is None or games_df.empty:
            return None

        # Aggregate counting stats (sum) by SEASON_WEEK_ID
        counting_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                         'FG3M', 'FG3A', 'FGM', 'FGA', 'FTM', 'FTA', 'DD']

        weekly = games_df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'NBA_SEASON', 'SEASON_WEEK_ID']).agg({
            **{stat: 'sum' for stat in counting_stats},
            'GAME_DATE': 'count'  # Games played that week
        }).reset_index()

        weekly.rename(columns={'GAME_DATE': 'GAMES_PLAYED'}, inplace=True)

        # Calculate weekly percentages
        weekly['FG_PCT'] = np.where(
            weekly['FGA'] > 0,
            weekly['FGM'] / weekly['FGA'],
            0
        )
        weekly['FT_PCT'] = np.where(
            weekly['FTA'] > 0,
            weekly['FTM'] / weekly['FTA'],
            0
        )
        weekly['FG3_PCT'] = np.where(
            weekly['FG3M'] > 0,
            weekly['FG3M'] / weekly['FG3A'],
            0
        )

        # Filter weeks with at least 2 games played
        weekly = weekly[weekly['GAMES_PLAYED'] >= 2]

        return weekly

    def calculate_player_variance(self, games_df, weekly_df):
        """
        Calculate per-game variance (performance consistency).

        This is more accurate than weekly variance because it separates:
        - Performance variance: How consistent is the player game-to-game?
        - Schedule variance: Games per week (handled separately in win probability)

        Weekly variance = per_game_variance × games_in_week

        Parameters:
        -----------
        games_df : DataFrame
            Game-by-game statistics
        weekly_df : DataFrame
            Weekly aggregated statistics (for percentages and weekly averages)

        Returns:
        --------
        dict : Player variance statistics by category
        """
        if games_df is None or games_df.empty:
            return None

        # Counting stats: use per-game variance
        counting_categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M']

        # Percentage stats: use game-level percentages
        percentage_categories = ['FG_PCT', 'FT_PCT', 'FG3_PCT']

        # DD: use per-game (0 or 1)

        variances = {}

        # Per-game counting stats
        for cat in counting_categories:
            if cat in games_df.columns:
                game_values = games_df[cat].values
                variances[cat] = {
                    'mean_per_game': np.mean(game_values),
                    'std_per_game': np.std(game_values, ddof=1),
                    'var_per_game': np.var(game_values, ddof=1),
                    'cv_per_game': np.std(game_values, ddof=1) / np.mean(game_values) if np.mean(game_values) > 0 else 0,
                    'games': len(game_values),
                    # Also include weekly stats for reference
                    'mean_weekly': weekly_df[cat].mean() if cat in weekly_df.columns else 0,
                    'weeks': len(weekly_df)
                }

        # Percentage stats: calculate per-game percentages
        for cat in percentage_categories:
            if cat in games_df.columns:
                game_percentages = games_df[cat].values
                # Filter out games with no attempts (0% means no attempts)
                valid_games = game_percentages[game_percentages > 0]

                if len(valid_games) > 0:
                    variances[cat] = {
                        'mean_per_game': np.mean(valid_games),
                        'std_per_game': np.std(valid_games, ddof=1),
                        'var_per_game': np.var(valid_games, ddof=1),
                        'cv_per_game': np.std(valid_games, ddof=1) / np.mean(valid_games) if np.mean(valid_games) > 0 else 0,
                        'games': len(valid_games),
                        'mean_weekly': weekly_df[cat].mean() if cat in weekly_df.columns else 0,
                        'weeks': len(weekly_df)
                    }

        # Double-doubles: per-game (binary)
        if 'DD' in games_df.columns:
            game_dd = games_df['DD'].values
            variances['DD'] = {
                'mean_per_game': np.mean(game_dd),
                'std_per_game': np.std(game_dd, ddof=1),
                'var_per_game': np.var(game_dd, ddof=1),
                'cv_per_game': 0,  # Not meaningful for binary
                'games': len(game_dd),
                'mean_weekly': weekly_df['DD'].mean() if 'DD' in weekly_df.columns else 0,
                'weeks': len(weekly_df)
            }

        return variances

    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"✓ Resuming from checkpoint: {len(checkpoint['completed_players'])} players already collected")
            return checkpoint
        return {
            'completed_players': [],
            'game_data_files': [],
            'weekly_data_files': [],
            'variance_data_files': []
        }

    def save_checkpoint(self, checkpoint, player_name, game_file, weekly_file, variance_file):
        """Save checkpoint after each player."""
        checkpoint['completed_players'].append(player_name)
        checkpoint['game_data_files'].append(game_file)
        checkpoint['weekly_data_files'].append(weekly_file)
        checkpoint['variance_data_files'].append(variance_file)

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def collect_league_data(self, min_weeks=20, max_players=200, resume=False, target_players_file=None):
        """
        Collect data for top NBA players across all specified seasons.

        Parameters:
        -----------
        min_weeks : int
            Minimum weeks of data required for a player
        max_players : int
            Maximum number of players to collect (top by games played)
        resume : bool
            Whether to resume from checkpoint
        target_players_file : str, optional
            Path to CSV with target players to prioritize

        Returns:
        --------
        tuple : (league_weekly_data, league_game_data, player_variances)
        """
        print("Fetching active players...")
        active_players = self.get_active_players(target_players_file=target_players_file)

        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {
            'completed_players': [],
            'game_data_files': [],
            'weekly_data_files': [],
            'variance_data_files': []
        }

        completed_set = set(checkpoint['completed_players'])

        all_weekly_data = []
        all_game_data = []
        player_variances = {}

        # Load existing data if resuming
        if resume and checkpoint['game_data_files']:
            print(f"Loading {len(checkpoint['game_data_files'])} existing data files...")
            for game_file, weekly_file, variance_file in zip(
                checkpoint['game_data_files'],
                checkpoint['weekly_data_files'],
                checkpoint['variance_data_files']
            ):
                if os.path.exists(game_file):
                    all_game_data.append(pd.read_csv(game_file))
                if os.path.exists(weekly_file):
                    all_weekly_data.append(pd.read_csv(weekly_file))
                if os.path.exists(variance_file):
                    with open(variance_file, 'r') as f:
                        player_variances.update(json.load(f))

        print(f"\nCollecting data for up to {max_players} players...")
        print(f"Already completed: {len(completed_set)} players\n")

        collected_count = len(completed_set)

        for idx, player in active_players.iterrows():
            player_id = player['id']
            player_name = player['full_name']

            # Skip if already completed
            if player_name in completed_set:
                continue

            # Check if we've reached the max player limit
            if collected_count >= max_players:
                break

            print(f"[{collected_count + 1}/{max_players}] {player_name}")

            player_weekly_data = []
            player_game_data = []

            for season in self.seasons:
                # Fetch game logs with retry logic
                games = self.fetch_player_gamelogs(player_id, player_name, season)

                if games is not None:
                    # Process games
                    processed = self.process_gamelogs(games)

                    if processed is not None:
                        # Add NBA season and week identifiers
                        processed_with_season = self.add_nba_season_week(processed)

                        if processed_with_season is not None:
                            # Store game-level data
                            player_game_data.append(processed_with_season)

                            # Aggregate to weekly
                            weekly = self.aggregate_to_weekly(processed_with_season)

                            if weekly is not None:
                                player_weekly_data.append(weekly)

            # Combine all seasons for this player
            if player_weekly_data and player_game_data:
                player_combined_weekly = pd.concat(player_weekly_data, ignore_index=True)
                player_combined_games = pd.concat(player_game_data, ignore_index=True)

                # Calculate variance if enough data
                if len(player_combined_weekly) >= min_weeks:
                    # Save individual player data immediately
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    player_slug = player_name.replace(' ', '_').lower()

                    game_file = os.path.join(self.temp_dir, f'game_{player_slug}_{timestamp}.csv')
                    weekly_file = os.path.join(self.temp_dir, f'weekly_{player_slug}_{timestamp}.csv')
                    variance_file = os.path.join(self.temp_dir, f'variance_{player_slug}_{timestamp}.json')

                    player_combined_games.to_csv(game_file, index=False)
                    player_combined_weekly.to_csv(weekly_file, index=False)

                    # Use game-level data for variance calculation
                    variances = self.calculate_player_variance(
                        player_combined_games,
                        player_combined_weekly
                    )

                    with open(variance_file, 'w') as f:
                        json.dump({player_name: variances}, f, indent=2)

                    # Update checkpoint
                    self.save_checkpoint(checkpoint, player_name, game_file, weekly_file, variance_file)

                    # Add to memory
                    all_weekly_data.append(player_combined_weekly)
                    all_game_data.append(player_combined_games)
                    player_variances[player_name] = variances

                    print(f"  ✓ Saved checkpoint ({len(checkpoint['completed_players'])} players total)")

                    # Only increment counter if player was actually saved
                    collected_count += 1
                else:
                    # Player didn't meet minimum weeks requirement, don't count them
                    print(f"  ✗ Skipped (only {len(player_combined_weekly)} weeks, need {min_weeks})")
            else:
                # No data for player, don't count them
                print(f"  ✗ Skipped (no data)")

        # Combine all player data
        if all_weekly_data:
            league_weekly_data = pd.concat(all_weekly_data, ignore_index=True)
        else:
            league_weekly_data = pd.DataFrame()

        if all_game_data:
            league_game_data = pd.concat(all_game_data, ignore_index=True)
        else:
            league_game_data = pd.DataFrame()

        return league_weekly_data, league_game_data, player_variances

    def save_data(self, league_weekly_data, league_game_data, player_variances, cleanup_temp=True):
        """
        Save collected data to files.

        Saves three files:
        1. Game-level data (granular)
        2. Weekly aggregated data (for H-scoring)
        3. Player variances (per-game variance)

        Parameters:
        -----------
        cleanup_temp : bool
            Whether to clean up temporary checkpoint files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save game-level data
        game_file = os.path.join(self.data_dir, f'league_game_data_{timestamp}.csv')
        league_game_data.to_csv(game_file, index=False)
        print(f"Saved game-level data to {game_file}")

        # Save weekly aggregated data
        weekly_file = os.path.join(self.data_dir, f'league_weekly_data_{timestamp}.csv')
        league_weekly_data.to_csv(weekly_file, index=False)
        print(f"Saved weekly data to {weekly_file}")

        # Save variance data
        variance_file = os.path.join(self.data_dir, f'player_variances_{timestamp}.json')
        with open(variance_file, 'w') as f:
            json.dump(player_variances, f, indent=2)
        print(f"Saved variance data to {variance_file}")

        # Clean up temporary directory if requested
        if cleanup_temp:
            import shutil
            if os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    print(f"Cleaned up temporary files from {self.temp_dir}")
                except:
                    pass

        return game_file, weekly_file, variance_file


if __name__ == "__main__":
    # Example usage
    collector = NBADataCollector(
        seasons=['2023-24'],
        data_dir='../data'
    )

    league_data, player_variances = collector.collect_league_data(
        min_weeks=20,
        max_players=200
    )

    collector.save_data(league_data, player_variances)

    print(f"\nCollected data for {len(player_variances)} players")
    print(f"Total weekly observations: {len(league_data)}")