"""
Enhanced scoring system that blends historical data with projections.

Key improvements:
1. Weight projections more heavily for young/improving players
2. Weight history more heavily for established veterans
3. Adjust for injury risk using GP projections
"""

import pandas as pd
import numpy as np
from .scoring import PlayerScoring


class ProjectionAwareScoringSystem(PlayerScoring):
    """Extends PlayerScoring to incorporate projections and injury risk."""

    def __init__(self, league_data, player_variances, roster_size, projections_file,
                 projection_weight=0.5, injury_penalty_strength=1.0):
        """
        Initialize with projection blending.

        Parameters:
        -----------
        league_data : DataFrame
            Historical weekly game log data
        player_variances : dict
            Per-game variances for each player
        roster_size : int
            Fantasy roster size (13)
        projections_file : str
            Path to projections CSV (fantasy_basketball_clean2.csv)
        projection_weight : float (0-1)
            Base weight for projections (will be adjusted per player)
            0.5 = equal weight to projections and history
        injury_penalty_strength : float
            How much to penalize injury-prone players (1.0 = moderate)
        """
        super().__init__(league_data, player_variances, roster_size)

        # Load projections
        self.projections = pd.read_csv(projections_file)
        self.base_projection_weight = projection_weight
        self.injury_penalty_strength = injury_penalty_strength

        # Map projection columns to our categories
        self.projection_column_map = {
            'PTS': 'PTS',
            'REB': 'TREB',  # Note: projections use TREB
            'AST': 'AST',
            'STL': 'STL',
            'BLK': 'BLK',
            'TOV': 'TO',    # Note: projections use TO
            'FG3M': '3PTM', # Note: projections use 3PTM
            'DD': 'DD',
            'FG_PCT': 'FG%',
            'FT_PCT': 'FT%',
            'FG3_PCT': '3P%'
        }

        # Estimate player ages/experience from data
        self._estimate_player_experience()

        # Calculate injury risk factors
        self._calculate_injury_risk()

    def _estimate_player_experience(self):
        """
        Estimate player experience based on number of seasons in dataset.
        More seasons = more experienced = trust history more.
        """
        self.player_experience = {}

        for player_name in self.league_data['PLAYER_NAME'].unique():
            player_data = self.league_data[self.league_data['PLAYER_NAME'] == player_name]
            num_seasons = player_data['NBA_SEASON'].nunique()
            self.player_experience[player_name] = num_seasons

    def _calculate_injury_risk(self):
        """
        Calculate injury risk factor based on projected GP.

        Full season = 82 games
        High risk: GP < 60 (73% of games)
        Moderate risk: GP 60-70
        Low risk: GP > 70
        """
        self.injury_risk = {}

        for _, row in self.projections.iterrows():
            player_name = row['PLAYER']
            gp_projected = row['GP']

            # Calculate availability percentage
            availability_pct = gp_projected / 82.0

            # Risk factor: 1.0 = no risk, <1.0 = penalized
            # This will be multiplied by projected stats
            if availability_pct >= 0.85:  # 70+ games
                risk_factor = 1.0
            elif availability_pct >= 0.73:  # 60-70 games
                # Linear scale from 1.0 to 0.9
                risk_factor = 0.9 + 0.1 * (availability_pct - 0.73) / 0.12
            else:  # < 60 games
                # More aggressive penalty below 60 games
                # 60 games = 0.9, 50 games = 0.75, 40 games = 0.6
                risk_factor = max(0.5, availability_pct * 1.25)

            # Apply injury penalty strength
            risk_factor = 1.0 - self.injury_penalty_strength * (1.0 - risk_factor)

            self.injury_risk[player_name] = {
                'gp_projected': gp_projected,
                'availability_pct': availability_pct,
                'risk_factor': risk_factor
            }

    def get_projection_weight(self, player_name):
        """
        Calculate player-specific projection weight.

        Young/improving players → higher weight on projections
        Veterans → higher weight on history

        Returns:
        --------
        float : Weight for projections (0-1)
        """
        # Get experience
        num_seasons = self.player_experience.get(player_name, 3)

        # Projection weight schedule:
        # 1-2 seasons (rookies/sophomores): 0.7 projection weight
        # 3 seasons: 0.5 (equal weight)
        # 4-5 seasons: 0.4
        # 6+ seasons (veterans): 0.3

        if num_seasons <= 2:
            # Young players - trust projections more
            weight = 0.7
        elif num_seasons == 3:
            # Emerging players - balanced
            weight = 0.5
        elif num_seasons <= 5:
            # Established players - trust history more
            weight = 0.4
        else:
            # Veterans - trust history most
            weight = 0.3

        return weight

    def get_blended_mean(self, player_name, category):
        """
        Calculate blended mean from historical data and projections.

        Returns:
        --------
        float : Blended weekly mean for category
        """
        # Get historical mean
        player_data = self.league_data[self.league_data['PLAYER_NAME'] == player_name]

        if player_data.empty:
            return None

        historical_mean = player_data[category].mean()

        # Try to get projection
        projection_col = self.projection_column_map.get(category)

        if projection_col is None:
            # No projection available for this category
            return historical_mean

        player_projection = self.projections[self.projections['PLAYER'] == player_name]

        if player_projection.empty:
            # No projection for this player
            return historical_mean

        projected_value = player_projection[projection_col].values[0]

        if np.isnan(projected_value):
            return historical_mean

        # Convert season total to weekly average
        gp_projected = player_projection['GP'].values[0]

        if gp_projected == 0:
            return historical_mean

        # Handle percentages vs counting stats differently
        if category in self.percentage_cats:
            # Percentages are already in correct format (e.g., 0.579 for 57.9%)
            # No need to convert to weekly - they're already percentages
            projected_weekly = projected_value
        else:
            # Counting stats: Convert season total to weekly average
            # Assume 3 games per week on average
            # Total season / (GP / 3) = weekly average
            weeks_projected = gp_projected / 3.0
            projected_weekly = projected_value / weeks_projected if weeks_projected > 0 else 0

        # Get projection weight for this player
        proj_weight = self.get_projection_weight(player_name)

        # Blend
        blended_mean = proj_weight * projected_weekly + (1 - proj_weight) * historical_mean

        # Apply injury risk adjustment
        injury_info = self.injury_risk.get(player_name, {'risk_factor': 1.0})
        risk_factor = injury_info['risk_factor']

        # Adjust blended mean by availability
        adjusted_mean = blended_mean * risk_factor

        return adjusted_mean

    def calculate_x_score(self, player_name, category):
        """
        Calculate X-score using blended mean and historical variance.

        Uses:
        - Blended mean (projections + history + injury adjustment)
        - Historical variance (consistency from past performance)
        """
        if category not in self.league_stats:
            return 0.0

        player_data = self.league_data[self.league_data['PLAYER_NAME'] == player_name]

        if player_data.empty:
            return 0.0

        # Get blended mean (incorporates projections + injury risk)
        mu_player = self.get_blended_mean(player_name, category)

        if mu_player is None or np.isnan(mu_player):
            return 0.0

        mu_league = self.league_stats[category]['mu_league']

        # Use historical variance (represents consistency)
        var_weekly = player_data[category].var()

        if np.isnan(var_weekly) or var_weekly <= 0 or len(player_data) < 3:
            sigma_within = np.sqrt(self.league_stats[category]['sigma_within_sq'])
        else:
            sigma_within = np.sqrt(var_weekly)

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

            if np.isnan(player_attempts) or np.isnan(league_attempts) or league_attempts == 0:
                return 0.0

            x_score = (player_attempts / league_attempts) * (mu_player - mu_league) / sigma_within
        else:
            # Standard X-score for counting stats
            if category == 'TOV':
                x_score = (mu_league - mu_player) / sigma_within
            else:
                x_score = (mu_player - mu_league) / sigma_within

        return x_score

    def get_player_info(self, player_name):
        """
        Get detailed player info including projection blending and injury risk.

        Returns:
        --------
        dict : Player information
        """
        info = {
            'player_name': player_name,
            'experience_seasons': self.player_experience.get(player_name, 0),
            'projection_weight': self.get_projection_weight(player_name),
            'injury_risk': self.injury_risk.get(player_name, {})
        }

        # Add category-by-category comparison
        info['categories'] = {}

        for cat in self.all_cats:
            player_data = self.league_data[self.league_data['PLAYER_NAME'] == player_name]

            if not player_data.empty:
                historical_mean = player_data[cat].mean()
                blended_mean = self.get_blended_mean(player_name, cat)
                x_score = self.calculate_x_score(player_name, cat)

                info['categories'][cat] = {
                    'historical_mean': historical_mean,
                    'blended_mean': blended_mean,
                    'x_score': x_score,
                    'difference': blended_mean - historical_mean if blended_mean else None
                }

        return info
