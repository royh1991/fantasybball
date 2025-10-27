"""
Weekly Fantasy Basketball Projection System

Uses adaptive Bayesian modeling to project weekly matchups.
Simulates 3 games per player, 500 matchup iterations.
Evaluates 11 categories: FG%, FT%, 3P%, 3PM, PTS, REB, AST, STL, BLK, TO, DD
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def parse_date(date_str: str) -> datetime:
    """Parse dates in format 'OCT 25, 2023' to datetime."""
    try:
        return datetime.strptime(date_str.strip(), "%b %d, %Y")
    except:
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except:
            return None


def adaptive_bayesian_update(prev_mean, prev_var, new_obs, obs_var, evolution_var):
    """Kalman-style Bayesian update for evolving stat rates."""
    K = (prev_var + evolution_var) / (prev_var + evolution_var + obs_var)
    new_mean = prev_mean + K * (new_obs - prev_mean)
    new_var = (1 - K) * (prev_var + evolution_var)
    return new_mean, new_var


def beta_params_from_mean_var(mean, var):
    """
    Convert mean and variance to Beta distribution parameters.

    For Beta(α, β):
        E[X] = α / (α + β) = mean
        Var[X] = αβ / ((α+β)²(α+β+1)) = var
    """
    # Ensure mean is in valid range
    mean = max(0.01, min(0.99, mean))

    # Ensure variance is not too large
    max_var = mean * (1 - mean) * 0.9
    var = min(var, max_var)
    var = max(var, 0.0001)

    # Method of moments
    common = (mean * (1 - mean) / var) - 1

    if common < 1:
        # Not enough data - use weak prior
        common = 2

    alpha = mean * common
    beta = (1 - mean) * common

    return max(alpha, 0.5), max(beta, 0.5)


def negative_binomial_params(mean, var):
    """
    Convert mean and variance to Negative Binomial parameters.

    For NegBin(r, p):
        E[X] = r(1-p)/p = mean
        Var[X] = r(1-p)/p² = var

    Returns (n, p) for np.random.negative_binomial(n, p)
    If var <= mean, returns None, None (use Poisson instead)
    """
    if var <= mean:
        return None, None

    # Negative binomial parameterization
    r = (mean * mean) / (var - mean)
    p = mean / var

    # Ensure valid parameters
    r = max(r, 0.1)
    p = max(0.01, min(0.99, p))

    return r, p


class FantasyProjectionModel:
    """
    Adaptive Bayesian model for all fantasy basketball stats.
    Projects: FGM, FGA, FTM, FTA, 3PM, 3PA, PTS, REB, AST, STL, BLK, TOV
    """

    def __init__(self, evolution_rate: float = 0.5):
        # Attempt stats (sample from Poisson)
        self.attempt_stats = ['FGA', 'FTA', 'FG3A']
        # Counting stats (sample from Poisson)
        self.counting_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        # Shooting percentages (for conditional sampling)
        self.shooting_pcts = ['FG_PCT', 'FT_PCT', 'FG3_PCT']

        self.evolution_rate = evolution_rate
        self.distributions = {}
        self.percentages = {}
        self.dd_rate = 0.0  # Historical double-double rate

    def fit_player(self, historical_data: pd.DataFrame, player_nba_name: str):
        """Fit adaptive model for a player using all their historical data."""
        # Mapping for filtering shooting percentages by attempts
        pct_to_attempt = {
            'FG_PCT': 'FGA',
            'FT_PCT': 'FTA',
            'FG3_PCT': 'FG3A'
        }

        # Case-insensitive comparison to handle mapping inconsistencies
        player_data = historical_data[historical_data['PLAYER_NAME'].str.lower() == player_nba_name.lower()].copy()

        if len(player_data) == 0:
            return False

        # Parse and sort by date
        player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
        player_data = player_data.dropna(subset=['parsed_date'])
        player_data = player_data.sort_values('parsed_date')

        # Split into training (all historical) and current season (for adaptive update)
        # We're in 2025-26 season, so use Oct 1, 2025 as cutoff
        cutoff_date = datetime(2025, 10, 1)
        all_historical = player_data[player_data['parsed_date'] < cutoff_date]
        current_season = player_data[player_data['parsed_date'] >= cutoff_date]

        # Use only recent games for training (last 150 games, or all if less than 150)
        # This ensures the model focuses on current ability, not early career
        if len(all_historical) > 150:
            training_data = all_historical.tail(150)
        else:
            training_data = all_historical

        # For rookies with no historical data, use all their current season games as training
        if len(training_data) < 5 and len(current_season) >= 5:
            # Rookie - use all current season data
            training_data = current_season
            current_season = pd.DataFrame()  # No update data

        if len(training_data) < 5:  # Still need minimum data
            return False

        # Initialize distributions for attempt stats and counting stats
        # Use recency weighting to capture trends (more recent games = higher weight)
        all_stats = self.attempt_stats + self.counting_stats

        n_games = len(training_data)
        # Exponential decay: more recent games get higher weight
        weights = 0.9 ** np.arange(n_games-1, -1, -1)
        weights = weights / weights.sum()  # Normalize to sum to 1

        for stat in all_stats:
            if stat not in training_data.columns:
                continue

            values = training_data[stat].values

            # Weighted mean and variance
            mean_val = np.average(values, weights=weights)
            var_val = np.average((values - mean_val)**2, weights=weights)

            # Effective sample size for uncertainty
            # With decay weighting, effective n is lower than actual n
            effective_n = 1.0 / np.sum(weights**2)  # Kish's effective sample size
            initial_uncertainty = var_val / effective_n

            self.distributions[stat] = {
                'mean': mean_val,
                'var': var_val,
                'obs_var': var_val,
                'posterior_mean': mean_val,
                'posterior_var': initial_uncertainty
            }

        # Initialize shooting percentages
        # CRITICAL FIX: Only include games where player attempted shots!
        # Mapping: FG_PCT requires FGA > 0, FT_PCT requires FTA > 0, FG3_PCT requires FG3A > 0
        for pct_stat in self.shooting_pcts:
            if pct_stat not in training_data.columns:
                continue

            # Get corresponding attempt column
            attempt_col = pct_to_attempt.get(pct_stat)
            if attempt_col not in training_data.columns:
                continue

            # FILTER: Only games where player attempted shots
            mask = (training_data[attempt_col] > 0) & \
                   (training_data[pct_stat].notna()) & \
                   (training_data[pct_stat] >= 0) & \
                   (training_data[pct_stat] <= 1)

            # Get filtered data with original index preserved
            filtered_indices = training_data.index[mask]
            values = training_data.loc[mask, pct_stat].values

            if len(values) == 0:
                continue

            # Apply recency weighting to filtered games
            # Map filtered indices back to positions in training_data
            n_total = len(training_data)
            positions = np.array([training_data.index.get_loc(idx) for idx in filtered_indices])

            # Calculate weights based on position (later games = higher weight)
            pct_weights = 0.9 ** (n_total - 1 - positions)
            pct_weights = pct_weights / pct_weights.sum()

            # Weighted mean and variance
            mean_val = np.average(values, weights=pct_weights)
            var_val = np.average((values - mean_val)**2, weights=pct_weights)

            # Effective sample size
            effective_n = 1.0 / np.sum(pct_weights**2)
            initial_uncertainty = var_val / effective_n

            self.percentages[pct_stat] = {
                'mean': mean_val,
                'var': var_val,
                'obs_var': var_val,
                'posterior_mean': mean_val,
                'posterior_var': initial_uncertainty
            }

        # Adaptive update if current season data available (for veterans)
        if len(current_season) > 0:
            n_update = min(10, len(current_season))
            update_data = current_season.head(n_update)

            # Update attempt stats and counting stats
            for stat in all_stats:
                if stat not in update_data.columns or stat not in self.distributions:
                    continue

                posterior_mean = self.distributions[stat]['posterior_mean']
                posterior_var = self.distributions[stat]['posterior_var']
                obs_var = self.distributions[stat]['obs_var']
                evolution_var = self.evolution_rate * obs_var / n_update

                for value in update_data[stat].values:
                    posterior_mean, posterior_var = adaptive_bayesian_update(
                        posterior_mean, posterior_var, value, obs_var, evolution_var
                    )

                self.distributions[stat]['posterior_mean'] = posterior_mean
                self.distributions[stat]['posterior_var'] = posterior_var

            # Update shooting percentages
            # CRITICAL FIX: Only update with games where player attempted shots
            for pct_stat in self.shooting_pcts:
                if pct_stat not in update_data.columns or pct_stat not in self.percentages:
                    continue

                # Get corresponding attempt column
                attempt_col = pct_to_attempt.get(pct_stat)
                if attempt_col not in update_data.columns:
                    continue

                # FILTER: Only games with attempts > 0
                mask = (update_data[attempt_col] > 0) & \
                       (update_data[pct_stat].notna()) & \
                       (update_data[pct_stat] >= 0) & \
                       (update_data[pct_stat] <= 1)

                values = update_data.loc[mask, pct_stat].values

                if len(values) == 0:
                    continue

                posterior_mean = self.percentages[pct_stat]['posterior_mean']
                posterior_var = self.percentages[pct_stat]['posterior_var']
                obs_var = self.percentages[pct_stat]['obs_var']
                evolution_var = self.evolution_rate * obs_var / len(values)

                for value in values:
                    posterior_mean, posterior_var = adaptive_bayesian_update(
                        posterior_mean, posterior_var, value, obs_var, evolution_var
                    )

                self.percentages[pct_stat]['posterior_mean'] = posterior_mean
                self.percentages[pct_stat]['posterior_var'] = posterior_var

        # Calculate historical double-double rate from training data
        # This will be used as a baseline in simulate_game()
        if all(stat in training_data.columns for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']):
            dd_count = 0
            for _, row in training_data.iterrows():
                dd_stats = [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']]
                if sum(s >= 10 for s in dd_stats) >= 2:
                    dd_count += 1
            self.dd_rate = dd_count / len(training_data) if len(training_data) > 0 else 0.0
        else:
            self.dd_rate = 0.0

        return True

    def fit_from_espn_projection(self, espn_row):
        """Fit model using ESPN season projections (for rookies/players without historical data)."""
        # Mapping for attempt and counting stats
        espn_to_stat = {
            'FGA': 'FGA',
            'FTA': 'FTA',
            '3PA': 'FG3A',
            'PTS': 'PTS',
            'TREB': 'REB',
            'AST': 'AST',
            'STL': 'STL',
            'BLK': 'BLK',
            'TO': 'TOV'
        }

        # Mapping for shooting percentages
        espn_to_pct = {
            'FG%': 'FG_PCT',
            'FT%': 'FT_PCT',
            '3P%': 'FG3_PCT'
        }

        games_played = espn_row['GP']
        if games_played == 0:
            return False

        # Convert season totals to per-game averages for attempt/counting stats
        for espn_col, stat_name in espn_to_stat.items():
            if espn_col not in espn_row or pd.isna(espn_row[espn_col]):
                continue

            per_game = espn_row[espn_col] / games_played

            # Use a reasonable variance estimate (assume CV ~0.4 for counting stats)
            variance = (0.4 * per_game) ** 2 if per_game > 0 else 0.1

            self.distributions[stat_name] = {
                'mean': per_game,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': per_game,
                'posterior_var': variance / 10  # Lower uncertainty for ESPN projections
            }

        # Set shooting percentages
        for espn_col, pct_name in espn_to_pct.items():
            if espn_col not in espn_row or pd.isna(espn_row[espn_col]):
                continue

            pct_value = espn_row[espn_col]

            # Use a reasonable variance estimate for percentages
            variance = 0.02  # ~14% std for shooting percentages

            self.percentages[pct_name] = {
                'mean': pct_value,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': pct_value,
                'posterior_var': variance / 10
            }

        # Estimate DD rate based on projected stats
        # Players averaging 10+ in 2+ categories likely get many DDs
        pts_pg = espn_row.get('PTS', 0) / games_played if games_played > 0 else 0
        reb_pg = espn_row.get('TREB', 0) / games_played if games_played > 0 else 0
        ast_pg = espn_row.get('AST', 0) / games_played if games_played > 0 else 0
        stl_pg = espn_row.get('STL', 0) / games_played if games_played > 0 else 0
        blk_pg = espn_row.get('BLK', 0) / games_played if games_played > 0 else 0

        stats_above_10 = sum([pts_pg >= 10, reb_pg >= 10, ast_pg >= 10, stl_pg >= 10, blk_pg >= 10])
        stats_close = sum([8 <= pts_pg < 10, 8 <= reb_pg < 10, 8 <= ast_pg < 10, 8 <= stl_pg < 10, 8 <= blk_pg < 10])

        # Estimate DD rate based on proximity to thresholds
        if stats_above_10 >= 3:
            self.dd_rate = 0.85  # Almost always DD (e.g., Jokić)
        elif stats_above_10 == 2:
            self.dd_rate = 0.70  # Very frequently DD (e.g., AD, KAT)
        elif stats_above_10 == 1 and stats_close >= 1:
            self.dd_rate = 0.40  # Sometimes DD (e.g., LeBron)
        elif stats_close >= 2:
            self.dd_rate = 0.20  # Occasionally DD
        else:
            self.dd_rate = 0.05  # Rarely DD

        return True

    def fit_replacement_level(self, position: str = 'SF'):
        """
        Create a replacement-level player model based on position.

        Uses league-average stats for the position. This is used for rookies
        with no historical data or ESPN projections.

        Replacement level stats are conservative estimates representing
        a typical deep-roster player at each position.
        """
        # Replacement level stats by position (per-game averages)
        # These are based on ~12th-15th man on roster statistics
        replacement_stats = {
            'PG': {
                'FGA': 7.0, 'FTA': 2.0, 'FG3A': 2.5,
                'PTS': 8.0, 'REB': 3.0, 'AST': 3.5, 'STL': 0.8, 'BLK': 0.2, 'TOV': 1.5,
                'FG_PCT': 0.430, 'FT_PCT': 0.750, 'FG3_PCT': 0.340
            },
            'SG': {
                'FGA': 7.5, 'FTA': 2.0, 'FG3A': 2.5,
                'PTS': 9.0, 'REB': 2.5, 'AST': 2.0, 'STL': 0.7, 'BLK': 0.2, 'TOV': 1.2,
                'FG_PCT': 0.435, 'FT_PCT': 0.780, 'FG3_PCT': 0.350
            },
            'SF': {
                'FGA': 7.5, 'FTA': 2.0, 'FG3A': 2.0,
                'PTS': 9.0, 'REB': 4.0, 'AST': 1.5, 'STL': 0.7, 'BLK': 0.4, 'TOV': 1.0,
                'FG_PCT': 0.445, 'FT_PCT': 0.760, 'FG3_PCT': 0.345
            },
            'PF': {
                'FGA': 7.0, 'FTA': 2.5, 'FG3A': 1.5,
                'PTS': 9.0, 'REB': 5.5, 'AST': 1.5, 'STL': 0.6, 'BLK': 0.6, 'TOV': 1.0,
                'FG_PCT': 0.460, 'FT_PCT': 0.740, 'FG3_PCT': 0.330
            },
            'C': {
                'FGA': 6.0, 'FTA': 2.5, 'FG3A': 0.5,
                'PTS': 8.0, 'REB': 6.0, 'AST': 1.0, 'STL': 0.5, 'BLK': 0.8, 'TOV': 1.2,
                'FG_PCT': 0.520, 'FT_PCT': 0.680, 'FG3_PCT': 0.280
            }
        }

        # Default to SF if position not recognized
        stats = replacement_stats.get(position, replacement_stats['SF'])

        # Set attempt stats
        for stat in ['FGA', 'FTA', 'FG3A']:
            value = stats[stat]
            variance = (0.5 * value) ** 2  # Higher variance for replacement level

            self.distributions[stat] = {
                'mean': value,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': value,
                'posterior_var': variance / 5  # Higher uncertainty
            }

        # Set counting stats
        for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
            value = stats[stat]
            variance = (0.5 * value) ** 2

            self.distributions[stat] = {
                'mean': value,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': value,
                'posterior_var': variance / 5
            }

        # Set shooting percentages
        for pct_stat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
            value = stats[pct_stat]
            variance = 0.03  # Higher variance for replacement level

            self.percentages[pct_stat] = {
                'mean': value,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': value,
                'posterior_var': variance / 5
            }

        # Replacement level players rarely get double-doubles
        self.dd_rate = 0.02  # 2% chance

        return True

    def simulate_game(self) -> Dict:
        """
        Simulate a single game with PROPER VARIANCE.

        Key improvements over original:
        1. Uses Negative Binomial for counting stats (allows variance > mean)
        2. Samples shooting % from Beta distribution (parameter uncertainty)
        3. Incorporates both obs_var and posterior_var
        """
        stats = {}

        # Step 1: Sample attempt stats using TOTAL VARIANCE
        for attempt_stat in self.attempt_stats:
            if attempt_stat in self.distributions:
                posterior_mean = self.distributions[attempt_stat]['posterior_mean']
                posterior_var = self.distributions[attempt_stat]['posterior_var']
                obs_var = self.distributions[attempt_stat]['obs_var']

                # Total variance = observation variance + parameter uncertainty
                # Use 1.5x obs_var to account for additional model uncertainty
                total_var = obs_var * 1.5 + posterior_var

                # Use Negative Binomial if overdispersed, else Poisson
                r, p = negative_binomial_params(posterior_mean, total_var)

                if r is not None and p is not None:
                    # Overdispersed - use Negative Binomial
                    value = np.random.negative_binomial(r, p)
                else:
                    # Not overdispersed - use Poisson
                    value = np.random.poisson(max(0, posterior_mean))

                stats[attempt_stat] = max(1, value)  # At least 1 attempt
            else:
                stats[attempt_stat] = 1

        # Step 2: Sample shooting percentages from Beta distribution
        # This incorporates game-to-game variation in shooting ability

        # FG%
        if 'FG_PCT' in self.percentages:
            mean = self.percentages['FG_PCT']['posterior_mean']
            var = self.percentages['FG_PCT']['posterior_var'] + \
                  self.percentages['FG_PCT']['obs_var'] * 1.2  # Use 120% of obs_var for extra variance

            alpha, beta = beta_params_from_mean_var(mean, var)
            fg_pct = np.random.beta(alpha, beta)
            fg_pct = max(0.0, min(1.0, fg_pct))  # Ensure valid probability
            stats['FGM'] = np.random.binomial(stats['FGA'], fg_pct)
        else:
            stats['FGM'] = 0

        # FT%
        if 'FT_PCT' in self.percentages:
            mean = self.percentages['FT_PCT']['posterior_mean']
            var = self.percentages['FT_PCT']['posterior_var'] + \
                  self.percentages['FT_PCT']['obs_var'] * 1.2  # Use 120% of obs_var

            alpha, beta = beta_params_from_mean_var(mean, var)
            ft_pct = np.random.beta(alpha, beta)
            ft_pct = max(0.0, min(1.0, ft_pct))  # Ensure valid probability
            stats['FTM'] = np.random.binomial(stats['FTA'], ft_pct)
        else:
            stats['FTM'] = 0

        # 3P%
        if 'FG3_PCT' in self.percentages:
            mean = self.percentages['FG3_PCT']['posterior_mean']
            var = self.percentages['FG3_PCT']['posterior_var'] + \
                  self.percentages['FG3_PCT']['obs_var'] * 1.2  # Use 120% of obs_var

            alpha, beta = beta_params_from_mean_var(mean, var)
            fg3_pct = np.random.beta(alpha, beta)
            fg3_pct = max(0.0, min(1.0, fg3_pct))  # Ensure valid probability
            stats['FG3M'] = np.random.binomial(stats['FG3A'], fg3_pct)
        else:
            stats['FG3M'] = 0

        # Step 3: Enforce constraint that 3PM can't exceed FGM
        stats['FG3M'] = min(stats['FG3M'], stats['FGM'])

        # Step 4: Sample counting stats with TOTAL VARIANCE using Negative Binomial
        for counting_stat in self.counting_stats:
            if counting_stat in self.distributions:
                posterior_mean = self.distributions[counting_stat]['posterior_mean']
                posterior_var = self.distributions[counting_stat]['posterior_var']
                obs_var = self.distributions[counting_stat]['obs_var']

                # Total variance includes both sources of uncertainty
                # Use 1.2x obs_var to account for additional model uncertainty
                total_var = obs_var * 1.2 + posterior_var

                # Use Negative Binomial for overdispersion
                r, p = negative_binomial_params(posterior_mean, total_var)

                if r is not None and p is not None:
                    value = np.random.negative_binomial(r, p)
                else:
                    value = np.random.poisson(max(0, posterior_mean))

                stats[counting_stat] = max(0, value)
            else:
                stats[counting_stat] = 0

        # Step 5: Calculate double-double using hybrid approach
        # Combines historical DD rate with proximity to thresholds
        dd_stats = [stats['PTS'], stats['REB'], stats['AST'], stats['STL'], stats['BLK']]
        stats_at_10 = sum(s >= 10 for s in dd_stats)
        stats_close = sum(8 <= s < 10 for s in dd_stats)

        # Calculate DD probability based on simulated stats and historical rate
        if stats_at_10 >= 2:
            # Already has DD based on simulated stats - very likely
            dd_prob = 0.95
        elif stats_at_10 == 1 and stats_close >= 2:
            # Close to DD with multiple stats near threshold
            # Use historical rate with moderate boost for correlation
            dd_prob = min(0.85, self.dd_rate * 1.3)
        elif stats_at_10 == 1 and stats_close == 1:
            # One stat at 10+, one close - use historical rate as-is
            dd_prob = self.dd_rate
        elif stats_at_10 == 1:
            # One stat at 10+, others far away - unlikely DD
            dd_prob = self.dd_rate * 0.4
        elif stats_close >= 2:
            # Multiple stats close but none at 10 - medium probability
            dd_prob = self.dd_rate * 0.7
        elif stats_close == 1:
            # One stat close - low probability
            dd_prob = self.dd_rate * 0.3
        else:
            # Far from DD - very rare
            dd_prob = self.dd_rate * 0.1

        # Sample DD based on calculated probability
        stats['DD'] = 1 if np.random.random() < dd_prob else 0

        return stats


def load_data():
    """Load all necessary data files."""
    print("Loading data files...")

    roster = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/roster_snapshots/roster_latest.csv')
    matchups = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/matchups_latest.csv')
    historical = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    mapping = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/mappings/player_mapping_latest.csv')
    espn_projections = pd.read_csv('/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv')

    print(f"  Roster: {len(roster)} player-team entries")
    print(f"  Matchups: {len(matchups)} matchups")
    print(f"  Historical: {len(historical)} game logs")
    print(f"  Mapping: {len(mapping)} player mappings")
    print(f"  ESPN Projections: {len(espn_projections)} players")

    return roster, matchups, historical, mapping, espn_projections


def fit_player_models(roster: pd.DataFrame, historical: pd.DataFrame,
                     mapping: pd.DataFrame, espn_projections: pd.DataFrame) -> Dict:
    """Fit adaptive Bayesian models for all rostered players."""
    print("\nFitting player models...")

    # Create mapping dict from ESPN name to NBA API name
    name_map = dict(zip(mapping['espn_name'], mapping['nba_api_name']))

    # Create ESPN projections lookup by player name
    espn_proj_dict = {row['PLAYER']: row for _, row in espn_projections.iterrows()}

    player_models = {}
    active_players = roster[roster['currently_rostered'] == True]

    for idx, player_row in active_players.iterrows():
        espn_name = player_row['player_name']
        injury_status = player_row['injury_status']

        # Skip injured players
        if injury_status in ['OUT']:
            print(f"  Skipping {espn_name} (OUT)")
            continue

        # Get NBA API name
        nba_name = name_map.get(espn_name)
        if not nba_name:
            print(f"  Warning: No mapping for {espn_name}")
            continue

        # Fit model - try historical data first, then ESPN projections
        model = FantasyProjectionModel(evolution_rate=0.5)
        success = model.fit_player(historical, nba_name)

        if not success:
            # Try ESPN projections as fallback
            if espn_name in espn_proj_dict:
                success = model.fit_from_espn_projection(espn_proj_dict[espn_name])
                if success:
                    print(f"  Using ESPN projections for {espn_name}")
                else:
                    print(f"  Warning: Insufficient data for {espn_name}")
            else:
                print(f"  Warning: No historical or ESPN data for {espn_name}")

        if success:
            player_models[espn_name] = model

    print(f"\nSuccessfully fitted {len(player_models)} player models")
    return player_models


def simulate_team_week(team_roster: List[str], player_models: Dict,
                      n_games: int = 3) -> Dict:
    """Simulate a team's weekly performance (n games per player)."""
    team_stats = {
        'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
        'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
    }

    for player_name in team_roster:
        if player_name not in player_models:
            continue

        model = player_models[player_name]

        # Simulate n_games for this player
        for _ in range(n_games):
            game_stats = model.simulate_game()
            for stat in team_stats:
                if stat in game_stats:
                    team_stats[stat] += game_stats[stat]

    return team_stats


def calculate_category_winner(team_a_stats: Dict, team_b_stats: Dict) -> Dict:
    """
    Compare two teams across 11 categories.
    Returns dict with category results.
    """
    results = {}

    # Percentage categories (higher is better)
    for pct_stat, makes, attempts in [('FG%', 'FGM', 'FGA'),
                                       ('FT%', 'FTM', 'FTA'),
                                       ('3P%', 'FG3M', 'FG3A')]:
        a_pct = team_a_stats[makes] / team_a_stats[attempts] if team_a_stats[attempts] > 0 else 0
        b_pct = team_b_stats[makes] / team_b_stats[attempts] if team_b_stats[attempts] > 0 else 0

        if a_pct > b_pct:
            results[pct_stat] = 'A'
        elif b_pct > a_pct:
            results[pct_stat] = 'B'
        else:
            results[pct_stat] = 'TIE'

    # Counting stats (higher is better, except TO which is lower is better)
    counting_stats = ['FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'DD']
    for stat in counting_stats:
        if team_a_stats[stat] > team_b_stats[stat]:
            results[stat] = 'A'
        elif team_b_stats[stat] > team_a_stats[stat]:
            results[stat] = 'B'
        else:
            results[stat] = 'TIE'

    # Turnovers (lower is better)
    if team_a_stats['TOV'] < team_b_stats['TOV']:
        results['TO'] = 'A'
    elif team_b_stats['TOV'] < team_a_stats['TOV']:
        results['TO'] = 'B'
    else:
        results['TO'] = 'TIE'

    return results


def simulate_matchup(team_a_roster: List[str], team_b_roster: List[str],
                    player_models: Dict, n_simulations: int = 500) -> Dict:
    """
    Simulate a matchup n_simulations times and calculate win probability.
    """
    team_a_wins = 0
    team_b_wins = 0
    ties = 0

    category_wins_a = {cat: 0 for cat in ['FG%', 'FT%', '3P%', 'FG3M', 'PTS', 'REB',
                                           'AST', 'STL', 'BLK', 'TO', 'DD']}
    category_wins_b = {cat: 0 for cat in category_wins_a.keys()}

    for _ in range(n_simulations):
        # Simulate weekly stats for both teams
        team_a_stats = simulate_team_week(team_a_roster, player_models, n_games=3)
        team_b_stats = simulate_team_week(team_b_roster, player_models, n_games=3)

        # Compare categories
        category_results = calculate_category_winner(team_a_stats, team_b_stats)

        # Count category wins
        a_cats = sum(1 for v in category_results.values() if v == 'A')
        b_cats = sum(1 for v in category_results.values() if v == 'B')

        if a_cats > b_cats:
            team_a_wins += 1
        elif b_cats > a_cats:
            team_b_wins += 1
        else:
            ties += 1

        # Track category win rates
        for cat, winner in category_results.items():
            if winner == 'A':
                category_wins_a[cat] += 1
            elif winner == 'B':
                category_wins_b[cat] += 1

    return {
        'team_a_wins': team_a_wins,
        'team_b_wins': team_b_wins,
        'ties': ties,
        'team_a_win_pct': team_a_wins / n_simulations,
        'team_b_win_pct': team_b_wins / n_simulations,
        'category_win_rates_a': {k: v / n_simulations for k, v in category_wins_a.items()},
        'category_win_rates_b': {k: v / n_simulations for k, v in category_wins_b.items()},
    }


def main():
    """Main execution."""
    print("="*80)
    print("WEEKLY FANTASY BASKETBALL PROJECTION SYSTEM")
    print("="*80)

    # Load data
    roster, matchups, historical, mapping, espn_projections = load_data()

    # Fit player models
    player_models = fit_player_models(roster, historical, mapping, espn_projections)

    # Process each matchup
    print("\n" + "="*80)
    print("MATCHUP PROJECTIONS")
    print("="*80)

    results = []

    for idx, matchup_row in matchups.iterrows():
        home_team_id = matchup_row['home_team_id']
        away_team_id = matchup_row['away_team_id']
        home_team_name = matchup_row['home_team_name']
        away_team_name = matchup_row['away_team_name']

        print(f"\n{home_team_name} vs {away_team_name}")
        print("-" * 60)

        # Get rosters
        home_roster = roster[(roster['fantasy_team_id'] == home_team_id) &
                           (roster['currently_rostered'] == True)]['player_name'].tolist()
        away_roster = roster[(roster['fantasy_team_id'] == away_team_id) &
                           (roster['currently_rostered'] == True)]['player_name'].tolist()

        print(f"  {home_team_name} roster: {len(home_roster)} players")
        print(f"  {away_team_name} roster: {len(away_roster)} players")

        # Simulate matchup
        print(f"  Running 500 simulations...")
        matchup_result = simulate_matchup(home_roster, away_roster, player_models, n_simulations=500)

        # Print results
        print(f"\n  PROJECTED WINNER: ", end="")
        if matchup_result['team_a_win_pct'] > matchup_result['team_b_win_pct']:
            print(f"{home_team_name} ({matchup_result['team_a_win_pct']:.1%})")
        else:
            print(f"{away_team_name} ({matchup_result['team_b_win_pct']:.1%})")

        print(f"  Win Probability: {home_team_name} {matchup_result['team_a_win_pct']:.1%} | " +
              f"{away_team_name} {matchup_result['team_b_win_pct']:.1%}")
        print(f"  Projected Score: {home_team_name} {matchup_result['team_a_wins']}/500 | " +
              f"{away_team_name} {matchup_result['team_b_wins']}/500")

        results.append({
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_win_pct': matchup_result['team_a_win_pct'],
            'away_win_pct': matchup_result['team_b_win_pct'],
            'home_wins': matchup_result['team_a_wins'],
            'away_wins': matchup_result['team_b_wins'],
            'ties': matchup_result['ties']
        })

    # Summary table
    print("\n" + "="*80)
    print("WEEKLY PROJECTIONS SUMMARY")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("PROJECTIONS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
