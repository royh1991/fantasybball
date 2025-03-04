import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class BasketballSimulator:
    def __init__(self, player_data, team_data=None, league_data=None):
        """
        Initialize the basketball simulator with historical game-by-game data
        
        Parameters:
        -----------
        player_data : pandas DataFrame
            Game-by-game player statistics with columns:
            - player_id: unique identifier for each player
            - game_id: unique identifier for each game
            - date: date of the game
            - team_id: player's team
            - opp_team_id: opponent team
            - home: boolean, True if home game
            - minutes: minutes played
            - fga, fgm: field goals attempted and made
            - fg3a, fg3m: 3-point field goals attempted and made
            - fta, ftm: free throws attempted and made
            - pts, reb, ast, stl, blk, tov: points, rebounds, assists, steals, blocks, turnovers
            - (optional) defender_id: primary defender
            - (optional) shot_distance: average shot distance
            - (optional) defender_distance: average defender distance
            - (optional) score_margin: final score margin
        
        team_data : pandas DataFrame, optional
            Team-level statistics for offensive and defensive ratings
            
        league_data : pandas DataFrame, optional
            League averages for each statistic
        """
        self.player_data = player_data
        self.team_data = team_data
        self.league_data = league_data
        
        # Extract unique players, teams, positions
        self.players = player_data['player_id'].unique()
        self.teams = player_data['team_id'].unique() if 'team_id' in player_data.columns else None
        self.positions = player_data['position'].unique() if 'position' in player_data.columns else None
        
        # Initialize models
        self.player_models = {}
        self.player_trends = {}
        self.team_adjustments = {}
        self.correlation_matrices = {}
        
        # Shooting categories to model
        self.shooting_cats = [('fgm', 'fga'), ('fg3m', 'fg3a'), ('ftm', 'fta')]
        
        # Box score categories to model with correlation structure
        self.box_score_cats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'minutes']
        
        # Precompute global means for fallback
        self._compute_global_means()
        
    def _compute_global_means(self):
        """Calculate league averages for all stats"""
        # Shooting percentages
        self.global_means = {}
        for makes, attempts in self.shooting_cats:
            self.global_means[f"{makes}_pct"] = (
                self.player_data[makes].sum() / self.player_data[attempts].sum()
            )
        
        # Box score categories
        for cat in self.box_score_cats:
            if cat in self.player_data.columns:
                self.global_means[cat] = self.player_data[cat].mean()
                
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        # Add shooting percentages
        for makes, attempts in self.shooting_cats:
            if makes in self.player_data.columns and attempts in self.player_data.columns:
                self.player_data[f"{makes}_pct"] = np.where(
                    self.player_data[attempts] > 0,
                    self.player_data[makes] / self.player_data[attempts],
                    np.nan
                )
        
        # Add rolling averages for detecting trends (7-game, 15-game, 30-game)
        player_groups = self.player_data.sort_values('date').groupby('player_id')
        
        windows = [7, 15, 30]
        for window in windows:
            # Rolling shooting percentages
            for makes, attempts in self.shooting_cats:
                if makes in self.player_data.columns and attempts in self.player_data.columns:
                    roll_makes = player_groups[makes].rolling(window).sum().reset_index(level=0, drop=True)
                    roll_attempts = player_groups[attempts].rolling(window).sum().reset_index(level=0, drop=True)
                    self.player_data[f"{makes}_pct_{window}g"] = np.where(
                        roll_attempts > 0,
                        roll_makes / roll_attempts,
                        np.nan
                    )
            
            # Rolling box score stats
            for cat in self.box_score_cats:
                if cat in self.player_data.columns:
                    self.player_data[f"{cat}_{window}g"] = (
                        player_groups[cat].rolling(window).mean().reset_index(level=0, drop=True)
                    )
        
        # Add days rest
        self.player_data['days_rest'] = player_groups['date'].diff().dt.days.fillna(3)
        
        # Add score margin context if not already present
        if 'score_margin' not in self.player_data.columns and 'team_score' in self.player_data.columns and 'opp_score' in self.player_data.columns:
            self.player_data['score_margin'] = self.player_data['team_score'] - self.player_data['opp_score']
        
        return self
    
    def fit_hierarchical_percentage_models(self, min_attempts=50):
        """
        Fit hierarchical Bayesian models for shooting percentages.
        This creates a multilevel model: league → position → team → player
        """
        # Iterate through each shooting category
        for makes, attempts in tqdm(self.shooting_cats, desc="Fitting percentage models"):
            # Skip if data not available
            if makes not in self.player_data.columns or attempts not in self.player_data.columns:
                continue
                
            # Filter to players with minimum attempts
            player_totals = self.player_data.groupby('player_id')[attempts].sum()
            valid_players = player_totals[player_totals >= min_attempts].index
            
            filtered_data = self.player_data[
                (self.player_data['player_id'].isin(valid_players)) & 
                (self.player_data[attempts] > 0)
            ].copy()
            
            # Prepare data for PyMC
            player_idx = pd.Categorical(filtered_data['player_id']).codes
            team_idx = pd.Categorical(filtered_data['team_id']).codes if 'team_id' in filtered_data.columns else None
            position_idx = pd.Categorical(filtered_data['position']).codes if 'position' in filtered_data.columns else None
            
            obs_makes = filtered_data[makes].values
            obs_attempts = filtered_data[attempts].values
            
            # Build the model
            with pm.Model() as model:
                # Hyperpriors for the league
                league_alpha = pm.Gamma('league_alpha', alpha=2.0, beta=0.1)
                league_beta = pm.Gamma('league_beta', alpha=2.0, beta=0.1)
                
                # Priors for positions if available
                if position_idx is not None:
                    n_positions = len(np.unique(position_idx))
                    position_offset = pm.Normal('position_offset', mu=0, sigma=0.1, shape=n_positions)
                
                # Priors for teams if available
                if team_idx is not None:
                    n_teams = len(np.unique(team_idx))
                    team_offset = pm.Normal('team_offset', mu=0, sigma=0.1, shape=n_teams)
                
                # Priors for players
                n_players = len(np.unique(player_idx))
                player_raw = pm.Normal('player_raw', mu=0, sigma=0.2, shape=n_players)
                
                # Calculate player rates with hierarchy
                logit_p = pm.math.logit(league_alpha / (league_alpha + league_beta))
                
                if position_idx is not None:
                    logit_p = logit_p + position_offset[position_idx]
                
                if team_idx is not None:
                    logit_p = logit_p + team_offset[team_idx]
                
                logit_p = logit_p + player_raw[player_idx]
                p = pm.math.invlogit(logit_p)
                
                # Likelihood
                pm.Binomial('obs', n=obs_attempts, p=p, observed=obs_makes)
                
                # Sample from the posterior
                trace = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept=0.9)
            
            # Store the model and trace
            self.player_models[f"{makes}_pct"] = {
                'model': model,
                'trace': trace,
                'players': {player: i for i, player in enumerate(filtered_data['player_id'].unique())}
            }
            
            # Extract and store player rates
            player_rates = az.summary(trace, var_names=['player_raw'])
            player_rates.index = filtered_data['player_id'].unique()
            self.player_models[f"{makes}_pct"]['rates'] = player_rates
            
        return self
    
    def fit_time_series_models(self, min_games=20):
        """
        Fit time series models to capture trends and seasonal patterns
        """
        # Process each player individually
        for player_id in tqdm(self.players, desc="Fitting time series models"):
            player_games = self.player_data[self.player_data['player_id'] == player_id].sort_values('date')
            
            # Skip if not enough games
            if len(player_games) < min_games:
                continue
            
            player_trends = {}
            
            # Fit ARIMA models for each shooting percentage
            for makes, attempts in self.shooting_cats:
                pct_col = f"{makes}_pct"
                if pct_col in player_games.columns:
                    # Filter out games with no attempts
                    valid_games = player_games[player_games[attempts] > 0].copy()
                    if len(valid_games) < min_games:
                        continue
                    
                    # Fit ARIMA model
                    try:
                        model = ARIMA(valid_games[pct_col].fillna(method='ffill'), order=(3, 0, 1))
                        model_fit = model.fit()
                        player_trends[pct_col] = model_fit
                    except:
                        # If model fails, use simpler model or skip
                        try:
                            model = ARIMA(valid_games[pct_col].fillna(method='ffill'), order=(1, 0, 0))
                            model_fit = model.fit()
                            player_trends[pct_col] = model_fit
                        except:
                            pass
            
            # Fit ARIMA models for box score stats
            for cat in self.box_score_cats:
                if cat in player_games.columns:
                    # Filter out games with no minutes
                    valid_games = player_games[player_games['minutes'] > 0].copy()
                    if len(valid_games) < min_games:
                        continue
                    
                    # Fit ARIMA model
                    try:
                        model = ARIMA(valid_games[cat].fillna(method='ffill'), order=(3, 0, 1))
                        model_fit = model.fit()
                        player_trends[cat] = model_fit
                    except:
                        # If model fails, use simpler model or skip
                        try:
                            model = ARIMA(valid_games[cat].fillna(method='ffill'), order=(1, 0, 0))
                            model_fit = model.fit()
                            player_trends[cat] = model_fit
                        except:
                            pass
            
            # Store player trends if any models were fit
            if player_trends:
                self.player_trends[player_id] = player_trends
        
        return self
    
    def fit_contextual_models(self):
        """
        Fit models that incorporate contextual variables like:
        - Home/away
        - Days rest
        - Shot distance
        - Defender distance
        - Score margin
        """
        # Define contextual variables to consider
        context_vars = [
            'home', 'days_rest', 'shot_distance', 'defender_distance', 'score_margin'
        ]
        
        # Filter to only available variables
        available_context = [var for var in context_vars if var in self.player_data.columns]
        
        if not available_context:
            print("No contextual variables available in the dataset")
            return self
        
        # Fit models for shooting percentages with contextual variables
        for makes, attempts in tqdm(self.shooting_cats, desc="Fitting contextual models"):
            pct_col = f"{makes}_pct"
            
            # Skip if data not available
            if makes not in self.player_data.columns or attempts not in self.player_data.columns:
                continue
            
            # Create datasets grouped by contexts for each player
            player_contexts = {}
            
            for player_id in self.players:
                player_games = self.player_data[self.player_data['player_id'] == player_id]
                
                if len(player_games) < 10:  # Minimum games threshold
                    continue
                
                contexts = {}
                
                # Home vs. Away
                if 'home' in available_context:
                    home_games = player_games[player_games['home'] == True]
                    away_games = player_games[player_games['home'] == False]
                    
                    if len(home_games) >= 5 and len(away_games) >= 5:
                        home_pct = home_games[makes].sum() / home_games[attempts].sum() if home_games[attempts].sum() > 0 else np.nan
                        away_pct = away_games[makes].sum() / away_games[attempts].sum() if away_games[attempts].sum() > 0 else np.nan
                        
                        contexts['home_effect'] = home_pct - away_pct
                
                # Days Rest
                if 'days_rest' in available_context:
                    rest_groups = player_games.groupby(pd.cut(player_games['days_rest'], [0, 1, 2, 10]))
                    rest_effects = {}
                    
                    for rest, group in rest_groups:
                        if len(group) >= 5 and group[attempts].sum() > 0:
                            rest_effects[str(rest)] = group[makes].sum() / group[attempts].sum()
                    
                    if rest_effects:
                        contexts['rest_effects'] = rest_effects
                
                # Shot Distance (binned)
                if 'shot_distance' in available_context:
                    distance_groups = player_games.groupby(pd.cut(player_games['shot_distance'], [0, 5, 15, 25, 40]))
                    distance_effects = {}
                    
                    for distance, group in distance_groups:
                        if len(group) >= 5 and group[attempts].sum() > 0:
                            distance_effects[str(distance)] = group[makes].sum() / group[attempts].sum()
                    
                    if distance_effects:
                        contexts['distance_effects'] = distance_effects
                
                # Score Margin Effect (close games vs. blowouts)
                if 'score_margin' in available_context:
                    close_games = player_games[player_games['score_margin'].abs() <= 5]
                    blowout_games = player_games[player_games['score_margin'].abs() > 15]
                    
                    if len(close_games) >= 5 and len(blowout_games) >= 5:
                        close_pct = close_games[makes].sum() / close_games[attempts].sum() if close_games[attempts].sum() > 0 else np.nan
                        blowout_pct = blowout_games[makes].sum() / blowout_games[attempts].sum() if blowout_games[attempts].sum() > 0 else np.nan
                        
                        contexts['clutch_effect'] = close_pct - blowout_pct
                
                if contexts:
                    player_contexts[player_id] = contexts
            
            # Store the contextual models
            self.player_models[f"{pct_col}_context"] = player_contexts
        
        return self

    def fit_correlation_structure(self):
        """
        Analyze and model the correlation structure between different box score stats
        """
        # Calculate correlation matrices for each player
        for player_id in tqdm(self.players, desc="Fitting correlation structures"):
            player_games = self.player_data[self.player_data['player_id'] == player_id]
            
            if len(player_games) < 20:  # Minimum games threshold
                continue
            
            # Filter to available box score categories
            available_cats = [cat for cat in self.box_score_cats if cat in player_games.columns]
            
            if len(available_cats) < 3:  # Need at least a few categories
                continue
            
            # Calculate correlation matrix
            corr_matrix = player_games[available_cats].corr().fillna(0)
            
            # Store correlation matrix
            self.correlation_matrices[player_id] = corr_matrix
        
        # If a player doesn't have enough data, use league average correlation
        if self.correlation_matrices:
            # Calculate league average correlation matrix
            all_corr_matrices = np.stack([matrix.values for matrix in self.correlation_matrices.values()])
            league_corr = np.nanmean(all_corr_matrices, axis=0)
            
            # Create a DataFrame with proper column names
            sample_cols = list(self.correlation_matrices.values())[0].columns
            self.league_correlation = pd.DataFrame(league_corr, index=sample_cols, columns=sample_cols)
        
        return self
    
    def fit_matchup_adjustments(self):
        """
        Calculate defensive adjustments for teams and individual defenders
        """
        if 'opp_team_id' not in self.player_data.columns:
            print("No opponent team data available for matchup adjustments")
            return self
        
        # Calculate team defense factors
        team_defense = {}
        
        for stat in tqdm(self.box_score_cats + [f"{makes}_pct" for makes, _ in self.shooting_cats], 
                        desc="Calculating team defense adjustments"):
            if stat not in self.player_data.columns:
                continue
            
            # Calculate league average
            league_avg = self.player_data[stat].mean()
            
            # Calculate team defense adjustments
            team_adjustments = {}
            
            for team_id in self.player_data['opp_team_id'].unique():
                team_games = self.player_data[self.player_data['opp_team_id'] == team_id]
                
                if len(team_games) < 10:
                    continue
                
                team_avg = team_games[stat].mean()
                
                # Defense adjustment factor (below 1 means better defense)
                adjustment = team_avg / league_avg if league_avg > 0 else 1.0
                
                team_adjustments[team_id] = adjustment
            
            team_defense[stat] = team_adjustments
        
        self.team_adjustments = team_defense
        
        # Calculate individual defender adjustments if data is available
        if 'defender_id' in self.player_data.columns:
            defender_adjustments = {}
            
            for stat in tqdm(self.box_score_cats + [f"{makes}_pct" for makes, _ in self.shooting_cats],
                            desc="Calculating defender adjustments"):
                if stat not in self.player_data.columns:
                    continue
                
                # Calculate league average
                league_avg = self.player_data[stat].mean()
                
                # Calculate defender adjustments
                individual_adjustments = {}
                
                for defender_id in self.player_data['defender_id'].unique():
                    defended_games = self.player_data[self.player_data['defender_id'] == defender_id]
                    
                    if len(defended_games) < 10:
                        continue
                    
                    defended_avg = defended_games[stat].mean()
                    
                    # Defense adjustment factor
                    adjustment = defended_avg / league_avg if league_avg > 0 else 1.0
                    
                    individual_adjustments[defender_id] = adjustment
                
                defender_adjustments[stat] = individual_adjustments
            
            self.defender_adjustments = defender_adjustments
        
        return self
    
    def predict_player_percentages(self, player_id, context=None):
        """
        Predict shooting percentages for a player based on all models
        
        Parameters:
        -----------
        player_id : str or int
            Player identifier
        context : dict, optional
            Contextual information like home/away, rest days, etc.
        
        Returns:
        --------
        dict
            Predicted shooting percentages
        """
        percentages = {}
        
        # For each shooting category
        for makes, attempts in self.shooting_cats:
            pct_col = f"{makes}_pct"
            
            # Base prediction - hierarchical model if available
            if pct_col in self.player_models and player_id in self.player_models[pct_col]['players']:
                # Get player index in the model
                player_idx = self.player_models[pct_col]['players'][player_id]
                
                # Get posterior mean
                player_summary = self.player_models[pct_col]['rates']
                logit_p = player_summary.loc[player_id, 'mean']
                
                # Convert back from logit
                base_pct = 1 / (1 + np.exp(-logit_p))
            else:
                # Fallback to simple mean if no model
                player_games = self.player_data[self.player_data['player_id'] == player_id]
                if len(player_games) >= 5 and player_games[attempts].sum() > 0:
                    base_pct = player_games[makes].sum() / player_games[attempts].sum()
                else:
                    base_pct = self.global_means.get(pct_col, 0.5)
            
            # Apply time series trend if available
            if player_id in self.player_trends and pct_col in self.player_trends[player_id]:
                model = self.player_trends[player_id][pct_col]
                trend_prediction = model.forecast(1)
                
                # Blend base prediction with trend (weighted)
                base_pct = 0.7 * base_pct + 0.3 * trend_prediction[0]
            
            # Apply contextual adjustments if available
            if context and f"{pct_col}_context" in self.player_models and player_id in self.player_models[f"{pct_col}_context"]:
                player_context = self.player_models[f"{pct_col}_context"][player_id]
                
                # Home/Away adjustment
                if 'home' in context and 'home_effect' in player_context:
                    home_effect = player_context['home_effect']
                    base_pct += (home_effect if context['home'] else -home_effect) * 0.5
                
                # Rest days adjustment
                if 'days_rest' in context and 'rest_effects' in player_context:
                    rest_effects = player_context['rest_effects']
                    rest_days = min(context['days_rest'], 3)  # Cap at 3+ days
                    rest_key = f"({rest_days}, {rest_days+1}]" if rest_days < 3 else "(3, 10]"
                    
                    if rest_key in rest_effects:
                        rest_pct = rest_effects[rest_key]
                        # Blend with base prediction
                        base_pct = 0.8 * base_pct + 0.2 * rest_pct
                
                # Shot distance adjustment (if available)
                if 'shot_distance' in context and 'distance_effects' in player_context:
                    distance_effects = player_context['distance_effects']
                    dist = context['shot_distance']
                    
                    # Find appropriate bin
                    if dist <= 5:
                        dist_key = "(0, 5]"
                    elif dist <= 15:
                        dist_key = "(5, 15]"
                    elif dist <= 25:
                        dist_key = "(15, 25]"
                    else:
                        dist_key = "(25, 40]"
                    
                    if dist_key in distance_effects:
                        dist_pct = distance_effects[dist_key]
                        # Blend with base prediction
                        base_pct = 0.7 * base_pct + 0.3 * dist_pct
                
                # Score margin (clutch) adjustment
                if 'score_margin' in context and 'clutch_effect' in player_context:
                    clutch_effect = player_context['clutch_effect']
                    margin = abs(context['score_margin'])
                    
                    # Apply clutch effect for close games
                    if margin <= 5:
                        base_pct += clutch_effect * 0.5
            
            # Apply defensive matchup adjustments
            if context:
                # Team defense adjustment
                if 'opp_team_id' in context and pct_col in self.team_adjustments and context['opp_team_id'] in self.team_adjustments[pct_col]:
                    team_factor = self.team_adjustments[pct_col][context['opp_team_id']]
                    # Blend adjustment (don't fully apply it)
                    base_pct *= (0.7 + 0.3 * team_factor)
                
                # Individual defender adjustment if available
                if hasattr(self, 'defender_adjustments') and 'defender_id' in context and pct_col in self.defender_adjustments and context['defender_id'] in self.defender_adjustments[pct_col]:
                    defender_factor = self.defender_adjustments[pct_col][context['defender_id']]
                    # Apply partial defender adjustment
                    base_pct *= (0.8 + 0.2 * defender_factor)
            
            # Ensure percentage is valid
            base_pct = max(0.0, min(1.0, base_pct))
            
            # Store final percentage
            percentages[pct_col] = base_pct
        
        return percentages
    
    def predict_player_box_score(self, player_id, minutes, context=None):
        """
        Predict a player's box score stats given minutes and context
        
        Parameters:
        -----------
        player_id : str or int
            Player identifier
        minutes : float
            Predicted minutes for the player
        context : dict, optional
            Contextual information like home/away, opp_team_id, etc.
        
        Returns:
        --------
        dict
            Predicted box score statistics
        """
        # Get base predictions
        predictions = {'minutes': minutes}
        
        # Get player's per-minute rates
        player_games = self.player_data[self.player_data['player_id'] == player_id]
        recent_games = player_games.sort_values('date').tail(20)
        
        # Calculate per-minute rates for each stat
        for cat in self.box_score_cats:
            if cat == 'minutes':
                continue
                
            if cat in player_games.columns:
                # Use recent games if available, otherwise all games
                if len(recent_games) >= 5:
                    per_min_rate = recent_games[cat].sum() / recent_games['minutes'].sum() if recent_games['minutes'].sum() > 0 else 0
                elif len(player_games) >= 5:
                    per_min_rate = player_games[cat].sum() / player_games['minutes'].sum() if player_games['minutes'].sum() > 0 else 0
                else:
                    per_min_rate = self.global_means.get(cat, 0) / self.global_means.get('minutes', 36)
                
                # Apply time-series adjustment if available
                if player_id in self.player_trends and cat in self.player_trends[player_id]:
                    model = self.player_trends[player_id][cat]
                    trend_prediction = model.forecast(1)[0]
                    
                    # Convert trend prediction to per-minute rate
                    if player_id in self.player_trends and 'minutes' in self.player_trends[player_id]:
                        pred_minutes = self.player_trends[player_id]['minutes'].forecast(1)[0]
                        trend_per_min = trend_prediction / pred_minutes if pred_minutes > 0 else per_min_rate
                    else:
                        trend_per_min = trend_prediction / self.global_means.get('minutes', 36)
                    
                    # Blend base rate with trend
                    per_min_rate = 0.7 * per_min_rate + 0.3 * trend_per_min
                
                # Apply matchup adjustments
                if context and 'opp_team_id' in context and cat in self.team_adjustments and context['opp_team_id'] in self.team_adjustments[cat]:
                    team_factor = self.team_adjustments[cat][context['opp_team_id']]
                    # Blend adjustment
                    per_min_rate *= (0.7 + 0.3 * team_factor)
                
                # Calculate raw prediction
                predictions[cat] = per_min_rate * minutes
        
        # Apply correlation structure for more realistic box scores
        if player_id in self.correlation_matrices:
            corr_matrix = self.correlation_matrices[player_id]
            
            # Generate correlated random adjustment factors
            available_cats = [cat for cat in self.box_score_cats if cat in corr_matrix.columns and cat in predictions and cat != 'minutes']
            
            if len(available_cats) >= 2:
                # Extract correlation submatrix for available categories
                sub_corr = corr_matrix.loc[available_cats, available_cats].values
                
                # Generate correlated normal random variables
                mean = np.zeros(len(available_cats))
                # Scale down the correlation to allow for some randomness
                scaled_corr = 0.7 * sub_corr + 0.3 * np.eye(len(available_cats))
                
                # Ensure the correlation matrix is positive semi-definite
                min_eig = np.min(np.linalg.eigvals(scaled_corr))
                if min_eig < 0:
                    scaled_corr -= 1.1 * min_eig * np.eye(len(available_cats))
                
                # Generate correlated random variables
                random_factors = np.random.multivariate_normal(mean, scaled_corr)
                
                # Convert to adjustment multipliers (centered around 1.0)
                adjustments = 1.0 + 0.15 * random_factors  # 15% random variation
                
                # Apply adjustments
                for i, cat in enumerate(available_cats):
                    predictions[cat] *= adjustments[i]
        
        # Ensure non-negative values and apply reasonable bounds
        for cat in predictions:
            if cat != 'minutes':
                predictions[cat] = max(0, predictions[cat])
                
                # Apply category-specific caps
                if cat == 'pts':
                    predictions[cat] = min(predictions[cat], 70)  # Cap points at a reasonable maximum
                elif cat in ['stl', 'blk']:
                    predictions[cat] = min(predictions[cat], 10)  # Cap steals/blocks
        
        return predictions
    
    def simulate_shot_attempts(self, player_id, minutes, context=None):
        """
        Simulate shot attempts for a player
        
        Parameters:
        -----------
        player_id : str or int
            Player identifier
        minutes : float
            Predicted minutes for the player
        context : dict, optional
            Contextual information
            
        Returns:
        --------
        dict
            Predicted attempts for each shooting category
        """
        attempts = {}
        
        player_games = self.player_data[self.player_data['player_id'] == player_id]
        recent_games = player_games.sort_values('date').tail(20)
        
        # For each shooting category
        for makes, attempts_col in self.shooting_cats:
            # Calculate per-minute attempt rate
            if len(recent_games) >= 5:
                per_min_rate = recent_games[attempts_col].sum() / recent_games['minutes'].sum() if recent_games['minutes'].sum() > 0 else 0
            elif len(player_games) >= 5:
                per_min_rate = player_games[attempts_col].sum() / player_games['minutes'].sum() if player_games['minutes'].sum() > 0 else 0
            else:
                per_min_rate = self.global_means.get(attempts_col, 0) / self.global_means.get('minutes', 36)
            
            # Apply time-series adjustment if available
            if player_id in self.player_trends and attempts_col in self.player_trends[player_id]:
                model = self.player_trends[player_id][attempts_col]
                trend_prediction = model.forecast(1)[0]
                
                # Convert trend prediction to per-minute rate
                if player_id in self.player_trends and 'minutes' in self.player_trends[player_id]:
                    pred_minutes = self.player_trends[player_id]['minutes'].forecast(1)[0]
                    trend_per_min = trend_prediction / pred_minutes if pred_minutes > 0 else per_min_rate
                else:
                    trend_per_min = trend_prediction / self.global_means.get('minutes', 36)
                
                # Blend base rate with trend
                per_min_rate = 0.7 * per_min_rate + 0.3 * trend_per_min
            
            # Apply matchup adjustments
            if context and 'opp_team_id' in context and attempts_col in self.team_adjustments and context['opp_team_id'] in self.team_adjustments[attempts_col]:
                team_factor = self.team_adjustments[attempts_col][context['opp_team_id']]
                # Blend adjustment
                per_min_rate *= (0.7 + 0.3 * team_factor)
            
            # Calculate expected attempts
            expected_attempts = per_min_rate * minutes
            
            # Simulate actual attempts using negative binomial
            # This allows for more variance than Poisson
            r = 3.0  # Shape parameter - controls dispersion
            p = r / (r + expected_attempts)  # Probability parameter
            
            attempts[attempts_col] = stats.nbinom.rvs(r, p)
        
        return attempts
    
    def simulate_game(self, player_id, context=None):
        """
        Simulate a full game for a player
        
        Parameters:
        -----------
        player_id : str or int
            Player identifier
        context : dict, optional
            Game context (home/away, opponent, etc.)
            
        Returns:
        --------
        dict
            Simulated box score
        """
        # Predict minutes first
        player_games = self.player_data[self.player_data['player_id'] == player_id]
        recent_games = player_games.sort_values('date').tail(20)
        
        if len(recent_games) >= 5:
            mean_minutes = recent_games['minutes'].mean()
            std_minutes = recent_games['minutes'].std()
        elif len(player_games) >= 5:
            mean_minutes = player_games['minutes'].mean()
            std_minutes = player_games['minutes'].std()
        else:
            mean_minutes = self.global_means.get('minutes', 36)
            std_minutes = 5.0
        
        # Apply time-series adjustment if available
        if player_id in self.player_trends and 'minutes' in self.player_trends[player_id]:
            model = self.player_trends[player_id]['minutes']
            trend_minutes = model.forecast(1)[0]
            
            # Blend historical average with trend
            mean_minutes = 0.6 * mean_minutes + 0.4 * trend_minutes
        
        # Simulate minutes with truncated normal distribution
        minutes = stats.truncnorm.rvs(
            (0 - mean_minutes) / std_minutes,  # Lower bound (0 minutes)
            (48 - mean_minutes) / std_minutes,  # Upper bound (48 minutes)
            loc=mean_minutes,
            scale=std_minutes
        )
        
        # Predict shooting percentages
        pcts = self.predict_player_percentages(player_id, context)
        
        # Simulate shot attempts
        shot_attempts = self.simulate_shot_attempts(player_id, minutes, context)
        
        # Simulate makes based on percentages and negative binomial
        shot_makes = {}
        for makes, attempts_col in self.shooting_cats:
            pct_col = f"{makes}_pct"
            if pct_col in pcts and attempts_col in shot_attempts:
                expected_makes = pcts[pct_col] * shot_attempts[attempts_col]
                
                # Simulate makes with binomial distribution
                shot_makes[makes] = stats.binom.rvs(shot_attempts[attempts_col], pcts[pct_col])
        
        # Predict other box score stats
        box_score = self.predict_player_box_score(player_id, minutes, context)
        
        # Replace predicted values with simulated makes/attempts
        for makes, attempts_col in self.shooting_cats:
            if makes in shot_makes and attempts_col in shot_attempts:
                box_score[makes] = shot_makes[makes]
                box_score[attempts_col] = shot_attempts[attempts_col]
        
        # Calculate points from shots
        if 'fgm' in shot_makes and 'fg3m' in shot_makes and 'ftm' in shot_makes:
            two_pointers = shot_makes['fgm'] - shot_makes['fg3m']
            points = (2 * two_pointers) + (3 * shot_makes['fg3m']) + shot_makes['ftm']
            box_score['pts'] = points
        
        # Round counts to integers
        for key in box_score:
            if key not in ['minutes'] + [f"{makes}_pct" for makes, _ in self.shooting_cats]:
                box_score[key] = round(box_score[key])
        
        # Calculate final percentages
        for makes, attempts_col in self.shooting_cats:
            pct_col = f"{makes}_pct"
            if makes in box_score and attempts_col in box_score and box_score[attempts_col] > 0:
                box_score[pct_col] = box_score[makes] / box_score[attempts_col]
            else:
                box_score[pct_col] = 0.0
        
        return box_score
    
    def simulate_player_season(self, player_id, n_games=82, context_generator=None):
        """
        Simulate a full season for a player
        
        Parameters:
        -----------
        player_id : str or int
            Player identifier
        n_games : int, default=82
            Number of games to simulate
        context_generator : callable, optional
            Function to generate game context for each simulation
            
        Returns:
        --------
        pandas.DataFrame
            Simulated season stats for each game
        """
        season_games = []
        
        for i in range(n_games):
            # Generate context for the game
            if context_generator:
                context = context_generator(i, player_id)
            else:
                # Default context - random home/away
                context = {'home': np.random.choice([True, False])}
                
                # Add random opponent
                if self.teams is not None:
                    context['opp_team_id'] = np.random.choice(self.teams)
            
            # Simulate the game
            game_stats = self.simulate_game(player_id, context)
            
            # Add game number and context
            game_stats['game_num'] = i + 1
            game_stats.update(context)
            
            season_games.append(game_stats)
        
        # Convert to DataFrame
        season_df = pd.DataFrame(season_games)
        
        # Add player identifier
        season_df['player_id'] = player_id
        
        return season_df
    
    def evaluate_model(self, test_data=None, n_simulations=100):
        """
        Evaluate the model's accuracy by comparing simulations to actual outcomes
        
        Parameters:
        -----------
        test_data : pandas.DataFrame, optional
            Holdout data for evaluation. If None, uses the most recent 10% of games.
        n_simulations : int, default=100
            Number of simulations to run for each player-game
            
        Returns:
        --------
        dict
            Evaluation metrics for different stats
        """
        if test_data is None:
            # Use most recent 10% of games as test data
            self.player_data = self.player_data.sort_values('date')
            split_idx = int(len(self.player_data) * 0.9)
            test_data = self.player_data.iloc[split_idx:]
            self.player_data = self.player_data.iloc[:split_idx]
            
            # Refit models with truncated data
            print("Refitting models with training data...")
            self.fit_hierarchical_percentage_models()
            self.fit_time_series_models()
            self.fit_contextual_models()
            self.fit_correlation_structure()
            self.fit_matchup_adjustments()
        
        # Evaluate model
        actual_vs_predicted = []
        
        for _, game in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating model"):
            player_id = game['player_id']
            
            # Create context from game
            context = {}
            for col in game.index:
                if col in ['home', 'opp_team_id', 'days_rest', 'shot_distance', 'defender_distance', 'score_margin', 'defender_id']:
                    context[col] = game[col]
            
            # Run multiple simulations
            simulations = []
            for _ in range(n_simulations):
                sim = self.simulate_game(player_id, context)
                simulations.append(sim)
            
            # Calculate mean prediction
            mean_pred = {}
            for key in simulations[0].keys():
                mean_pred[key] = np.mean([s[key] for s in simulations])
            
            # Store actual vs. predicted
            comparison = {'player_id': player_id}
            for key in mean_pred.keys():
                if key in game.index:
                    comparison[f"{key}_actual"] = game[key]
                    comparison[f"{key}_predicted"] = mean_pred[key]
            
            actual_vs_predicted.append(comparison)
        
        results_df = pd.DataFrame(actual_vs_predicted)
        
        # Calculate metrics
        metrics = {}
        for key in self.box_score_cats + [makes for makes, _ in self.shooting_cats]:
            if f"{key}_actual" in results_df.columns and f"{key}_predicted" in results_df.columns:
                # Mean Absolute Error
                mae = np.mean(np.abs(results_df[f"{key}_actual"] - results_df[f"{key}_predicted"]))
                
                # Mean Absolute Percentage Error
                non_zero = results_df[results_df[f"{key}_actual"] > 0]
                mape = np.mean(np.abs((non_zero[f"{key}_actual"] - non_zero[f"{key}_predicted"]) / non_zero[f"{key}_actual"])) if len(non_zero) > 0 else np.nan
                
                # R-squared
                ss_total = np.sum((results_df[f"{key}_actual"] - results_df[f"{key}_actual"].mean()) ** 2)
                ss_residual = np.sum((results_df[f"{key}_actual"] - results_df[f"{key}_predicted"]) ** 2)
                r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan
                
                metrics[key] = {
                    'MAE': mae,
                    'MAPE': mape,
                    'R^2': r2
                }
        
        return metrics, results_df
    
    def plot_simulation_results(self, sim_results, actual=None, player_id=None, stat='pts'):
        """
        Plot the distribution of simulated results for a player
        
        Parameters:
        -----------
        sim_results : pandas.DataFrame
            Results from simulate_player_season
        actual : pandas.DataFrame, optional
            Actual results for comparison
        player_id : str or int, optional
            Player to plot (only needed if actual data provided)
        stat : str, default='pts'
            Statistic to plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot simulated distribution
        sns.histplot(sim_results[stat], kde=True, label='Simulated')
        
        # Plot actual distribution if provided
        if actual is not None and player_id is not None:
            player_actual = actual[actual['player_id'] == player_id]
            if len(player_actual) > 0 and stat in player_actual.columns:
                sns.histplot(player_actual[stat], color='red', kde=True, alpha=0.6, label='Actual')
        
        plt.title(f'Distribution of {stat} for Player {player_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def explain_player_prediction(self, player_id, context=None):
        """
        Provide an explanation of the prediction for a player
        
        Parameters:
        -----------
        player_id : str or int
            Player identifier
        context : dict, optional
            Game context
            
        Returns:
        --------
        dict
            Explanation of the prediction components
        """
        explanation = {"player_id": player_id, "components": {}}
        
        # Baseline predictions
        player_games = self.player_data[self.player_data['player_id'] == player_id]
        recent_games = player_games.sort_values('date').tail(20)
        
        # Calculate recent averages
        if len(recent_games) >= 5:
            recent_avg = {
                "period": f"Last {len(recent_games)} games",
                "stats": {}
            }
            
            for cat in self.box_score_cats:
                if cat in recent_games.columns:
                    recent_avg["stats"][cat] = recent_games[cat].mean()
            
            for makes, attempts in self.shooting_cats:
                pct_col = f"{makes}_pct"
                if makes in recent_games.columns and attempts in recent_games.columns and recent_games[attempts].sum() > 0:
                    recent_avg["stats"][pct_col] = recent_games[makes].sum() / recent_games[attempts].sum()
            
            explanation["components"]["recent_average"] = recent_avg
        
        # Time series components
        if player_id in self.player_trends:
            trends = {
                "description": "Time series model predictions (trend/momentum)",
                "stats": {}
            }
            
            for stat, model in self.player_trends[player_id].items():
                if hasattr(model, 'forecast'):
                    trends["stats"][stat] = model.forecast(1)[0]
            
            explanation["components"]["trend"] = trends
        
        # Contextual effects
        if context:
            context_effects = {
                "description": "Contextual adjustments",
                "factors": {}
            }
            
            # Home/Away effect
            if 'home' in context:
                for makes, _ in self.shooting_cats:
                    pct_col = f"{makes}_pct"
                    context_key = f"{pct_col}_context"
                    
                    if context_key in self.player_models and player_id in self.player_models[context_key] and 'home_effect' in self.player_models[context_key][player_id]:
                        home_effect = self.player_models[context_key][player_id]['home_effect']
                        context_effects["factors"]["home_away"] = {
                            "type": "Home" if context['home'] else "Away",
                            "effect": f"{home_effect:.3f}" if context['home'] else f"{-home_effect:.3f}"
                        }
            
            # Matchup effects
            if 'opp_team_id' in context:
                opponent_effects = {}
                
                for stat in self.box_score_cats + [f"{makes}_pct" for makes, _ in self.shooting_cats]:
                    if stat in self.team_adjustments and context['opp_team_id'] in self.team_adjustments[stat]:
                        team_factor = self.team_adjustments[stat][context['opp_team_id']]
                        opponent_effects[stat] = f"{(team_factor - 1) * 100:.1f}%"
                
                if opponent_effects:
                    context_effects["factors"]["opponent"] = {
                        "team_id": context['opp_team_id'],
                        "effects": opponent_effects
                    }
            
            explanation["components"]["context"] = context_effects
        
        # Simulate a single game and provide summary
        simulated_game = self.simulate_game(player_id, context)
        explanation["simulated_outcome"] = simulated_game
        
        return explanation


# Example usage:
if __name__ == "__main__":
    # Load your historical data
    # Sample code - replace with your actual data loading
    player_data = pd.read_csv('player_game_logs.csv')
    
    # Initialize simulator
    simulator = BasketballSimulator(player_data)
    
    # Preprocess data
    simulator.preprocess_data()
    
    # Fit all models
    simulator.fit_hierarchical_percentage_models()
    simulator.fit_time_series_models()
    simulator.fit_contextual_models()
    simulator.fit_correlation_structure()
    simulator.fit_matchup_adjustments()
    
    # Simulate a player season
    player_id = player_data['player_id'].iloc[0]  # Example player
    simulated_season = simulator.simulate_player_season(player_id)
    
    # Evaluate model
    metrics, comparison = simulator.evaluate_model()
    
    # Print metrics
    for stat, metric in metrics.items():
        print(f"{stat}: MAE = {metric['MAE']:.3f}, R² = {metric['R^2']:.3f}")
    
    # Plot results
    simulator.plot_simulation_results(simulated_season, player_data, player_id, 'pts')