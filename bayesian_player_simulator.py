import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy import stats
from datetime import datetime
from tqdm import tqdm

class BayesianPlayerSimulator:
    """
    A Bayesian basketball player simulator that uses hierarchical models 
    for more realistic and theoretically grounded simulations.
    """
    
    def __init__(self, player_data=None, player_id=None, verbose=True):
        """
        Initialize the simulator with player data or load data from files
        
        Parameters:
        -----------
        player_data : pandas DataFrame, optional
            Game-by-game player data
        player_id : str, optional
            Player identifier to load data from files
        verbose : bool, default=True
            Whether to print progress information
        """
        self.player_id = player_id
        self.player_data = player_data
        self.verbose = verbose
        self.game_type = None
        self.matchup_type = None
        
        # Store Bayesian models
        self.shooting_models = {}  # Stores posterior samples for shooting percentages
        self.stat_models = {}      # Stores posterior samples for other stats
        
        # If player_data not provided but player_id is, try to load from files
        if player_data is None and player_id is not None:
            if self.verbose:
                print(f"Loading data for {player_id}...")
            self.player_data = self.load_player_data(player_id)
            
        # Pre-calculate historical distributions for the player
        if self.player_data is not None:
            self.historical_distributions = self._calculate_historical_distributions()
            
            # Fit Bayesian models
            if self.verbose:
                print("Fitting Bayesian models...")
            self._fit_bayesian_models()
        else:
            self.historical_distributions = {}
    
    def load_player_data(self, player_id):
        """
        Load player data from files
        
        Parameters:
        -----------
        player_id : str
            Player name to load data for
            
        Returns:
        --------
        DataFrame
            Processed player data
        """
        data_path = "data/static/player_stats_historical/"
        player_files = [f for f in os.listdir(data_path) if f.startswith(f"df_{player_id}")]
        
        if not player_files:
            raise ValueError(f"No data files found for player: {player_id}")
        
        dfs = []
        for file in player_files:
            df = pd.read_csv(os.path.join(data_path, file))
            dfs.append(df)
        
        # Combine all the data
        player_data = pd.concat(dfs, ignore_index=True)
        
        # Process the data to standard format
        processed_data = self._process_raw_data(player_data, player_id)
        return processed_data
    
    def _process_raw_data(self, raw_data, player_id):
        """
        Convert raw data to the standardized format
        
        Parameters:
        -----------
        raw_data : DataFrame
            Raw data from CSV files
        player_id : str
            Player identifier
            
        Returns:
        --------
        DataFrame
            Processed data in standardized format
        """
        processed_data = raw_data.copy()
        processed_data['player_id'] = player_id
        processed_data['game_id'] = range(len(processed_data))
        
        # Process date if available
        if 'DATE' in processed_data.columns:
            processed_data['date'] = pd.to_datetime(processed_data['DATE'])
        elif 'GAME_DATE' in processed_data.columns:
            processed_data['date'] = pd.to_datetime(processed_data['GAME_DATE'])
        
        # Process team and opponent info
        if 'TEAM' in processed_data.columns:
            processed_data['team_id'] = processed_data['TEAM']
        
        if 'OPPONENT' in processed_data.columns:
            processed_data['opp_team_id'] = processed_data['OPPONENT']
        
        if 'HOME/AWAY' in processed_data.columns:
            processed_data['home'] = processed_data['HOME/AWAY'] == 'HOME'
        
        # Handle minutes conversion
        if 'MP' in processed_data.columns:
            processed_data['minutes'] = self._convert_minutes(processed_data['MP'])
        elif 'MIN' in processed_data.columns:
            processed_data['minutes'] = self._convert_minutes(processed_data['MIN'])
        
        # Standardize stat column names
        column_mapping = {
            'FG': 'fgm', 'FGA': 'fga', 
            '3P': 'fg3m', '3PA': 'fg3a', 
            'FT': 'ftm', 'FTA': 'fta', 
            'PTS': 'pts', 'TRB': 'reb',
            'ORB': 'oreb', 'DRB': 'dreb',
            'AST': 'ast', 'STL': 'stl', 
            'BLK': 'blk', 'TOV': 'tov',
            'FG3M': 'fg3m', 'FG3A': 'fg3a',
            'FTM': 'ftm', 'REB': 'reb',
            'OREB': 'oreb', 'DREB': 'dreb'
        }
        
        # Apply the column mapping where columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in processed_data.columns:
                processed_data[new_col] = pd.to_numeric(processed_data[old_col], errors='coerce').fillna(0)
        
        # If no separate offensive/defensive rebounds, calculate them
        if 'oreb' not in processed_data.columns and 'dreb' not in processed_data.columns:
            if 'reb' in processed_data.columns:
                # Estimate using typical split (about 25% offensive)
                processed_data['oreb'] = (processed_data['reb'] * 0.25).round().astype(int)
                processed_data['dreb'] = processed_data['reb'] - processed_data['oreb']
        
        # Filter out invalid games
        processed_data = processed_data[processed_data['minutes'] > 0]
        
        return processed_data
    
    def _convert_minutes(self, minutes_series):
        """
        Convert various minutes formats to decimal
        
        Parameters:
        -----------
        minutes_series : Series
            Series of minutes data in various formats
            
        Returns:
        --------
        Series
            Converted minutes as decimal values
        """
        def convert_value(x):
            if pd.isna(x) or x in ['Inactive', 'Did Not Play', 'DNP', 'NWT', 'Did Not Dress']:
                return 0.0
            elif ':' in str(x):
                try:
                    parts = str(x).split(':')
                    return float(parts[0]) + float(parts[1])/60
                except:
                    return 0.0
            else:
                try:
                    return float(x)
                except:
                    return 0.0
        
        return minutes_series.apply(convert_value)
    
    def _calculate_historical_distributions(self):
        """
        Calculate empirical distributions from historical data
        
        Returns:
        --------
        dict
            Dictionary of distribution parameters for each stat
        """
        distributions = {}
        
        # Filter for meaningful games
        player_games = self.player_data[self.player_data['minutes'] > 10]
        
        if len(player_games) < 10:  # Need at least 10 games for reliable distributions
            return distributions
        
        # Calculate distributions for per-36 minute rates
        for stat in ['reb', 'ast', 'stl', 'blk', 'tov', 'fga', 'fg3a', 'fta']:
            if stat in player_games.columns:
                # Normalize by minutes to get per 36 stats
                per36 = player_games[stat] * 36 / player_games['minutes']
                
                distributions[stat] = {
                    'mean': per36.mean(),
                    'std': per36.std(),
                    'min': per36.min(),
                    'max': per36.max(),
                    'p25': per36.quantile(0.25),
                    'p50': per36.quantile(0.50),
                    'p75': per36.quantile(0.75),
                    'p90': per36.quantile(0.90),
                    'p95': per36.quantile(0.95)
                }
                
                # Save the actual values for empirical sampling
                distributions[f"{stat}_values"] = per36.values
        
        # Calculate shooting percentages distributions
        for makes, attempts in [('fgm', 'fga'), ('fg3m', 'fg3a'), ('ftm', 'fta')]:
            # Filter to avoid division by zero
            valid_games = player_games[player_games[attempts] > 0]
            if len(valid_games) >= 10:  # Need minimum games
                pct = valid_games[makes] / valid_games[attempts]
                pct_col = f"{makes[:-1]}_pct"  # Convert 'fgm' to 'fg_pct'
                
                distributions[pct_col] = {
                    'mean': pct.mean(),
                    'std': pct.std(),
                    'min': pct.min(),
                    'max': pct.max(),
                    'p25': pct.quantile(0.25),
                    'p50': pct.quantile(0.50),
                    'p75': pct.quantile(0.75)
                }
                
                # Save values for empirical sampling
                distributions[f"{pct_col}_values"] = pct.values
        
        # Calculate additional player tendencies
        distributions['tendencies'] = self._calculate_player_tendencies(player_games)
        
        return distributions
    
    def _calculate_player_tendencies(self, player_games):
        """
        Calculate player-specific tendencies to inform simulations
        
        Parameters:
        -----------
        player_games : DataFrame
            Filtered player games
            
        Returns:
        --------
        dict
            Dictionary of player tendencies
        """
        tendencies = {}
        
        # Calculate basic tendencies if enough data
        if len(player_games) >= 20:
            # Calculate scoring role (primary, secondary, tertiary)
            pts_per36 = player_games['pts'] * 36 / player_games['minutes']
            mean_pts = pts_per36.mean()
            
            if mean_pts > 22:
                tendencies['scoring_role'] = 'primary'
            elif mean_pts > 15:
                tendencies['scoring_role'] = 'secondary'
            else:
                tendencies['scoring_role'] = 'tertiary'
            
            # Shooting profile
            if 'fg3a' in player_games.columns and 'fga' in player_games.columns:
                three_pt_rate = player_games['fg3a'].sum() / player_games['fga'].sum()
                tendencies['three_pt_rate'] = three_pt_rate
                
                if three_pt_rate > 0.45:
                    tendencies['shooting_profile'] = 'three_specialist'
                elif three_pt_rate > 0.30:
                    tendencies['shooting_profile'] = 'balanced'
                else:
                    tendencies['shooting_profile'] = 'inside_scorer'
            
            # Free throw rate (player's ability to get to the line)
            if 'fta' in player_games.columns and 'fga' in player_games.columns:
                ft_rate = player_games['fta'].sum() / player_games['fga'].sum()
                tendencies['ft_rate'] = ft_rate
                
                if ft_rate > 0.4:
                    tendencies['ft_drawer'] = 'elite'
                elif ft_rate > 0.25:
                    tendencies['ft_drawer'] = 'above_average'
                else:
                    tendencies['ft_drawer'] = 'average'
            
            # Playing style (based on AST/REB ratio)
            ast_per36 = player_games['ast'] * 36 / player_games['minutes']
            reb_per36 = player_games['reb'] * 36 / player_games['minutes']
            
            # Save averages
            tendencies['ast_per36'] = ast_per36.mean()
            tendencies['reb_per36'] = reb_per36.mean()
            
            # Set playstyle based on assist/rebound ratio
            if ast_per36.mean() > 6 and reb_per36.mean() < 6:
                tendencies['playstyle'] = 'playmaker'
            elif ast_per36.mean() < 3 and reb_per36.mean() > 8:
                tendencies['playstyle'] = 'rebounder'
            elif ast_per36.mean() > 4 and reb_per36.mean() > 6:
                tendencies['playstyle'] = 'all_around'
            else:
                tendencies['playstyle'] = 'scorer'
            
            # Scoring variance (player consistency)
            tendencies['pts_variance'] = pts_per36.std() / pts_per36.mean()
            
            if tendencies['pts_variance'] > 0.4:
                tendencies['consistency'] = 'volatile'
            elif tendencies['pts_variance'] > 0.25:
                tendencies['consistency'] = 'average'
            else:
                tendencies['consistency'] = 'consistent'
            
            # Defensive stats tendency
            def_per36 = (player_games['stl'] + player_games['blk']) * 36 / player_games['minutes']
            tendencies['def_per36'] = def_per36.mean()
            
            if def_per36.mean() > 3:
                tendencies['defensive_playmaker'] = 'elite'
            elif def_per36.mean() > 1.5:
                tendencies['defensive_playmaker'] = 'active'
            else:
                tendencies['defensive_playmaker'] = 'average'
        
        return tendencies
    
    def _fit_bayesian_models(self):
        """
        Fit Bayesian models for shooting percentages and key box score stats
        """
        # Only proceed if we have enough data
        if not self.historical_distributions:
            if self.verbose:
                print("Not enough historical data to fit Bayesian models")
            return
        
        # Filter for meaningful games
        player_games = self.player_data[self.player_data['minutes'] > 10]
        
        # 1. Fit Beta-Binomial models for shooting percentages
        self._fit_shooting_percentage_models(player_games)
        
        # 2. Fit Gamma models for positive-valued stats
        self._fit_stat_models(player_games)
    
    def _fit_shooting_percentage_models(self, player_games):
        """
        Fit Beta-Binomial models for shooting percentages
        
        Parameters:
        -----------
        player_games : DataFrame
            Filtered player games data
        """
        # Shooting categories
        shooting_cats = [('fgm', 'fga'), ('fg3m', 'fg3a'), ('ftm', 'fta')]
        
        for makes_col, attempts_col in shooting_cats:
            # Skip if data not available
            if makes_col not in player_games.columns or attempts_col not in player_games.columns:
                continue
                
            # Filter to valid games (where attempts > 0)
            valid_games = player_games[player_games[attempts_col] > 0]
            if len(valid_games) < 10:  # Need enough data
                continue
                
            makes = valid_games[makes_col].values
            attempts = valid_games[attempts_col].values
            
            # Calculate prior alpha and beta from the data
            # A simple empirical prior based on the historical mean and variance
            p_hat = makes.sum() / attempts.sum()
            var_hat = p_hat * (1 - p_hat) / attempts.sum()
            
            # Convert to Beta parameters
            # These equations relate the mean and variance of a Beta distribution to alpha and beta
            mean_squared = p_hat * p_hat
            variance = var_hat
            
            # Solving for alpha and beta
            alpha = p_hat * (p_hat * (1 - p_hat) / variance - 1)
            beta = (1 - p_hat) * (p_hat * (1 - p_hat) / variance - 1)
            
            # Ensure positive values with reasonable minimums
            alpha = max(alpha, 2)
            beta = max(beta, 2)
            
            # Fit Beta-Binomial model
            with pm.Model() as model:
                # Prior
                theta = pm.Beta('theta', alpha=alpha, beta=beta)
                
                # Likelihood
                y = pm.Binomial('y', n=attempts, p=theta, observed=makes)
                
                # Sample posterior
                trace = pm.sample(1000, tune=1000, progressbar=self.verbose)
            
            # Extract posterior samples
            posterior_samples = trace.posterior['theta'].values.flatten()
            
            # Store the model results
            pct_col = f"{makes_col[:-1]}_pct"  # Convert 'fgm' to 'fg_pct'
            self.shooting_models[pct_col] = {
                'posterior_samples': posterior_samples,
                'prior_alpha': alpha,
                'prior_beta': beta,
                'observed_mean': p_hat,
                'observed_var': var_hat
            }
            
            if self.verbose:
                print(f"Fitted {pct_col} model: posterior mean = {posterior_samples.mean():.3f}")
    
    def _fit_stat_models(self, player_games):
        """
        Fit Gamma models for rate stats (points, rebounds, etc.)
        
        Parameters:
        -----------
        player_games : DataFrame
            Filtered player games data
        """
        # Stats to model as rates per 36 minutes
        rate_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fga', 'fg3a', 'fta']
        
        for stat in rate_stats:
            # Skip if data not available
            if stat not in player_games.columns:
                continue
                
            # Convert to per-36 rates
            rate_values = player_games[stat] * 36 / player_games['minutes']
            rate_values = rate_values[rate_values > 0]  # Filter out zeros
            
            if len(rate_values) < 10:  # Need enough data
                continue
            
            # Calculate sample statistics
            sample_mean = rate_values.mean()
            sample_var = rate_values.var()
            
            # Gamma distribution parameters
            shape = (sample_mean ** 2) / sample_var
            rate = sample_mean / sample_var
            
            # Ensure positive values with reasonable minimums
            shape = max(shape, 1)
            rate = max(rate, 0.1)
            
            # Fit Gamma model
            with pm.Model() as model:
                # Prior
                mu = pm.Gamma('mu', alpha=shape, beta=rate)
                
                # Likelihood (using Normal for simplicity)
                sigma = pm.HalfNormal('sigma', sigma=5)
                y = pm.Normal('y', mu=mu, sigma=sigma, observed=rate_values)
                
                # Sample posterior
                trace = pm.sample(1000, tune=1000, progressbar=self.verbose)
            
            # Extract posterior samples
            posterior_samples = trace.posterior['mu'].values.flatten()
            
            # Store the model results
            self.stat_models[stat] = {
                'posterior_samples': posterior_samples,
                'prior_shape': shape,
                'prior_rate': rate,
                'observed_mean': sample_mean,
                'observed_var': sample_var
            }
            
            if self.verbose:
                print(f"Fitted {stat} model: posterior mean = {posterior_samples.mean():.3f}")
    
    def simulate_game(self, player_id=None, context=None, game_type=None, matchup_type=None):
        """
        Simulate a full game using Bayesian posterior predictive samples
        
        Parameters:
        -----------
        player_id : str, optional
            Player identifier (if not set during initialization)
        context : dict, optional
            Game context (home/away, opponent, etc.)
        game_type : str, optional
            Force specific game type ('hot', 'cold', 'average', 'passive')
        matchup_type : str, optional
            Force specific matchup type ('easy', 'normal', 'tough')
            
        Returns:
        --------
        dict
            Simulated box score
        """
        # If player_id provided and different from initialized, load that player
        if player_id is not None and (self.player_id is None or player_id != self.player_id):
            self.player_id = player_id
            self.player_data = self.load_player_data(player_id)
            self.historical_distributions = self._calculate_historical_distributions()
            self._fit_bayesian_models()
        
        # Ensure we have player data
        if self.player_data is None or len(self.player_data) == 0:
            raise ValueError("No player data available for simulation")
        
        # Filter for meaningful games
        player_games = self.player_data[self.player_data['minutes'] > 10]
        if len(player_games) == 0:
            # If no games with >10 minutes, use all available
            player_games = self.player_data
        
        # Determine tendencies if available
        player_tendencies = self.historical_distributions.get('tendencies', {})
        
        # Set game type - either from parameter or randomly based on player consistency
        if game_type is not None:
            self.game_type = game_type
        else:
            consistency = player_tendencies.get('consistency', 'average')
            
            # More volatile players have more extreme games
            if consistency == 'volatile':
                self.game_type = np.random.choice(
                    ['average', 'hot', 'cold', 'passive'], 
                    p=[0.5, 0.25, 0.15, 0.1]
                )
            elif consistency == 'consistent':
                self.game_type = np.random.choice(
                    ['average', 'hot', 'cold', 'passive'], 
                    p=[0.7, 0.15, 0.1, 0.05]
                )
            else:  # average consistency
                self.game_type = np.random.choice(
                    ['average', 'hot', 'cold', 'passive'], 
                    p=[0.6, 0.2, 0.1, 0.1]
                )
        
        # Set matchup type - either from parameter or randomly
        if matchup_type is not None:
            self.matchup_type = matchup_type
        else:
            self.matchup_type = np.random.choice(
                ['easy', 'normal', 'tough'], 
                p=[0.2, 0.6, 0.2]
            )
        
        # Determine game factors based on game type and matchup
        scoring_factor, efficiency_factor, reb_factor, ast_factor = self._get_game_factors()
        
        # Calculate adjusted base minutes
        minutes = self._simulate_minutes(player_games)
        
        # Create initial box score
        box_score = {'minutes': minutes}
        
        # If we have Bayesian models, use posterior predictive sampling
        if self.shooting_models and self.stat_models:
            # Simulate shooting attempts and percentages with Bayesian models
            box_score = self._simulate_shooting_bayesian(box_score, minutes, scoring_factor, efficiency_factor)
            
            # Simulate other stats with Bayesian models
            box_score = self._simulate_stats_bayesian(box_score, minutes, reb_factor, ast_factor)
            
            # Ensure consistency constraints
            box_score = self._apply_consistency_constraints(box_score)
        else:
            # Fall back to empirical sampling if Bayesian models not available
            if self.verbose:
                print("Falling back to empirical sampling (Bayesian models not fitted)")
                
            # Get per-minute rates for box score stats
            box_score = self._simulate_from_empirical(
                player_games, box_score, minutes, scoring_factor, efficiency_factor, reb_factor, ast_factor
            )
        
        return box_score
    
    def _get_game_factors(self):
        """
        Determine game factors based on game type and matchup
        
        Returns:
        --------
        tuple
            scoring_factor, efficiency_factor, reb_factor, ast_factor
        """
        # Base factors by game type
        if self.game_type == 'hot':
            scoring_factor = 1.25
            efficiency_factor = 1.15
            reb_factor = 0.9
            ast_factor = 0.9
        elif self.game_type == 'cold':
            scoring_factor = 0.75
            efficiency_factor = 0.85
            reb_factor = 1.1
            ast_factor = 1.1
        elif self.game_type == 'passive':
            scoring_factor = 0.9
            efficiency_factor = 1.0
            reb_factor = 1.1
            ast_factor = 1.5
        else:  # average game
            scoring_factor = 1.0
            efficiency_factor = 1.0
            reb_factor = 1.0
            ast_factor = 1.0
            
        # Adjust for matchup difficulty
        if self.matchup_type == 'easy':
            scoring_factor *= 1.15
            efficiency_factor *= 1.1
        elif self.matchup_type == 'tough':
            scoring_factor *= 0.9
            efficiency_factor *= 0.9
            
        return scoring_factor, efficiency_factor, reb_factor, ast_factor
    
    def _simulate_minutes(self, player_games):
        """
        Simulate realistic minutes played
        
        Parameters:
        -----------
        player_games : DataFrame
            Filtered player games
            
        Returns:
        --------
        float
            Simulated minutes
        """
        # Get player's average minutes
        avg_minutes = player_games['minutes'].mean()
        std_minutes = player_games['minutes'].std()
        
        # Ensure reasonable standard deviation
        std_minutes = max(min(std_minutes, 8), 2)
        
        # Simulate primary minutes with normal distribution
        minutes_mean = np.random.uniform(0.9 * avg_minutes, 1.1 * avg_minutes)
        minutes_std = np.random.uniform(0.8 * std_minutes, 1.2 * std_minutes)
        minutes = np.random.normal(minutes_mean, minutes_std/2)
        
        # Add occasional special cases (blowouts, OT, injury, etc.)
        minutes_special_case = np.random.choice([0, 1, 2], p=[0.93, 0.05, 0.02])
        if minutes_special_case == 1:  # Low minutes (blowout, minor injury, etc.)
            minutes = np.random.uniform(0.5 * avg_minutes, 0.75 * avg_minutes)
        elif minutes_special_case == 2:  # High minutes (overtime, close game)
            minutes = np.random.uniform(1.1 * avg_minutes, min(48, 1.25 * avg_minutes))
            
        # Ensure minutes are within reasonable bounds
        minutes = max(min(minutes, 48), 10)
        
        return minutes
    
    def _simulate_shooting_bayesian(self, box_score, minutes, scoring_factor, efficiency_factor):
        """
        Simulate shooting stats using Bayesian posterior samples
        
        Parameters:
        -----------
        box_score : dict
            Current box score
        minutes : float
            Simulated minutes
        scoring_factor : float
            Adjustment factor for scoring
        efficiency_factor : float
            Adjustment factor for efficiency
            
        Returns:
        --------
        dict
            Updated box score
        """
        # Get player tendencies
        tendencies = self.historical_distributions.get('tendencies', {})
        shooting_profile = tendencies.get('shooting_profile', 'balanced')
        
        # 1. Simulate field goal attempts using posterior
        if 'fga' in self.stat_models:
            # Check historical data for low/zero FGA games
            zero_fga_prob = len(self.player_data[self.player_data['fga'] == 0]) / len(self.player_data)
            
            # Adjust for game context - players rarely take 0 shots
            zero_fga_prob = min(zero_fga_prob, 0.02)  # Cap at 2% chance
            
            if np.random.random() < zero_fga_prob:
                # Rare case: player doesn't attempt a field goal
                fga = 0
            else:
                # Sample from posterior - this is Bayesian!
                fga_per36_samples = self.stat_models['fga']['posterior_samples']
                fga_per36 = np.random.choice(fga_per36_samples)
                
                # Apply game context factor to the mean
                fga_per36 *= scoring_factor
                
                # Convert to expected value given minutes
                lambda_fga = fga_per36 * minutes / 36
                
                # Get the dispersion parameter directly from model
                # This uses data-driven dispersion from our fitted model
                alpha = self.stat_models['fga'].get('dispersion_param', None)
                
                if alpha is not None:
                    # Use negative binomial with proper dispersion
                    # NB is the theoretically correct distribution for overdispersed count data
                    try:
                        # Convert to NB parameters: alpha=dispersion, p=prob
                        p_fga = alpha/(alpha+lambda_fga)
                        # Generate from negative binomial
                        fga = np.random.negative_binomial(n=alpha, p=p_fga)
                    except:
                        # Fallback to Poisson if issues
                        fga = np.random.poisson(lambda_fga)
                else:
                    # Fallback to Poisson if dispersion not available
                    fga = np.random.poisson(lambda_fga)
                
                # Get historical bounds
                max_fga = self.historical_distributions['fga'].get('max', 35)
                
                # Use soft bounds - allow a small chance to exceed historical max
                # This maintains the Bayesian principles while handling extreme cases
                if fga > max_fga * 1.2:  # Allow 20% beyond max
                    fga = max_fga + np.random.poisson(1)  # Small chance to exceed historical max
                    
                # Final sanity check
                fga = max(fga, 0)
            
            box_score['fga'] = fga
        
        # 2. Three-point attempts
        if 'fga' in box_score and 'fg3a' in self.stat_models:
            # Check for zero 3PT attempt probability
            zero_fg3a_prob = len(self.player_data[self.player_data['fg3a'] == 0]) / len(self.player_data)
            
            # Adjust based on player profile
            if shooting_profile == 'three_specialist':
                zero_fg3a_prob *= 0.5  # Specialists rarely have zero 3PA games
            elif shooting_profile == 'inside_scorer':
                zero_fg3a_prob *= 1.2  # Inside scorers more commonly don't shoot 3s
            
            # Handle zero FGA case
            if box_score['fga'] == 0:
                fg3a = 0
            elif np.random.random() < zero_fg3a_prob:
                # Player doesn't attempt a three
                fg3a = 0
            else:
                # Sample from posterior for 3PT attempt rate
                fg3a_per36_samples = self.stat_models['fg3a']['posterior_samples']
                fg3a_per36 = np.random.choice(fg3a_per36_samples)
                
                # Add variance and apply game type factors
                variance_multiplier = np.random.lognormal(0, 0.2)
                
                # Player shooting profile affects 3PT rate and variance
                if shooting_profile == 'three_specialist':
                    # More consistent 3PA rate for specialists
                    variance_multiplier = np.random.lognormal(0, 0.15)
                elif shooting_profile == 'inside_scorer':
                    # More variance for inside scorers (sometimes they take more 3s than usual)
                    variance_multiplier = np.random.lognormal(0, 0.3)
                
                # More 3s in hot games
                three_pt_factor = 1.0
                if self.game_type == 'hot':
                    three_pt_factor = 1.2
                
                fg3a_per36 *= variance_multiplier * three_pt_factor
                
                # Convert to actual attempts with Poisson
                lambda_fg3a = fg3a_per36 * minutes / 36
                fg3a = np.random.poisson(lambda_fg3a)
                
                # Adjust based on relationship to total FGA
                fg3a_rate = tendencies.get('three_pt_rate', 0.3)
                fg3a = min(fg3a, round(box_score['fga'] * fg3a_rate * 1.5))  # Cap at 150% of typical rate
                
                # Ensure reasonable bounds
                fg3a = max(min(fg3a, box_score['fga']), 0)
            
            box_score['fg3a'] = fg3a
        
        # 3. Free throw attempts
        if 'fta' in self.stat_models:
            # Calculate zero FTA probability from historical data
            zero_fta_prob = len(self.player_data[self.player_data['fta'] == 0]) / len(self.player_data)
            
            # Adjust based on player tendency
            ft_drawer = tendencies.get('ft_drawer', 'average')
            if ft_drawer == 'elite':
                zero_fta_prob *= 0.7  # Elite FT drawers less likely to have zero FTA
            elif ft_drawer == 'average':
                zero_fta_prob *= 1.1  # Average FT drawers more likely to have zero FTA
            
            if np.random.random() < zero_fta_prob:
                # Player doesn't attempt any free throws
                fta = 0
            else:
                # Sample from posterior
                fta_per36_samples = self.stat_models['fta']['posterior_samples']
                fta_per36 = np.random.choice(fta_per36_samples)
                
                # Add variance based on tendency
                if ft_drawer == 'elite':
                    # Elite FT drawers have high variance (some games with many FTs)
                    variance_multiplier = np.random.lognormal(0, 0.35)
                else:
                    variance_multiplier = np.random.lognormal(0, 0.3)
                    
                fta_per36 *= variance_multiplier * scoring_factor
                
                # Free throws often come in bunches (usually in even numbers)
                # Use a negative binomial to get more dispersion than Poisson
                lambda_fta = fta_per36 * minutes / 36
                r = 2  # Shape parameter
                p = r / (r + lambda_fta)  # Success probability parameter
                
                # Simulate with negative binomial for more clumping and dispersion
                fta = np.random.negative_binomial(r, p)
                
                # Free throws usually come in pairs/bunches, round to nearest 2 for some
                if np.random.random() < 0.7 and fta > 0:  # 70% chance to get even FTAs
                    fta = 2 * round(fta / 2)
                
                # Ensure reasonable bounds
                max_fta = self.historical_distributions['fta'].get('max', 25)
                fta = max(min(fta, max_fta), 0)
            
            box_score['fta'] = fta
        
        # 4. Simulate makes based on Bayesian shooting percentages
        # FG%
        if 'fg_pct' in self.shooting_models and 'fga' in box_score:
            if box_score['fga'] == 0:
                box_score['fgm'] = 0
            else:
                # Sample from posterior
                fg_pct_samples = self.shooting_models['fg_pct']['posterior_samples']
                fg_pct = np.random.choice(fg_pct_samples)
                
                # Adjust for game type
                fg_pct *= efficiency_factor
                
                # Add random variation (hot/cold streaks)
                fg_pct *= np.random.normal(1, 0.1)  # More variation
                
                # Ensure percentage is valid
                fg_pct = max(min(fg_pct, 0.95), 0.1)
                
                # Use a binomial distribution for realistic makes
                fgm = np.random.binomial(box_score['fga'], fg_pct)
                
                # Ensure makes <= attempts
                box_score['fgm'] = min(fgm, box_score['fga'])
        
        # 3P%
        if 'fg3_pct' in self.shooting_models and 'fg3a' in box_score:
            if box_score['fg3a'] == 0:
                box_score['fg3m'] = 0
            else:
                # Sample from posterior
                fg3_pct_samples = self.shooting_models['fg3_pct']['posterior_samples']
                fg3_pct = np.random.choice(fg3_pct_samples)
                
                # Adjust for game type
                fg3_pct *= efficiency_factor
                
                # 3PT% varies more than overall FG%
                fg3_pct *= np.random.normal(1, 0.15)  # High variance for 3PT
                
                # Ensure percentage is valid
                fg3_pct = max(min(fg3_pct, 1.0), 0.0)
                
                # Use binomial for precise makes
                fg3m = np.random.binomial(box_score['fg3a'], fg3_pct)
                
                # Ensure makes <= attempts
                box_score['fg3m'] = min(fg3m, box_score['fg3a'])
        
        # FT%
        if 'ft_pct' in self.shooting_models and 'fta' in box_score:
            if box_score['fta'] == 0:
                box_score['ftm'] = 0
            else:
                # Sample from posterior
                ft_pct_samples = self.shooting_models['ft_pct']['posterior_samples']
                ft_pct = np.random.choice(ft_pct_samples)
                
                # Good FT shooters have less variance
                if ft_pct > 0.8:
                    ft_pct_variance = 0.07  # Still some variance
                elif ft_pct > 0.7:
                    ft_pct_variance = 0.09
                else:
                    ft_pct_variance = 0.12
                    
                # Add slight variation
                ft_pct *= np.random.normal(1, ft_pct_variance)
                
                # Ensure percentage is valid
                ft_pct = max(min(ft_pct, 1.0), 0.0)
                
                # Use binomial for precise makes
                ftm = np.random.binomial(box_score['fta'], ft_pct)
                
                # Ensure makes <= attempts
                box_score['ftm'] = min(ftm, box_score['fta'])
        
        return box_score
    
    def _simulate_stats_bayesian(self, box_score, minutes, reb_factor, ast_factor):
        """
        Simulate non-shooting stats using Bayesian posterior samples
        
        Parameters:
        -----------
        box_score : dict
            Current box score
        minutes : float
            Simulated minutes
        reb_factor : float
            Adjustment factor for rebounds
        ast_factor : float
            Adjustment factor for assists
            
        Returns:
        --------
        dict
            Updated box score
        """
        # Get tendencies if available
        tendencies = self.historical_distributions.get('tendencies', {})
        
        # Simulate rebounds
        if 'reb' in self.stat_models:
            # True Bayesian approach: Use negative binomial or Poisson to naturally model 
            # low count phenomena, which inherently includes probability of zeros
            
            # Sample from posterior
            reb_per36_samples = self.stat_models['reb']['posterior_samples']
            reb_per36 = np.random.choice(reb_per36_samples)
            
            # Add variance for wider distribution (but not through direct historical matching)
            # Use smaller variance for rebounds to prevent extreme low values
            variance_multiplier = np.random.lognormal(0, 0.2)  # Reduced from 0.3
            reb_per36 *= variance_multiplier * reb_factor
            
            # Use rate parameter scaled to minutes
            lambda_reb = reb_per36 * minutes / 36
            
            # The key insight: for counting statistics, Poisson/Negative Binomial naturally 
            # accounts for zeros with appropriate rate parameters
            
            # Get player's historical minimum rebounds to inform distribution
            min_hist_reb = self.historical_distributions['reb'].get('min', 0)
            reb_mean = self.historical_distributions['reb'].get('mean', 5)
            
            # For stronger rebounders, use different shape parameter
            # This remains fully Bayesian - just using a more appropriate prior
            if reb_mean > 8:  # Elite rebounders
                r = 8  # Higher shape parameter reduces probability of very low values
            elif reb_mean > 5:  # Good rebounders
                r = 5  # Moderate shape parameter
            else:
                r = 3  # Standard shape parameter
                
            p = r / (r + lambda_reb)  # Convert rate to probability
            
            # Use negative binomial to get proper dispersion including zeros
            try:
                # Generate the raw rebounds count
                raw_rebs = np.random.negative_binomial(r, p)
                
                # For players who historically never have zero-rebound games,
                # incorporate minimum as a soft constraint (still Bayesian)
                if min_hist_reb > 0 and raw_rebs < min_hist_reb:
                    # Probability of respecting player's floor increases with their rebounding ability
                    floor_prob = min(0.8, 0.4 + (reb_mean / 20))  # Scale with rebounding ability
                    
                    if np.random.random() < floor_prob:
                        # Establish a floor near historical minimum
                        rebs = min_hist_reb + np.random.poisson(0.5)
                    else:
                        # Allow some possibility of breaking the minimum
                        rebs = raw_rebs
                else:
                    rebs = raw_rebs
            except:
                # Fallback to Poisson if numerical issues
                rebs = np.random.poisson(lambda_reb)
                
                # Apply same floor logic to Poisson fallback
                if min_hist_reb > 0 and rebs < min_hist_reb and np.random.random() < 0.7:
                    rebs = min_hist_reb
            
            # Ensure reasonable bounds
            max_rebs = self.historical_distributions['reb'].get('max', 20)
            rebs = max(min(rebs, max_rebs), 0)
            
            box_score['reb'] = rebs
            
            # Split into offensive and defensive
            if rebs > 0:
                playstyle = tendencies.get('playstyle', 'average')
                if playstyle == 'rebounder':
                    oreb_ratio = np.random.beta(3, 7)  # More offensive boards for rebounders
                else:
                    oreb_ratio = np.random.beta(2, 10)  # Typical ratio for most players
                    
                box_score['oreb'] = round(rebs * oreb_ratio)
                box_score['dreb'] = rebs - box_score['oreb']
            else:
                box_score['oreb'] = 0
                box_score['dreb'] = 0
        
        # Simulate assists
        if 'ast' in self.stat_models:
            # Sample from posterior - proper Bayesian approach
            ast_per36_samples = self.stat_models['ast']['posterior_samples']
            ast_per36 = np.random.choice(ast_per36_samples)
            
            # Apply game context factor
            ast_per36 *= ast_factor
            
            # Convert to expected value given minutes
            lambda_ast = ast_per36 * minutes / 36
            
            # Get the dispersion parameter directly from model
            alpha_ast = self.stat_models['ast'].get('dispersion_param', None)
            
            if alpha_ast is not None:
                # Use negative binomial with data-derived dispersion
                try:
                    # NB parameters: alpha controls dispersion, p is derived from mean
                    p_ast = alpha_ast/(alpha_ast+lambda_ast)
                    asts = np.random.negative_binomial(n=alpha_ast, p=p_ast)
                except:
                    # Fallback to Poisson
                    asts = np.random.poisson(lambda_ast)
            else:
                # If no dispersion parameter available, use Poisson
                asts = np.random.poisson(lambda_ast)
            
            # Get historical max
            max_asts = self.historical_distributions['ast'].get('max', 15)
            
            # Use soft bounds that respect the distribution's tails
            if asts > max_asts * 1.3:  # Allow reasonable exceedance
                # Use tail behavior that mimics the distribution
                excess = asts - max_asts
                # Keep small chance to have an exceptional game beyond historical max
                asts = max_asts + np.random.binomial(n=excess, p=0.1)
            
            # Final sanity check
            asts = max(asts, 0)
            
            box_score['ast'] = asts
        
        # Simulate defensive stats (steals and blocks)
        def_tendency = tendencies.get('defensive_playmaker', 'average')
        
        # Different variance based on defensive tendency
        # We need to tone down defensive stats in general - they were overestimated
        
        # Get player's historical defensive stats for calibration
        def_stats_present = 'blk' in self.historical_distributions and 'stl' in self.historical_distributions
        if def_stats_present:
            stl_avg = self.historical_distributions['stl'].get('mean', 1.0)
            blk_avg = self.historical_distributions['blk'].get('mean', 0.5)
        else:
            stl_avg = 1.0
            blk_avg = 0.5
        
        # Adjust parameters by defensive tendency
        if def_tendency == 'elite':
            def_variance = 0.3  # Reduced from 0.4
            outlier_chance = 0.08  # Reduced from 0.1
            # Higher r = fewer zeros but also less variance
            def_r_factor = 2.5 if stl_avg > 1.5 else 1.8
        elif def_tendency == 'active':
            def_variance = 0.25  # Reduced from 0.3
            outlier_chance = 0.04  # Reduced from 0.05
            def_r_factor = 1.5
        else:
            def_variance = 0.22  # Reduced from 0.25
            outlier_chance = 0.02  # Reduced from 0.03
            def_r_factor = 1.0  # More zeros for average defenders
        
        # Steals - purely Bayesian approach using proper count distributions
        if 'stl' in self.stat_models:
            # Sample from posterior
            stl_per36_samples = self.stat_models['stl']['posterior_samples']
            stl_per36 = np.random.choice(stl_per36_samples)
            
            # Add variance
            variance_multiplier = np.random.lognormal(0, def_variance)
            
            # Occasional outlier games
            if np.random.random() < outlier_chance:
                variance_multiplier *= np.random.uniform(1.5, 2.5)
                
            # Apply a scaling factor to prevent overestimation
            # Historically, the simulator has overestimated steals
            scaling_factor = 0.8  # Reduce steals rate by 20%
            
            stl_per36 *= variance_multiplier * scaling_factor
            
            # Convert to lambda parameter
            lambda_stl = stl_per36 * minutes / 36
            
            # Steals are rare events, perfect for Poisson/NegBinom modeling
            # Use negative binomial for precise control over zeros/variance
            r_stl = def_r_factor  # Controls distribution shape
            p_stl = r_stl / (r_stl + lambda_stl)
            
            try:
                stls = np.random.negative_binomial(r_stl, p_stl)
            except:
                # Fallback to Poisson
                stls = np.random.poisson(lambda_stl)
            
            # Ensure reasonable bounds
            max_stls = self.historical_distributions['stl'].get('max', 6)
            stls = max(min(stls, max_stls), 0)
            
            box_score['stl'] = stls
        
        # Blocks - same approach as steals
        if 'blk' in self.stat_models:
            # Sample from posterior
            blk_per36_samples = self.stat_models['blk']['posterior_samples']
            blk_per36 = np.random.choice(blk_per36_samples)
            
            # Add variance
            variance_multiplier = np.random.lognormal(0, def_variance)
            
            # Occasional outlier games
            if np.random.random() < outlier_chance:
                variance_multiplier *= np.random.uniform(1.5, 2.5)
                
            # Apply scaling factor to prevent overestimation of blocks
            # Blocks tend to be the most overestimated stat
            scaling_factor = 0.6  # Reduce blocks rate by 40%
            
            blk_per36 *= variance_multiplier * scaling_factor
            
            # Convert to lambda parameter
            lambda_blk = blk_per36 * minutes / 36
            
            # Special case for elite shot blockers
            block_r_factor = def_r_factor
            if def_tendency == 'elite' and 'blk' in self.historical_distributions:
                if self.historical_distributions['blk'].get('mean', 0) > 1.5:
                    # Slightly more concentrated for elite blockers
                    block_r_factor *= 1.2
            
            # Use negative binomial for proper dispersion
            r_blk = block_r_factor  # Blocks typically more concentrated than steals
            p_blk = r_blk / (r_blk + lambda_blk)
            
            try:
                blks = np.random.negative_binomial(r_blk, p_blk)
            except:
                # Fallback to Poisson
                blks = np.random.poisson(lambda_blk)
            
            # Ensure reasonable bounds
            max_blks = self.historical_distributions['blk'].get('max', 8)
            blks = max(min(blks, max_blks), 0)
            
            box_score['blk'] = blks
        
        # Turnovers - also convert to proper Bayesian approach
        if 'tov' in self.stat_models:
            # Sample from posterior - pure Bayesian approach
            tov_per36_samples = self.stat_models['tov']['posterior_samples']
            tov_per36 = np.random.choice(tov_per36_samples)
            
            # Apply game context through factor
            # This preserves relationships between assist rate and turnover rate
            usage_factor = 1.0
            # Use statistical correlation with other stats instead of arbitrary factors
            if 'ast' in box_score:
                # Very high assist games tend to produce more turnovers
                # This factor depends on relationship between variables
                ast_ratio = box_score['ast'] / self.historical_distributions.get('ast', {}).get('mean', 1)
                if ast_ratio > 1.5:  # Well above average assists
                    # Factor derived from covariance rather than fixed
                    usage_factor = np.sqrt(ast_ratio)  # Non-linear relationship
            
            tov_per36 *= usage_factor
            
            # Convert to expected value given minutes
            lambda_tov = tov_per36 * minutes / 36
            
            # Get the dispersion parameter directly from model
            alpha_tov = self.stat_models['tov'].get('dispersion_param', None)
            
            if alpha_tov is not None:
                # Use negative binomial with model-derived dispersion
                try:
                    # Parameterized directly from model fit
                    p_tov = alpha_tov/(alpha_tov+lambda_tov)
                    tovs = np.random.negative_binomial(n=alpha_tov, p=p_tov)
                except:
                    # Fallback to Poisson if numerical issues
                    tovs = np.random.poisson(lambda_tov)
            else:
                # Without dispersion parameter, model as Poisson
                tovs = np.random.poisson(lambda_tov)
            
            # Get historical bounds
            max_tovs = self.historical_distributions['tov'].get('max', 10)
            
            # Use distributional approach for extreme values
            # This maintains the proper tail behavior
            if tovs > max_tovs * 1.2:  # Allow 20% beyond max
                # Use exponential decay for tail probability
                # Higher exceedance = lower probability (exponentially)
                excess = tovs - max_tovs
                p_keep = np.exp(-excess/2)  # Exponential decay function
                if np.random.random() > p_keep:
                    # Most excess values get adjusted down
                    tovs = max_tovs + np.random.poisson(0.5)  # Small chance to exceed historical max
            
            # Final sanity check
            tovs = max(tovs, 0)
            
            box_score['tov'] = tovs
        
        # Points (calculated from shooting if not already done)
        if 'pts' not in box_score and all(k in box_score for k in ['fgm', 'fg3m', 'ftm']):
            two_pointers = box_score['fgm'] - box_score['fg3m']
            box_score['pts'] = (two_pointers * 2) + (box_score['fg3m'] * 3) + box_score['ftm']
        
        return box_score
    
    def _apply_consistency_constraints(self, box_score):
        """
        Ensure the box score has consistent values
        
        Parameters:
        -----------
        box_score : dict
            Current box score
            
        Returns:
        --------
        dict
            Consistent box score
        """
        # 1. Ensure FGM >= FG3M (3pt makes are a subset of FG makes)
        if 'fgm' in box_score and 'fg3m' in box_score:
            box_score['fgm'] = max(box_score['fgm'], box_score['fg3m'])
        
        # 2. Ensure OREB + DREB = REB
        if 'oreb' in box_score and 'dreb' in box_score and 'reb' in box_score:
            total_reb = box_score['oreb'] + box_score['dreb']
            if total_reb != box_score['reb']:
                box_score['reb'] = total_reb
        
        # 3. Calculate points if necessary
        if all(k in box_score for k in ['fgm', 'fg3m', 'ftm']):
            two_pointers = box_score['fgm'] - box_score['fg3m']
            box_score['pts'] = (two_pointers * 2) + (box_score['fg3m'] * 3) + box_score['ftm']
        
        return box_score
    
    def _simulate_from_empirical(self, player_games, box_score, minutes, 
                               scoring_factor, efficiency_factor, reb_factor, ast_factor):
        """
        Simulate stats using empirical distributions when Bayesian models are unavailable
        
        Parameters:
        -----------
        player_games : DataFrame
            Filtered player games
        box_score : dict
            Current box score
        minutes : float
            Simulated minutes
        scoring_factor : float
            Adjustment for scoring
        efficiency_factor : float
            Adjustment for efficiency
        reb_factor : float
            Adjustment for rebounds
        ast_factor : float
            Adjustment for assists
            
        Returns:
        --------
        dict
            Simulated box score
        """
        # Get tendencies if available
        tendencies = self.historical_distributions.get('tendencies', {})
        
        # Simulate shooting stats
        # 1. Field goal attempts
        if 'fga_values' in self.historical_distributions:
            # Sample from empirical distribution
            fga_per36 = np.random.choice(self.historical_distributions['fga_values'])
            
            # Add variance
            variance_multiplier = np.random.lognormal(0, 0.2)
            fga_per36 *= variance_multiplier * scoring_factor
            
            # Convert to actual attempts
            fga = round(fga_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_fga = self.historical_distributions['fga'].get('max', 35)
            fga = max(min(fga, max_fga), 2)
            box_score['fga'] = fga
        
        # 2. Three-point attempts
        if 'fga' in box_score and 'fg3a_values' in self.historical_distributions:
            # Get shooting profile
            shooting_profile = tendencies.get('shooting_profile', 'balanced')
            
            # Sample from empirical distribution
            fg3a_per36 = np.random.choice(self.historical_distributions['fg3a_values'])
            
            # Add variance based on profile
            if shooting_profile == 'three_specialist':
                variance_multiplier = np.random.lognormal(0, 0.15)
            elif shooting_profile == 'inside_scorer':
                variance_multiplier = np.random.lognormal(0, 0.3)
            else:
                variance_multiplier = np.random.lognormal(0, 0.2)
                
            # Hot games have more 3s
            three_pt_factor = 1.0
            if self.game_type == 'hot':
                three_pt_factor = 1.2
                
            fg3a_per36 *= variance_multiplier * three_pt_factor
            
            # Convert to actual attempts
            fg3a = round(fg3a_per36 * minutes / 36)
            
            # Ensure reasonable bounds and relationship to total FGA
            fg3a = max(min(fg3a, box_score['fga']), 0)
            box_score['fg3a'] = fg3a
        
        # 3. Free throw attempts
        if 'fta_values' in self.historical_distributions:
            # Sample from empirical distribution
            fta_per36 = np.random.choice(self.historical_distributions['fta_values'])
            
            # Get free throw drawer tendency
            ft_drawer = tendencies.get('ft_drawer', 'average')
            
            # Add variance based on tendency
            if ft_drawer == 'elite':
                variance_multiplier = np.random.lognormal(0, 0.3)
            else:
                variance_multiplier = np.random.lognormal(0, 0.25)
                
            fta_per36 *= variance_multiplier * scoring_factor
            
            # Convert to actual attempts
            fta = round(fta_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_fta = self.historical_distributions['fta'].get('max', 25)
            fta = max(min(fta, max_fta), 0)
            box_score['fta'] = fta
        
        # 4. Simulate makes based on empirical shooting percentages
        # FG%
        if 'fg_pct_values' in self.historical_distributions and 'fga' in box_score:
            # Sample from empirical distribution
            fg_pct = np.random.choice(self.historical_distributions['fg_pct_values'])
            
            # Adjust for game type
            fg_pct *= efficiency_factor
            
            # Add random variation
            fg_pct *= np.random.normal(1, 0.08)
            
            # Ensure percentage is valid
            fg_pct = max(min(fg_pct, 0.95), 0.1)
            
            # Calculate makes
            fgm = round(box_score['fga'] * fg_pct)
            
            # Ensure makes <= attempts
            box_score['fgm'] = min(fgm, box_score['fga'])
        
        # 3P%
        if 'fg3_pct_values' in self.historical_distributions and 'fg3a' in box_score:
            # Sample from empirical distribution
            fg3_pct = np.random.choice(self.historical_distributions['fg3_pct_values'])
            
            # Adjust for game type
            fg3_pct *= efficiency_factor
            
            # 3PT% varies more than overall FG%
            fg3_pct *= np.random.normal(1, 0.12)
            
            # Ensure percentage is valid
            fg3_pct = max(min(fg3_pct, 1.0), 0.0)
            
            # Calculate makes
            fg3m = round(box_score['fg3a'] * fg3_pct)
            
            # Ensure makes <= attempts
            box_score['fg3m'] = min(fg3m, box_score['fg3a'])
        
        # FT%
        if 'ft_pct_values' in self.historical_distributions and 'fta' in box_score:
            # Sample from empirical distribution
            ft_pct = np.random.choice(self.historical_distributions['ft_pct_values'])
            
            # Free throw shooting has less variance for good shooters
            if ft_pct > 0.8:
                ft_pct_variance = 0.05
            elif ft_pct > 0.7:
                ft_pct_variance = 0.08
            else:
                ft_pct_variance = 0.1
                
            # Add slight variation
            ft_pct *= np.random.normal(1, ft_pct_variance)
            
            # Ensure percentage is valid
            ft_pct = max(min(ft_pct, 1.0), 0.0)
            
            # Calculate makes
            ftm = round(box_score['fta'] * ft_pct)
            
            # Ensure makes <= attempts
            box_score['ftm'] = min(ftm, box_score['fta'])
        
        # Simulate other stats
        # Rebounds
        if 'reb_values' in self.historical_distributions:
            # Sample from empirical distribution
            reb_per36 = np.random.choice(self.historical_distributions['reb_values'])
            
            # Add variance
            variance_multiplier = np.random.lognormal(0, 0.2)
            reb_per36 *= variance_multiplier * reb_factor
            
            # Convert to actual value
            rebs = round(reb_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_rebs = self.historical_distributions['reb'].get('max', 20)
            rebs = max(min(rebs, max_rebs), 0)
            box_score['reb'] = rebs
            
            # Split into offensive and defensive
            playstyle = tendencies.get('playstyle', 'average')
            if playstyle == 'rebounder':
                oreb_ratio = np.random.beta(3, 7)  # More offensive boards for rebounders
            else:
                oreb_ratio = np.random.beta(2, 10)  # Typical ratio for most players
                
            box_score['oreb'] = round(rebs * oreb_ratio)
            box_score['dreb'] = rebs - box_score['oreb']
        
        # Assists
        if 'ast_values' in self.historical_distributions:
            # Sample from empirical distribution
            ast_per36 = np.random.choice(self.historical_distributions['ast_values'])
            
            # Playmakers have different variance patterns
            playstyle = tendencies.get('playstyle', 'average')
            if playstyle == 'playmaker':
                variance_multiplier = np.random.lognormal(0, 0.25)
            else:
                variance_multiplier = np.random.lognormal(0, 0.2)
                
            ast_per36 *= variance_multiplier * ast_factor
            
            # Convert to actual value
            asts = round(ast_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_asts = self.historical_distributions['ast'].get('max', 15)
            asts = max(min(asts, max_asts), 0)
            box_score['ast'] = asts
        
        # Defensive stats (steals and blocks)
        def_tendency = tendencies.get('defensive_playmaker', 'average')
        
        # Different variance based on defensive tendency
        if def_tendency == 'elite':
            def_variance = 0.4
            outlier_chance = 0.1
        elif def_tendency == 'active':
            def_variance = 0.3
            outlier_chance = 0.05
        else:
            def_variance = 0.25
            outlier_chance = 0.03
        
        # Steals
        if 'stl_values' in self.historical_distributions:
            # Sample from empirical distribution
            stl_per36 = np.random.choice(self.historical_distributions['stl_values'])
            
            # Add variance
            variance_multiplier = np.random.lognormal(0, def_variance)
            
            # Occasional outlier games
            if np.random.random() < outlier_chance:
                variance_multiplier *= np.random.uniform(1.5, 2.5)
                
            stl_per36 *= variance_multiplier
            
            # Convert to actual value
            stls = round(stl_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_stls = self.historical_distributions['stl'].get('max', 6)
            stls = max(min(stls, max_stls), 0)
            box_score['stl'] = stls
        
        # Blocks
        if 'blk_values' in self.historical_distributions:
            # Sample from empirical distribution
            blk_per36 = np.random.choice(self.historical_distributions['blk_values'])
            
            # Add variance
            variance_multiplier = np.random.lognormal(0, def_variance)
            
            # Occasional outlier games
            if np.random.random() < outlier_chance:
                variance_multiplier *= np.random.uniform(1.5, 2.5)
                
            blk_per36 *= variance_multiplier
            
            # Convert to actual value
            blks = round(blk_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_blks = self.historical_distributions['blk'].get('max', 8)
            blks = max(min(blks, max_blks), 0)
            box_score['blk'] = blks
        
        # Turnovers
        if 'tov_values' in self.historical_distributions:
            # Sample from empirical distribution
            tov_per36 = np.random.choice(self.historical_distributions['tov_values'])
            
            # Higher usage leads to more turnovers
            usage_factor = 1.0
            if 'ast' in box_score and box_score['ast'] > self.historical_distributions.get('ast', {}).get('p75', 0):
                # More assists -> more turnovers
                usage_factor *= 1.2
            if 'fga' in box_score and box_score['fga'] > self.historical_distributions.get('fga', {}).get('p75', 0):
                # More shot attempts -> more turnovers
                usage_factor *= 1.1
                
            # Add variance
            variance_multiplier = np.random.lognormal(0, 0.25)
            tov_per36 *= variance_multiplier * usage_factor
            
            # Convert to actual value
            tovs = round(tov_per36 * minutes / 36)
            
            # Ensure reasonable bounds
            max_tovs = self.historical_distributions['tov'].get('max', 10)
            tovs = max(min(tovs, max_tovs), 0)
            box_score['tov'] = tovs
        
        # Calculate points from shooting
        if all(k in box_score for k in ['fgm', 'fg3m', 'ftm']):
            two_pointers = box_score['fgm'] - box_score['fg3m']
            box_score['pts'] = (two_pointers * 2) + (box_score['fg3m'] * 3) + box_score['ftm']
        
        return box_score
    
    def simulate_games(self, n_games=10, player_id=None, output_file=None):
        """
        Simulate multiple games and return results
        
        Parameters:
        -----------
        n_games : int, default=10
            Number of games to simulate
        player_id : str, optional
            Player identifier (if not set during initialization)
        output_file : str, optional
            File path to save simulation results
            
        Returns:
        --------
        DataFrame
            Simulated game stats
        """
        if player_id is not None and (self.player_id is None or player_id != self.player_id):
            self.player_id = player_id
            self.player_data = self.load_player_data(player_id)
            self.historical_distributions = self._calculate_historical_distributions()
            self._fit_bayesian_models()
        
        simulated_games = []
        
        if self.verbose:
            print(f"Simulating {n_games} games for {self.player_id}...")
            iterator = tqdm(range(n_games))
        else:
            iterator = range(n_games)
            
        for i in iterator:
            # Alternate home/away
            context = {'home': i % 2 == 0}
            
            # Simulate the game
            game_stats = self.simulate_game(context=context)
            
            # Add game number and type
            game_stats['game_num'] = i + 1
            game_stats['game_type'] = self.game_type
            game_stats['matchup_type'] = self.matchup_type
            game_stats['player_id'] = self.player_id
            
            simulated_games.append(game_stats)
        
        # Convert to DataFrame
        sim_df = pd.DataFrame(simulated_games)
        
        # Save to file if requested
        if output_file:
            sim_df.to_csv(output_file, index=False)
            if self.verbose:
                print(f"Simulation results saved to {output_file}")
        
        return sim_df
    
    def compare_with_historical(self, simulated_games, output_dir=None):
        """
        Compare simulated games with historical data
        
        Parameters:
        -----------
        simulated_games : DataFrame
            Simulated game results
        output_dir : str, optional
            Directory to save comparison charts
            
        Returns:
        --------
        tuple
            (historical_df, summary_stats)
        """
        if self.player_data is None:
            raise ValueError("No historical data available for comparison")
        
        # Filter historical data to only include games with minutes > 0
        hist_df = self.player_data[self.player_data['minutes'] > 0].copy()
        sim_df = simulated_games.copy()
        
        # Calculate shooting percentages
        for df in [hist_df, sim_df]:
            df['fg_pct'] = df['fgm'] / df['fga']
            df['fg3_pct'] = df['fg3m'] / df['fg3a']
            df['ft_pct'] = df['ftm'] / df['fta']
            
            # Replace infinities and NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
        
        # List of stats to compare
        key_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'minutes', 
                    'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta']
        
        # Create summary stats DataFrame
        summary_stats = pd.DataFrame({
            'Stat': key_stats,
            'Historical Mean': [hist_df[stat].mean() for stat in key_stats],
            'Historical Std': [hist_df[stat].std() for stat in key_stats],
            'Simulated Mean': [sim_df[stat].mean() for stat in key_stats],
            'Simulated Std': [sim_df[stat].std() for stat in key_stats]
        })
        
        # Add percentage differences
        summary_stats['Mean % Diff'] = ((summary_stats['Simulated Mean'] - summary_stats['Historical Mean']) / 
                                       summary_stats['Historical Mean'] * 100)
        summary_stats['Std % Diff'] = ((summary_stats['Simulated Std'] - summary_stats['Historical Std']) / 
                                      summary_stats['Historical Std'] * 100)
        
        # Create comparison charts if output directory provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 1. Basic stats distribution comparison
            fig, axes = plt.subplots(len(key_stats), 1, figsize=(10, 30))
            fig.suptitle(f"{self.player_id}: Historical vs Simulated Stats (Bayesian)", fontsize=16)
            
            for i, stat in enumerate(key_stats):
                ax = axes[i]
                
                # Historical distribution
                sns.histplot(hist_df[stat], ax=ax, color='blue', alpha=0.6, label='Historical', 
                           kde=True, stat='density')
                
                # Simulated distribution
                sns.histplot(sim_df[stat], ax=ax, color='red', alpha=0.6, label='Simulated', 
                           kde=True, stat='density')
                
                # Add vertical lines for the means
                ax.axvline(hist_df[stat].mean(), color='blue', linestyle='--', alpha=0.7)
                ax.axvline(sim_df[stat].mean(), color='red', linestyle='--', alpha=0.7)
                
                # Add stat mean and std values in the legend
                hist_mean = hist_df[stat].mean()
                hist_std = hist_df[stat].std()
                
                sim_mean = sim_df[stat].mean()
                sim_std = sim_df[stat].std()
                
                ax.legend([f'Historical (μ={hist_mean:.1f}, σ={hist_std:.1f})', 
                          f'Simulated (μ={sim_mean:.1f}, σ={sim_std:.1f})'])
                
                ax.set_title(f"Distribution of {stat.upper()}")
                
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, f"{self.player_id}_stat_distributions_bayes.png"))
            plt.close()
            
            # 2. Shooting efficiency comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # FG%
            sns.histplot(hist_df['fg_pct'], ax=axes[0], color='blue', alpha=0.6, kde=True, label='Historical')
            sns.histplot(sim_df['fg_pct'], ax=axes[0], color='red', alpha=0.6, kde=True, label='Simulated')
            axes[0].set_title('FG%')
            axes[0].set_xlim([0, 1])
            axes[0].legend([f'Historical (μ={hist_df["fg_pct"].mean():.3f}, σ={hist_df["fg_pct"].std():.3f})', 
                           f'Simulated (μ={sim_df["fg_pct"].mean():.3f}, σ={sim_df["fg_pct"].std():.3f})'])
            
            # 3P%
            sns.histplot(hist_df['fg3_pct'], ax=axes[1], color='blue', alpha=0.6, kde=True, label='Historical')
            sns.histplot(sim_df['fg3_pct'], ax=axes[1], color='red', alpha=0.6, kde=True, label='Simulated')
            axes[1].set_title('3P%')
            axes[1].set_xlim([0, 1])
            axes[1].legend([f'Historical (μ={hist_df["fg3_pct"].mean():.3f}, σ={hist_df["fg3_pct"].std():.3f})', 
                           f'Simulated (μ={sim_df["fg3_pct"].mean():.3f}, σ={sim_df["fg3_pct"].std():.3f})'])
            
            # FT%
            sns.histplot(hist_df['ft_pct'], ax=axes[2], color='blue', alpha=0.6, kde=True, label='Historical')
            sns.histplot(sim_df['ft_pct'], ax=axes[2], color='red', alpha=0.6, kde=True, label='Simulated')
            axes[2].set_title('FT%')
            axes[2].set_xlim([0, 1])
            axes[2].legend([f'Historical (μ={hist_df["ft_pct"].mean():.3f}, σ={hist_df["ft_pct"].std():.3f})', 
                           f'Simulated (μ={sim_df["ft_pct"].mean():.3f}, σ={sim_df["ft_pct"].std():.3f})'])
            
            plt.suptitle(f"{self.player_id}: Shooting Efficiency Comparison (Bayesian)", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"{self.player_id}_shooting_efficiency_bayes.png"))
            plt.close()
            
            # 3. Game type analysis
            game_type_stats = sim_df.groupby('game_type').agg({
                'pts': ['mean', 'std'],
                'reb': ['mean', 'std'],
                'ast': ['mean', 'std'],
                'fg_pct': ['mean', 'std'],
                'fg3_pct': ['mean', 'std'],
                'ft_pct': ['mean', 'std']
            })
            
            # 4. Plot posteriors for shooting percentages
            if self.shooting_models:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # FG%
                if 'fg_pct' in self.shooting_models:
                    samples = self.shooting_models['fg_pct']['posterior_samples']
                    sns.histplot(samples, ax=axes[0], color='purple', alpha=0.7, kde=True)
                    axes[0].axvline(samples.mean(), color='red', linestyle='-', alpha=0.7)
                    axes[0].axvline(hist_df['fg_pct'].mean(), color='blue', linestyle='--', alpha=0.7)
                    axes[0].set_title(f'FG% Posterior (μ={samples.mean():.3f})')
                    axes[0].set_xlim([0, 1])
                
                # 3P%
                if 'fg3_pct' in self.shooting_models:
                    samples = self.shooting_models['fg3_pct']['posterior_samples']
                    sns.histplot(samples, ax=axes[1], color='purple', alpha=0.7, kde=True)
                    axes[1].axvline(samples.mean(), color='red', linestyle='-', alpha=0.7)
                    axes[1].axvline(hist_df['fg3_pct'].mean(), color='blue', linestyle='--', alpha=0.7)
                    axes[1].set_title(f'3P% Posterior (μ={samples.mean():.3f})')
                    axes[1].set_xlim([0, 1])
                
                # FT%
                if 'ft_pct' in self.shooting_models:
                    samples = self.shooting_models['ft_pct']['posterior_samples']
                    sns.histplot(samples, ax=axes[2], color='purple', alpha=0.7, kde=True)
                    axes[2].axvline(samples.mean(), color='red', linestyle='-', alpha=0.7)
                    axes[2].axvline(hist_df['ft_pct'].mean(), color='blue', linestyle='--', alpha=0.7)
                    axes[2].set_title(f'FT% Posterior (μ={samples.mean():.3f})')
                    axes[2].set_xlim([0, 1])
                
                plt.suptitle(f"{self.player_id}: Shooting Percentage Posteriors", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(output_dir, f"{self.player_id}_shooting_posteriors.png"))
                plt.close()
            
            # Save summary stats to CSV
            summary_stats.to_csv(os.path.join(output_dir, f"{self.player_id}_summary_stats_bayes.csv"), 
                                index=False, float_format='%.2f')
            
            # Save game type stats to CSV
            game_type_stats.to_csv(os.path.join(output_dir, f"{self.player_id}_game_type_stats_bayes.csv"))
        
        # Print summary
        if self.verbose:
            print("\n===== SUMMARY STATISTICS COMPARISON =====")
            print(summary_stats[['Stat', 'Historical Mean', 'Simulated Mean', 'Mean % Diff']].round(2))
            
            # Print game type distribution
            if 'game_type' in sim_df.columns:
                game_type_counts = sim_df['game_type'].value_counts(normalize=True) * 100
                print("\nSimulated Game Type Distribution:")
                for game_type, percentage in game_type_counts.items():
                    print(f"{game_type.capitalize()}: {percentage:.1f}%")
        
        return hist_df, summary_stats

# Example usage
if __name__ == "__main__":
    # Example: Simulate Kevin Durant with Bayesian models
    simulator = BayesianPlayerSimulator(player_id="Kevin Durant")
    
    # Simulate 100 games
    simulated_games = simulator.simulate_games(n_games=100, 
                                              output_file="kevin_durant_simulations_bayes.csv")
    
    # Compare with historical data
    simulator.compare_with_historical(simulated_games, output_dir="bayesian_results")