import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bayesian_player_simulator import BayesianPlayerSimulator

class ImprovedBayesianSimulator(BayesianPlayerSimulator):
    """
    Modified version of the BayesianPlayerSimulator that uses better statistical models
    for count data, particularly for rare events like high field goal attempts.
    """
    
    def simulate_game(self, context=None):
        """
        Override the simulate_game method to use improved statistical approaches
        
        Parameters:
        -----------
        context : dict, optional
            Context information - simplified to reduce extreme variations
            
        Returns:
        --------
        dict
            Simulated box score stats
        """
        # Simplify game type distribution - less extreme variations
        self.game_type = np.random.choice(['average', 'hot', 'cold'], 
                                         p=[0.8, 0.1, 0.1])
        self.matchup_type = 'normal'  # Use normal matchup type
        
        # Simplified context - reduce sources of extreme variation
        fixed_context = {'home': True if context is None else context.get('home', True)} 
        
        # Instead of using parent's simulate_game, we'll replicate its structure
        # but with our improved box score simulation
        
        # Determine minutes played from parent's method
        minutes = 0
        
        # Simple minutes simulation based on historical data
        if 'minutes' in self.historical_distributions:
            # Get basic stats about player's minutes
            minutes_stats = self.historical_distributions['minutes']
            avg_minutes = minutes_stats.get('mean', 30.0)
            min_minutes = minutes_stats.get('min', 20.0)
            max_minutes = minutes_stats.get('max', 40.0)
            std_minutes = minutes_stats.get('std', 5.0)
            
            # Generate minutes, slightly biased toward average
            minutes = np.random.normal(avg_minutes, std_minutes * 0.8)
            minutes = max(min_minutes, min(max_minutes, minutes))
        else:
            # Default minutes if no historical data available
            minutes = 30.0 + np.random.normal(0, 4.0)
            minutes = max(20.0, min(40.0, minutes))
        
        # Round to reasonable value
        minutes = round(minutes * 100) / 100
        
        # Use our improved box score simulation
        box_score = self._simulate_box_score_improved(minutes)
        box_score['minutes'] = minutes
        
        return box_score
        
    def _simulate_box_score_improved(self, minutes):
        """
        Improved simulation of box score stats - more Bayesian approach
        using better statistical models for count data
        
        Parameters:
        -----------
        minutes : float
            Simulated minutes played
            
        Returns:
        --------
        dict
            Simulated box score statistics
        """
        box_score = {}
        
        # Get player tendencies
        tendencies = self.historical_distributions.get('tendencies', {})
        
        # Determine game type factors (minor adjustments for hot/cold games)
        pts_factor = 1.0
        if self.game_type == 'hot':
            pts_factor = 1.15
        elif self.game_type == 'cold':
            pts_factor = 0.85
            
        # When simulating shot attempts, use a true negative binomial distribution
        # This is more appropriate for count data and naturally handles the distribution's tail
        if 'fga' in self.stat_models:
            # Get the posterior samples for FGA per 36 min
            fga_per36_samples = self.stat_models['fga']['posterior_samples']
            
            # Sample from the posterior - this preserves the Bayesian nature
            fga_per36 = np.random.choice(fga_per36_samples)
            
            # Scale by minutes
            fga_lambda = fga_per36 * minutes / 36.0
            
            # For FGA, use a shape parameter that controls the tail weight
            # Higher r = more consistent, tighter distribution
            # This is directly estimated from the historical data variance
            r = 10.0  # Default conservative value for shape
            if 'fga' in self.historical_distributions:
                # Use coefficient of variation to determine appropriate shape
                fga_mean = self.historical_distributions['fga']['mean']
                fga_std = self.historical_distributions['fga']['std']
                if fga_mean > 0 and fga_std > 0:
                    cv = fga_std / fga_mean
                    # Negative binomial shape parameter is approximately 1/(CV^2)
                    r = max(5.0, min(25.0, 1.0 / (cv * cv)))
            
            # Convert to probability parameter for negative binomial
            p = r / (r + fga_lambda * pts_factor)
            
            # Generate FGA with proper distributional characteristics
            try:
                # This naturally handles the tail of the distribution
                fga = np.random.negative_binomial(r, p)
            except:
                # Fallback to a more conservative approach
                fga = np.random.poisson(fga_lambda * pts_factor)
                
            box_score['fga'] = fga
            
            # Now handle 3PA using similar logic but with correlation to FGA
            if 'fg3a' in self.stat_models:
                # Get historical 3-point tendency
                three_pt_rate = tendencies.get('three_pt_rate', 0.3)
                
                # Use a beta distribution to model the proportion of 3PA to FGA
                # Parameters estimated from historical data
                alpha = three_pt_rate * 20
                beta = (1 - three_pt_rate) * 20
                
                # Sample proportion from beta distribution
                prop_3pa = np.random.beta(alpha, beta)
                
                # Apply to FGA, ensuring we don't exceed total FGA
                fg3a = min(fga, round(fga * prop_3pa))
                box_score['fg3a'] = fg3a
                
                # Non-3PA shots are simply the remainder
                non_3pa = fga - fg3a
                
                # Now simulate makes - use binomial for exact counts
                if 'fg_pct' in self.shooting_models:
                    # Get posterior mean for FG%
                    # We need to fall back on a simpler approach since posterior samples
                    # might not be directly accessible in the format we need
                    fg_pct = self.historical_distributions.get('fg_pct', {}).get('mean', 0.45)
                    # Add some noise around the posterior mean
                    fg_pct = np.random.normal(fg_pct, 0.03)
                    
                    # For non-3PT shots
                    if non_3pa > 0:
                        fgm_non3 = np.random.binomial(non_3pa, fg_pct)
                    else:
                        fgm_non3 = 0
                        
                    if fg3a > 0:
                        # Get 3P% from historical mean
                        fg3_pct = self.historical_distributions.get('fg3_pct', {}).get('mean', 0.35)
                        # Add some noise
                        fg3_pct = np.random.normal(fg3_pct, 0.04)
                        
                        # Simulate 3PT makes
                        fg3m = np.random.binomial(fg3a, fg3_pct)
                    else:
                        fg3m = 0
                        
                    # Total FGM is sum of 2PT and 3PT makes
                    fgm = fgm_non3 + fg3m
                    
                    box_score['fgm'] = fgm
                    box_score['fg3m'] = fg3m
        
        # Similar approach for free throws
        if 'fta' in self.stat_models:
            # Get posterior samples for FTA per 36 min
            fta_per36_samples = self.stat_models['fta']['posterior_samples']
            fta_per36 = np.random.choice(fta_per36_samples)
            
            # Scale by minutes and apply game factor
            fta_lambda = fta_per36 * minutes / 36.0 * pts_factor
            
            # Use shape parameter based on historical variance
            r_fta = 8.0
            if 'fta' in self.historical_distributions:
                fta_mean = self.historical_distributions['fta']['mean']
                fta_std = self.historical_distributions['fta']['std']
                if fta_mean > 0 and fta_std > 0:
                    cv = fta_std / fta_mean
                    r_fta = max(4.0, min(20.0, 1.0 / (cv * cv)))
            
            p_fta = r_fta / (r_fta + fta_lambda)
            
            try:
                fta = np.random.negative_binomial(r_fta, p_fta)
            except:
                fta = np.random.poisson(fta_lambda)
                
            box_score['fta'] = fta
            
            # Simulate makes
            if fta > 0:
                # Get FT% from historical mean
                ft_pct = self.historical_distributions.get('ft_pct', {}).get('mean', 0.75)
                # Add some noise
                ft_pct = np.random.normal(ft_pct, 0.03)
                # Ensure percentage is valid
                ft_pct = max(0.5, min(0.95, ft_pct))
                ftm = np.random.binomial(fta, ft_pct)
                box_score['ftm'] = ftm
            else:
                box_score['ftm'] = 0
        
        # Simulate other box score stats
        self._simulate_other_stats_improved(box_score, minutes)
        
        # Calculate points
        if all(k in box_score for k in ['fgm', 'fg3m', 'ftm']):
            two_pointers = box_score['fgm'] - box_score['fg3m']
            box_score['pts'] = (two_pointers * 2) + (box_score['fg3m'] * 3) + box_score['ftm']
        
        return box_score
        
    def _simulate_other_stats_improved(self, box_score, minutes):
        """
        Simulate other box score stats using improved Bayesian models
        
        Parameters:
        -----------
        box_score : dict
            Current box score to update
        minutes : float
            Minutes played
        """
        # Improved simulation for rebounds, assists, steals, blocks, turnovers
        # using appropriate negative binomial distributions
        
        for stat in ['reb', 'ast', 'stl', 'blk', 'tov']:
            if stat in self.stat_models:
                # Get posterior samples
                stat_per36_samples = self.stat_models[stat]['posterior_samples']
                stat_per36 = np.random.choice(stat_per36_samples)
                
                # Scale by minutes
                stat_lambda = stat_per36 * minutes / 36.0
                
                # Calculate shape parameter based on historical variance
                r_stat = 5.0  # Default
                if stat in self.historical_distributions:
                    stat_mean = self.historical_distributions[stat]['mean']
                    stat_std = self.historical_distributions[stat]['std']
                    if stat_mean > 0 and stat_std > 0:
                        cv = stat_std / stat_mean
                        # Different stats have different natural variability
                        if stat in ['stl', 'blk']:  # More variable
                            r_stat = max(2.0, min(10.0, 1.0 / (cv * cv)))
                        else:  # Less variable
                            r_stat = max(3.0, min(15.0, 1.0 / (cv * cv)))
                
                p_stat = r_stat / (r_stat + stat_lambda)
                
                try:
                    stat_value = np.random.negative_binomial(r_stat, p_stat)
                except:
                    stat_value = np.random.poisson(stat_lambda)
                    
                box_score[stat] = stat_value
                
                # Handle specific stat details (like offensive/defensive rebounds)
                if stat == 'reb' and stat_value > 0:
                    # Use beta distribution for ratio
                    oreb_ratio = np.random.beta(2, 8)  # Typically more defensive than offensive
                    box_score['oreb'] = round(stat_value * oreb_ratio)
                    box_score['dreb'] = stat_value - box_score['oreb']

def run_player_simulation(player_name, num_games=100, output_dir=None, verbose=True):
    """
    Run a Bayesian simulation for a specific player.
    
    Parameters:
    -----------
    player_name : str
        Name of the player to simulate
    num_games : int, default=100
        Number of games to simulate
    output_dir : str, optional
        Base directory for outputs. If None, will use '[player_name]_sim_results'
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    tuple
        (simulator, simulated_games, historical_data, summary_stats)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = f"{player_name.replace(' ', '_').lower()}_sim_results"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize simulator using our improved Bayesian model
    if verbose:
        print(f"Initializing Bayesian simulator for {player_name}...")
    
    simulator = ImprovedBayesianSimulator(player_id=player_name, verbose=verbose)
    
    # Simulate games
    if verbose:
        print(f"Simulating {num_games} games for {player_name}...")
    
    simulated_games = simulator.simulate_games(
        n_games=num_games, 
        output_file=os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_games.csv")
    )
    
    # Compare with historical data
    if verbose:
        print("\nComparing simulated games with historical data...")
    
    hist_df, summary_stats = simulator.compare_with_historical(
        simulated_games, 
        output_dir=output_dir
    )
    
    # Create additional visualizations
    if verbose:
        print("\nGenerating additional visualizations...")
    
    create_visualizations(simulator, hist_df, simulated_games, output_dir, player_name)
    
    # Print summary statistics
    if verbose:
        print_summary(simulator, hist_df, simulated_games, summary_stats)
    
    return simulator, simulated_games, hist_df, summary_stats

def create_visualizations(simulator, hist_df, simulated_games, output_dir, player_name):
    """
    Create additional visualizations for the player simulation.
    
    Parameters:
    -----------
    simulator : BayesianPlayerSimulator
        The simulator instance
    hist_df : DataFrame
        Historical player data
    simulated_games : DataFrame
        Simulated game data
    output_dir : str
        Directory for outputs
    player_name : str
        Name of the player
    """
    # Define box score stats to analyze
    box_score_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta']
    
    # Calculate percentiles for historical data
    hist_percentiles = {}
    for stat in box_score_stats + ['minutes']:
        hist_percentiles[stat] = {
            'mean': hist_df[stat].mean(),
            'std': hist_df[stat].std(),
            'min': hist_df[stat].min(),
            'p25': hist_df[stat].quantile(0.25),
            'p50': hist_df[stat].quantile(0.50),
            'p75': hist_df[stat].quantile(0.75),
            'max': hist_df[stat].max()
        }
    
    # Calculate percentiles for simulated data
    sim_percentiles = {}
    for stat in box_score_stats + ['minutes']:
        sim_percentiles[stat] = {
            'mean': simulated_games[stat].mean(),
            'std': simulated_games[stat].std(),
            'min': simulated_games[stat].min(),
            'p25': simulated_games[stat].quantile(0.25),
            'p50': simulated_games[stat].quantile(0.50),
            'p75': simulated_games[stat].quantile(0.75),
            'max': simulated_games[stat].max()
        }
    
    # 1. Create side-by-side box plots for all stats
    fig, axes = plt.subplots(len(box_score_stats), 1, figsize=(12, 4*len(box_score_stats)))
    
    for i, stat in enumerate(box_score_stats):
        ax = axes[i]
        
        # Box plot comparing historical vs simulated
        data = pd.DataFrame({
            'Historical': hist_df[stat],
            'Simulated': simulated_games[stat]
        })
        
        sns.boxplot(data=data, ax=ax, palette=['blue', 'red'])
        
        # Add means as points
        plt.scatter([0, 1], [hist_percentiles[stat]['mean'], sim_percentiles[stat]['mean']], 
                   marker='o', color='yellow', s=100, zorder=3)
        
        # Add statistical details as text
        hist_text = f"μ={hist_percentiles[stat]['mean']:.2f}, σ={hist_percentiles[stat]['std']:.2f}"
        sim_text = f"μ={sim_percentiles[stat]['mean']:.2f}, σ={sim_percentiles[stat]['std']:.2f}"
        
        ax.annotate(hist_text, xy=(0, ax.get_ylim()[1]*0.9), 
                   xytext=(0, ax.get_ylim()[1]*0.9), ha='center')
        ax.annotate(sim_text, xy=(1, ax.get_ylim()[1]*0.9), 
                   xytext=(1, ax.get_ylim()[1]*0.9), ha='center')
        
        ax.set_title(f"{stat.upper()}", fontsize=14)
    
    plt.suptitle(f"{player_name}: Historical vs Bayesian Simulated Stats", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_boxplot_comparison.png"))
    plt.close()
    
    # 2. Create correlation plots for key stat relationships
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    # Determine if player is a three-point specialist
    tendencies = simulator.historical_distributions.get('tendencies', {})
    is_three_specialist = tendencies.get('shooting_profile', '') == 'three_specialist'
    
    # 1. PTS vs FGA
    ax = axes[0]
    ax.scatter(hist_df['fga'], hist_df['pts'], color='blue', alpha=0.4, label='Historical')
    ax.scatter(simulated_games['fga'], simulated_games['pts'], color='red', alpha=0.4, label='Simulated')
    
    hist_corr = np.corrcoef(hist_df['fga'], hist_df['pts'])[0, 1]
    sim_corr = np.corrcoef(simulated_games['fga'], simulated_games['pts'])[0, 1]
    
    ax.set_xlabel('FGA', fontsize=12)
    ax.set_ylabel('PTS', fontsize=12)
    ax.set_title(f'PTS vs FGA (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
    ax.legend()
    
    # 2. REB vs Minutes (or 3PA vs PTS for 3PT specialists)
    ax = axes[1]
    if is_three_specialist:
        ax.scatter(hist_df['fg3a'], hist_df['pts'], color='blue', alpha=0.4, label='Historical')
        ax.scatter(simulated_games['fg3a'], simulated_games['pts'], color='red', alpha=0.4, label='Simulated')
        
        hist_corr = np.corrcoef(hist_df['fg3a'], hist_df['pts'])[0, 1]
        sim_corr = np.corrcoef(simulated_games['fg3a'], simulated_games['pts'])[0, 1]
        
        ax.set_xlabel('3PA', fontsize=12)
        ax.set_ylabel('PTS', fontsize=12)
        ax.set_title(f'PTS vs 3PA (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
    else:
        ax.scatter(hist_df['minutes'], hist_df['reb'], color='blue', alpha=0.4, label='Historical')
        ax.scatter(simulated_games['minutes'], simulated_games['reb'], color='red', alpha=0.4, label='Simulated')
        
        hist_corr = np.corrcoef(hist_df['minutes'], hist_df['reb'])[0, 1]
        sim_corr = np.corrcoef(simulated_games['minutes'], simulated_games['reb'])[0, 1]
        
        ax.set_xlabel('Minutes', fontsize=12)
        ax.set_ylabel('REB', fontsize=12)
        ax.set_title(f'REB vs Minutes (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
    ax.legend()
    
    # 3. AST vs TOV
    ax = axes[2]
    ax.scatter(hist_df['ast'], hist_df['tov'], color='blue', alpha=0.4, label='Historical')
    ax.scatter(simulated_games['ast'], simulated_games['tov'], color='red', alpha=0.4, label='Simulated')
    
    hist_corr = np.corrcoef(hist_df['ast'], hist_df['tov'])[0, 1]
    sim_corr = np.corrcoef(simulated_games['ast'], simulated_games['tov'])[0, 1]
    
    ax.set_xlabel('AST', fontsize=12)
    ax.set_ylabel('TOV', fontsize=12)
    ax.set_title(f'TOV vs AST (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
    ax.legend()
    
    # 4. Custom fourth plot based on player style
    ax = axes[3]
    playstyle = tendencies.get('playstyle', 'average')
    
    if playstyle == 'playmaker':
        # For playmakers, show AST vs Minutes
        ax.scatter(hist_df['minutes'], hist_df['ast'], color='blue', alpha=0.4, label='Historical')
        ax.scatter(simulated_games['minutes'], simulated_games['ast'], color='red', alpha=0.4, label='Simulated')
        
        hist_corr = np.corrcoef(hist_df['minutes'], hist_df['ast'])[0, 1]
        sim_corr = np.corrcoef(simulated_games['minutes'], simulated_games['ast'])[0, 1]
        
        ax.set_xlabel('Minutes', fontsize=12)
        ax.set_ylabel('AST', fontsize=12)
        ax.set_title(f'AST vs Minutes (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
    elif playstyle == 'rebounder':
        # For rebounders, show OREB vs DREB
        ax.scatter(hist_df['oreb'], hist_df['dreb'], color='blue', alpha=0.4, label='Historical')
        ax.scatter(simulated_games['oreb'], simulated_games['dreb'], color='red', alpha=0.4, label='Simulated')
        
        hist_corr = np.corrcoef(hist_df['oreb'], hist_df['dreb'])[0, 1]
        sim_corr = np.corrcoef(simulated_games['oreb'], simulated_games['dreb'])[0, 1]
        
        ax.set_xlabel('OREB', fontsize=12)
        ax.set_ylabel('DREB', fontsize=12)
        ax.set_title(f'DREB vs OREB (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
    else:
        # Default to FG% vs FT%
        hist_df['fg_pct'] = hist_df['fgm'] / hist_df['fga']
        simulated_games['fg_pct'] = simulated_games['fgm'] / simulated_games['fga']
        hist_df['ft_pct'] = hist_df['ftm'] / hist_df['fta']
        simulated_games['ft_pct'] = simulated_games['ftm'] / simulated_games['fta']
        
        # Replace infinities and NaN
        hist_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        hist_df.dropna(subset=['fg_pct', 'ft_pct'], inplace=True)
        simulated_games.replace([np.inf, -np.inf], np.nan, inplace=True)
        simulated_games.dropna(subset=['fg_pct', 'ft_pct'], inplace=True)
        
        ax.scatter(hist_df['fg_pct'], hist_df['ft_pct'], color='blue', alpha=0.4, label='Historical')
        ax.scatter(simulated_games['fg_pct'], simulated_games['ft_pct'], color='red', alpha=0.4, label='Simulated')
        
        try:
            hist_corr = np.corrcoef(hist_df['fg_pct'], hist_df['ft_pct'])[0, 1]
            sim_corr = np.corrcoef(simulated_games['fg_pct'], simulated_games['ft_pct'])[0, 1]
            
            ax.set_xlabel('FG%', fontsize=12)
            ax.set_ylabel('FT%', fontsize=12)
            ax.set_title(f'FT% vs FG% (Corr: Hist={hist_corr:.2f}, Sim={sim_corr:.2f})', fontsize=14)
        except:
            ax.set_xlabel('FG%', fontsize=12)
            ax.set_ylabel('FT%', fontsize=12)
            ax.set_title(f'FT% vs FG%', fontsize=14)
    
    ax.legend()
    
    plt.suptitle(f"{player_name}: Statistical Relationships in Bayesian Simulation", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_stat_relationships.png"))
    plt.close()
    
    # 3. Create special visualizations for 3-point specialists
    if is_three_specialist:
        plt.figure(figsize=(10, 6))
        bins = np.arange(0, 15, 1)
        
        plt.hist(hist_df['fg3m'], bins=bins, alpha=0.5, label='Historical', color='blue')
        plt.hist(simulated_games['fg3m'], bins=bins, alpha=0.5, label='Simulated', color='red')
        
        plt.axvline(hist_df['fg3m'].mean(), color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Historical Mean: {hist_df["fg3m"].mean():.2f}')
        plt.axvline(simulated_games['fg3m'].mean(), color='red', linestyle='dashed', linewidth=2,
                    label=f'Simulated Mean: {simulated_games["fg3m"].mean():.2f}')
        
        plt.title(f"{player_name}: 3-Point Makes Distribution", fontsize=14)
        plt.xlabel("3PM", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_3pm_distribution.png"))
        plt.close()
    
    # 4. Create game type analysis
    if 'game_type' in simulated_games.columns:
        # Group key stats by game type
        game_type_stats = simulated_games.groupby('game_type')[box_score_stats].mean().reset_index()
        
        # Create a sorted bar chart for key stats
        key_stats = ['pts', 'reb', 'ast', 'fg3m'] if is_three_specialist else ['pts', 'reb', 'ast']
        
        for stat in key_stats:
            plt.figure(figsize=(10, 6))
            ordered_data = game_type_stats.sort_values(stat, ascending=False)
            
            bars = plt.bar(ordered_data['game_type'], ordered_data[stat])
            
            # Add historical mean reference line
            hist_mean = hist_percentiles[stat]['mean']
            plt.axhline(y=hist_mean, linestyle='--', color='black', 
                       label=f'Historical Mean: {hist_mean:.2f}')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.title(f'{player_name}: Average {stat.upper()} by Game Type', fontsize=14)
            plt.ylabel(stat.upper())
            plt.xlabel('Game Type')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_{stat}_by_game_type.png"))
            plt.close()
    
    # 5. Save all percentile data
    combined_percentiles = {
        'Historical': hist_percentiles,
        'Simulated': sim_percentiles
    }
    
    # Convert to a more pandas-friendly format
    percentile_rows = []
    for stat in box_score_stats + ['minutes']:
        row = {'Stat': stat}
        row.update({f'hist_{metric}': hist_percentiles[stat][metric] for metric in hist_percentiles[stat]})
        row.update({f'sim_{metric}': sim_percentiles[stat][metric] for metric in sim_percentiles[stat]})
        percentile_rows.append(row)
    
    percentile_df = pd.DataFrame(percentile_rows)
    percentile_df.to_csv(os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_percentiles.csv"), index=False)

def print_summary(simulator, hist_df, simulated_games, summary_stats):
    """
    Print a summary of the simulation results.
    
    Parameters:
    -----------
    simulator : BayesianPlayerSimulator
        The simulator instance
    hist_df : DataFrame
        Historical player data
    simulated_games : DataFrame
        Simulated game data
    summary_stats : DataFrame
        Summary statistics comparison
    """
    box_score_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta']
    
    # Calculate percentiles for historical data
    hist_percentiles = {}
    for stat in box_score_stats + ['minutes']:
        hist_percentiles[stat] = {
            'mean': hist_df[stat].mean(),
            'std': hist_df[stat].std(),
        }
    
    # Calculate percentiles for simulated data
    sim_percentiles = {}
    for stat in box_score_stats + ['minutes']:
        sim_percentiles[stat] = {
            'mean': simulated_games[stat].mean(),
            'std': simulated_games[stat].std(),
        }
    
    print("\nBayesian simulation complete!")
    print("\nKey findings:")
    for stat in box_score_stats:
        hist_mean = hist_percentiles[stat]['mean']
        sim_mean = sim_percentiles[stat]['mean']
        pct_diff = ((sim_mean - hist_mean) / hist_mean) * 100
        
        hist_std = hist_percentiles[stat]['std']
        sim_std = sim_percentiles[stat]['std']
        std_pct_diff = ((sim_std - hist_std) / hist_std) * 100
        
        print(f"{stat.upper()}: Hist mean={hist_mean:.2f}, Sim mean={sim_mean:.2f} ({pct_diff:+.1f}%) | " 
              f"Hist std={hist_std:.2f}, Sim std={sim_std:.2f} ({std_pct_diff:+.1f}%)")
    
    # Print tendencies
    print("\nPlayer Tendencies detected:")
    for key, value in simulator.historical_distributions.get('tendencies', {}).items():
        if isinstance(value, (str, int, float)):
            print(f"- {key}: {value}")
    
    # For 3-point specialists, print additional 3-point metrics
    tendencies = simulator.historical_distributions.get('tendencies', {})
    is_three_specialist = tendencies.get('shooting_profile', '') == 'three_specialist'
    
    if is_three_specialist:
        hist_3pt_rate = hist_df['fg3a'].sum() / hist_df['fga'].sum()
        sim_3pt_rate = simulated_games['fg3a'].sum() / simulated_games['fga'].sum()
        
        hist_3pt_pct = hist_df['fg3m'].sum() / hist_df['fg3a'].sum()
        sim_3pt_pct = simulated_games['fg3m'].sum() / simulated_games['fg3a'].sum()
        
        print(f"\n3-Point Metrics:")
        print(f"- 3PA Rate: Historical {hist_3pt_rate:.3f} vs Simulated {sim_3pt_rate:.3f} ({(sim_3pt_rate-hist_3pt_rate)/hist_3pt_rate*100:+.1f}%)")
        print(f"- 3PT%: Historical {hist_3pt_pct:.3f} vs Simulated {sim_3pt_pct:.3f} ({(sim_3pt_pct-hist_3pt_pct)/hist_3pt_pct*100:+.1f}%)")
        print(f"- 3PM per game: Historical {hist_df['fg3m'].mean():.2f} vs Simulated {simulated_games['fg3m'].mean():.2f} ({(simulated_games['fg3m'].mean()-hist_df['fg3m'].mean())/hist_df['fg3m'].mean()*100:+.1f}%)")

def batch_simulate_players(player_list, num_games=100, output_dir="batch_simulations", verbose=True):
    """
    Run batch simulations for multiple players.
    
    Parameters:
    -----------
    player_list : list
        List of player names to simulate
    num_games : int, default=100
        Number of games to simulate for each player
    output_dir : str, default="batch_simulations"
        Base directory for outputs
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    dict
        Dictionary mapping player names to their simulation results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    
    for i, player_name in enumerate(player_list):
        if verbose:
            print(f"\n[{i+1}/{len(player_list)}] Processing {player_name}...")
        
        player_dir = os.path.join(output_dir, player_name.replace(' ', '_').lower())
        
        try:
            simulator, simulated_games, hist_df, summary_stats = run_player_simulation(
                player_name, 
                num_games=num_games, 
                output_dir=player_dir,
                verbose=verbose
            )
            
            results[player_name] = {
                'simulator': simulator,
                'simulated_games': simulated_games,
                'historical_data': hist_df,
                'summary_stats': summary_stats
            }
            
            if verbose:
                print(f"Successfully simulated {player_name}")
        except Exception as e:
            if verbose:
                print(f"Error simulating {player_name}: {str(e)}")
    
    # Create comparative analysis across players
    if verbose:
        print("\nCreating cross-player analysis...")
    
    create_cross_player_analysis(results, output_dir)
    
    return results

def create_cross_player_analysis(results, output_dir):
    """
    Create comparative analysis across multiple players.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping player names to their simulation results
    output_dir : str
        Base directory for outputs
    """
    # Extract key metrics from each player's simulation
    player_metrics = []
    
    for player_name, data in results.items():
        simulator = data['simulator']
        hist_df = data['historical_data']
        sim_df = data['simulated_games']
        
        # Extract tendencies
        tendencies = simulator.historical_distributions.get('tendencies', {})
        
        # Calculate basic metrics
        hist_pts_mean = hist_df['pts'].mean()
        sim_pts_mean = sim_df['pts'].mean()
        pts_diff_pct = ((sim_pts_mean - hist_pts_mean) / hist_pts_mean) * 100
        
        # Calculate shooting percentages
        hist_fg_pct = hist_df['fgm'].sum() / hist_df['fga'].sum()
        sim_fg_pct = sim_df['fgm'].sum() / sim_df['fga'].sum()
        
        hist_3pt_pct = hist_df['fg3m'].sum() / hist_df['fg3a'].sum() if hist_df['fg3a'].sum() > 0 else 0
        sim_3pt_pct = sim_df['fg3m'].sum() / sim_df['fg3a'].sum() if sim_df['fg3a'].sum() > 0 else 0
        
        hist_ft_pct = hist_df['ftm'].sum() / hist_df['fta'].sum() if hist_df['fta'].sum() > 0 else 0
        sim_ft_pct = sim_df['ftm'].sum() / sim_df['fta'].sum() if sim_df['fta'].sum() > 0 else 0
        
        # Collect metrics
        metrics = {
            'player_name': player_name,
            'scoring_role': tendencies.get('scoring_role', ''),
            'playstyle': tendencies.get('playstyle', ''),
            'shooting_profile': tendencies.get('shooting_profile', ''),
            'hist_pts_mean': hist_pts_mean,
            'sim_pts_mean': sim_pts_mean,
            'pts_diff_pct': pts_diff_pct,
            'hist_fg_pct': hist_fg_pct,
            'sim_fg_pct': sim_fg_pct,
            'hist_3pt_pct': hist_3pt_pct,
            'sim_3pt_pct': sim_3pt_pct,
            'hist_ft_pct': hist_ft_pct,
            'sim_ft_pct': sim_ft_pct,
            'hist_reb_mean': hist_df['reb'].mean(),
            'sim_reb_mean': sim_df['reb'].mean(),
            'hist_ast_mean': hist_df['ast'].mean(),
            'sim_ast_mean': sim_df['ast'].mean(),
            'hist_stl_mean': hist_df['stl'].mean(),
            'sim_stl_mean': sim_df['stl'].mean(),
            'hist_blk_mean': hist_df['blk'].mean(),
            'sim_blk_mean': sim_df['blk'].mean(),
            'hist_tov_mean': hist_df['tov'].mean(),
            'sim_tov_mean': sim_df['tov'].mean(),
        }
        
        player_metrics.append(metrics)
    
    # Create a DataFrame with all player metrics
    if player_metrics:
        metrics_df = pd.DataFrame(player_metrics)
        metrics_df.to_csv(os.path.join(output_dir, "player_simulation_comparison.csv"), index=False)
        
        # Create visualizations comparing players
        create_cross_player_visualizations(metrics_df, output_dir)

def create_cross_player_visualizations(metrics_df, output_dir):
    """
    Create visualizations comparing different players.
    
    Parameters:
    -----------
    metrics_df : DataFrame
        DataFrame with metrics for each player
    output_dir : str
        Base directory for outputs
    """
    # 1. Points prediction accuracy
    plt.figure(figsize=(12, 8))
    
    # Sort by historical points
    sorted_df = metrics_df.sort_values('hist_pts_mean', ascending=False).head(15)  # Top 15 players
    
    x = np.arange(len(sorted_df))
    width = 0.35
    
    # Plot historical and simulated points side by side
    plt.bar(x - width/2, sorted_df['hist_pts_mean'], width, label='Historical', color='blue', alpha=0.7)
    plt.bar(x + width/2, sorted_df['sim_pts_mean'], width, label='Simulated', color='red', alpha=0.7)
    
    # Add percent difference as text
    for i, row in enumerate(sorted_df.itertuples()):
        plt.text(i, max(row.hist_pts_mean, row.sim_pts_mean) + 1, 
                f"{row.pts_diff_pct:+.1f}%", ha='center', rotation=45)
    
    plt.xlabel('Player')
    plt.ylabel('Points per Game')
    plt.title('Comparison of Historical vs Simulated Points per Game')
    plt.xticks(x, sorted_df['player_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "points_comparison.png"))
    plt.close()
    
    # 2. Shooting percentage accuracy
    plt.figure(figsize=(15, 10))
    
    # Create subplot for each shooting percentage
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Field Goal %
    sorted_fg = metrics_df.sort_values('hist_fg_pct', ascending=False).head(15)
    axes[0].bar(np.arange(len(sorted_fg)) - width/2, sorted_fg['hist_fg_pct'], width, label='Historical', color='blue', alpha=0.7)
    axes[0].bar(np.arange(len(sorted_fg)) + width/2, sorted_fg['sim_fg_pct'], width, label='Simulated', color='red', alpha=0.7)
    axes[0].set_xlabel('Player')
    axes[0].set_ylabel('FG%')
    axes[0].set_title('Field Goal Percentage Comparison')
    axes[0].set_xticks(np.arange(len(sorted_fg)))
    axes[0].set_xticklabels(sorted_fg['player_name'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 3-Point %
    sorted_3pt = metrics_df.sort_values('hist_3pt_pct', ascending=False).head(15)
    axes[1].bar(np.arange(len(sorted_3pt)) - width/2, sorted_3pt['hist_3pt_pct'], width, label='Historical', color='blue', alpha=0.7)
    axes[1].bar(np.arange(len(sorted_3pt)) + width/2, sorted_3pt['sim_3pt_pct'], width, label='Simulated', color='red', alpha=0.7)
    axes[1].set_xlabel('Player')
    axes[1].set_ylabel('3P%')
    axes[1].set_title('3-Point Percentage Comparison')
    axes[1].set_xticks(np.arange(len(sorted_3pt)))
    axes[1].set_xticklabels(sorted_3pt['player_name'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Free Throw %
    sorted_ft = metrics_df.sort_values('hist_ft_pct', ascending=False).head(15)
    axes[2].bar(np.arange(len(sorted_ft)) - width/2, sorted_ft['hist_ft_pct'], width, label='Historical', color='blue', alpha=0.7)
    axes[2].bar(np.arange(len(sorted_ft)) + width/2, sorted_ft['sim_ft_pct'], width, label='Simulated', color='red', alpha=0.7)
    axes[2].set_xlabel('Player')
    axes[2].set_ylabel('FT%')
    axes[2].set_title('Free Throw Percentage Comparison')
    axes[2].set_xticks(np.arange(len(sorted_ft)))
    axes[2].set_xticklabels(sorted_ft['player_name'], rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shooting_pct_comparison.png"))
    plt.close()
    
    # 3. Grouped by player type/role
    if 'playstyle' in metrics_df.columns:
        # Calculate average points by playstyle
        playstyle_pts = metrics_df.groupby('playstyle')[['hist_pts_mean', 'sim_pts_mean']].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(playstyle_pts))
        
        plt.bar(x - width/2, playstyle_pts['hist_pts_mean'], width, label='Historical', color='blue', alpha=0.7)
        plt.bar(x + width/2, playstyle_pts['sim_pts_mean'], width, label='Simulated', color='red', alpha=0.7)
        
        plt.xlabel('Player Playstyle')
        plt.ylabel('Average Points per Game')
        plt.title('Points per Game by Playstyle')
        plt.xticks(x, playstyle_pts['playstyle'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "points_by_playstyle.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bayesian player simulations')
    parser.add_argument('--player', '-p', type=str, help='Player name to simulate (use comma for multiple players)')
    parser.add_argument('--games', '-g', type=int, default=100, help='Number of games to simulate')
    parser.add_argument('--output', '-o', type=str, default='player_simulations', help='Output directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Run in quiet mode')
    
    args = parser.parse_args()
    
    if args.player:
        # Single player or comma-separated list
        players = [name.strip() for name in args.player.split(',')]
        
        if len(players) == 1:
            # Single player mode
            run_player_simulation(
                players[0],
                num_games=args.games,
                output_dir=os.path.join(args.output, players[0].replace(' ', '_').lower()),
                verbose=not args.quiet
            )
        else:
            # Batch mode for specified players
            batch_simulate_players(
                players,
                num_games=args.games,
                output_dir=args.output,
                verbose=not args.quiet
            )
    else:
        # Example players if none specified
        example_players = ["LeBron James", "Stephen Curry", "Kevin Durant"]
        print(f"No player specified. Running example simulation with: {', '.join(example_players)}")
        
        batch_simulate_players(
            example_players,
            num_games=args.games,
            output_dir=args.output,
            verbose=not args.quiet
        )