import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
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

def run_player_simulation(player_name, num_games=500, output_dir=None, verbose=True):
    """
    Run a Bayesian simulation for a specific player and save results to CSV.
    
    Parameters:
    -----------
    player_name : str
        Name of the player to simulate
    num_games : int, default=500
        Number of games to simulate
    output_dir : str, optional
        Base directory for outputs. If None, will use timestamped folder
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    DataFrame
        Simulated game data
    """
    # Set up output directory with timestamp if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"run_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize simulator - use our improved Bayesian simulator
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
    
    # Add player name column for easy identification
    simulated_games['player_name'] = player_name
    
    if verbose:
        print(f"Simulation complete. Results saved to {output_dir}")
    
    return simulated_games

def batch_simulate_players(player_list, num_games=500, output_dir=None, verbose=True):
    """
    Run batch simulations for multiple players and combine results into a single CSV.
    
    Parameters:
    -----------
    player_list : list
        List of player names to simulate
    num_games : int, default=500
        Number of games to simulate for each player
    output_dir : str, optional
        Base directory for outputs. If None, will use timestamped folder
    verbose : bool, default=True
        Whether to print progress information
    
    Returns:
    --------
    DataFrame
        Combined simulated game data for all players
    """
    # Set up output directory with timestamp if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"run_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_simulated_games = []
    
    for i, player_name in enumerate(player_list):
        if verbose:
            print(f"\n[{i+1}/{len(player_list)}] Processing {player_name}...")
        
        try:
            simulated_games = run_player_simulation(
                player_name, 
                num_games=num_games, 
                output_dir=output_dir,
                verbose=verbose
            )
            
            all_simulated_games.append(simulated_games)
            
            if verbose:
                print(f"Successfully simulated {player_name}")
        except Exception as e:
            if verbose:
                print(f"Error simulating {player_name}: {str(e)}")
    
    # Combine all simulations into a single DataFrame
    if all_simulated_games:
        combined_df = pd.concat(all_simulated_games, ignore_index=True)
        combined_output_file = os.path.join(output_dir, "all_player_simulations.csv")
        combined_df.to_csv(combined_output_file, index=False)
        
        if verbose:
            print(f"\nCombined simulations saved to {combined_output_file}")
        
        return combined_df
    else:
        if verbose:
            print("\nNo successful simulations to combine.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run slim Bayesian player simulations')
    parser.add_argument('--player', '-p', type=str, help='Player name to simulate (use comma for multiple players)')
    parser.add_argument('--games', '-g', type=int, default=500, help='Number of games to simulate')
    parser.add_argument('--output', '-o', type=str, help='Output directory (default: timestamped folder)')
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
                output_dir=args.output,
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