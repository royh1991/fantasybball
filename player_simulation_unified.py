import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from bayesian_player_simulator import BayesianPlayerSimulator

class PureBayesianSimulator(BayesianPlayerSimulator):
    """
    A simulator that strictly adheres to Bayesian principles, using only
    posterior distributions without additional heuristics or categorizations.
    """
    
    def simulate_game(self, context=None, game_type=None, player_id=None):
        """
        Override the simulate_game method to use pure posterior sampling
        
        Parameters:
        -----------
        context : dict, optional
            Context information for simulation
        game_type : str, optional
            Type of game to simulate
        player_id : str, optional
            Player ID for simulation
            
        Returns:
        --------
        dict
            Simulated box score stats
        """
        # Let parent class handle the simulation mechanics
        return super().simulate_game(context=context, game_type=game_type, player_id=player_id)
        
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
    
    # 2. REB vs Minutes
    ax = axes[1]
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
    
    # 4. FG% vs FT%
    ax = axes[3]
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
    
    # 3. Save all percentile data
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
    
    # Print overall stats from posterior means
    print("\nPosterior distribution statistics:")
    for key, model in simulator.stat_models.items():
        if 'posterior_mean' in model:
            print(f"{key}: {model['posterior_mean']:.3f}")

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

def run_player_simulation(player_name, num_games=100, output_dir=None, verbose=True, slim=False):
    """
    Run a Bayesian simulation for a specific player.
    
    Parameters:
    -----------
    player_name : str
        Name of the player to simulate
    num_games : int, default=100
        Number of games to simulate
    output_dir : str, optional
        Base directory for outputs. If None, will use timestamped folder or player name
    verbose : bool, default=True
        Whether to print progress information
    slim : bool, default=False
        If True, only generates simulation data without visualizations
    
    Returns:
    --------
    tuple or DataFrame
        If slim=False: (simulator, simulated_games, historical_data, summary_stats)
        If slim=True: simulated_games DataFrame only
    """
    # Set up output directory based on mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        if slim:
            base_dir = "slim_simulations"
            output_dir = os.path.join(base_dir, f"{player_name.replace(' ', '_').lower()}_{timestamp}")
        else:
            base_dir = "full_simulations"
            output_dir = os.path.join(base_dir, f"{player_name.replace(' ', '_').lower()}_{timestamp}")
        
        # Create base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize simulator using pure Bayesian method
    if verbose:
        print(f"Initializing Bayesian simulator for {player_name}...")
    
    simulator = PureBayesianSimulator(player_id=player_name, verbose=verbose)
    
    # Simulate games
    if verbose:
        print(f"Simulating {num_games} games for {player_name}...")
    
    simulated_games = simulator.simulate_games(
        n_games=num_games, 
        output_file=os.path.join(output_dir, f"{player_name.replace(' ', '_').lower()}_games.csv")
    )
    
    # Add player name column for easy identification in batch mode
    simulated_games['player_name'] = player_name
    
    # If in slim mode, just return the simulated games
    if slim:
        if verbose:
            print(f"Simulation complete. Results saved to {output_dir}")
        return simulated_games
    
    # Otherwise continue with comparisons and visualizations
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

def batch_simulate_players(player_list, num_games=100, output_dir=None, verbose=True, slim=False):
    """
    Run batch simulations for multiple players.
    
    Parameters:
    -----------
    player_list : list
        List of player names to simulate
    num_games : int, default=100
        Number of games to simulate for each player
    output_dir : str, optional
        Base directory for outputs
    verbose : bool, default=True
        Whether to print progress information
    slim : bool, default=False
        If True, only generates simulation data without visualizations
    
    Returns:
    --------
    dict or DataFrame
        If slim=False: Dictionary mapping player names to their simulation results
        If slim=True: Combined DataFrame with all players' simulated games
    """
    # Set up output directory based on mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        if slim:
            base_dir = "slim_simulations"
            output_dir = os.path.join(base_dir, f"batch_{timestamp}")
        else:
            base_dir = "full_simulations"
            output_dir = os.path.join(base_dir, f"batch_{timestamp}")
        
        # Create base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {}
    all_simulated_games = []
    
    for i, player_name in enumerate(player_list):
        if verbose:
            print(f"\n[{i+1}/{len(player_list)}] Processing {player_name}...")
        
        try:
            if slim:
                player_dir = output_dir  # All in same directory for slim mode
                simulated_games = run_player_simulation(
                    player_name, 
                    num_games=num_games, 
                    output_dir=player_dir,
                    verbose=verbose,
                    slim=True
                )
                all_simulated_games.append(simulated_games)
            else:
                player_dir = os.path.join(output_dir, player_name.replace(' ', '_').lower())
                simulator, simulated_games, hist_df, summary_stats = run_player_simulation(
                    player_name, 
                    num_games=num_games, 
                    output_dir=player_dir,
                    verbose=verbose,
                    slim=False
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
    
    if slim:
        # Combine all simulations into a single DataFrame for slim mode
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
    else:
        # Create comparative analysis across players for full mode
        if verbose and len(results) > 1:
            print("\nCreating cross-player analysis...")
        
        if len(results) > 1:
            create_cross_player_analysis(results, output_dir)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bayesian player simulations')
    parser.add_argument('--player', '-p', type=str, help='Player name to simulate (use comma for multiple players)')
    parser.add_argument('--games', '-g', type=int, default=100, help='Number of games to simulate')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Run in quiet mode')
    parser.add_argument('--slim', '-s', action='store_true', help='Run in slim mode (only generate data, no visualizations)')
    
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
                verbose=not args.quiet,
                slim=args.slim
            )
        else:
            # Batch mode for specified players
            batch_simulate_players(
                players,
                num_games=args.games,
                output_dir=args.output,
                verbose=not args.quiet,
                slim=args.slim
            )
    else:
        # Example players if none specified
        example_players = ["LeBron James", "Stephen Curry", "Kevin Durant"]
        print(f"No player specified. Running example simulation with: {', '.join(example_players)}")
        
        batch_simulate_players(
            example_players,
            num_games=args.games,
            output_dir=args.output,
            verbose=not args.quiet,
            slim=args.slim
        )