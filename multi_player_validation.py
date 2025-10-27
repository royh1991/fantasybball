"""
Multi-Player Fantasy Basketball Model Validation

Validates Adaptive Bayesian model across 10 diverse players from all-stars to regular starters.
Uses first 10 games of 2024-25 to adaptively update beliefs, then validates on remaining games.
Stats: FGM, REB, FTA, AST, FG3A
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats as scipy_stats
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


def load_and_filter_data(csv_path: str, player_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load player data and split into training, updating, and validation sets."""
    df = pd.read_csv(csv_path)
    player_data = df[df['PLAYER_NAME'] == player_name].copy()

    if len(player_data) == 0:
        return None, None, None

    player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
    player_data = player_data.dropna(subset=['parsed_date'])
    player_data = player_data.sort_values('parsed_date')

    cutoff_date = datetime(2024, 10, 1)
    training_data = player_data[player_data['parsed_date'] < cutoff_date].copy()
    season_2024 = player_data[player_data['parsed_date'] >= cutoff_date].copy()

    # Split 2024-25 into update (first 10) and validation (rest)
    n_update_games = min(10, len(season_2024))
    update_data = season_2024.head(n_update_games).copy()
    validation_data = season_2024.iloc[n_update_games:].copy()

    return training_data, update_data, validation_data


def adaptive_bayesian_update(prev_mean, prev_var, new_obs, obs_var, evolution_var):
    """Kalman-style Bayesian update for evolving stat rates."""
    K = (prev_var + evolution_var) / (prev_var + evolution_var + obs_var)
    new_mean = prev_mean + K * (new_obs - prev_mean)
    new_var = (1 - K) * (prev_var + evolution_var)
    return new_mean, new_var


class AdaptiveBayesianModel:
    """Bayesian model with adaptive updating based on recent games."""

    def __init__(self, stats_to_model: List[str], evolution_rate: float = 0.5):
        self.stats = stats_to_model
        self.evolution_rate = evolution_rate
        self.distributions = {}

    def fit_prior(self, training_data: pd.DataFrame):
        """Fit prior distribution from historical data."""
        for stat in self.stats:
            if stat not in training_data.columns:
                continue

            values = training_data[stat].values
            mean_val = np.mean(values)
            var_val = np.var(values)
            n_games = len(values)
            initial_uncertainty = var_val / n_games

            self.distributions[stat] = {
                'mean': mean_val,
                'var': var_val,
                'obs_var': var_val,
                'posterior_mean': mean_val,
                'posterior_var': initial_uncertainty
            }

    def adaptive_update(self, update_data: pd.DataFrame):
        """Update beliefs using first N games of new season."""
        for stat in self.stats:
            if stat not in update_data.columns or stat not in self.distributions:
                continue

            posterior_mean = self.distributions[stat]['posterior_mean']
            posterior_var = self.distributions[stat]['posterior_var']
            obs_var = self.distributions[stat]['obs_var']
            evolution_var = self.evolution_rate * obs_var / len(update_data)

            for value in update_data[stat].values:
                posterior_mean, posterior_var = adaptive_bayesian_update(
                    posterior_mean, posterior_var, value, obs_var, evolution_var
                )

            self.distributions[stat]['posterior_mean'] = posterior_mean
            self.distributions[stat]['posterior_var'] = posterior_var

    def simulate_games(self, n_simulations: int = 200, use_posterior: bool = True) -> pd.DataFrame:
        """Simulate games using either prior or posterior distribution."""
        simulated_games = {stat: [] for stat in self.stats}

        for _ in range(n_simulations):
            for stat in self.stats:
                dist_info = self.distributions[stat]
                mean = dist_info['posterior_mean'] if use_posterior else dist_info['mean']
                value = np.random.poisson(mean)
                simulated_games[stat].append(max(0, value))

        return pd.DataFrame(simulated_games)


def calculate_validation_metrics(simulations: pd.DataFrame,
                                 validation_data: pd.DataFrame,
                                 stats: List[str]) -> Dict:
    """Calculate validation metrics."""
    metrics = {}

    for stat in stats:
        if stat not in validation_data.columns:
            continue

        sim_values = simulations[stat].values
        actual_values = validation_data[stat].values

        mae = np.abs(sim_values.mean() - actual_values.mean())

        # Calculate distribution similarity
        bins = np.linspace(
            min(sim_values.min(), actual_values.min()),
            max(sim_values.max(), actual_values.max()),
            20
        )
        sim_hist, _ = np.histogram(sim_values, bins=bins, density=True)
        actual_hist, _ = np.histogram(actual_values, bins=bins, density=True)

        sim_hist = sim_hist + 1e-10
        actual_hist = actual_hist + 1e-10
        sim_hist = sim_hist / sim_hist.sum()
        actual_hist = actual_hist / actual_hist.sum()

        m = 0.5 * (sim_hist + actual_hist)
        js_div = 0.5 * (scipy_stats.entropy(sim_hist, m) + scipy_stats.entropy(actual_hist, m))
        dist_similarity = 1 - js_div

        in_ci = (actual_values.mean() >= np.percentile(sim_values, 10) and
                 actual_values.mean() <= np.percentile(sim_values, 90))

        metrics[stat] = {
            'mae': mae,
            'dist_similarity': dist_similarity,
            'actual_mean': actual_values.mean(),
            'sim_mean': sim_values.mean(),
            'in_80pct_ci': in_ci
        }

    return metrics


def validate_player(player_name: str, csv_path: str, stats: List[str],
                   evolution_rate: float = 0.5, n_simulations: int = 200) -> Dict:
    """Run adaptive Bayesian validation for a single player."""
    print(f"\n{'='*70}")
    print(f"Validating: {player_name}")
    print(f"{'='*70}")

    training_data, update_data, validation_data = load_and_filter_data(csv_path, player_name)

    if training_data is None or len(update_data) == 0 or len(validation_data) == 0:
        print(f"  ERROR: Insufficient data for {player_name}")
        return None

    print(f"  Training: {len(training_data)} | Update: {len(update_data)} | Validation: {len(validation_data)} games")

    # Initialize adaptive model
    model = AdaptiveBayesianModel(stats, evolution_rate=evolution_rate)
    model.fit_prior(training_data)

    # Update with first 10 games
    model.adaptive_update(update_data)

    # Simulate with updated beliefs
    simulations = model.simulate_games(n_simulations, use_posterior=True)
    metrics = calculate_validation_metrics(simulations, validation_data, stats)

    # Print summary
    avg_mae = np.mean([m['mae'] for m in metrics.values()])
    avg_similarity = np.mean([m['dist_similarity'] for m in metrics.values()])
    all_in_ci = all([m['in_80pct_ci'] for m in metrics.values()])

    print(f"  Avg MAE: {avg_mae:.2f} | Avg Similarity: {avg_similarity:.1%} | All in CI: {all_in_ci}")

    return {
        'player_name': player_name,
        'training_games': len(training_data),
        'update_games': len(update_data),
        'validation_games': len(validation_data),
        'metrics': metrics,
        'avg_mae': avg_mae,
        'avg_similarity': avg_similarity,
        'all_in_ci': all_in_ci,
        'simulations': simulations,
        'validation_data': validation_data
    }


def create_summary_visualization(results: List[Dict], stats: List[str],
                                output_path: str = 'multi_player_summary.png'):
    """Create summary visualization across all players."""
    print("\nCreating summary visualization...")

    # Need 2 overview plots + 5 stat plots = 7 total
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    # Plot 1: Average MAE by Player
    ax = axes[0]
    players = [r['player_name'] for r in results]
    avg_maes = [r['avg_mae'] for r in results]
    colors = ['green' if mae < 1.0 else 'orange' if mae < 1.5 else 'red' for mae in avg_maes]
    ax.barh(players, avg_maes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average MAE', fontsize=11)
    ax.set_title('Model Error by Player (Lower is Better)', fontsize=12, fontweight='bold')
    ax.axvline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Average Distribution Similarity by Player
    ax = axes[1]
    avg_sims = [r['avg_similarity'] * 100 for r in results]
    colors = ['green' if sim > 95 else 'orange' if sim > 90 else 'red' for sim in avg_sims]
    ax.barh(players, avg_sims, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Distribution Similarity (%)', fontsize=11)
    ax.set_title('Model Accuracy by Player (Higher is Better)', fontsize=12, fontweight='bold')
    ax.axvline(95, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(80, 100)

    # Plots 3-7: MAE by Stat across all players
    for idx, stat in enumerate(stats):
        ax = axes[idx + 2]
        stat_maes = [r['metrics'][stat]['mae'] if stat in r['metrics'] else 0
                     for r in results]
        colors = ['green' if mae < 0.75 else 'orange' if mae < 1.5 else 'red'
                 for mae in stat_maes]
        ax.barh(players, stat_maes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('MAE', fontsize=10)
        ax.set_title(f'{stat} Prediction Error', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Multi-Player Model Validation Summary',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary visualization saved to: {output_path}")


def create_detailed_table(results: List[Dict], stats: List[str]) -> pd.DataFrame:
    """Create detailed results table."""
    rows = []
    for r in results:
        row = {
            'Player': r['player_name'],
            'Training': r['training_games'],
            'Update': r['update_games'],
            'Validation': r['validation_games'],
            'Avg MAE': f"{r['avg_mae']:.2f}",
            'Avg Similarity': f"{r['avg_similarity']:.1%}",
            'All in CI': 'Yes' if r['all_in_ci'] else 'No'
        }

        # Add per-stat MAE
        for stat in stats:
            if stat in r['metrics']:
                row[f'{stat}_MAE'] = f"{r['metrics'][stat]['mae']:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Main execution function."""
    print("="*70)
    print("MULTI-PLAYER ADAPTIVE BAYESIAN VALIDATION")
    print("="*70)

    # Configuration
    players = [
        "Shai Gilgeous-Alexander",  # All-Star elite scorer
        "Anthony Edwards",           # All-Star athletic wing
        "Trae Young",                # All-Star elite playmaker
        "James Harden",              # All-Star veteran
        "Bam Adebayo",               # All-Star big man
        "Pascal Siakam",             # All-Star forward
        "Derrick White",             # Solid starter 3&D
        "Mikal Bridges",             # Elite 3&D wing
        "Jarrett Allen",             # Starting center
        "Buddy Hield"                # Role player shooter
    ]

    stats_to_model = ['FGM', 'REB', 'FTA', 'AST', 'FG3A']
    evolution_rate = 0.5  # Moderate reactivity to role changes
    n_simulations = 200

    csv_path = '/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_20251023_175056.csv'

    print(f"\nUsing adaptive Bayesian updating:")
    print(f"  - First 10 games of 2024-25 used to update beliefs")
    print(f"  - Remaining games used for validation")
    print(f"  - Evolution rate: {evolution_rate}")

    # Run validation for all players
    results = []
    for player in players:
        result = validate_player(player, csv_path, stats_to_model, evolution_rate, n_simulations)
        if result:
            results.append(result)

    # Create summary table
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)
    summary_df = create_detailed_table(results, stats_to_model)
    print(summary_df.to_string(index=False))

    # Create visualizations
    create_summary_visualization(results, stats_to_model)

    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL MODEL PERFORMANCE")
    print("="*70)
    avg_mae_all = np.mean([r['avg_mae'] for r in results])
    avg_sim_all = np.mean([r['avg_similarity'] for r in results])
    pct_in_ci = sum([r['all_in_ci'] for r in results]) / len(results) * 100

    print(f"Average MAE across all players: {avg_mae_all:.2f}")
    print(f"Average similarity across all players: {avg_sim_all:.1%}")
    print(f"Percentage of players with all stats in 80% CI: {pct_in_ci:.0f}%")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)
    print(f"Total players validated: {len(results)}")
    print(f"Total simulations: {len(results) * n_simulations}")
    print(f"Summary visualization: multi_player_summary.png")


if __name__ == "__main__":
    main()
