"""
Generate Individual Player Validation Reports

Creates detailed visualizations and metrics for specific players.
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


def load_and_filter_data(csv_path: str, player_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load player data and split into training and validation sets."""
    df = pd.read_csv(csv_path)
    player_data = df[df['PLAYER_NAME'] == player_name].copy()

    if len(player_data) == 0:
        return None, None

    player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
    player_data = player_data.dropna(subset=['parsed_date'])
    player_data = player_data.sort_values('parsed_date')

    cutoff_date = datetime(2024, 10, 1)
    training_data = player_data[player_data['parsed_date'] < cutoff_date].copy()
    validation_data = player_data[player_data['parsed_date'] >= cutoff_date].copy()

    return training_data, validation_data


class SimpleBayesianModel:
    """Simplified Bayesian model using Poisson/Negative Binomial distributions."""

    def __init__(self, stats_to_model: List[str]):
        self.stats = stats_to_model
        self.distributions = {}

    def fit(self, training_data: pd.DataFrame):
        """Fit distributions for each stat."""
        for stat in self.stats:
            if stat not in training_data.columns:
                continue

            values = training_data[stat].values
            mean_val = np.mean(values)
            var_val = np.var(values)
            cv = np.sqrt(var_val) / mean_val if mean_val > 0 else 0

            if cv > 1.5 and var_val > mean_val:
                dist_type = 'negbin'
                p = 1 - mean_val / var_val
                n = mean_val * (1 - p) / p if p > 0 and p < 1 else mean_val
            else:
                dist_type = 'poisson'
                n = mean_val
                p = None

            self.distributions[stat] = {
                'type': dist_type,
                'mean': mean_val,
                'var': var_val,
                'cv': cv,
                'n': n,
                'p': p
            }

    def simulate_games(self, n_simulations: int = 200) -> pd.DataFrame:
        """Simulate n games for the player."""
        simulated_games = {stat: [] for stat in self.stats}

        for _ in range(n_simulations):
            for stat in self.stats:
                dist_info = self.distributions[stat]

                if dist_info['type'] == 'negbin' and dist_info['p'] is not None:
                    n, p = dist_info['n'], dist_info['p']
                    if n > 0 and 0 < p < 1:
                        value = np.random.negative_binomial(n, 1-p)
                    else:
                        value = np.random.poisson(dist_info['mean'])
                else:
                    value = np.random.poisson(dist_info['mean'])

                simulated_games[stat].append(max(0, value))

        return pd.DataFrame(simulated_games)

    def get_summary_stats(self, simulations: pd.DataFrame) -> Dict:
        """Get summary statistics from simulations."""
        summary = {}
        for stat in self.stats:
            summary[stat] = {
                'mean': simulations[stat].mean(),
                'median': simulations[stat].median(),
                'std': simulations[stat].std(),
                'min': simulations[stat].min(),
                'max': simulations[stat].max(),
                'p10': simulations[stat].quantile(0.10),
                'p25': simulations[stat].quantile(0.25),
                'p75': simulations[stat].quantile(0.75),
                'p90': simulations[stat].quantile(0.90),
            }
        return summary


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
            'actual_std': actual_values.std(),
            'sim_std': sim_values.std(),
            'in_80pct_ci': in_ci
        }

    return metrics


def create_visualization(simulations: pd.DataFrame,
                        validation_data: pd.DataFrame,
                        stats: List[str],
                        player_name: str,
                        output_path: str):
    """Create visualization comparing simulated vs actual distributions."""
    print(f"  Creating visualization: {output_path}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        ax = axes[idx]

        sim_values = simulations[stat].values
        actual_values = validation_data[stat].values

        all_values = np.concatenate([sim_values, actual_values])
        bins = np.linspace(all_values.min(), all_values.max(), 25)

        ax.hist(sim_values, bins=bins, alpha=0.5, color='blue',
                label=f'Simulated (n={len(sim_values)})', density=True, edgecolor='black')

        ax.hist(actual_values, bins=bins, alpha=0.5, color='red',
                label=f'Actual 2024-25 (n={len(actual_values)})', density=True, edgecolor='black')

        ax.axvline(sim_values.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Sim Mean: {sim_values.mean():.1f}')
        ax.axvline(actual_values.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Actual Mean: {actual_values.mean():.1f}')

        ax.set_xlabel(stat, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{stat} Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(stats) < len(axes):
        axes[-1].set_visible(False)

    plt.suptitle(f'{player_name}: Simulated vs Actual 2024-25 Performance',
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def generate_player_report(player_name: str, csv_path: str, stats: List[str],
                          n_simulations: int = 200):
    """Generate complete validation report for a player."""
    print(f"\n{'='*70}")
    print(f"GENERATING REPORT: {player_name}")
    print(f"{'='*70}")

    training_data, validation_data = load_and_filter_data(csv_path, player_name)

    if training_data is None or len(validation_data) == 0:
        print(f"ERROR: Insufficient data for {player_name}")
        return

    print(f"Training: {len(training_data)} games | Validation: {len(validation_data)} games")

    # Fit model
    print("Fitting model...")
    model = SimpleBayesianModel(stats)
    model.fit(training_data)

    for stat in stats:
        if stat in model.distributions:
            d = model.distributions[stat]
            print(f"  {stat}: mean={d['mean']:.2f}, var={d['var']:.2f}, cv={d['cv']:.2f}, dist={d['type']}")

    # Run simulations
    print(f"\nSimulating {n_simulations} games...")
    simulations = model.simulate_games(n_simulations)

    # Get summary statistics
    print("\n" + "="*70)
    print("SIMULATION SUMMARY STATISTICS")
    print("="*70)
    summary = model.get_summary_stats(simulations)
    for stat in stats:
        print(f"\n{stat}:")
        print(f"  Mean: {summary[stat]['mean']:.2f}")
        print(f"  Median: {summary[stat]['median']:.1f}")
        print(f"  Std: {summary[stat]['std']:.2f}")
        print(f"  Range: [{summary[stat]['min']:.0f}, {summary[stat]['max']:.0f}]")
        print(f"  80% CI: [{summary[stat]['p10']:.1f}, {summary[stat]['p90']:.1f}]")

    # Calculate validation metrics
    print("\n" + "="*70)
    print("VALIDATION METRICS (vs 2024-25 Season)")
    print("="*70)
    metrics = calculate_validation_metrics(simulations, validation_data, stats)

    for stat in stats:
        if stat in metrics:
            m = metrics[stat]
            print(f"\n{stat}:")
            print(f"  Actual Mean: {m['actual_mean']:.2f}")
            print(f"  Simulated Mean: {m['sim_mean']:.2f}")
            print(f"  Mean Absolute Error: {m['mae']:.2f}")
            print(f"  Actual Std: {m['actual_std']:.2f}")
            print(f"  Simulated Std: {m['sim_std']:.2f}")
            print(f"  Distribution Similarity: {m['dist_similarity']:.2%}")
            print(f"  Actual mean in 80% CI: {'Yes' if m['in_80pct_ci'] else 'No'}")

    # Create visualization
    safe_name = player_name.replace(' ', '_').replace('-', '_').lower()
    viz_path = f'{safe_name}_validation_viz.png'
    create_visualization(simulations, validation_data, stats, player_name, viz_path)

    print("\n" + "="*70)
    print(f"REPORT COMPLETE: {viz_path}")
    print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("INDIVIDUAL PLAYER VALIDATION REPORTS")
    print("="*70)

    # Configuration
    players = [
        "Shai Gilgeous-Alexander",
        "Anthony Edwards",
        "Mikal Bridges",
        "Pascal Siakam"
    ]

    stats_to_model = ['FGM', 'REB', 'FTA', 'AST', 'FG3A']
    n_simulations = 200

    csv_path = '/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_20251023_175056.csv'

    # Generate report for each player
    for player in players:
        generate_player_report(player, csv_path, stats_to_model, n_simulations)

    print("\n" + "="*70)
    print("ALL REPORTS COMPLETE!")
    print("="*70)
    print(f"Generated {len(players)} individual validation reports")


if __name__ == "__main__":
    main()
