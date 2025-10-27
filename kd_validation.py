"""
Kevin Durant Fantasy Basketball Model Validation

Simplified validation focusing on 4 stats: FGM, REB, FTA, AST
Training on pre-2024-25 data, validating on 2024-25 season.
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
        # Try uppercase month format
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except:
            print(f"Could not parse date: {date_str}")
            return None


def load_and_filter_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Kevin Durant data and split into training and validation sets.

    Returns:
        training_data: Games before 2024-25 season (before Oct 2024)
        validation_data: Games from 2024-25 season
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Filter for Kevin Durant
    kd_data = df[df['PLAYER_NAME'] == 'Kevin Durant'].copy()
    print(f"Found {len(kd_data)} Kevin Durant games")

    # Parse dates
    kd_data['parsed_date'] = kd_data['GAME_DATE'].apply(parse_date)
    kd_data = kd_data.dropna(subset=['parsed_date'])

    # Sort by date
    kd_data = kd_data.sort_values('parsed_date')

    # Split: training is before Oct 2024, validation is 2024-25 season
    cutoff_date = datetime(2024, 10, 1)

    training_data = kd_data[kd_data['parsed_date'] < cutoff_date].copy()
    validation_data = kd_data[kd_data['parsed_date'] >= cutoff_date].copy()

    print(f"Training set: {len(training_data)} games (before Oct 2024)")
    print(f"Validation set: {len(validation_data)} games (2024-25 season)")

    return training_data, validation_data


class SimpleBayesianModel:
    """
    Simplified Bayesian model for 4 statistics.
    Uses Poisson/Negative Binomial distributions.
    """

    def __init__(self, stats_to_model: List[str]):
        self.stats = stats_to_model
        self.distributions = {}

    def fit(self, training_data: pd.DataFrame, recency_weight: bool = True):
        """
        Fit distributions for each stat.

        Args:
            training_data: Historical game logs
            recency_weight: Whether to weight recent games more heavily
        """
        print("\nFitting model...")

        for stat in self.stats:
            if stat not in training_data.columns:
                print(f"Warning: {stat} not found in data")
                continue

            values = training_data[stat].values

            # Apply recency weighting if requested
            if recency_weight and len(values) > 10:
                # Use last 10 games with decay factor 0.9
                n_recent = min(10, len(values))
                recent_values = values[-n_recent:]

                # Calculate weights: most recent = 1.0, decay backwards
                weights = 0.9 ** np.arange(n_recent-1, -1, -1)
                weights = weights / weights.sum() * len(recent_values)  # Normalize

                # Weighted statistics
                mean_val = np.average(recent_values, weights=weights)
                # Approximate variance with weights
                var_val = np.average((recent_values - mean_val)**2, weights=weights)
            else:
                mean_val = np.mean(values)
                var_val = np.var(values)

            # Determine if overdispersed (use Negative Binomial vs Poisson)
            cv = np.sqrt(var_val) / mean_val if mean_val > 0 else 0

            if cv > 1.5 and var_val > mean_val:
                # Use Negative Binomial for overdispersed data
                dist_type = 'negbin'
                # For NegBin: mean = n*p/(1-p), variance = n*p/(1-p)^2
                # Solve for n and p
                if var_val > mean_val:
                    p = 1 - mean_val / var_val
                    n = mean_val * (1 - p) / p if p > 0 and p < 1 else mean_val
                else:
                    # Fall back to Poisson
                    dist_type = 'poisson'
                    n, p = mean_val, None
            else:
                # Use Poisson
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

            print(f"  {stat}: mean={mean_val:.2f}, var={var_val:.2f}, cv={cv:.2f}, dist={dist_type}")

    def simulate_games(self, n_simulations: int = 200) -> pd.DataFrame:
        """
        Simulate n games for the player.

        Returns:
            DataFrame with simulated stats
        """
        print(f"\nSimulating {n_simulations} games...")

        simulated_games = {stat: [] for stat in self.stats}

        for _ in range(n_simulations):
            for stat in self.stats:
                dist_info = self.distributions[stat]

                if dist_info['type'] == 'negbin' and dist_info['p'] is not None:
                    # Sample from Negative Binomial
                    n, p = dist_info['n'], dist_info['p']
                    if n > 0 and 0 < p < 1:
                        value = np.random.negative_binomial(n, 1-p)
                    else:
                        value = np.random.poisson(dist_info['mean'])
                else:
                    # Sample from Poisson
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
    """
    Calculate how well simulations match actual validation data.
    """
    metrics = {}

    for stat in stats:
        if stat not in validation_data.columns:
            continue

        sim_values = simulations[stat].values
        actual_values = validation_data[stat].values

        # Mean Absolute Error
        mae = np.abs(sim_values.mean() - actual_values.mean())

        # RMSE (comparing distributions)
        # Create histograms and compare
        bins = np.linspace(
            min(sim_values.min(), actual_values.min()),
            max(sim_values.max(), actual_values.max()),
            20
        )
        sim_hist, _ = np.histogram(sim_values, bins=bins, density=True)
        actual_hist, _ = np.histogram(actual_values, bins=bins, density=True)

        # Distribution similarity (1 - Jensen-Shannon divergence)
        # Add small constant to avoid log(0)
        sim_hist = sim_hist + 1e-10
        actual_hist = actual_hist + 1e-10
        sim_hist = sim_hist / sim_hist.sum()
        actual_hist = actual_hist / actual_hist.sum()

        # KL divergence
        m = 0.5 * (sim_hist + actual_hist)
        js_div = 0.5 * (scipy_stats.entropy(sim_hist, m) + scipy_stats.entropy(actual_hist, m))
        dist_similarity = 1 - js_div

        # Check if actual mean falls within simulation credible interval
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
                        output_path: str = 'kd_validation_viz.png'):
    """
    Create visualization comparing simulated vs actual distributions.
    """
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        ax = axes[idx]

        sim_values = simulations[stat].values
        actual_values = validation_data[stat].values

        # Create bins for histogram
        all_values = np.concatenate([sim_values, actual_values])
        bins = np.linspace(all_values.min(), all_values.max(), 25)

        # Plot simulated distribution
        ax.hist(sim_values, bins=bins, alpha=0.5, color='blue',
                label=f'Simulated (n={len(sim_values)})', density=True, edgecolor='black')

        # Plot actual distribution
        ax.hist(actual_values, bins=bins, alpha=0.5, color='red',
                label=f'Actual 2024-25 (n={len(actual_values)})', density=True, edgecolor='black')

        # Add mean lines
        ax.axvline(sim_values.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Sim Mean: {sim_values.mean():.1f}')
        ax.axvline(actual_values.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Actual Mean: {actual_values.mean():.1f}')

        ax.set_xlabel(stat, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{stat} Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Kevin Durant: Simulated vs Actual 2024-25 Performance',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    return output_path


def main():
    """Main execution function."""
    print("="*70)
    print("KEVIN DURANT FANTASY BASKETBALL MODEL VALIDATION")
    print("="*70)

    # Configuration
    stats_to_model = ['FGM', 'REB', 'FTA', 'AST']
    n_simulations = 200

    # Check which data file to use
    # Try the timestamped file first as it has 2024-25 data
    import os
    data_dir = '/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/'

    csv_files = [
        os.path.join(data_dir, 'historical_gamelogs_20251023_175056.csv'),
        os.path.join(data_dir, 'historical_gamelogs_latest.csv')
    ]

    csv_path = None
    for file in csv_files:
        if os.path.exists(file):
            csv_path = file
            print(f"\nUsing data file: {csv_path}")
            break

    if csv_path is None:
        print("ERROR: Could not find data file!")
        return

    # Load and filter data
    training_data, validation_data = load_and_filter_data(csv_path)

    if len(validation_data) == 0:
        print("\nWARNING: No 2024-25 validation data found in latest file.")
        print("Trying timestamped file...")
        csv_path = csv_files[1]
        if os.path.exists(csv_path):
            print(f"Using: {csv_path}")
            training_data, validation_data = load_and_filter_data(csv_path)

    if len(validation_data) == 0:
        print("ERROR: No validation data found. Cannot proceed.")
        return

    # Build and fit model
    model = SimpleBayesianModel(stats_to_model)
    model.fit(training_data, recency_weight=False)  # Turned off - was overfitting to last 10 games

    # Run simulations
    simulations = model.simulate_games(n_simulations)

    # Get summary statistics
    print("\n" + "="*70)
    print("SIMULATION SUMMARY STATISTICS")
    print("="*70)
    summary = model.get_summary_stats(simulations)
    for stat in stats_to_model:
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
    metrics = calculate_validation_metrics(simulations, validation_data, stats_to_model)

    for stat in stats_to_model:
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
    viz_path = create_visualization(simulations, validation_data, stats_to_model)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)
    print(f"Total simulations: {n_simulations}")
    print(f"Training games: {len(training_data)}")
    print(f"Validation games: {len(validation_data)}")
    print(f"Visualization: {viz_path}")


if __name__ == "__main__":
    main()
