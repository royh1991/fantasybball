"""
Adaptive Bayesian Validation using Kalman-style updates

Uses first 10 games of 2024-25 to adaptively update beliefs about each stat's
true rate, then simulates remaining games.
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
    """
    Load player data and split into training, updating, and validation sets.

    Returns:
        training_data: Games before 2024-25 season
        update_data: First 10 games of 2024-25 (for adaptive updating)
        validation_data: Remaining 2024-25 games (for validation)
    """
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
    """
    Kalman-style Bayesian update for evolving stat rates.

    Args:
        prev_mean: Current estimate of stat's true mean
        prev_var: Uncertainty in current estimate
        new_obs: Newly observed value from game
        obs_var: Measurement noise (game-to-game variance)
        evolution_var: How much true rate drifts over time

    Returns:
        new_mean: Updated estimate
        new_var: Updated uncertainty
    """
    # Kalman gain (how much to trust new observation)
    K = (prev_var + evolution_var) / (prev_var + evolution_var + obs_var)

    # Update mean and variance
    new_mean = prev_mean + K * (new_obs - prev_mean)
    new_var = (1 - K) * (prev_var + evolution_var)

    return new_mean, new_var


class AdaptiveBayesianModel:
    """
    Bayesian model with adaptive updating based on recent games.
    """

    def __init__(self, stats_to_model: List[str], evolution_rate: float = 0.5):
        """
        Args:
            stats_to_model: List of stat names to model
            evolution_rate: Controls how quickly we believe true rates change
                           Higher = more reactive to new data
                           Lower = more stable, resistant to noise
        """
        self.stats = stats_to_model
        self.evolution_rate = evolution_rate
        self.distributions = {}

    def fit_prior(self, training_data: pd.DataFrame):
        """Fit prior distribution from historical data."""
        print("Fitting prior from historical data...")

        for stat in self.stats:
            if stat not in training_data.columns:
                continue

            values = training_data[stat].values
            mean_val = np.mean(values)
            var_val = np.var(values)

            # Initial uncertainty based on sample size
            # Larger sample = more confident in estimate
            n_games = len(values)
            initial_uncertainty = var_val / n_games

            self.distributions[stat] = {
                'mean': mean_val,
                'var': var_val,
                'obs_var': var_val,  # Game-to-game variance
                'posterior_mean': mean_val,
                'posterior_var': initial_uncertainty
            }

            print(f"  {stat}: prior_mean={mean_val:.2f}, obs_var={var_val:.2f}")

    def adaptive_update(self, update_data: pd.DataFrame):
        """
        Update beliefs using first N games of new season.

        This is where we learn about role changes, skill development, etc.
        """
        print(f"\nAdaptive updating with {len(update_data)} games...")

        for stat in self.stats:
            if stat not in update_data.columns or stat not in self.distributions:
                continue

            prior_mean = self.distributions[stat]['mean']
            posterior_mean = self.distributions[stat]['posterior_mean']
            posterior_var = self.distributions[stat]['posterior_var']
            obs_var = self.distributions[stat]['obs_var']

            # Evolution variance: how much we expect true rate to drift per game
            evolution_var = self.evolution_rate * obs_var / len(update_data)

            # Sequentially update with each game
            for idx, value in enumerate(update_data[stat].values):
                posterior_mean, posterior_var = adaptive_bayesian_update(
                    posterior_mean, posterior_var, value, obs_var, evolution_var
                )

            # Store updated beliefs
            self.distributions[stat]['posterior_mean'] = posterior_mean
            self.distributions[stat]['posterior_var'] = posterior_var

            change = posterior_mean - prior_mean
            pct_change = (change / prior_mean * 100) if prior_mean > 0 else 0

            print(f"  {stat}: {prior_mean:.2f} → {posterior_mean:.2f} ({pct_change:+.1f}%)")

    def simulate_games(self, n_simulations: int = 200, use_posterior: bool = True) -> pd.DataFrame:
        """
        Simulate games using either prior or posterior distribution.

        Args:
            n_simulations: Number of games to simulate
            use_posterior: If True, use updated beliefs. If False, use prior.
        """
        simulated_games = {stat: [] for stat in self.stats}

        for _ in range(n_simulations):
            for stat in self.stats:
                dist_info = self.distributions[stat]

                if use_posterior:
                    mean = dist_info['posterior_mean']
                else:
                    mean = dist_info['mean']

                # Sample from Poisson distribution
                # (Could use NegBin for overdispersed stats, but keeping simple for now)
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


def create_comparison_viz(player_name: str, stats: List[str],
                          prior_sims: pd.DataFrame, posterior_sims: pd.DataFrame,
                          validation_data: pd.DataFrame, output_path: str):
    """Create visualization comparing prior, posterior, and actual."""
    print(f"  Creating comparison visualization: {output_path}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        ax = axes[idx]

        prior_values = prior_sims[stat].values
        posterior_values = posterior_sims[stat].values
        actual_values = validation_data[stat].values

        all_values = np.concatenate([prior_values, posterior_values, actual_values])
        bins = np.linspace(all_values.min(), all_values.max(), 20)

        # Plot prior (baseline)
        ax.hist(prior_values, bins=bins, alpha=0.3, color='gray',
                label=f'Prior (no update)', density=True, edgecolor='black')

        # Plot posterior (with adaptive updates)
        ax.hist(posterior_values, bins=bins, alpha=0.5, color='blue',
                label=f'Posterior (updated)', density=True, edgecolor='black')

        # Plot actual
        ax.hist(actual_values, bins=bins, alpha=0.5, color='red',
                label=f'Actual (n={len(actual_values)})', density=True, edgecolor='black')

        # Mean lines
        ax.axvline(prior_values.mean(), color='gray', linestyle='--', linewidth=2,
                   label=f'Prior: {prior_values.mean():.1f}')
        ax.axvline(posterior_values.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Posterior: {posterior_values.mean():.1f}')
        ax.axvline(actual_values.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Actual: {actual_values.mean():.1f}')

        ax.set_xlabel(stat, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{stat} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(stats) < len(axes):
        axes[-1].set_visible(False)

    plt.suptitle(f'{player_name}: Adaptive Bayesian Updating (First 10 Games → Predict Rest)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def validate_player(player_name: str, csv_path: str, stats: List[str],
                   evolution_rate: float = 0.5, n_simulations: int = 200):
    """Run adaptive Bayesian validation for a player."""
    print(f"\n{'='*70}")
    print(f"ADAPTIVE BAYESIAN VALIDATION: {player_name}")
    print(f"{'='*70}")

    training_data, update_data, validation_data = load_and_filter_data(csv_path, player_name)

    if training_data is None or len(update_data) == 0 or len(validation_data) == 0:
        print(f"ERROR: Insufficient data for {player_name}")
        return None

    print(f"Training: {len(training_data)} games")
    print(f"Update: {len(update_data)} games (first 10 of 2024-25)")
    print(f"Validation: {len(validation_data)} games (remaining 2024-25)")

    # Initialize model
    model = AdaptiveBayesianModel(stats, evolution_rate=evolution_rate)

    # Fit prior from historical data
    model.fit_prior(training_data)

    # Simulate with prior (baseline - no adaptation)
    print(f"\nSimulating {n_simulations} games with PRIOR (no updates)...")
    prior_sims = model.simulate_games(n_simulations, use_posterior=False)

    # Update with first 10 games
    model.adaptive_update(update_data)

    # Simulate with posterior (after adaptation)
    print(f"\nSimulating {n_simulations} games with POSTERIOR (after updates)...")
    posterior_sims = model.simulate_games(n_simulations, use_posterior=True)

    # Calculate metrics for both
    print("\n" + "="*70)
    print("PRIOR METRICS (No Adaptation)")
    print("="*70)
    prior_metrics = calculate_validation_metrics(prior_sims, validation_data, stats)

    for stat in stats:
        if stat in prior_metrics:
            m = prior_metrics[stat]
            print(f"{stat}: MAE={m['mae']:.2f}, Similarity={m['dist_similarity']:.1%}")

    print("\n" + "="*70)
    print("POSTERIOR METRICS (With Adaptive Updates)")
    print("="*70)
    posterior_metrics = calculate_validation_metrics(posterior_sims, validation_data, stats)

    for stat in stats:
        if stat in posterior_metrics:
            m = posterior_metrics[stat]
            print(f"{stat}: MAE={m['mae']:.2f}, Similarity={m['dist_similarity']:.1%}")

    print("\n" + "="*70)
    print("IMPROVEMENT (Posterior vs Prior)")
    print("="*70)
    for stat in stats:
        if stat in prior_metrics and stat in posterior_metrics:
            prior_mae = prior_metrics[stat]['mae']
            post_mae = posterior_metrics[stat]['mae']
            improvement = prior_mae - post_mae
            pct_improvement = (improvement / prior_mae * 100) if prior_mae > 0 else 0

            symbol = "✓" if improvement > 0 else "✗"
            print(f"{stat}: {prior_mae:.2f} → {post_mae:.2f} ({pct_improvement:+.1f}%) {symbol}")

    # Create visualization
    safe_name = player_name.replace(' ', '_').replace('-', '_').lower()
    viz_path = f'{safe_name}_adaptive_validation.png'
    create_comparison_viz(player_name, stats, prior_sims, posterior_sims,
                         validation_data, viz_path)

    print(f"\nVisualization saved: {viz_path}")


def main():
    """Main execution function."""
    print("="*70)
    print("ADAPTIVE BAYESIAN VALIDATION")
    print("="*70)

    players = [
        "Anthony Edwards",           # Should improve 3PA prediction
        "Shai Gilgeous-Alexander",  # Should improve 3PA prediction
        "Mikal Bridges",             # Should improve FTA prediction
        "Kevin Durant"               # Should NOT overreact to noise
    ]

    stats_to_model = ['FGM', 'REB', 'FTA', 'AST', 'FG3A']
    evolution_rate = 0.5  # Moderate reactivity
    n_simulations = 200

    csv_path = '/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_20251023_175056.csv'

    for player in players:
        validate_player(player, csv_path, stats_to_model, evolution_rate, n_simulations)

    print("\n" + "="*70)
    print("ALL VALIDATIONS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
