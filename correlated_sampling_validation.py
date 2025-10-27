"""
Compare Independent vs Correlated Sampling for FG%

Shows that correlated sampling (Binomial) produces more realistic FG% variance
compared to independent Poisson sampling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel

def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str.strip(), "%b %d, %Y")
    except:
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except:
            return None


class IndependentSamplingModel:
    """Old approach: independently sample FGM and FGA from Poisson."""

    def __init__(self):
        self.distributions = {}

    def fit_player(self, historical_data: pd.DataFrame, player_name: str):
        player_data = historical_data[historical_data['PLAYER_NAME'] == player_name].copy()
        if len(player_data) == 0:
            return False

        player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
        player_data = player_data.dropna(subset=['parsed_date'])

        for stat in ['FGM', 'FGA']:
            values = player_data[stat].values
            self.distributions[stat] = {
                'posterior_mean': np.mean(values)
            }
        return True

    def simulate_game(self):
        fgm = np.random.poisson(max(0, self.distributions['FGM']['posterior_mean']))
        fga = np.random.poisson(max(0, self.distributions['FGA']['posterior_mean']))
        fga = max(1, fga)
        fgm = min(fgm, fga)  # Enforce constraint
        return {'FGM': fgm, 'FGA': fga, 'FG_PCT': fgm / fga}


def main():
    player_name = "Kevin Durant"

    print("="*70)
    print("COMPARING INDEPENDENT VS CORRELATED SAMPLING")
    print(f"Player: {player_name}")
    print("="*70)

    # Load data
    csv_path = '/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv'
    df = pd.read_csv(csv_path)
    player_data = df[df['PLAYER_NAME'] == player_name].copy()

    # Get actual FG% distribution
    player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
    player_data = player_data.dropna(subset=['parsed_date'])
    actual_fg_pct = player_data['FG_PCT'].values
    actual_fg_pct = actual_fg_pct[~np.isnan(actual_fg_pct)]

    print(f"\nActual games: {len(actual_fg_pct)}")
    print(f"Actual FG% mean: {np.mean(actual_fg_pct):.3f}")
    print(f"Actual FG% std: {np.std(actual_fg_pct):.3f}")

    # Method 1: Independent sampling (old)
    print("\n" + "-"*70)
    print("METHOD 1: INDEPENDENT SAMPLING (OLD)")
    print("-"*70)

    old_model = IndependentSamplingModel()
    old_model.fit_player(df, player_name)

    print(f"FGM mean: {old_model.distributions['FGM']['posterior_mean']:.2f}")
    print(f"FGA mean: {old_model.distributions['FGA']['posterior_mean']:.2f}")
    print(f"Implied FG%: {old_model.distributions['FGM']['posterior_mean'] / old_model.distributions['FGA']['posterior_mean']:.3f}")

    old_sims = [old_model.simulate_game()['FG_PCT'] for _ in range(200)]
    print(f"\nSimulated FG% mean: {np.mean(old_sims):.3f}")
    print(f"Simulated FG% std: {np.std(old_sims):.3f}")
    print(f"Variance ratio (sim/actual): {np.std(old_sims) / np.std(actual_fg_pct):.2f}x")

    # Method 2: Correlated sampling (new)
    print("\n" + "-"*70)
    print("METHOD 2: CORRELATED SAMPLING (NEW)")
    print("-"*70)

    new_model = FantasyProjectionModel()
    new_model.fit_player(df, player_name)

    if 'FG_PCT' in new_model.percentages:
        print(f"FG% mean: {new_model.percentages['FG_PCT']['posterior_mean']:.3f}")
    if 'FGA' in new_model.distributions:
        print(f"FGA mean: {new_model.distributions['FGA']['posterior_mean']:.2f}")

    new_sims = []
    for _ in range(200):
        game = new_model.simulate_game()
        if game['FGA'] > 0:
            new_sims.append(game['FGM'] / game['FGA'])

    print(f"\nSimulated FG% mean: {np.mean(new_sims):.3f}")
    print(f"Simulated FG% std: {np.std(new_sims):.3f}")
    print(f"Variance ratio (sim/actual): {np.std(new_sims) / np.std(actual_fg_pct):.2f}x")

    # Create comparison visualization
    print("\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    bins = np.linspace(0, 1, 30)

    # Actual distribution
    ax = axes[0]
    ax.hist(actual_fg_pct, bins=bins, alpha=0.7, color='red', density=True, edgecolor='black')
    ax.axvline(np.mean(actual_fg_pct), color='darkred', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(actual_fg_pct):.3f}')
    ax.axvline(np.mean(actual_fg_pct) - np.std(actual_fg_pct), color='darkred', linestyle=':', linewidth=1.5)
    ax.axvline(np.mean(actual_fg_pct) + np.std(actual_fg_pct), color='darkred', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Field Goal %', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'ACTUAL FG%\n(n={len(actual_fg_pct)} games)\nStd: {np.std(actual_fg_pct):.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Independent sampling
    ax = axes[1]
    ax.hist(old_sims, bins=bins, alpha=0.7, color='orange', density=True, edgecolor='black')
    ax.axvline(np.mean(old_sims), color='darkorange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(old_sims):.3f}')
    ax.axvline(np.mean(old_sims) - np.std(old_sims), color='darkorange', linestyle=':', linewidth=1.5)
    ax.axvline(np.mean(old_sims) + np.std(old_sims), color='darkorange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Field Goal %', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'INDEPENDENT SAMPLING (OLD)\n(200 simulations)\nStd: {np.std(old_sims):.3f} ({np.std(old_sims)/np.std(actual_fg_pct):.2f}x actual)',
                 fontsize=13, fontweight='bold', color='darkred')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Correlated sampling
    ax = axes[2]
    ax.hist(new_sims, bins=bins, alpha=0.7, color='green', density=True, edgecolor='black')
    ax.axvline(np.mean(new_sims), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(new_sims):.3f}')
    ax.axvline(np.mean(new_sims) - np.std(new_sims), color='darkgreen', linestyle=':', linewidth=1.5)
    ax.axvline(np.mean(new_sims) + np.std(new_sims), color='darkgreen', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Field Goal %', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'CORRELATED SAMPLING (NEW)\n(200 simulations)\nStd: {np.std(new_sims):.3f} ({np.std(new_sims)/np.std(actual_fg_pct):.2f}x actual)',
                 fontsize=13, fontweight='bold', color='darkgreen')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.suptitle(f'{player_name}: Independent vs Correlated Sampling Comparison',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sampling_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: sampling_comparison.png")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nActual FG% Std:         {np.std(actual_fg_pct):.3f}")
    print(f"Independent Sampling:   {np.std(old_sims):.3f} ({np.std(old_sims)/np.std(actual_fg_pct):.2f}x) ❌ TOO HIGH")
    print(f"Correlated Sampling:    {np.std(new_sims):.3f} ({np.std(new_sims)/np.std(actual_fg_pct):.2f}x) ✓ MUCH BETTER")

    improvement = (np.std(old_sims) - np.std(new_sims)) / np.std(old_sims) * 100
    print(f"\nVariance reduction: {improvement:.1f}%")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
