"""
Kevin Durant FG% Validation

Tests if the NEW correlated sampling approach produces realistic FG% distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Import the new correlated sampling model
sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel


def parse_date(date_str: str) -> datetime:
    """Parse dates in format 'OCT 25, 2023' to datetime."""
    try:
        return datetime.strptime(date_str.strip(), "%b %d, %Y")
    except:
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except:
            return None


def create_fg_pct_validation_viz(player_name: str,
                                  sim_fgm, sim_fga, sim_fg_pct,
                                  actual_fgm, actual_fga, actual_fg_pct):
    """Create comprehensive FG% validation visualization."""

    # Check if we have actual data
    has_actual = len(actual_fgm) > 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Component stats (FGM, FGA)
    # FGM distribution
    ax = axes[0, 0]
    if has_actual:
        bins_fgm = np.linspace(
            min(min(sim_fgm), min(actual_fgm)),
            max(max(sim_fgm), max(actual_fgm)),
            20
        )
    else:
        bins_fgm = np.linspace(min(sim_fgm), max(sim_fgm), 20)

    ax.hist(sim_fgm, bins=bins_fgm, alpha=0.5, color='blue',
            label=f'Simulated (μ={np.mean(sim_fgm):.1f})', density=True, edgecolor='black')
    if has_actual:
        ax.hist(actual_fgm, bins=bins_fgm, alpha=0.5, color='red',
                label=f'Actual (μ={np.mean(actual_fgm):.1f})', density=True, edgecolor='black')
    ax.axvline(np.mean(sim_fgm), color='blue', linestyle='--', linewidth=2)
    if has_actual:
        ax.axvline(np.mean(actual_fgm), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Field Goals Made', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('FGM Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # FGA distribution
    ax = axes[0, 1]
    if has_actual:
        bins_fga = np.linspace(
            min(min(sim_fga), min(actual_fga)),
            max(max(sim_fga), max(actual_fga)),
            20
        )
    else:
        bins_fga = np.linspace(min(sim_fga), max(sim_fga), 20)

    ax.hist(sim_fga, bins=bins_fga, alpha=0.5, color='blue',
            label=f'Simulated (μ={np.mean(sim_fga):.1f})', density=True, edgecolor='black')
    if has_actual:
        ax.hist(actual_fga, bins=bins_fga, alpha=0.5, color='red',
                label=f'Actual (μ={np.mean(actual_fga):.1f})', density=True, edgecolor='black')
    ax.axvline(np.mean(sim_fga), color='blue', linestyle='--', linewidth=2)
    if has_actual:
        ax.axvline(np.mean(actual_fga), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Field Goal Attempts', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('FGA Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # FG% distribution (main result)
    ax = axes[0, 2]
    if has_actual:
        bins_pct = np.linspace(
            min(min(sim_fg_pct), min(actual_fg_pct)),
            max(max(sim_fg_pct), max(actual_fg_pct)),
            25
        )
    else:
        bins_pct = np.linspace(min(sim_fg_pct), max(sim_fg_pct), 25)

    ax.hist(sim_fg_pct, bins=bins_pct, alpha=0.5, color='blue',
            label=f'Simulated (μ={np.mean(sim_fg_pct):.3f})', density=True, edgecolor='black')
    if has_actual:
        ax.hist(actual_fg_pct, bins=bins_pct, alpha=0.5, color='red',
                label=f'Actual (μ={np.mean(actual_fg_pct):.3f})', density=True, edgecolor='black')
    ax.axvline(np.mean(sim_fg_pct), color='blue', linestyle='--', linewidth=2)
    if has_actual:
        ax.axvline(np.mean(actual_fg_pct), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Field Goal Percentage', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('FG% Distribution (KEY METRIC)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Scatter plots and stats
    # FGM vs FGA scatter (simulated)
    ax = axes[1, 0]
    ax.scatter(sim_fga, sim_fgm, alpha=0.3, color='blue', s=30, label='Simulated')
    # Add constraint line (FGM = FGA)
    max_val = max(max(sim_fga), max(sim_fgm))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='FGM = FGA (limit)')
    ax.set_xlabel('FGA', fontsize=11)
    ax.set_ylabel('FGM', fontsize=11)
    ax.set_title('Simulated: FGM vs FGA', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # FGM vs FGA scatter (actual)
    ax = axes[1, 1]
    if has_actual:
        ax.scatter(actual_fga, actual_fgm, alpha=0.5, color='red', s=30, label='Actual')
        max_val = max(max(actual_fga), max(actual_fgm))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='FGM = FGA (limit)')
        ax.set_xlabel('FGA', fontsize=11)
        ax.set_ylabel('FGM', fontsize=11)
        ax.set_title('Actual: FGM vs FGA', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No validation data available\n(Player has <10 2024-25 games)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Actual Data Not Available', fontsize=12, fontweight='bold')
        ax.axis('off')

    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    if has_actual:
        summary_text = f"""
    VALIDATION SUMMARY
    ==================

    Sample Sizes:
    • Simulated: {len(sim_fg_pct)} games
    • Actual: {len(actual_fg_pct)} games

    FG% Statistics:
    • Simulated Mean: {np.mean(sim_fg_pct):.3f}
    • Actual Mean: {np.mean(actual_fg_pct):.3f}
    • Difference: {abs(np.mean(sim_fg_pct) - np.mean(actual_fg_pct)):.3f}

    • Simulated Std: {np.std(sim_fg_pct):.3f}
    • Actual Std: {np.std(actual_fg_pct):.3f}

    FGM Statistics:
    • Simulated Mean: {np.mean(sim_fgm):.2f}
    • Actual Mean: {np.mean(actual_fgm):.2f}

    FGA Statistics:
    • Simulated Mean: {np.mean(sim_fga):.2f}
    • Actual Mean: {np.mean(actual_fga):.2f}

    Methodology (NEW):
    • FGA ~ Poisson(λ)
    • FGM ~ Binomial(FGA, FG%)
    • FG% = FGM / FGA (realistic!)
        """
    else:
        summary_text = f"""
    SIMULATION SUMMARY
    ==================

    Sample Size:
    • Simulated: {len(sim_fg_pct)} games

    FG% Distribution:
    • Mean: {np.mean(sim_fg_pct):.3f}
    • Std: {np.std(sim_fg_pct):.3f}
    • Min: {np.min(sim_fg_pct):.3f}
    • Max: {np.max(sim_fg_pct):.3f}

    FGM Distribution:
    • Mean: {np.mean(sim_fgm):.2f}
    • Std: {np.std(sim_fgm):.2f}

    FGA Distribution:
    • Mean: {np.mean(sim_fga):.2f}
    • Std: {np.std(sim_fga):.2f}

    Methodology (NEW):
    • FGA ~ Poisson(λ)
    • FGM ~ Binomial(FGA, FG%)
    • FG% = FGM / FGA (realistic!)

    Note: No validation data
    (insufficient 2024-25 games)
        """

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.suptitle(f'{player_name}: FG% Validation (NEW: Correlated Binomial Sampling)',
                 fontsize=14, fontweight='bold', y=0.995, color='darkgreen')
    plt.tight_layout()
    plt.savefig('kd_fg_pct_validation.png', dpi=300, bbox_inches='tight')
    print("Saved: kd_fg_pct_validation.png")


def main():
    player_name = "Kevin Durant"

    print("="*70)
    print(f"{player_name.upper()} FG% VALIDATION (NEW CORRELATED SAMPLING)")
    print("Testing if correlated sampling produces realistic FG% distributions")
    print("="*70)

    # Load data
    csv_path = '/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv'
    df = pd.read_csv(csv_path)

    # Initialize new model with correlated sampling
    model = FantasyProjectionModel(evolution_rate=0.5)

    print(f"\nFitting model for {player_name}...")
    success = model.fit_player(df, player_name)

    if not success:
        print(f"ERROR: Could not fit model for {player_name}")
        return

    if 'FGA' in model.distributions:
        print(f"  FGA posterior mean: {model.distributions['FGA']['posterior_mean']:.2f}")
    if 'FG_PCT' in model.percentages:
        print(f"  FG% posterior mean: {model.percentages['FG_PCT']['posterior_mean']:.3f}")

    print(f"  Historical FG% (training): {df[df['PLAYER_NAME']==player_name]['FG_PCT'].mean():.3f}")

    # Simulate games using NEW correlated approach
    print("\nSimulating 200 games with CORRELATED sampling...")
    sim_fgm = []
    sim_fga = []
    sim_fg_pct = []

    for _ in range(200):
        game = model.simulate_game()
        sim_fgm.append(game['FGM'])
        sim_fga.append(game['FGA'])
        if game['FGA'] > 0:
            sim_fg_pct.append(game['FGM'] / game['FGA'])
        else:
            sim_fg_pct.append(0)

    # Get actual stats from historical data
    print("Getting actual historical data...")
    player_data = df[df['PLAYER_NAME'] == player_name].copy()
    player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
    player_data = player_data.dropna(subset=['parsed_date'])

    actual_fgm = player_data['FGM'].values
    actual_fga = player_data['FGA'].values
    actual_fg_pct = player_data['FG_PCT'].values
    actual_fg_pct = actual_fg_pct[~np.isnan(actual_fg_pct)]

    print(f"\nValidation set: {len(actual_fg_pct)} games")

    # Print comparison
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Simulated FG%: {np.mean(sim_fg_pct):.3f} (±{np.std(sim_fg_pct):.3f})")
    print(f"Actual FG%:    {np.mean(actual_fg_pct):.3f} (±{np.std(actual_fg_pct):.3f})")
    print(f"Difference:    {abs(np.mean(sim_fg_pct) - np.mean(actual_fg_pct)):.3f}")
    print()
    print(f"Simulated FGM: {np.mean(sim_fgm):.2f}")
    print(f"Actual FGM:    {np.mean(actual_fgm):.2f}")
    print()
    print(f"Simulated FGA: {np.mean(sim_fga):.2f}")
    print(f"Actual FGA:    {np.mean(actual_fga):.2f}")

    # Create visualization
    print("\nCreating visualization...")
    create_fg_pct_validation_viz(
        player_name,
        sim_fgm, sim_fga, sim_fg_pct,
        actual_fgm, actual_fga, actual_fg_pct
    )

    print("\n" + "="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
