"""
Create Comprehensive Overview Report for All Matchups

Generates visualizations and detailed analysis of all simulated matchups.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def create_overview_visualizations():
    """Create comprehensive overview visualizations."""

    # Load summary data
    summary = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/all_matchups_summary.csv')

    # Parse team names
    summary['home_team'] = summary['matchup'].str.split(' vs ').str[0]
    summary['away_team'] = summary['matchup'].str.split(' vs ').str[1]

    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Win Probability Bar Chart
    ax1 = fig.add_subplot(gs[0, :])

    x = np.arange(len(summary))
    width = 0.35

    bars1 = ax1.barh(x - width/2, summary['team_a_win_pct'] * 100, width,
                     label='Home Team', color='blue', alpha=0.7)
    bars2 = ax1.barh(x + width/2, summary['team_b_win_pct'] * 100, width,
                     label='Away Team', color='red', alpha=0.7)

    ax1.set_xlabel('Win Probability (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Matchup', fontsize=12, fontweight='bold')
    ax1.set_title('Week 6 Matchup Win Probabilities', fontsize=14, fontweight='bold')
    ax1.set_yticks(x)
    ax1.set_yticklabels([f"{row['home_team']} vs\\n{row['away_team']}"
                         for _, row in summary.iterrows()], fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for idx, (bar1, bar2, row) in enumerate(zip(bars1, bars2, summary.iterrows())):
        _, data = row
        ax1.text(bar1.get_width() + 1, bar1.get_y() + bar1.get_height()/2,
                f'{data["team_a_win_pct"]*100:.1f}%',
                ha='left', va='center', fontsize=9, fontweight='bold')
        ax1.text(bar2.get_width() + 1, bar2.get_y() + bar2.get_height()/2,
                f'{data["team_b_win_pct"]*100:.1f}%',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # 2. Average Categories Won
    ax2 = fig.add_subplot(gs[1, 0])

    x = np.arange(len(summary))
    ax2.barh(x - width/2, summary['team_a_avg_cats_won'], width,
            label='Home Team', color='blue', alpha=0.7)
    ax2.barh(x + width/2, summary['team_b_avg_cats_won'], width,
            label='Away Team', color='red', alpha=0.7)
    ax2.axvline(6, color='green', linestyle='--', linewidth=2, label='Needed to Win (6)')

    ax2.set_xlabel('Average Categories Won', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Matchup', fontsize=11, fontweight='bold')
    ax2.set_title('Average Categories Won (out of 11)', fontsize=12, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(range(1, len(summary) + 1), fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    # 3. Player & Game Count Comparison
    ax3 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(summary))
    ax3.scatter(summary['team_a_total_games'], summary['team_b_total_games'],
               s=200, alpha=0.6, c=range(len(summary)), cmap='viridis', edgecolors='black', linewidths=2)

    for idx, row in summary.iterrows():
        ax3.annotate(f"{idx+1}", (row['team_a_total_games'], row['team_b_total_games']),
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Add diagonal line (equal games)
    max_games = max(summary['team_a_total_games'].max(), summary['team_b_total_games'].max())
    ax3.plot([0, max_games], [0, max_games], 'k--', alpha=0.3, linewidth=1)

    ax3.set_xlabel('Home Team Total Games Played', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Away Team Total Games Played', fontsize=11, fontweight='bold')
    ax3.set_title('Game Count Comparison (numbered by matchup)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Competitiveness Analysis
    ax4 = fig.add_subplot(gs[2, :])

    summary['competitiveness'] = 1 - abs(summary['team_a_win_pct'] - summary['team_b_win_pct'])
    summary_sorted = summary.sort_values('competitiveness', ascending=False)

    colors = ['green' if c > 0.6 else 'yellow' if c > 0.3 else 'red'
             for c in summary_sorted['competitiveness']]

    bars = ax4.barh(range(len(summary_sorted)), summary_sorted['competitiveness'] * 100,
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    ax4.set_xlabel('Competitiveness Score (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Matchup', fontsize=12, fontweight='bold')
    ax4.set_title('Matchup Competitiveness (100% = perfectly even, 0% = complete mismatch)',
                 fontsize=13, fontweight='bold')
    ax4.set_yticks(range(len(summary_sorted)))
    ax4.set_yticklabels([f"{row['home_team']} vs {row['away_team']}"
                         for _, row in summary_sorted.iterrows()], fontsize=9)
    ax4.grid(axis='x', alpha=0.3)

    # Add value labels
    for idx, (bar, comp) in enumerate(zip(bars, summary_sorted['competitiveness'])):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{comp*100:.1f}%',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Competitive (>60%)'),
        Patch(facecolor='yellow', alpha=0.7, label='Moderate (30-60%)'),
        Patch(facecolor='red', alpha=0.7, label='Mismatch (<30%)')
    ]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.suptitle('Fantasy Basketball Week 6 - Complete Matchup Overview',
                fontsize=16, fontweight='bold', y=0.995)

    output_path = '/Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/OVERVIEW_REPORT.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_markdown_report():
    """Create detailed markdown report."""

    summary = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/all_matchups_summary.csv')

    # Parse team names
    summary['home_team'] = summary['matchup'].str.split(' vs ').str[0]
    summary['away_team'] = summary['matchup'].str.split(' vs ').str[1]
    summary['competitiveness'] = 1 - abs(summary['team_a_win_pct'] - summary['team_b_win_pct'])

    report = f"""# Week 6 Fantasy Basketball Matchup Simulations

## Executive Summary

This report presents the results of 500 Monte Carlo simulations for each of the 7 Week 6 matchups.
Each simulation uses actual game data and player performance models to predict category winners.

**Simulation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Matchups:** {len(summary)}
**Simulations Per Matchup:** 500
**Categories Evaluated:** 11 (FG%, FT%, 3P%, 3PM, PTS, REB, AST, STL, BLK, TO, DD)

---

## Matchup Results

"""

    for idx, row in summary.iterrows():
        matchup_num = idx + 1
        competitive = row['competitiveness']

        if competitive > 0.6:
            comp_label = "🟢 COMPETITIVE"
            comp_color = "green"
        elif competitive > 0.3:
            comp_label = "🟡 MODERATE"
            comp_color = "yellow"
        else:
            comp_label = "🔴 MISMATCH"
            comp_color = "red"

        report += f"""### {matchup_num}. {row['home_team']} vs {row['away_team']} {comp_label}

| Metric | {row['home_team']} | {row['away_team']} |
|--------|-------------|-------------|
| **Win Probability** | **{row['team_a_win_pct']*100:.1f}%** | **{row['team_b_win_pct']*100:.1f}%** |
| Wins (out of 500) | {row['team_a_wins']} | {row['team_b_wins']} |
| Ties | {row['ties']} | {row['ties']} |
| Avg Categories Won | {row['team_a_avg_cats_won']:.2f} / 11 | {row['team_b_avg_cats_won']:.2f} / 11 |
| Players with Games | {row['team_a_players']} | {row['team_b_players']} |
| Total Games Played | {row['team_a_total_games']} | {row['team_b_total_games']} |

**Competitiveness Score:** {competitive*100:.1f}%

"""

        # Load category summary if available
        matchup_folder = f"{row['home_team']}_vs_{row['away_team']}_week6".replace(' ', '_').replace('.', '')
        category_summary_path = f"/Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/{matchup_folder}/category_summary.json"

        if os.path.exists(category_summary_path):
            with open(category_summary_path, 'r') as f:
                cat_summary = json.load(f)

            report += f"""**Category Breakdown:**

| Category | {row['home_team']} Win % | {row['away_team']} Win % | Ties % |
|----------|-------------|-------------|---------|
"""

            for cat, data in cat_summary.items():
                report += f"| {cat} | {data['team_a_win_pct']*100:.1f}% | {data['team_b_win_pct']*100:.1f}% | {data['ties']/500*100:.1f}% |\n"

        report += "\n---\n\n"

    # Add statistical summary
    report += f"""## Statistical Summary

### Competitiveness Distribution
- **Competitive Matchups** (>60%): {sum(summary['competitiveness'] > 0.6)} matchups
- **Moderate Matchups** (30-60%): {sum((summary['competitiveness'] > 0.3) & (summary['competitiveness'] <= 0.6))} matchups
- **Mismatches** (<30%): {sum(summary['competitiveness'] <= 0.3)} matchups

### Average Statistics
- **Mean Win Probability Spread:** {(abs(summary['team_a_win_pct'] - summary['team_b_win_pct']).mean() * 100):.1f}%
- **Average Categories Won (Home):** {summary['team_a_avg_cats_won'].mean():.2f}
- **Average Categories Won (Away):** {summary['team_b_avg_cats_won'].mean():.2f}
- **Average Games Per Team:** {((summary['team_a_total_games'] + summary['team_b_total_games']) / 2).mean():.1f}

### Most Competitive Matchup
"""
    most_comp = summary.loc[summary['competitiveness'].idxmax()]
    report += f"**{most_comp['home_team']} vs {most_comp['away_team']}** ({most_comp['competitiveness']*100:.1f}% competitive)\n\n"

    report += f"""### Biggest Mismatch
"""
    least_comp = summary.loc[summary['competitiveness'].idxmin()]
    report += f"**{least_comp['home_team']} vs {least_comp['away_team']}** ({least_comp['competitiveness']*100:.1f}% competitive)\n\n"

    report += f"""---

## Methodology

### Simulation Approach
1. **Data Source:** Actual games played during Week 6 for each player
2. **Player Models:** Bayesian projection models fitted on historical data (2019-2025)
3. **Simulations:** 500 Monte Carlo simulations per matchup
4. **Categories:** 11 standard fantasy basketball categories

### Model Details
- **Shooting Stats:** Correlated binomial sampling (FGM|FGA, FTM|FTA, 3PM|3PA)
- **Counting Stats:** Poisson distribution sampling (PTS, REB, AST, STL, BLK, TOV, DD)
- **Category Winners:** Direct comparison of aggregated team totals
- **Matchup Winner:** Team winning 6+ categories

### Limitations
- Based on player models from historical data - actual performance may vary
- Does not account for specific matchups, home court advantage, or coaching decisions
- Player injury status from game data may not reflect actual playing status

---

## Files Generated

Each matchup has its own folder containing:
- `category_distributions.png` - Visual distribution of all 11 categories
- `all_simulations.csv` - Raw data from all 500 simulations
- `summary.json` - High-level statistics
- `category_summary.json` - Category-by-category win rates

**Master Files:**
- `all_matchups_summary.csv` - Summary of all matchups
- `OVERVIEW_REPORT.png` - Visual overview of all matchups
- `OVERVIEW_REPORT.md` - This report

---

*Generated by Fantasy Basketball Matchup Simulation System*
*Location: `/Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/`*
"""

    output_path = '/Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/OVERVIEW_REPORT.md'
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved: {output_path}")

    return summary


def main():
    print("="*80)
    print("CREATING OVERVIEW REPORT")
    print("="*80)

    print("\nGenerating overview visualizations...")
    create_overview_visualizations()

    print("\nGenerating markdown report...")
    summary = create_markdown_report()

    print("\n" + "="*80)
    print("OVERVIEW REPORT COMPLETE!")
    print("="*80)

    print("\nKey Findings:")
    print(f"  Total Matchups: {len(summary)}")
    print(f"  Competitive (>60%): {sum(summary['competitiveness'] > 0.6)}")
    print(f"  Moderate (30-60%): {sum((summary['competitiveness'] > 0.3) & (summary['competitiveness'] <= 0.6))}")
    print(f"  Mismatches (<30%): {sum(summary['competitiveness'] <= 0.3)}")

    print("\nFiles created:")
    print("  - OVERVIEW_REPORT.png (visual summary)")
    print("  - OVERVIEW_REPORT.md (detailed report)")

    print(f"\nAll matchup results: /Users/rhu/fantasybasketball2/fantasy_2026/matchup_simulations/")


if __name__ == "__main__":
    main()
