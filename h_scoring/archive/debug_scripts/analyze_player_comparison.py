"""
Deep analysis comparing player valuations: Why does H-scoring prefer certain players?

This script compares players to understand H-scoring methodology.
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_final import HScoreOptimizerFinal
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


class PlayerComparisonAnalyzer:
    """Analyze and compare player valuations in H-scoring."""

    def __init__(self, data_file, variance_file):
        """Initialize analyzer."""
        print("=" * 80)
        print("PLAYER COMPARISON ANALYZER")
        print("=" * 80)

        # Load data
        print(f"\nLoading data from {data_file}")
        self.league_data = pd.read_csv(data_file)

        print(f"Loading variances from {variance_file}")
        with open(variance_file, 'r') as f:
            self.player_variances = json.load(f)

        # Initialize scoring system
        print("Initializing scoring system...")
        self.scoring = PlayerScoring(
            self.league_data,
            self.player_variances,
            roster_size=13
        )

        # Initialize covariance calculator
        print("Calculating covariance matrix...")
        self.cov_calc = CovarianceCalculator(
            self.league_data,
            self.scoring
        )

        # Get setup parameters
        self.setup_params = self.cov_calc.get_setup_params()

        # Initialize H-score optimizer
        print("Initializing H-score optimizer...")
        self.optimizer = HScoreOptimizerFinal(
            self.setup_params,
            self.scoring,
            omega=0.7,
            gamma=0.25
        )

        # Calculate player averages
        self._calculate_player_averages()

        # Load ADP data
        adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'
        if os.path.exists(adp_file):
            self.adp_data = pd.read_csv(adp_file)
            self.adp_data['PLAYER'] = self.adp_data['PLAYER'].str.strip()
        else:
            self.adp_data = None

        print("✓ Initialization complete!\n")

    def _calculate_player_averages(self):
        """Calculate TRUE per-game averages for all players."""
        # Data contains weekly totals, need to divide by games played

        # Group by player and sum totals and games
        player_totals = self.league_data.groupby('PLAYER_NAME').agg({
            'PTS': 'sum',
            'REB': 'sum',
            'AST': 'sum',
            'STL': 'sum',
            'BLK': 'sum',
            'FG3M': 'sum',
            'FGM': 'sum',
            'FGA': 'sum',
            'FTM': 'sum',
            'FTA': 'sum',
            'FG3A': 'sum',
            'TOV': 'sum',
            'DD': 'sum',
            'GAMES_PLAYED': 'sum'
        }).reset_index()

        # Calculate per-game averages
        self.player_averages = player_totals.copy()
        for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3A', 'TOV', 'DD']:
            self.player_averages[col] = player_totals[col] / player_totals['GAMES_PLAYED']

        # Calculate percentages from totals
        self.player_averages['FG_PCT'] = (
            player_totals['FGM'] / player_totals['FGA'] * 100
        ).fillna(0)
        self.player_averages['FT_PCT'] = (
            player_totals['FTM'] / player_totals['FTA'] * 100
        ).fillna(0)
        self.player_averages['FG3_PCT'] = (
            player_totals['FG3M'] / player_totals['FG3A'] * 100
        ).fillna(0)

    def get_player_stats(self, player_name):
        """Get comprehensive stats for a player."""
        stats = self.player_averages[self.player_averages['PLAYER_NAME'] == player_name]
        if stats.empty:
            return None
        return stats.iloc[0]

    def get_player_adp(self, player_name):
        """Get ADP for a player."""
        if self.adp_data is None:
            return None

        # Try exact and fuzzy match
        match = self.adp_data[self.adp_data['PLAYER'].str.lower() == player_name.lower()]
        if not match.empty:
            return match.iloc[0]['ADP']
        return None

    def compare_players(self, player1_name, player2_name):
        """
        Deep comparison of two players.

        Shows:
        1. Raw stats comparison
        2. G-scores (variance-adjusted value)
        3. X-scores (optimization basis)
        4. H-scores in different contexts
        5. Category-by-category breakdown
        """
        print("\n" + "=" * 80)
        print(f"COMPARING: {player1_name} vs {player2_name}")
        print("=" * 80)

        # Get stats
        p1_stats = self.get_player_stats(player1_name)
        p2_stats = self.get_player_stats(player2_name)

        if p1_stats is None:
            print(f"Error: {player1_name} not found in dataset")
            return
        if p2_stats is None:
            print(f"Error: {player2_name} not found in dataset")
            return

        # Get ADP
        p1_adp = self.get_player_adp(player1_name)
        p2_adp = self.get_player_adp(player2_name)

        print(f"\n{player1_name}: ADP {p1_adp:.1f}" if p1_adp else f"\n{player1_name}: ADP N/A")
        print(f"{player2_name}: ADP {p2_adp:.1f}" if p2_adp else f"{player2_name}: ADP N/A")

        # Section 1: Raw Stats Comparison
        print("\n" + "-" * 80)
        print("1. RAW STATS COMPARISON (Per-Game Averages)")
        print("-" * 80)

        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'TOV', 'DD']

        print(f"\n{'Category':<12} {player1_name:<20} {player2_name:<20} {'Winner':<15}")
        print("-" * 80)

        for cat in categories:
            val1 = p1_stats[cat]
            val2 = p2_stats[cat]

            if cat == 'TOV':  # Lower is better
                winner = player1_name if val1 < val2 else player2_name
                diff = val2 - val1  # Positive means p1 better
            else:
                winner = player1_name if val1 > val2 else player2_name
                diff = val1 - val2

            winner_str = f"✓ {winner}" if abs(diff) > 0.01 else "Tie"

            print(f"{cat:<12} {val1:<20.2f} {val2:<20.2f} {winner_str:<15}")

        # Section 2: G-Scores (Variance-Adjusted)
        print("\n" + "-" * 80)
        print("2. G-SCORES (Variance-Adjusted Value)")
        print("-" * 80)
        print("G-scores account for player consistency. Higher is better.")
        print()

        p1_g_scores = self.scoring.calculate_all_g_scores(player1_name)
        p2_g_scores = self.scoring.calculate_all_g_scores(player2_name)

        print(f"{'Category':<12} {player1_name:<20} {player2_name:<20} {'Winner':<15}")
        print("-" * 80)

        for cat in categories:
            g1 = p1_g_scores.get(cat, 0)
            g2 = p2_g_scores.get(cat, 0)

            winner = player1_name if g1 > g2 else player2_name
            winner_str = f"✓ {winner}" if abs(g1 - g2) > 0.1 else "Tie"

            print(f"{cat:<12} {g1:<20.3f} {g2:<20.3f} {winner_str:<15}")

        print(f"\n{'TOTAL G-SCORE':<12} {p1_g_scores['TOTAL']:<20.3f} {p2_g_scores['TOTAL']:<20.3f}")

        # Section 3: X-Scores (Optimization Basis)
        print("\n" + "-" * 80)
        print("3. X-SCORES (Optimization Input)")
        print("-" * 80)
        print("X-scores are the simplified values used in H-score optimization.")
        print()

        p1_x_scores = self.scoring.calculate_all_x_scores(player1_name)
        p2_x_scores = self.scoring.calculate_all_x_scores(player2_name)

        print(f"{'Category':<12} {player1_name:<20} {player2_name:<20} {'Winner':<15}")
        print("-" * 80)

        for cat in categories:
            x1 = p1_x_scores.get(cat, 0)
            x2 = p2_x_scores.get(cat, 0)

            winner = player1_name if x1 > x2 else player2_name
            winner_str = f"✓ {winner}" if abs(x1 - x2) > 0.1 else "Tie"

            print(f"{cat:<12} {x1:<20.3f} {x2:<20.3f} {winner_str:<15}")

        # Section 4: H-Scores (Context-Dependent)
        print("\n" + "-" * 80)
        print("4. H-SCORES (Dynamic Valuation)")
        print("-" * 80)
        print("H-scores change based on draft context. Testing in different scenarios:")
        print()

        scenarios = [
            ("Empty roster (Pick 1)", [], 0),
            ("After drafting Jokić", ["Nikola Jokić"], 1),
            ("After drafting SGA + Harden", ["Shai Gilgeous-Alexander", "James Harden"], 2),
            ("Mid-draft (5 picks)", ["Nikola Jokić", "James Harden", "Anthony Davis",
                                      "Stephen Curry", "LeBron James"], 5),
        ]

        for scenario_name, my_team, picks_made in scenarios:
            print(f"\nScenario: {scenario_name}")
            print("-" * 40)

            # Show current team composition if not empty
            if my_team:
                print(f"Current team: {', '.join(my_team)}")

                # Calculate team X-scores to show what categories you have
                team_x = np.zeros(len(categories))
                for player in my_team:
                    player_x_scores = self.scoring.calculate_all_x_scores(player)
                    for i, cat in enumerate(categories):
                        team_x[i] += player_x_scores.get(cat, 0)

                # Show team strengths and weaknesses
                print("\nTeam category status:")
                for i, cat in enumerate(categories):
                    val = team_x[i]
                    status = "💪 Strong" if val > 2 else "⚠️  Weak" if val < 1 else "➡️  Average"
                    print(f"  {cat:<12} X-score: {val:6.2f}  {status}")

                print()

            # Calculate H-scores
            h1, weights1 = self.optimizer.evaluate_player(
                player1_name,
                my_team,
                opponent_teams=[],
                picks_made=picks_made,
                total_picks=13,
                last_weights=None,
                format='each_category'
            )

            h2, weights2 = self.optimizer.evaluate_player(
                player2_name,
                my_team,
                opponent_teams=[],
                picks_made=picks_made,
                total_picks=13,
                last_weights=None,
                format='each_category'
            )

            winner = player1_name if h1 > h2 else player2_name
            diff = abs(h1 - h2)

            print(f"{player1_name}: {h1:.4f}")
            print(f"{player2_name}: {h2:.4f}")
            print(f"Winner: ✓ {winner} (+{diff:.4f})")

            # Show what each player contributes
            if my_team:  # Only for non-empty roster
                p1_x_scores = self.scoring.calculate_all_x_scores(player1_name)
                p2_x_scores = self.scoring.calculate_all_x_scores(player2_name)

                print("\nCategory contribution comparison:")
                print(f"{'Category':<12} {'Team Need':<12} {player1_name:<20} {player2_name:<20} {'Better Fit':<15}")
                print("-" * 80)

                for i, cat in enumerate(categories):
                    need = "High" if team_x[i] < 1 else "Med" if team_x[i] < 2 else "Low"
                    p1_contrib = p1_x_scores.get(cat, 0)
                    p2_contrib = p2_x_scores.get(cat, 0)

                    # Who fills the need better?
                    if team_x[i] < 2:  # If we need this category
                        better = player1_name if p1_contrib > p2_contrib else player2_name
                        better_str = f"✓ {better}"
                    else:
                        better_str = "(don't need)"

                    print(f"{cat:<12} {need:<12} {p1_contrib:<20.2f} {p2_contrib:<20.2f} {better_str:<15}")

        # Section 5: Key Insights
        print("\n" + "=" * 80)
        print("5. KEY INSIGHTS")
        print("=" * 80)

        insights = []

        # Efficiency comparison
        if p1_stats['FG_PCT'] > p2_stats['FG_PCT']:
            insights.append(f"✓ {player1_name} is more efficient (FG%: {p1_stats['FG_PCT']:.1f}% vs {p2_stats['FG_PCT']:.1f}%)")
        else:
            insights.append(f"✓ {player2_name} is more efficient (FG%: {p2_stats['FG_PCT']:.1f}% vs {p1_stats['FG_PCT']:.1f}%)")

        # Turnovers
        if p1_stats['TOV'] < p2_stats['TOV']:
            tov_diff = p2_stats['TOV'] - p1_stats['TOV']
            insights.append(f"✓ {player1_name} protects the ball better ({tov_diff:.1f} fewer turnovers/game)")
        else:
            tov_diff = p1_stats['TOV'] - p2_stats['TOV']
            insights.append(f"✓ {player2_name} protects the ball better ({tov_diff:.1f} fewer turnovers/game)")

        # Consistency (from variance)
        p1_variance = self.player_variances.get(player1_name, {})
        p2_variance = self.player_variances.get(player2_name, {})

        if p1_variance and p2_variance:
            try:
                # Handle nested variance structure
                p1_values = []
                p2_values = []

                for v in p1_variance.values():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        p1_values.append(v)

                for v in p2_variance.values():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        p2_values.append(v)

                if p1_values and p2_values:
                    p1_avg_variance = np.mean(p1_values)
                    p2_avg_variance = np.mean(p2_values)

                    if p1_avg_variance < p2_avg_variance:
                        insights.append(f"✓ {player1_name} is more consistent (lower variance)")
                    else:
                        insights.append(f"✓ {player2_name} is more consistent (lower variance)")
            except Exception as e:
                pass  # Skip variance comparison if data structure issues

        # Category coverage
        p1_strong_cats = sum(1 for cat in categories if p1_g_scores.get(cat, 0) > 1.0)
        p2_strong_cats = sum(1 for cat in categories if p2_g_scores.get(cat, 0) > 1.0)

        insights.append(f"✓ {player1_name} contributes strongly in {p1_strong_cats} categories")
        insights.append(f"✓ {player2_name} contributes strongly in {p2_strong_cats} categories")

        # ADP value
        if p1_adp and p2_adp:
            if p1_adp > p2_adp:  # Higher ADP = later pick = better value
                insights.append(f"✓ {player1_name} available later (ADP {p1_adp:.1f} vs {p2_adp:.1f})")
            else:
                insights.append(f"✓ {player2_name} available later (ADP {p2_adp:.1f} vs {p1_adp:.1f})")

        # H-score preference
        if p1_g_scores['TOTAL'] > p2_g_scores['TOTAL']:
            insights.append(f"\n⭐ H-scoring prefers {player1_name} due to:")
            insights.append(f"   - Higher total G-score ({p1_g_scores['TOTAL']:.2f} vs {p2_g_scores['TOTAL']:.2f})")
            insights.append(f"   - Better category balance and efficiency")
            if p1_adp and p2_adp and p1_adp > p2_adp:
                insights.append(f"   - Available at better draft position (ADP {p1_adp:.1f})")
        else:
            insights.append(f"\n⭐ H-scoring prefers {player2_name} due to:")
            insights.append(f"   - Higher total G-score ({p2_g_scores['TOTAL']:.2f} vs {p1_g_scores['TOTAL']:.2f})")
            insights.append(f"   - Better category balance and efficiency")
            if p1_adp and p2_adp and p2_adp > p1_adp:
                insights.append(f"   - Available at better draft position (ADP {p2_adp:.1f})")

        for insight in insights:
            print(insight)

        return {
            'player1': player1_name,
            'player2': player2_name,
            'p1_stats': p1_stats,
            'p2_stats': p2_stats,
            'p1_g_scores': p1_g_scores,
            'p2_g_scores': p2_g_scores,
            'p1_x_scores': p1_x_scores,
            'p2_x_scores': p2_x_scores,
            'insights': insights
        }

    def generate_comparison_chart(self, player1_name, player2_name, output_file=None):
        """Generate visual comparison chart."""
        print(f"\nGenerating comparison chart...")

        # Get data
        p1_g_scores = self.scoring.calculate_all_g_scores(player1_name)
        p2_g_scores = self.scoring.calculate_all_g_scores(player2_name)

        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'TOV', 'DD']

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Chart 1: G-Score Comparison (Bar Chart)
        ax1 = axes[0, 0]
        g1_vals = [p1_g_scores.get(cat, 0) for cat in categories]
        g2_vals = [p2_g_scores.get(cat, 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width/2, g1_vals, width, label=player1_name, alpha=0.8, color='#2E86AB')
        ax1.bar(x + width/2, g2_vals, width, label=player2_name, alpha=0.8, color='#A23B72')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel('Category', fontweight='bold')
        ax1.set_ylabel('G-Score', fontweight='bold')
        ax1.set_title('G-Score Comparison (Variance-Adjusted Value)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Chart 2: Radar Chart - G-Scores
        ax2 = axes[0, 1]
        ax2.remove()
        ax2 = fig.add_subplot(222, projection='polar')

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        g1_vals_plot = g1_vals + g1_vals[:1]
        g2_vals_plot = g2_vals + g2_vals[:1]
        angles_plot = angles + angles[:1]

        ax2.plot(angles_plot, g1_vals_plot, 'o-', linewidth=2, label=player1_name, color='#2E86AB')
        ax2.fill(angles_plot, g1_vals_plot, alpha=0.25, color='#2E86AB')
        ax2.plot(angles_plot, g2_vals_plot, 'o-', linewidth=2, label=player2_name, color='#A23B72')
        ax2.fill(angles_plot, g2_vals_plot, alpha=0.25, color='#A23B72')

        ax2.set_xticks(angles)
        ax2.set_xticklabels(categories)
        ax2.set_title('Category Profile (Radar)', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True)

        # Chart 3: Raw Stats Comparison (Selected Categories)
        ax3 = axes[1, 0]
        p1_stats = self.get_player_stats(player1_name)
        p2_stats = self.get_player_stats(player2_name)

        main_cats = ['PTS', 'REB', 'AST', 'FG_PCT', 'FT_PCT', 'TOV']
        p1_raw = [p1_stats[cat] for cat in main_cats]
        p2_raw = [p2_stats[cat] for cat in main_cats]

        # Normalize for visualization (except TOV which we invert)
        max_vals = [max(p1_raw[i], p2_raw[i]) for i in range(len(main_cats))]
        p1_normalized = [p1_raw[i] / max_vals[i] * 100 if max_vals[i] > 0 else 0 for i in range(len(main_cats))]
        p2_normalized = [p2_raw[i] / max_vals[i] * 100 if max_vals[i] > 0 else 0 for i in range(len(main_cats))]

        # Invert TOV (lower is better)
        p1_normalized[5] = 100 - p1_normalized[5]
        p2_normalized[5] = 100 - p2_normalized[5]

        x = np.arange(len(main_cats))
        ax3.bar(x - width/2, p1_normalized, width, label=player1_name, alpha=0.8, color='#2E86AB')
        ax3.bar(x + width/2, p2_normalized, width, label=player2_name, alpha=0.8, color='#A23B72')
        ax3.set_xlabel('Category', fontweight='bold')
        ax3.set_ylabel('Normalized Value (% of Max)', fontweight='bold')
        ax3.set_title('Raw Stats Comparison (Normalized)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(main_cats, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Chart 4: Summary Stats
        ax4 = axes[1, 1]
        ax4.axis('off')

        p1_adp = self.get_player_adp(player1_name)
        p2_adp = self.get_player_adp(player2_name)

        p1_adp_str = f"{p1_adp:.1f}" if p1_adp else "N/A"
        p2_adp_str = f"{p2_adp:.1f}" if p2_adp else "N/A"
        winner_name = player1_name if p1_g_scores['TOTAL'] > p2_g_scores['TOTAL'] else player2_name

        summary_text = f"""
        COMPARISON SUMMARY
        ========================================

        {player1_name}:
          • ADP: {p1_adp_str}
          • Total G-Score: {p1_g_scores['TOTAL']:.2f}
          • PTS: {p1_stats['PTS']:.1f} | REB: {p1_stats['REB']:.1f} | AST: {p1_stats['AST']:.1f}
          • FG: {p1_stats['FG_PCT']:.1f}% | FT: {p1_stats['FT_PCT']:.1f}%
          • TOV: {p1_stats['TOV']:.1f}

        {player2_name}:
          • ADP: {p2_adp_str}
          • Total G-Score: {p2_g_scores['TOTAL']:.2f}
          • PTS: {p2_stats['PTS']:.1f} | REB: {p2_stats['REB']:.1f} | AST: {p2_stats['AST']:.1f}
          • FG: {p2_stats['FG_PCT']:.1f}% | FT: {p2_stats['FT_PCT']:.1f}%
          • TOV: {p2_stats['TOV']:.1f}

        ========================================
        WINNER: {winner_name}
        (Based on Total G-Score)
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        if output_file is None:
            output_file = f'comparison_{player1_name.replace(" ", "_")}_vs_{player2_name.replace(" ", "_")}.png'

        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Chart saved to: {output_file}")

        plt.close()


def main():
    """Main entry point."""
    # Find data files
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    # Initialize analyzer
    analyzer = PlayerComparisonAnalyzer(data_file, variance_file)

    # Compare Kevin Durant vs Karl-Anthony Towns
    print("\n" + "=" * 80)
    print("ANALYZING: Why does H-scoring prefer Kevin Durant over Karl-Anthony Towns?")
    print("=" * 80)

    result = analyzer.compare_players("Kevin Durant", "Karl-Anthony Towns")

    # Generate visual comparison
    analyzer.generate_comparison_chart("Kevin Durant", "Karl-Anthony Towns")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nTo compare other players, modify the compare_players() call in main().")


if __name__ == "__main__":
    main()
