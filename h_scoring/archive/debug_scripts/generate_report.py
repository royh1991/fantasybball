"""
Generate comprehensive draft and season report with visualizations.

Usage:
    python generate_report.py --draft-results draft_results_20251001_094006.json
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


class DraftReportGenerator:
    """Generate comprehensive draft report with visualizations."""

    def __init__(self, draft_results_file):
        """
        Initialize report generator.

        Parameters:
        -----------
        draft_results_file : str
            Path to draft results JSON
        """
        self.draft_file = draft_results_file

        # Extract timestamp from filename
        timestamp = draft_results_file.replace('draft_results_', '').replace('.json', '')
        self.timestamp = timestamp

        # Load draft results
        print(f"Loading draft results from {draft_results_file}")
        with open(draft_results_file, 'r') as f:
            self.draft_data = json.load(f)

        # Look for corresponding season results
        season_file = f'season_results_{timestamp}.csv'
        if os.path.exists(season_file):
            print(f"Loading season results from {season_file}")
            self.season_data = pd.read_csv(season_file)
        else:
            print(f"Warning: No season results found at {season_file}")
            self.season_data = None

        # Categories for H-scoring
        self.categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'TOV', 'DD']

        # Load player data
        self._load_player_data()

        # Calculate team stats
        self._calculate_team_stats()

    def _load_player_data(self):
        """Load player statistics from data files."""
        data_dir = 'data'
        weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])

        if not weekly_files:
            raise FileNotFoundError("No player data files found!")

        data_file = os.path.join(data_dir, weekly_files[-1])
        print(f"Loading player data from {data_file}")

        self.player_data = pd.read_csv(data_file)

        # Calculate season averages per player
        self.player_averages = self.player_data.groupby('PLAYER_NAME').agg({
            'PTS': 'mean',
            'REB': 'mean',
            'AST': 'mean',
            'STL': 'mean',
            'BLK': 'mean',
            'FG3M': 'mean',
            'FGM': 'mean',
            'FGA': 'mean',
            'FTM': 'mean',
            'FTA': 'mean',
            'FG3A': 'mean',
            'TOV': 'mean'
        }).reset_index()

        # Calculate percentages
        self.player_averages['FG_PCT'] = (
            self.player_averages['FGM'] / self.player_averages['FGA'] * 100
        ).fillna(0)
        self.player_averages['FT_PCT'] = (
            self.player_averages['FTM'] / self.player_averages['FTA'] * 100
        ).fillna(0)
        self.player_averages['FG3_PCT'] = (
            self.player_averages['FG3M'] / self.player_averages['FG3A'] * 100
        ).fillna(0)

        # Calculate double-doubles (simplified: PTS >= 10 and REB >= 10)
        self.player_averages['DD'] = (
            ((self.player_averages['PTS'] >= 10) & (self.player_averages['REB'] >= 10)) |
            ((self.player_averages['PTS'] >= 10) & (self.player_averages['AST'] >= 10)) |
            ((self.player_averages['REB'] >= 10) & (self.player_averages['AST'] >= 10))
        ).astype(float) * 0.5  # Rough estimate

        # Load ADP data
        adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'
        if os.path.exists(adp_file):
            self.adp_data = pd.read_csv(adp_file)
            # Normalize names
            self.adp_data['PLAYER'] = self.adp_data['PLAYER'].str.strip()
        else:
            self.adp_data = None

    def _calculate_team_stats(self):
        """Calculate projected stats for all teams."""
        self.team_stats = {}

        for team_id, roster in self.draft_data['all_teams'].items():
            team_num = int(team_id.replace('Team_', ''))

            # Get stats for all players on roster
            team_players = self.player_averages[
                self.player_averages['PLAYER_NAME'].isin(roster)
            ]

            if len(team_players) == 0:
                print(f"Warning: No stats found for Team {team_num}")
                continue

            # Sum counting stats, weighted average for percentages
            stats = {}
            for cat in self.categories:
                if cat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                    # Weighted average by attempts
                    if cat == 'FG_PCT':
                        total_fgm = team_players['FGM'].sum()
                        total_fga = team_players['FGA'].sum()
                        stats[cat] = (total_fgm / total_fga * 100) if total_fga > 0 else 0
                    elif cat == 'FT_PCT':
                        total_ftm = team_players['FTM'].sum()
                        total_fta = team_players['FTA'].sum()
                        stats[cat] = (total_ftm / total_fta * 100) if total_fta > 0 else 0
                    elif cat == 'FG3_PCT':
                        total_fg3m = team_players['FG3M'].sum()
                        total_fg3a = team_players['FG3A'].sum()
                        stats[cat] = (total_fg3m / total_fg3a * 100) if total_fg3a > 0 else 0
                else:
                    # Sum counting stats
                    stats[cat] = team_players[cat].sum()

            self.team_stats[team_num] = {
                'roster': roster,
                'stats': stats,
                'roster_size': len(team_players)
            }

        print(f"Calculated stats for {len(self.team_stats)} teams")

    def _get_player_adp(self, player_name):
        """Get ADP for a player."""
        if self.adp_data is None:
            return None

        # Try exact match
        match = self.adp_data[self.adp_data['PLAYER'] == player_name]
        if not match.empty:
            return match.iloc[0]['ADP']

        # Try fuzzy match
        player_lower = player_name.lower()
        for _, row in self.adp_data.iterrows():
            if row['PLAYER'].lower() == player_lower:
                return row['ADP']

        return None

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def generate_category_heatmap(self):
        """Generate category strength heatmap for all teams."""
        print("\nGenerating category strength heatmap...")

        # Create matrix of team stats
        teams = sorted(self.team_stats.keys())
        matrix = []

        for team_num in teams:
            stats = self.team_stats[team_num]['stats']
            matrix.append([stats[cat] for cat in self.categories])

        matrix = np.array(matrix)

        # Normalize each category (z-score)
        normalized_matrix = np.zeros_like(matrix)
        for i, cat in enumerate(self.categories):
            col = matrix[:, i]
            mean = col.mean()
            std = col.std()

            if std > 0:
                if cat == 'TOV':  # Lower is better
                    normalized_matrix[:, i] = -(col - mean) / std
                else:
                    normalized_matrix[:, i] = (col - mean) / std
            else:
                normalized_matrix[:, i] = 0

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))

        # Your team position
        your_position = self.draft_data['settings']['your_position']

        # Create custom colormap (red for weak, white for average, green for strong)
        cmap = sns.diverging_palette(10, 130, as_cmap=True)

        sns.heatmap(
            normalized_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            vmin=-2.5,
            vmax=2.5,
            xticklabels=self.categories,
            yticklabels=[f'Team {t}{"*" if t == your_position else ""}' for t in teams],
            cbar_kws={'label': 'Standard Deviations from Mean'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Category Strength Heatmap\n(Green = Strong, Red = Weak)\n* = Your Team',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Team', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_team_comparison_chart(self):
        """Generate radar chart comparing your team vs league average."""
        print("Generating team comparison radar chart...")

        your_position = self.draft_data['settings']['your_position']
        your_stats = self.team_stats[your_position]['stats']

        # Calculate league averages
        league_avg = {}
        for cat in self.categories:
            values = [self.team_stats[t]['stats'][cat] for t in self.team_stats.keys()]
            league_avg[cat] = np.mean(values)

        # Normalize to percentiles
        your_percentiles = []
        for cat in self.categories:
            values = [self.team_stats[t]['stats'][cat] for t in self.team_stats.keys()]
            values_sorted = sorted(values, reverse=(cat != 'TOV'))  # TOV: lower is better
            your_value = your_stats[cat]

            if cat == 'TOV':
                percentile = (values_sorted.index(your_value) + 1) / len(values_sorted) * 100
            else:
                percentile = (len(values_sorted) - values_sorted.index(your_value)) / len(values_sorted) * 100

            your_percentiles.append(percentile)

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(self.categories), endpoint=False).tolist()
        your_percentiles += your_percentiles[:1]  # Complete the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, your_percentiles, 'o-', linewidth=2, color='#2E86AB', label='Your Team')
        ax.fill(angles, your_percentiles, alpha=0.25, color='#2E86AB')
        ax.plot(angles, [50] * len(angles), '--', linewidth=1, color='gray', alpha=0.5, label='League Average (50th %ile)')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.categories, size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=9)
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_title(f'Your Team (Team {your_position}) - Category Percentiles',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_draft_value_chart(self):
        """Generate chart showing draft picks and their ADP vs pick number."""
        print("Generating draft value chart...")

        your_position = self.draft_data['settings']['your_position']
        your_team = self.draft_data['your_team']

        # Get ADP for each pick
        picks_data = []
        for pick_num, player in enumerate(your_team, start=1):
            adp = self._get_player_adp(player)
            if adp:
                # Calculate actual pick number in snake draft
                if pick_num % 2 == 1:  # Odd rounds
                    actual_pick = (pick_num - 1) * 12 + your_position
                else:  # Even rounds (snake)
                    actual_pick = pick_num * 12 - your_position + 1

                picks_data.append({
                    'round': pick_num,
                    'player': player,
                    'adp': adp,
                    'pick': actual_pick,
                    'value': adp - actual_pick  # Positive = got value
                })

        if not picks_data:
            print("Warning: No ADP data available")
            return None

        df = pd.DataFrame(picks_data)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Chart 1: ADP vs Pick Number
        colors = ['green' if v > 0 else 'red' for v in df['value']]
        ax1.scatter(df['pick'], df['adp'], c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
        ax1.plot([0, 160], [0, 160], '--', color='gray', alpha=0.5, label='ADP = Pick (Fair Value)')

        # Annotate best values
        top_values = df.nlargest(3, 'value')
        for _, row in top_values.iterrows():
            ax1.annotate(row['player'], xy=(row['pick'], row['adp']),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax1.set_xlabel('Actual Pick Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Draft Position (ADP)', fontsize=12, fontweight='bold')
        ax1.set_title('Draft Value Analysis\n(Green = Value Pick, Red = Reach)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Chart 2: Value by Round
        ax2.bar(df['round'], df['value'], color=['green' if v > 0 else 'red' for v in df['value']],
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Draft Round', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Draft Value (ADP - Pick)', fontsize=12, fontweight='bold')
        ax2.set_title('Draft Value by Round\n(Positive = Value, Negative = Reach)', fontsize=13, fontweight='bold')
        ax2.set_xticks(df['round'])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_season_performance_chart(self):
        """Generate season performance summary."""
        if self.season_data is None:
            return None

        print("Generating season performance chart...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Chart 1: Win percentages
        teams = self.season_data.sort_values('rank')
        your_position = self.draft_data['settings']['your_position']

        colors = ['#2E86AB' if tid == your_position else '#A23B72' for tid in teams['team_id']]

        ax1.barh(range(len(teams)), teams['win_pct'] * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_yticks(range(len(teams)))
        ax1.set_yticklabels([f"Team {tid}{'*' if tid == your_position else ''}" for tid in teams['team_id']])
        ax1.set_xlabel('Win Percentage', fontsize=12, fontweight='bold')
        ax1.set_title('Season Performance - Win Percentage\n* = Your Team', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        for i, (pct, tid) in enumerate(zip(teams['win_pct'], teams['team_id'])):
            ax1.text(pct * 100 + 1, i, f'{pct*100:.1f}%', va='center', fontsize=9)

        # Chart 2: Total wins
        ax2.barh(range(len(teams)), teams['wins'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(teams)))
        ax2.set_yticklabels([f"Team {tid}{'*' if tid == your_position else ''}" for tid in teams['team_id']])
        ax2.set_xlabel('Total Wins', fontsize=12, fontweight='bold')
        ax2.set_title(f'Season Performance - Total Wins\n(Out of {teams["wins"].iloc[0] + teams["losses"].iloc[0]} matchups)',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add win count labels
        for i, wins in enumerate(teams['wins']):
            ax2.text(wins + 20, i, f'{wins}', va='center', fontsize=9)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_win_probability_matrix(self):
        """Generate head-to-head win probability matrix."""
        print("Generating win probability matrix...")

        teams = sorted(self.team_stats.keys())
        n_teams = len(teams)
        win_prob_matrix = np.zeros((n_teams, n_teams))

        # Calculate win probabilities for each matchup
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i == j:
                    win_prob_matrix[i, j] = 0.5  # Tie with self
                else:
                    # Count categories where team1 beats team2
                    wins = 0
                    for cat in self.categories:
                        val1 = self.team_stats[team1]['stats'][cat]
                        val2 = self.team_stats[team2]['stats'][cat]

                        if cat == 'TOV':  # Lower is better
                            if val1 < val2:
                                wins += 1
                        else:
                            if val1 > val2:
                                wins += 1

                    win_prob_matrix[i, j] = wins / len(self.categories)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        your_position = self.draft_data['settings']['your_position']

        sns.heatmap(
            win_prob_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            xticklabels=[f'Team {t}{"*" if t == your_position else ""}' for t in teams],
            yticklabels=[f'Team {t}{"*" if t == your_position else ""}' for t in teams],
            cbar_kws={'label': 'Win Probability'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Head-to-Head Win Probability Matrix\n(Row Team vs Column Team)\n* = Your Team',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Opponent Team', fontsize=12, fontweight='bold')
        ax.set_ylabel('Your Team', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_html_report(self, output_file=None):
        """Generate complete HTML report."""
        print("\n" + "=" * 80)
        print("GENERATING DRAFT REPORT")
        print("=" * 80)

        if output_file is None:
            output_file = f'draft_report_{self.timestamp}.html'

        # Generate all visualizations
        heatmap_img = self.generate_category_heatmap()
        radar_img = self.generate_team_comparison_chart()
        value_img = self.generate_draft_value_chart()
        win_matrix_img = self.generate_win_probability_matrix()
        season_img = self.generate_season_performance_chart() if self.season_data is not None else None

        # Your team info
        your_position = self.draft_data['settings']['your_position']
        your_team = self.draft_data['your_team']

        # Season stats if available
        your_season_stats = ""
        if self.season_data is not None:
            your_row = self.season_data[self.season_data['team_id'] == your_position]
            if not your_row.empty:
                rank = your_row.iloc[0]['rank']
                wins = your_row.iloc[0]['wins']
                losses = your_row.iloc[0]['losses']
                win_pct = your_row.iloc[0]['win_pct']
                your_season_stats = f"""
                <div class="season-summary">
                    <h3>Season Performance</h3>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-value">#{rank}</div>
                            <div class="stat-label">Final Rank</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{wins}-{losses}</div>
                            <div class="stat-label">Record</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{win_pct*100:.1f}%</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                    </div>
                </div>
                """

        # Build roster table
        roster_rows = ""
        for i, player in enumerate(your_team, start=1):
            adp = self._get_player_adp(player)
            adp_str = f"{adp:.1f}" if adp else "N/A"

            # Calculate actual pick number
            if i % 2 == 1:
                actual_pick = (i - 1) * 12 + your_position
            else:
                actual_pick = i * 12 - your_position + 1

            value = ""
            if adp:
                diff = adp - actual_pick
                if diff > 5:
                    value = f'<span class="value-good">+{diff:.1f}</span>'
                elif diff < -5:
                    value = f'<span class="value-bad">{diff:.1f}</span>'
                else:
                    value = f'<span class="value-neutral">{diff:+.1f}</span>'

            roster_rows += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{player}</strong></td>
                <td>{adp_str}</td>
                <td>{actual_pick}</td>
                <td>{value}</td>
            </tr>
            """

        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Fantasy Basketball Draft Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 10px;
                    font-size: 2.5em;
                }}
                .subtitle {{
                    text-align: center;
                    color: #7f8c8d;
                    margin-bottom: 30px;
                    font-size: 1.1em;
                }}
                .season-summary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .season-summary h3 {{
                    margin-top: 0;
                    font-size: 1.8em;
                }}
                .stat-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-top: 20px;
                }}
                .stat-box {{
                    background: rgba(255,255,255,0.2);
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .stat-label {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .section {{
                    margin-bottom: 50px;
                }}
                .section h2 {{
                    color: #34495e;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                .roster-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .roster-table th {{
                    background: #34495e;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                .roster-table td {{
                    padding: 10px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                .roster-table tr:hover {{
                    background: #f8f9fa;
                }}
                .value-good {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .value-bad {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .value-neutral {{
                    color: #95a5a6;
                }}
                .chart {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .chart img {{
                    max-width: 100%;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .footer {{
                    text-align: center;
                    color: #7f8c8d;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏀 Fantasy Basketball Draft Report</h1>
                <div class="subtitle">Team {your_position} • Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>

                {your_season_stats}

                <div class="section">
                    <h2>Your Roster</h2>
                    <table class="roster-table">
                        <thead>
                            <tr>
                                <th>Round</th>
                                <th>Player</th>
                                <th>ADP</th>
                                <th>Pick #</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {roster_rows}
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>Category Strength Analysis</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{heatmap_img}" alt="Category Heatmap">
                    </div>
                </div>

                <div class="section">
                    <h2>Your Team vs League</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{radar_img}" alt="Team Comparison">
                    </div>
                </div>

                <div class="section">
                    <h2>Head-to-Head Win Probabilities</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{win_matrix_img}" alt="Win Probability Matrix">
                    </div>
                </div>

                <div class="section">
                    <h2>Draft Value Analysis</h2>
                    <div class="chart">
                        <img src="data:image/png;base64,{value_img}" alt="Draft Value">
                    </div>
                </div>

                {"<div class='section'><h2>Season Results</h2><div class='chart'><img src='data:image/png;base64," + season_img + "' alt='Season Performance'></div></div>" if season_img else ""}

                <div class="footer">
                    <p>Report generated using H-scoring optimization algorithm</p>
                    <p><em>Draft file: {self.draft_file}</em></p>
                </div>
            </div>
        </body>
        </html>
        """

        # Save HTML
        with open(output_file, 'w') as f:
            f.write(html)

        print(f"\n✓ Report saved to: {output_file}")
        print(f"\nOpen in browser:")
        print(f"  file://{os.path.abspath(output_file)}")

        return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate fantasy basketball draft report')
    parser.add_argument('--draft-results', type=str, required=False,
                       help='Path to draft results JSON file')

    args = parser.parse_args()

    # If no file specified, use most recent
    if args.draft_results is None:
        draft_files = sorted([f for f in os.listdir('.') if f.startswith('draft_results_')])
        if not draft_files:
            print("Error: No draft results files found!")
            print("Please run: python simulate_draft.py or python simulate_season.py")
            return

        args.draft_results = draft_files[-1]
        print(f"Using most recent draft results: {args.draft_results}")

    # Generate report
    generator = DraftReportGenerator(args.draft_results)
    output_file = generator.generate_html_report()

    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
