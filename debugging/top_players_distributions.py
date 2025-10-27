"""
Top Players Distribution Analysis

Shows the distribution of individual game stats for the top 6 players from each team.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the player game data
team_a_games = pd.read_csv('/Users/rhu/fantasybasketball2/debugging/team_a_all_player_games.csv')
team_b_games = pd.read_csv('/Users/rhu/fantasybasketball2/debugging/team_b_all_player_games.csv')

# Get top players by average points
team_a_top = team_a_games.groupby('player_name')['PTS'].mean().nlargest(4).index.tolist()
team_b_top = team_b_games.groupby('player_name')['PTS'].mean().nlargest(4).index.tolist()

# Create visualization
fig, axes = plt.subplots(4, 2, figsize=(16, 18))

all_players = list(zip(team_a_top, ['blue']*4, ['Team Boricua Squad']*4)) + \
              list(zip(team_b_top, ['red']*4, ['KL2 LLC']*4))

for idx, (player, color, team) in enumerate(all_players):
    row = idx % 4
    col = idx // 4
    ax = axes[row, col]

    # Get player data
    if team == 'Team Boricua Squad':
        player_data = team_a_games[team_a_games['player_name'] == player]
    else:
        player_data = team_b_games[team_b_games['player_name'] == player]

    # Create histogram of points per game
    pts = player_data['PTS'].values
    ax.hist(pts, bins=25, color=color, alpha=0.7, edgecolor='black', density=True)

    # Add mean and std lines
    mean_pts = pts.mean()
    std_pts = pts.std()

    ax.axvline(mean_pts, color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {mean_pts:.1f}')
    ax.axvline(mean_pts - std_pts, color='orange', linestyle=':', linewidth=1.5)
    ax.axvline(mean_pts + std_pts, color='orange', linestyle=':', linewidth=1.5)

    # Add percentile markers
    p25, p75 = np.percentile(pts, [25, 75])
    ax.axvline(p25, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(p75, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Points per Game', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{player} ({team})\n{len(pts)} games | μ={mean_pts:.1f} σ={std_pts:.1f}',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text box with additional stats
    other_stats = f"REB: {player_data['REB'].mean():.1f}\nAST: {player_data['AST'].mean():.1f}\n3PM: {player_data['FG3M'].mean():.1f}"
    ax.text(0.98, 0.97, other_stats, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Top 4 Players Distribution (Individual Game Stats across 1500 games)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/rhu/fantasybasketball2/debugging/top_players_distributions.png',
            dpi=300, bbox_inches='tight')
print("Saved: /Users/rhu/fantasybasketball2/debugging/top_players_distributions.png")

# Print detailed percentile analysis
print("\n" + "="*80)
print("DETAILED PERCENTILE ANALYSIS FOR TOP PLAYERS")
print("="*80)

for player, color, team in all_players:
    if team == 'Team Boricua Squad':
        player_data = team_a_games[team_a_games['player_name'] == player]
    else:
        player_data = team_b_games[team_b_games['player_name'] == player]

    pts = player_data['PTS'].values

    print(f"\n{player} ({team}):")
    print(f"  Games: {len(pts)}")
    print(f"  Mean: {pts.mean():.2f} ± {pts.std():.2f}")
    print(f"  Min: {pts.min()}")
    print(f"  10th percentile: {np.percentile(pts, 10):.1f}")
    print(f"  25th percentile: {np.percentile(pts, 25):.1f}")
    print(f"  Median: {np.percentile(pts, 50):.1f}")
    print(f"  75th percentile: {np.percentile(pts, 75):.1f}")
    print(f"  90th percentile: {np.percentile(pts, 90):.1f}")
    print(f"  Max: {pts.max()}")
    print(f"  Other averages: REB={player_data['REB'].mean():.1f}, AST={player_data['AST'].mean():.1f}, "
          f"3PM={player_data['FG3M'].mean():.1f}, FG%={player_data['FGM'].sum()/player_data['FGA'].sum():.3f}")

print("\n" + "="*80)
