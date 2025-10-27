"""
Roster Depth Impact Visualization

Shows how the missing players affect Team Boricua Squad's projections.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read roster analysis
roster_df = pd.read_csv('/Users/rhu/fantasybasketball2/debugging/roster_analysis.csv')

# Separate teams
boricua = roster_df[roster_df['team'] == 'Team Boricua Squad'].copy()
kl2 = roster_df[roster_df['team'] == 'KL2 LLC'].copy()

# Fill NaN with 0 for visualization
boricua_filled = boricua.fillna(0)
kl2_filled = kl2.fillna(0)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Points per player
ax = axes[0, 0]
x_boricua = np.arange(len(boricua))
gap = 1  # Gap between teams
x_kl2 = np.arange(len(kl2)) + len(boricua) + gap

colors_boricua = ['blue' if m else 'lightgray' for m in boricua['has_model']]
colors_kl2 = ['red' if m else 'lightgray' for m in kl2['has_model']]

ax.barh(x_boricua, boricua_filled['pts'], color=colors_boricua, alpha=0.7, label='Team Boricua Squad')
ax.barh(x_kl2, kl2_filled['pts'], color=colors_kl2, alpha=0.7, label='KL2 LLC')

# Create labels with gap
all_labels = list(boricua['player']) + [''] + list(kl2['player'])
all_positions = list(x_boricua) + [len(boricua)] + list(x_kl2)

ax.set_yticks(all_positions)
ax.set_yticklabels(all_labels, fontsize=9)
ax.set_xlabel('Points Per Game', fontsize=11)
ax.set_title('Points Per Game by Player (Gray = Not Modeled)', fontsize=12, fontweight='bold')
ax.axhline(y=len(boricua) + gap/2 - 0.5, color='black', linestyle='--', linewidth=2)
ax.grid(axis='x', alpha=0.3)
ax.legend()

# Plot 2: Cumulative stats comparison
ax = axes[0, 1]
stats = ['pts', 'reb', 'ast']
boricua_totals = [boricua[boricua['has_model'] == True][stat].sum() * 3 for stat in stats]
kl2_totals = [kl2[kl2['has_model'] == True][stat].sum() * 3 for stat in stats]

x = np.arange(len(stats))
width = 0.35

bars1 = ax.bar(x - width/2, boricua_totals, width, label='Team Boricua Squad', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, kl2_totals, width, label='KL2 LLC', color='red', alpha=0.7)

ax.set_ylabel('Total Per Week (3 games/player)', fontsize=11)
ax.set_title('Weekly Team Totals (Only Modeled Players)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Points', 'Rebounds', 'Assists'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Modeled vs Not Modeled
ax = axes[1, 0]
teams = ['Team Boricua\nSquad', 'KL2 LLC']
modeled = [boricua['has_model'].sum(), kl2['has_model'].sum()]
not_modeled = [(~boricua['has_model']).sum(), (~kl2['has_model']).sum()]

x = np.arange(len(teams))
width = 0.6

bars1 = ax.bar(x, modeled, width, label='Modeled', color='green', alpha=0.7)
bars2 = ax.bar(x, not_modeled, width, bottom=modeled, label='Not Modeled (OUT)', color='gray', alpha=0.7)

ax.set_ylabel('Number of Players', fontsize=11)
ax.set_title('Roster Depth: Modeled vs OUT Players', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(teams)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (m, nm) in enumerate(zip(modeled, not_modeled)):
    ax.text(i, m/2, str(m), ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(i, m + nm/2, str(nm), ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(i, m + nm + 0.3, f'{m}/{m+nm}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 4: Player-Games Advantage
ax = axes[1, 1]
player_games_boricua = boricua['has_model'].sum() * 3
player_games_kl2 = kl2['has_model'].sum() * 3

bars = ax.bar(['Team Boricua Squad', 'KL2 LLC'],
              [player_games_boricua, player_games_kl2],
              color=['blue', 'red'], alpha=0.7)

ax.set_ylabel('Total Player-Games per Week', fontsize=11)
ax.set_title('Total Player-Games per Week (3 games × modeled players)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels and percentage
for i, (bar, pg) in enumerate(zip(bars, [player_games_boricua, player_games_kl2])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)} player-games',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add gap annotation
gap_pct = (player_games_kl2 - player_games_boricua) / player_games_boricua * 100
ax.text(0.5, max(player_games_boricua, player_games_kl2) * 0.5,
        f'KL2 LLC has {gap_pct:.0f}% more\nplayer-games',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.suptitle('Roster Depth Analysis: Why KL2 LLC Dominates',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/Users/rhu/fantasybasketball2/debugging/roster_depth_impact.png',
            dpi=300, bbox_inches='tight')
print("Saved: /Users/rhu/fantasybasketball2/debugging/roster_depth_impact.png")

# Print summary
print("\n" + "="*70)
print("ROSTER DEPTH SUMMARY")
print("="*70)
print(f"\nTeam Boricua Squad:")
print(f"  Modeled players: {boricua['has_model'].sum()}/{len(boricua)}")
print(f"  Player-games per week: {player_games_boricua}")
print(f"  Weekly PTS: {boricua_totals[0]:.1f}")
print(f"  Weekly REB: {boricua_totals[1]:.1f}")
print(f"  Weekly AST: {boricua_totals[2]:.1f}")

print(f"\nKL2 LLC:")
print(f"  Modeled players: {kl2['has_model'].sum()}/{len(kl2)}")
print(f"  Player-games per week: {player_games_kl2}")
print(f"  Weekly PTS: {kl2_totals[0]:.1f}")
print(f"  Weekly REB: {kl2_totals[1]:.1f}")
print(f"  Weekly AST: {kl2_totals[2]:.1f}")

print(f"\nGaps:")
print(f"  Player-games: KL2 LLC +{gap_pct:.0f}%")
print(f"  Points: KL2 LLC +{kl2_totals[0] - boricua_totals[0]:.1f} ({(kl2_totals[0]/boricua_totals[0] - 1)*100:.0f}%)")
print(f"  Rebounds: KL2 LLC +{kl2_totals[1] - boricua_totals[1]:.1f} ({(kl2_totals[1]/boricua_totals[1] - 1)*100:.0f}%)")
print(f"  Assists: KL2 LLC +{kl2_totals[2] - boricua_totals[2]:.1f} ({(kl2_totals[2]/boricua_totals[2] - 1)*100:.0f}%)")

print(f"\nMissing Players from Team Boricua Squad:")
for _, player in boricua[boricua['has_model'] == False].iterrows():
    print(f"  - {player['player']}")

print("\n" + "="*70)
