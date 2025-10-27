#!/usr/bin/env python3
"""
Restart draft using actual picks from other teams.
You can re-do your picks while others keep their actual selections.
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os
import json
from datetime import datetime

# Actual draft order (from your league)
ACTUAL_DRAFT = {
    # Round 1
    1: ("Victor Wembanyama", "Coolest Beans"),
    2: ("Nikola Jokic", "Hardwood Hustlers"),
    3: ("Shai Gilgeous-Alexander", "Jay Stat"),
    4: ("Luka Doncic", "Team Boricua Squad"),
    5: ("Anthony Davis", "Not Like Russ"),  # YOUR PICK
    6: ("Giannis Antetokounmpo", "Nadim the Dream"),
    7: ("James Harden", "Team perez"),
    8: ("Cade Cunningham", "Enter the Dragon"),
    9: ("Anthony Edwards", "LF da broccoli"),
    10: ("Karl-Anthony Towns", "TEAM TOO ICEY BOY 12"),
    11: ("Stephen Curry", "Return of Burrito"),
    12: ("Trae Young", "Team Menyo"),
    13: ("Devin Booker", "IN MAMBA WE TRUST"),
    14: ("Amen Thompson", "BDE"),

    # Round 2
    15: ("Domantas Sabonis", "BDE"),
    16: ("Kevin Durant", "IN MAMBA WE TRUST"),
    17: ("Evan Mobley", "Team Menyo"),
    18: ("Ivica Zubac", "Return of Burrito"),
    19: ("Tyrese Maxey", "TEAM TOO ICEY BOY 12"),
    20: ("Chet Holmgren", "LF da broccoli"),
    21: ("Josh Giddey", "Enter the Dragon"),
    22: ("LaMelo Ball", "Team perez"),
    23: ("Jaylen Brown", "Nadim the Dream"),
    24: ("Nikola Vucevic", "Not Like Russ"),  # YOUR PICK
    25: ("Jalen Williams", "Team Boricua Squad"),
    26: ("Donovan Mitchell", "Jay Stat"),
    27: ("Jalen Brunson", "Hardwood Hustlers"),
    28: ("Pascal Siakam", "Coolest Beans"),

    # Round 3
    29: ("LeBron James", "Coolest Beans"),
    30: ("Joel Embiid", "Hardwood Hustlers"),
    31: ("Paolo Banchero", "Jay Stat"),
    32: ("Scottie Barnes", "Team Boricua Squad"),
    33: ("Jamal Murray", "Not Like Russ"),  # YOUR PICK
    34: ("Alperen Sengun", "Nadim the Dream"),
    35: ("Bam Adebayo", "Team perez"),
    36: ("Jalen Johnson", "Enter the Dragon"),
    37: ("Derrick White", "LF da broccoli"),
    38: ("Desmond Bane", "TEAM TOO ICEY BOY 12"),
    39: ("Cooper Flagg", "Return of Burrito"),
    40: ("De'Aaron Fox", "Team Menyo"),
    41: ("Ja Morant", "IN MAMBA WE TRUST"),
    42: ("Trey Murphy III", "BDE"),

    # Round 4
    43: ("Dyson Daniels", "BDE"),
    44: ("Zion Williamson", "IN MAMBA WE TRUST"),
    45: ("Franz Wagner", "Team Menyo"),
    46: ("Austin Reaves", "Return of Burrito"),
    47: ("Myles Turner", "TEAM TOO ICEY BOY 12"),
    48: ("Mikal Bridges", "LF da broccoli"),
    49: ("Walker Kessler", "Enter the Dragon"),
    50: ("Jaren Jackson Jr.", "Team perez"),
    51: ("Coby White", "Nadim the Dream"),
    52: ("Jordan Poole", "Not Like Russ"),  # YOUR PICK
    53: ("Josh Hart", "Team Boricua Squad"),
    54: ("Jalen Duren", "Jay Stat"),
    55: ("Jarrett Allen", "Hardwood Hustlers"),
    56: ("Zach LaVine", "Coolest Beans"),

    # Round 5
    57: ("Miles Bridges", "Coolest Beans"),
    58: ("Brandon Miller", "Hardwood Hustlers"),
    59: ("Darius Garland", "Jay Stat"),
    60: ("Michael Porter Jr.", "Team Boricua Squad"),
    61: ("Reed Sheppard", "Not Like Russ"),  # YOUR PICK
    62: ("DeMar DeRozan", "Nadim the Dream"),
    63: ("Deni Avdija", "Team perez"),
    64: ("Jalen Green", "Enter the Dragon"),
    65: ("OG Anunoby", "LF da broccoli"),
    66: ("Tyler Herro", "TEAM TOO ICEY BOY 12"),
    67: ("Kawhi Leonard", "Return of Burrito"),
    68: ("Kristaps Porzingis", "Team Menyo"),
}

# Your picks in the draft (14-team snake)
YOUR_PICKS = [5, 24, 33, 52, 61, 80, 89, 108, 117, 136, 145, 164, 173]

# Team names in order
TEAM_NAMES = [
    "Coolest Beans",           # 1
    "Hardwood Hustlers",       # 2
    "Jay Stat",                # 3
    "Team Boricua Squad",      # 4
    "Not Like Russ",           # 5 - YOU
    "Nadim the Dream",         # 6
    "Team perez",              # 7
    "Enter the Dragon",        # 8
    "LF da broccoli",          # 9
    "TEAM TOO ICEY BOY 12",    # 10
    "Return of Burrito",       # 11
    "Team Menyo",              # 12
    "IN MAMBA WE TRUST",       # 13
    "BDE"                      # 14
]


class RestartDraft:
    def __init__(self, data_file, variance_file, adp_file):
        """Initialize with actual draft history."""
        self.adp_df = pd.read_csv(adp_file)
        self.adp_df = self.adp_df.sort_values('ADP')

        # Initialize H-scoring
        print("\n" + "="*80)
        print("RESTARTING DRAFT WITH ACTUAL PICKS")
        print("="*80)
        print(f"\nLoading data...")

        self.assistant = DraftAssistant(
            data_file=data_file,
            variance_file=variance_file,
            format='each_category'
        )

        # Track all teams
        self.all_teams = {name: [] for name in TEAM_NAMES}
        self.drafted_players = set()

        # Auto-save file
        self.save_file = f"draft_restart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        print(f"✓ Initialization complete!")
        print(f"\nDraft will be auto-saved to: {self.save_file}")

    def _find_player_in_adp(self, player_name):
        """Find player in ADP rankings."""
        player_lower = player_name.strip().lower()

        for idx, row in self.adp_df.iterrows():
            if row['PLAYER'].lower() == player_lower:
                return row['PLAYER'], row['ADP'], row['POS']

        # Try partial match
        for idx, row in self.adp_df.iterrows():
            if player_lower in row['PLAYER'].lower() or row['PLAYER'].lower() in player_lower:
                return row['PLAYER'], row['ADP'], row['POS']

        return player_name, None, None

    def _get_available_players(self, top_n=50):
        """Get available players sorted by ADP."""
        available = []
        for idx, row in self.adp_df.iterrows():
            player_name = row['PLAYER']
            if player_name not in self.drafted_players:
                available.append({
                    'PLAYER': player_name,
                    'ADP': row['ADP'],
                    'POS': row['POS']
                })
                if len(available) >= top_n:
                    break
        return available

    def _your_turn(self, pick_num):
        """Handle your pick with H-score recommendations."""
        print("\n" + "="*80)
        print("★ YOUR TURN ★")
        print("="*80)

        # Show your current roster
        print(f"\n【 YOUR ROSTER ({len(self.all_teams['Not Like Russ'])}/13) 】")
        for i, player in enumerate(self.all_teams['Not Like Russ'], 1):
            _, adp, pos = self._find_player_in_adp(player)
            adp_str = f"ADP {adp:.1f}" if adp else "ADP N/A"
            pos_str = pos if pos else "N/A"
            print(f"  {i}. {player:30s} ({pos_str:10s}) {adp_str}")

        if not self.all_teams['Not Like Russ']:
            print("  (No players yet)")

        # Get available players
        available = self._get_available_players(top_n=60)
        available_names = [p['PLAYER'] for p in available]

        # Update opponent rosters
        opponent_rosters = [
            self.all_teams[name] for name in TEAM_NAMES
            if name != 'Not Like Russ'
        ]
        self.assistant.update_opponent_rosters(opponent_rosters)

        # Get H-score recommendations
        print(f"\n⚙️  Calculating H-scores for top {len(available_names)} available players...")
        recommendations = self.assistant.recommend_pick(
            available_players=available_names,
            top_n=20
        )

        # Display recommendations
        print("\n" + "="*80)
        print("📊 TOP 10 H-SCORE RECOMMENDATIONS")
        print("="*80)
        print(f"\n{'Rank':<6} {'Player':<30} {'Pos':<10} {'ADP':<8} {'H-Score':<10}")
        print("-"*80)

        for idx, row in recommendations.head(10).iterrows():
            player = row['PLAYER_NAME']
            h_score = row['H_SCORE']

            # Get position and ADP
            _, adp, pos = self._find_player_in_adp(player)
            adp_str = f"{adp:.1f}" if adp else "N/A"
            pos_str = pos if pos else "N/A"

            print(f"{idx+1:<6} {player:<30} {pos_str:<10} {adp_str:<8} {h_score:.4f}")

        # Show category analysis for top pick
        print("\n" + "-"*80)
        print(f"📈 TOP RECOMMENDATION ANALYSIS: {recommendations.iloc[0]['PLAYER_NAME']}")
        print("-"*80)

        top_player = recommendations.iloc[0]['PLAYER_NAME']
        print(f"\nCategory contributions (X-Scores):")

        categories = self.assistant.optimizer.categories
        player_x = []
        for cat in categories:
            x = self.assistant.scoring.calculate_x_score(top_player, cat)
            if abs(x) > 0.1:
                player_x.append((cat, x))

        player_x.sort(key=lambda x: x[1], reverse=True)
        for cat, x in player_x:
            symbol = "▲" if x > 0 else "▼"
            print(f"  {cat:10s}: {x:6.2f} {symbol}")

        # Prompt for your pick
        print("\n" + "="*80)
        while True:
            player_input = input("Enter player you're drafting: ").strip()

            if not player_input:
                print("❌ Please enter a player name")
                continue

            # Find the player
            full_name, adp, pos = self._find_player_in_adp(player_input)

            # Confirm
            if adp:
                print(f"\n✓ Found: {full_name} ({pos}) - ADP {adp:.1f}")
            else:
                print(f"\n⚠️  Player not found in ADP list: {full_name}")

            confirm = input("Confirm this pick? (y/n): ").strip().lower()
            if confirm == 'y':
                return full_name

    def _execute_pick(self, pick_num, player_name, team_name):
        """Execute a draft pick."""
        self.all_teams[team_name].append(player_name)
        self.drafted_players.add(player_name)

        if team_name == 'Not Like Russ':
            self.assistant.draft_player(player_name)

    def _save_draft_state(self):
        """Auto-save draft state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'all_teams': self.all_teams,
            'drafted_players': list(self.drafted_players)
        }

        with open(self.save_file, 'w') as f:
            json.dump(state, f, indent=2)

    def run_restart(self, start_from_pick=1):
        """Restart draft from specific pick number."""
        print("\n" + "="*80)
        print("🏀 RESTARTING DRAFT 🏀")
        print("="*80)
        print(f"\nYou are: 'Not Like Russ' (Position 5)")
        print(f"Starting from pick #{start_from_pick}")
        print(f"\n")

        # Process all picks
        for pick_num in range(1, 183):  # 14 teams * 13 rounds = 182 picks

            # Skip picks before start_from_pick
            if pick_num < start_from_pick:
                if pick_num in ACTUAL_DRAFT:
                    player_name, team_name = ACTUAL_DRAFT[pick_num]
                    self._execute_pick(pick_num, player_name, team_name)
                continue

            round_num = ((pick_num - 1) // 14) + 1

            print(f"\n\n{'='*80}")
            print(f"Pick {pick_num} (Round {round_num})")
            print(f"{'='*80}")

            # Is it your turn?
            if pick_num in YOUR_PICKS:
                # Your pick - let H-scoring recommend
                player = self._your_turn(pick_num)
                team_name = "Not Like Russ"
            else:
                # Use actual pick from draft history
                if pick_num in ACTUAL_DRAFT:
                    player, team_name = ACTUAL_DRAFT[pick_num]

                    _, adp, pos = self._find_player_in_adp(player)
                    adp_str = f"(ADP {adp:.1f})" if adp else ""
                    pos_str = f"({pos})" if pos else ""

                    print(f"\n{team_name} selects: {player} {pos_str} {adp_str}")
                else:
                    # We don't have this pick in history - skip
                    print(f"\n(Pick {pick_num} not in history - ending)")
                    break

            # Execute the pick
            self._execute_pick(pick_num, player, team_name)

            # Confirm
            _, adp, pos = self._find_player_in_adp(player)
            adp_str = f"(ADP {adp:.1f})" if adp else ""
            print(f"\n✓ DRAFTED: {team_name} → {player} {adp_str}")

            # Auto-save
            self._save_draft_state()

            # Stop after round 5 (we only have data through pick 68)
            if pick_num >= 68:
                print("\n\n" + "="*80)
                print("📋 REACHED END OF AVAILABLE DRAFT HISTORY")
                print("="*80)
                print("\nContinuing draft with remaining picks...\n")
                # Continue for your remaining picks
                if pick_num >= 182:
                    break

        # Show results
        self._show_results()

    def _show_results(self):
        """Show final results."""
        print("\n\n" + "="*80)
        print("📊 YOUR FINAL ROSTER")
        print("="*80)

        for i, player in enumerate(self.all_teams['Not Like Russ'], 1):
            _, adp, pos = self._find_player_in_adp(player)
            adp_str = f"ADP {adp:.1f}" if adp else ""
            pos_str = pos if pos else "N/A"
            print(f"  {i}. {player:30s} ({pos_str:10s}) {adp_str}")

        print(f"\n\n✓ Draft saved to: {self.save_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Restart draft from specific pick')
    parser.add_argument('-p', '--pick', type=int, default=52,
                       help='Pick number to start from (default: 52 - your round 4 pick)')
    args = parser.parse_args()

    # Find data files
    data_dir = 'data'

    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("❌ Error: No data files found!")
        print("Please run: python collect_full_data.py")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])
    adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    # Run restart
    draft = RestartDraft(data_file, variance_file, adp_file)
    draft.run_restart(start_from_pick=args.pick)


if __name__ == "__main__":
    main()
