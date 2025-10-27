#!/usr/bin/env python3
"""
Interactive Draft Assistant CLI for H-Scoring Algorithm

Usage:
    python draft_assistant_cli.py

This will guide you through your draft, showing H-score recommendations
when it's your turn, and allowing you to input other teams' picks.
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os
import json
from datetime import datetime
import sys


class InteractiveDraftAssistant:
    """Interactive CLI for fantasy basketball draft."""

    def __init__(self, data_file, variance_file, adp_file):
        """Initialize the draft assistant."""

        # League settings
        self.num_teams = 14
        self.roster_size = 13
        self.your_position = 5

        # Team names in draft order
        self.team_names = [
            "coolest",
            "hardowod",
            "jay stat",
            "team boruica",
            "my team",  # Position 5
            "Nadim",
            "team perez",
            "enter the dragon",
            "LF da brocoli",
            "team too",
            "retutrn burito",
            "team menyo",
            "mamba",
            "bde"
        ]

        # Load ADP for reference
        self.adp_df = pd.read_csv(adp_file)
        self.adp_df = self.adp_df.sort_values('ADP')

        # Initialize H-scoring
        print("\n" + "="*80)
        print("INITIALIZING H-SCORING DRAFT ASSISTANT")
        print("="*80)
        print(f"\nLoading data...")

        self.assistant = DraftAssistant(
            data_file=data_file,
            variance_file=variance_file,
            format='each_category'
        )

        # Track draft state
        self.all_teams = {name: [] for name in self.team_names}
        self.drafted_players = set()
        self.current_round = 1
        self.current_pick = 0

        # Auto-save file
        self.save_file = f"draft_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        print(f"✓ Initialization complete!")
        print(f"\nDraft will be auto-saved to: {self.save_file}")

    def _get_draft_order(self, round_num):
        """Get draft order for a given round (snake draft)."""
        if round_num % 2 == 1:
            # Odd rounds: 1->14
            return list(range(14))
        else:
            # Even rounds: 14->1
            return list(range(13, -1, -1))

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

    def _show_team_summary(self, team_name):
        """Show current roster for a team."""
        roster = self.all_teams[team_name]
        if not roster:
            return "  (No players yet)"

        result = []
        for i, player in enumerate(roster, 1):
            _, adp, pos = self._find_player_in_adp(player)
            adp_str = f"ADP {adp:.1f}" if adp else "ADP N/A"
            pos_str = pos if pos else "N/A"
            result.append(f"  {i}. {player:30s} ({pos_str:10s}) {adp_str}")
        return "\n".join(result)

    def _your_turn(self):
        """Handle your pick with H-score recommendations."""
        print("\n" + "="*80)
        print("★ YOUR TURN ★")
        print("="*80)

        # Show your current roster
        print(f"\n【 YOUR CURRENT ROSTER ({len(self.all_teams['my team'])}/13) 】")
        print(self._show_team_summary('my team'))

        # Get available players
        available = self._get_available_players(top_n=60)
        available_names = [p['PLAYER'] for p in available]

        # Update opponent rosters for context
        opponent_rosters = [
            self.all_teams[name] for name in self.team_names
            if name != 'my team'
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

    def _opponent_pick(self, team_name):
        """Handle opponent's pick."""
        print(f"\n{'─'*80}")
        print(f"Team: {team_name}")
        print(f"{'─'*80}")

        # Show their current roster
        print(f"\n【 {team_name.upper()} ROSTER ({len(self.all_teams[team_name])}/13) 】")
        print(self._show_team_summary(team_name))

        # Show top available by ADP for reference
        available = self._get_available_players(top_n=10)
        print(f"\n📋 Top 10 available by ADP:")
        for i, p in enumerate(available, 1):
            print(f"  {i}. {p['PLAYER']:30s} ({p['POS']:10s}) ADP {p['ADP']:.1f}")

        # Prompt for their pick
        print("\n" + "─"*80)
        while True:
            player_input = input(f"Who did {team_name} draft? ").strip()

            if not player_input:
                print("❌ Please enter a player name")
                continue

            # Find the player
            full_name, adp, pos = self._find_player_in_adp(player_input)

            # Show what we found
            if adp:
                print(f"✓ {full_name} ({pos}) - ADP {adp:.1f}")
            else:
                print(f"⚠️  {full_name} (not in ADP list)")

            confirm = input("Correct? (y/n): ").strip().lower()
            if confirm == 'y':
                return full_name

    def _execute_pick(self, team_name, player_name):
        """Execute a draft pick."""
        self.all_teams[team_name].append(player_name)
        self.drafted_players.add(player_name)

        if team_name == 'my team':
            self.assistant.draft_player(player_name)

        self.current_pick += 1

    def _save_draft_state(self):
        """Auto-save draft state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'current_round': self.current_round,
            'current_pick': self.current_pick,
            'all_teams': self.all_teams,
            'drafted_players': list(self.drafted_players)
        }

        with open(self.save_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_draft_state(self, save_file):
        """Load draft state from file."""
        with open(save_file, 'r') as f:
            state = json.load(f)

        self.current_round = state['current_round']
        self.current_pick = state['current_pick']
        self.all_teams = state['all_teams']
        self.drafted_players = set(state['drafted_players'])

        # Update H-scoring assistant with drafted players
        my_team = self.all_teams.get('my team', [])
        for player in my_team:
            self.assistant.draft_player(player)

        print(f"\n✓ Loaded draft state from {save_file}")
        print(f"  Current pick: {self.current_pick}")
        print(f"  Current round: {self.current_round}")
        print(f"  Your roster: {len(my_team)} players")

    def run_draft(self):
        """Run the interactive draft."""
        print("\n" + "="*80)
        print("🏀 FANTASY BASKETBALL DRAFT 🏀")
        print("="*80)
        print(f"\nLeague: 14 teams, 13 roster spots")
        print(f"Your team: '{self.team_names[self.your_position - 1]}' (Position {self.your_position})")
        print(f"\nStarting draft...\n")

        try:
            for round_num in range(1, self.roster_size + 1):
                self.current_round = round_num

                print("\n" + "="*80)
                print(f"ROUND {round_num}")
                print("="*80)

                # Get draft order for this round
                order = self._get_draft_order(round_num)

                for position_idx in order:
                    team_name = self.team_names[position_idx]

                    print(f"\n\n{'='*80}")
                    print(f"Pick {self.current_pick + 1} (Round {round_num})")
                    print(f"{'='*80}")

                    if team_name == 'my team':
                        # Your pick
                        player = self._your_turn()
                    else:
                        # Opponent's pick
                        player = self._opponent_pick(team_name)

                    # Execute the pick
                    self._execute_pick(team_name, player)

                    # Confirm
                    _, adp, pos = self._find_player_in_adp(player)
                    adp_str = f"(ADP {adp:.1f})" if adp else ""
                    print(f"\n✓ DRAFTED: {team_name} → {player} {adp_str}")

                    # Auto-save after each pick
                    self._save_draft_state()

            # Draft complete
            self._show_final_results()

        except KeyboardInterrupt:
            print("\n\n⚠️  Draft interrupted. Progress saved to:", self.save_file)
            sys.exit(0)

    def _show_final_results(self):
        """Show final draft results."""
        print("\n\n" + "="*80)
        print("🎉 DRAFT COMPLETE! 🎉")
        print("="*80)

        print(f"\n\n【 YOUR FINAL ROSTER 】")
        print("="*80)
        print(self._show_team_summary('my team'))

        print(f"\n\n【 ALL TEAMS 】")
        print("="*80)
        for team_name in self.team_names:
            print(f"\n{team_name.upper()}:")
            roster = self.all_teams[team_name]
            for i, player in enumerate(roster[:3], 1):
                _, adp, pos = self._find_player_in_adp(player)
                adp_str = f"ADP {adp:.1f}" if adp else ""
                print(f"  {i}. {player} ({pos}) {adp_str}")
            if len(roster) > 3:
                print(f"  ... and {len(roster) - 3} more")

        print(f"\n\n✓ Draft saved to: {self.save_file}")


def main():
    """Main entry point."""
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

    # Check for existing save files
    save_files = sorted([f for f in os.listdir('.') if f.startswith('draft_session_') and f.endswith('.json')])

    if save_files:
        print("\n" + "="*80)
        print("FOUND EXISTING DRAFT SESSIONS")
        print("="*80)
        for i, f in enumerate(save_files, 1):
            # Get timestamp from filename
            timestamp = f.replace('draft_session_', '').replace('.json', '')
            print(f"  {i}. {f} ({timestamp})")

        print("\n" + "="*80)
        choice = input("\nResume from saved draft? (enter number or 'n' for new): ").strip()

        if choice != 'n' and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(save_files):
                # Resume existing draft
                draft = InteractiveDraftAssistant(data_file, variance_file, adp_file)
                draft._load_draft_state(save_files[idx])
                draft.run_draft()
                return

    # Start new draft
    draft = InteractiveDraftAssistant(data_file, variance_file, adp_file)
    draft.run_draft()


if __name__ == "__main__":
    main()
