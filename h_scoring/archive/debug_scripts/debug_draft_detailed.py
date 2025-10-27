"""
Detailed draft debugging script with H-score calculations.

Shows:
- Detailed H-score calculations for each pick
- Category X-scores
- Team composition before/after
- Why each player was selected
"""

import pandas as pd
import numpy as np
from simulate_draft import DraftSimulator
from draft_assistant import DraftAssistant
import os
import json
import sys
from datetime import datetime


class DetailedDraftDebugger(DraftSimulator):
    """Extended DraftSimulator with detailed H-score logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Detailed logging
        self.pick_logs = []
        self.output_lines = []

    def log(self, message):
        """Log message to both console and output buffer."""
        print(message)
        self.output_lines.append(message)

    def _your_pick_h_score(self):
        """Your pick using H-scoring with detailed logging."""
        # Get available players
        available_adp = self._get_available_adp_players()

        if not available_adp:
            return None

        # Get top candidates by ADP (evaluate top 50 to save time)
        top_candidates = [p['PLAYER'] for p in available_adp[:50]]

        self.log(f"\n  Evaluating top {len(top_candidates)} candidates with H-scoring...")

        # Get H-score recommendations
        # Update opponent rosters for context
        opponent_rosters = [
            self.all_teams[i] for i in range(1, self.num_teams + 1)
            if i != self.your_position
        ]
        self.assistant.update_opponent_rosters(opponent_rosters)

        # Evaluate candidates
        recommendations = self.assistant.recommend_pick(
            available_players=top_candidates,
            top_n=20  # Get more candidates for analysis
        )

        if recommendations.empty:
            # Fallback to ADP if no recommendations
            return available_adp[0]['PLAYER']

        # Pick top H-score player
        best_player = recommendations.iloc[0]['PLAYER_NAME']

        # DETAILED LOGGING - Show top 10 and detailed analysis
        self.log(f"\n  {'='*80}")
        self.log(f"  TOP 10 H-SCORE RECOMMENDATIONS:")
        self.log(f"  {'='*80}")

        for idx, row in recommendations.head(10).iterrows():
            adp_name, adp_rank = self._find_player_in_adp(row['PLAYER_NAME'])
            adp_str = f"ADP: {adp_rank:5.1f}" if adp_rank else "ADP:   N/A"

            self.log(f"    {idx+1:2d}. {row['PLAYER_NAME']:25s} | H: {row['H_SCORE']:6.4f} | G: {row['G_SCORE']:6.4f} | {adp_str}")

        # Show detailed analysis for top pick
        self.log(f"\n  {'='*80}")
        self.log(f"  DETAILED ANALYSIS - YOUR PICK: {best_player}")
        self.log(f"  {'='*80}")

        # Get player X-scores and details
        top_pick = recommendations.iloc[0]

        # Get player from assistant's scoring system
        from modules.scoring import PlayerScoring
        scoring = self.assistant.scoring

        # Show X-scores by category
        self.log(f"\n  X-Scores by Category:")
        self.log(f"  {'-'*60}")

        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'DD', 'FG_PCT', 'FT_PCT', 'FG3_PCT']

        for cat in categories:
            x_score = scoring.calculate_x_score(best_player, cat)
            g_score = scoring.calculate_g_score(best_player, cat)
            self.log(f"    {cat:8s}: X = {x_score:7.3f} | G = {g_score:7.3f}")

        # Show current team state
        self.log(f"\n  Current Team Composition ({len(self.all_teams[self.your_position])} players):")
        self.log(f"  {'-'*60}")

        if len(self.all_teams[self.your_position]) > 0:
            for i, player in enumerate(self.all_teams[self.your_position], 1):
                self.log(f"    {i}. {player}")
        else:
            self.log(f"    (Empty - First pick)")

        # Calculate team X-scores before and after
        self.log(f"\n  Team X-Score Totals (Before → After Adding {best_player}):")
        self.log(f"  {'-'*60}")

        team_before = {cat: 0 for cat in categories}
        team_after = {cat: 0 for cat in categories}

        # Current team totals
        for player in self.all_teams[self.your_position]:
            for cat in categories:
                team_before[cat] += scoring.calculate_x_score(player, cat)
                team_after[cat] += scoring.calculate_x_score(player, cat)

        # Add new player
        for cat in categories:
            team_after[cat] += scoring.calculate_x_score(best_player, cat)

        for cat in categories:
            delta = team_after[cat] - team_before[cat]
            self.log(f"    {cat:8s}: {team_before[cat]:7.2f} → {team_after[cat]:7.2f} (Δ {delta:+7.2f})")

        # Show why this pick vs next best
        if len(recommendations) > 1:
            next_best = recommendations.iloc[1]
            h_diff = top_pick['H_SCORE'] - next_best['H_SCORE']
            g_diff = top_pick['G_SCORE'] - next_best['G_SCORE']

            self.log(f"\n  Why {best_player} over {next_best['PLAYER_NAME']}?")
            self.log(f"  {'-'*60}")
            self.log(f"    H-Score Advantage: {h_diff:+.4f}")
            self.log(f"    G-Score Advantage: {g_diff:+.4f}")

            # Compare key categories
            self.log(f"\n    Category Comparison:")
            comparison_cats = ['PTS', 'REB', 'AST', 'BLK', 'DD', 'FG_PCT']
            for cat in comparison_cats:
                x1 = scoring.calculate_x_score(best_player, cat)
                x2 = scoring.calculate_x_score(next_best['PLAYER_NAME'], cat)
                diff = x1 - x2
                if abs(diff) > 0.5:  # Only show significant differences
                    symbol = "✓" if diff > 0 else "✗"
                    self.log(f"      {symbol} {cat:8s}: {x1:6.2f} vs {x2:6.2f} (Δ {diff:+6.2f})")

        self.log(f"  {'='*80}\n")

        # Store detailed pick info
        pick_info = {
            'pick_number': len(self.all_teams[self.your_position]) + 1,
            'player': best_player,
            'h_score': top_pick['H_SCORE'],
            'g_score': top_pick['G_SCORE'],
            'adp_rank': adp_rank if adp_rank else None,
            'team_before': list(self.all_teams[self.your_position]),
            'x_scores': {cat: scoring.calculate_x_score(best_player, cat) for cat in categories}
        }
        self.pick_logs.append(pick_info)

        return best_player

    def run_draft(self):
        """Run the full draft simulation with detailed logging."""
        self.log("=" * 80)
        self.log("DETAILED DRAFT DEBUGGING - H-SCORE CALCULATIONS")
        self.log("=" * 80)
        self.log(f"\nSettings:")
        self.log(f"  - {self.num_teams} teams, {self.roster_size} roster spots")
        self.log(f"  - You are drafting at position {self.your_position}")
        self.log(f"  - Snake draft format")
        self.log(f"  - Opponents use Balanced ADP strategy")
        self.log(f"  - You use H-scoring strategy")
        self.log(f"\nStarting draft...\n")

        pick_num = 0

        for round_num in range(1, self.roster_size + 1):
            self.log("-" * 80)
            self.log(f"ROUND {round_num}")
            self.log("-" * 80)

            # Determine order for this round
            if round_num % 2 == 1:  # Odd round
                order = list(range(1, self.num_teams + 1))
            else:  # Even round (snake)
                order = list(range(self.num_teams, 0, -1))

            for team_num in order:
                pick_num += 1

                if team_num == self.your_position:
                    # Your pick using H-scoring with detailed logging
                    self.log(f"\n{'#'*80}")
                    self.log(f"# Pick {pick_num} - YOUR TURN (Team {team_num}, Round {round_num})")
                    self.log(f"{'#'*80}")

                    player = self._your_pick_h_score()

                    if player:
                        adp_name, adp_rank = self._find_player_in_adp(player)
                        adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                        self.log(f"\n  ✓✓✓ YOU DRAFTED: {player} {adp_str} ✓✓✓\n")
                        self._execute_pick(team_num, player)
                    else:
                        self.log(f"  ✗ No available players!")

                else:
                    # Opponent pick using ADP (less verbose)
                    player = self._opponent_pick_adp(team_num)

                    if player:
                        adp_name, adp_rank = self._find_player_in_adp(player)
                        adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                        self.log(f"Pick {pick_num:3d} - Team {team_num:2d}: {player:30s} {adp_str}")
                        self._execute_pick(team_num, player)

        # Draft complete
        self.log("\n" + "=" * 80)
        self.log("DRAFT COMPLETE")
        self.log("=" * 80)

        self._show_final_summary()

    def _show_final_summary(self):
        """Show final draft summary with analysis."""
        self.log(f"\n{'='*80}")
        self.log(f"YOUR FINAL TEAM (Position {self.your_position})")
        self.log(f"{'='*80}\n")

        your_team = self.all_teams[self.your_position]

        self.log(f"{'Round':>5} {'Overall':>8} {'Player':30s} {'ADP':>8} {'H-Score':>10} {'G-Score':>10}")
        self.log("-" * 80)

        for idx, player in enumerate(your_team, 1):
            adp_name, adp_rank = self._find_player_in_adp(player)
            adp_str = f"{adp_rank:.1f}" if adp_rank else "N/A"

            # Find pick info
            pick_info = next((p for p in self.pick_logs if p['player'] == player), None)
            h_score_str = f"{pick_info['h_score']:.4f}" if pick_info else "N/A"
            g_score_str = f"{pick_info['g_score']:.4f}" if pick_info else "N/A"

            # Calculate overall pick number (approximate)
            if idx <= 6:  # First 6 picks
                overall = (idx - 1) * 12 + self.your_position
            elif idx <= 12:  # Picks 7-12
                overall = 6 * 12 + ((idx - 7) * 12) + (13 - self.your_position)
            else:
                overall = 12 * 12 + ((idx - 13) * 12) + self.your_position

            self.log(f"{idx:5d} {overall:8d} {player:30s} {adp_str:>8s} {h_score_str:>10s} {g_score_str:>10s}")

        # Calculate team totals
        self.log(f"\n{'='*80}")
        self.log(f"TEAM X-SCORE TOTALS BY CATEGORY")
        self.log(f"{'='*80}\n")

        from modules.scoring import PlayerScoring
        scoring = self.assistant.scoring

        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'DD', 'FG_PCT', 'FT_PCT', 'FG3_PCT']
        team_totals = {cat: 0 for cat in categories}

        for player in your_team:
            for cat in categories:
                team_totals[cat] += scoring.calculate_x_score(player, cat)

        # Sort by absolute value
        sorted_cats = sorted(categories, key=lambda c: abs(team_totals[c]), reverse=True)

        self.log(f"{'Category':12s} {'Total X-Score':>15s} {'Strength'}")
        self.log("-" * 50)

        for cat in sorted_cats:
            total = team_totals[cat]
            if total > 10:
                strength = "★★★ Elite"
            elif total > 5:
                strength = "★★ Strong"
            elif total > 0:
                strength = "★ Average"
            elif total > -5:
                strength = "⚠ Weak"
            else:
                strength = "✗ Punt"

            self.log(f"{cat:12s} {total:15.2f} {strength}")

        # Save results
        self._save_detailed_results()

    def _save_detailed_results(self):
        """Save detailed results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save raw output
        raw_output_file = f'draft_debug_raw_{timestamp}.txt'
        with open(raw_output_file, 'w') as f:
            f.write('\n'.join(self.output_lines))

        print(f"\n✓ Raw output saved to: {raw_output_file}")

        # Save structured JSON
        results = {
            'timestamp': timestamp,
            'settings': {
                'your_position': self.your_position,
                'num_teams': self.num_teams,
                'roster_size': self.roster_size
            },
            'your_picks': self.pick_logs,
            'your_team': self.all_teams[self.your_position],
            'all_teams': {f'Team_{i}': players for i, players in self.all_teams.items()}
        }

        json_file = f'draft_debug_detailed_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Detailed JSON saved to: {json_file}")

        return raw_output_file, json_file


def main():
    """Run detailed draft debugging."""

    # Find most recent data files
    data_dir = 'data'

    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        return

    # Use most recent files
    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print(f"Using data files:")
    print(f"  - {data_file}")
    print(f"  - {variance_file}\n")

    # ADP file
    adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    # Initialize and run detailed debugging
    debugger = DetailedDraftDebugger(
        adp_file=adp_file,
        data_file=data_file,
        variance_file=variance_file,
        your_position=6,
        num_teams=12,
        roster_size=13
    )

    debugger.run_draft()


if __name__ == "__main__":
    main()
