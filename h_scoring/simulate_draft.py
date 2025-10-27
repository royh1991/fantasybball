"""
Simulate a fantasy basketball draft.

- 12 teams, snake draft
- You draft at position 6
- Opponents use ADP strategy (pick highest available ADP)
- You use H-scoring strategy
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os
import json
import random
import time


class DraftSimulator:
    """Simulate a fantasy basketball draft."""

    def __init__(self, adp_file, data_file, variance_file, your_position=6, num_teams=12, roster_size=13):
        """
        Initialize draft simulator.

        Parameters:
        -----------
        adp_file : str
            Path to ADP rankings CSV
        data_file : str
            Path to weekly player data
        variance_file : str
            Path to player variances JSON
        your_position : int
            Your draft position (1-indexed)
        num_teams : int
            Number of teams in league
        roster_size : int
            Roster size per team
        """
        # Seed random number generator with time + process ID for uniqueness
        random.seed(time.time() * os.getpid())

        self.your_position = your_position
        self.num_teams = num_teams
        self.roster_size = roster_size

        # Load ADP rankings
        self.adp_df = pd.read_csv(adp_file)
        self.adp_df = self.adp_df.sort_values('ADP')

        # Initialize H-scoring assistant for your picks
        self.assistant = DraftAssistant(
            data_file=data_file,
            variance_file=variance_file,
            format='each_category'
        )

        # Track draft state
        self.all_teams = {i: [] for i in range(1, num_teams + 1)}
        self.drafted_players = set()
        self.current_round = 1
        self.draft_order = []

        # Generate snake draft order
        self._generate_draft_order()

    def _generate_draft_order(self):
        """Generate snake draft order."""
        for round_num in range(1, self.roster_size + 1):
            if round_num % 2 == 1:  # Odd rounds: 1->12
                self.draft_order.extend(range(1, self.num_teams + 1))
            else:  # Even rounds: 12->1
                self.draft_order.extend(range(self.num_teams, 0, -1))

    def _normalize_player_name(self, name):
        """Normalize player names for matching."""
        # Remove extra whitespace and convert to lowercase
        return ' '.join(name.strip().lower().split())

    def _find_player_in_adp(self, player_name):
        """Find player in ADP rankings (fuzzy match)."""
        normalized_target = self._normalize_player_name(player_name)

        for idx, row in self.adp_df.iterrows():
            normalized_adp = self._normalize_player_name(row['PLAYER'])
            if normalized_adp == normalized_target:
                return row['PLAYER'], row['ADP']

        return None, None

    def _get_available_adp_players(self):
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
        return available

    def _opponent_pick_adp(self, team_num):
        """
        Opponent picks with balanced roster building + randomness.

        Strategy:
        - Early rounds (1-4): Pick from top 5-10 with randomness
        - Mid rounds (5-9): Weight towards needed positions
        - Late rounds (10-13): Balance roster + fill gaps
        """
        import random

        available = self._get_available_adp_players()

        if not available:
            return None

        # Get current roster for this team
        current_roster = self.all_teams[team_num]
        round_num = len(current_roster) + 1

        # Count positions already drafted
        position_counts = {'G': 0, 'F': 0, 'C': 0}
        for player_name in current_roster:
            player_info = next((p for p in self.adp_df.iterrows() if p[1]['PLAYER'] == player_name), None)
            if player_info:
                pos = player_info[1]['POS']
                if 'G' in pos:
                    position_counts['G'] += 1
                if 'F' in pos:
                    position_counts['F'] += 1
                if 'C' in pos:
                    position_counts['C'] += 1

        # Determine pool size based on round with MORE variance
        if round_num <= 4:
            # Early rounds: WIDER pool to create draft diversity
            pool_size = random.randint(5, 12)
            position_weight = 0.3  # Light position weighting
            noise_range = 1.5  # Larger noise to shake up picks
        elif round_num <= 9:
            # Mid rounds: balance value and need
            pool_size = random.randint(8, 15)
            position_weight = 0.5  # Moderate position weighting
            noise_range = 1.0
        else:
            # Late rounds: fill gaps
            pool_size = random.randint(10, 20)
            position_weight = 0.8  # Strong position weighting
            noise_range = 0.8

        # Get top candidates
        pool_size = min(pool_size, len(available))
        candidates = available[:pool_size]

        # Score each candidate (lower is better)
        candidate_scores = []
        for i, candidate in enumerate(candidates):
            # Base score from ADP rank
            adp_score = i  # 0 for best ADP, higher for worse

            # Position need score
            pos = candidate['POS']
            position_need = 0

            # Calculate how much we need each position (lower count = more need)
            if 'C' in pos and position_counts['C'] < 2:  # Need at least 2 centers
                position_need -= 2
            if 'G' in pos and position_counts['G'] < 4:  # Need at least 4 guards
                position_need -= 1
            if 'F' in pos and position_counts['F'] < 4:  # Need at least 4 forwards
                position_need -= 1

            # Combined score (lower is better)
            total_score = adp_score + (position_need * position_weight)

            # Add LARGER random noise to create real variance
            random_noise = random.uniform(-noise_range, noise_range)
            total_score += random_noise

            candidate_scores.append((candidate, total_score))

        # Sort by score and pick the best
        candidate_scores.sort(key=lambda x: x[1])
        best_candidate = candidate_scores[0][0]

        return best_candidate['PLAYER']

    def _your_pick_h_score(self):
        """Your pick using H-scoring."""
        # Get available players
        available_adp = self._get_available_adp_players()

        if not available_adp:
            return None

        # Get top candidates by ADP (evaluate top 50 to save time)
        top_candidates = [p['PLAYER'] for p in available_adp[:50]]

        print(f"\n  Evaluating top {len(top_candidates)} candidates with H-scoring...")

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
            top_n=10
        )

        if recommendations.empty:
            # Fallback to ADP if no recommendations
            return available_adp[0]['PLAYER']

        # Pick top H-score player
        best_player = recommendations.iloc[0]['PLAYER_NAME']

        # Show top 5 for context
        print(f"\n  Top 5 H-Score Recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            adp_name, adp_rank = self._find_player_in_adp(row['PLAYER_NAME'])
            adp_str = f"ADP: {adp_rank:.1f}" if adp_rank else "ADP: N/A"
            print(f"    {idx+1}. {row['PLAYER_NAME']:25s} | H-Score: {row['H_SCORE']:.2f} | G-Score: {row['G_SCORE']:.2f} | {adp_str}")

        return best_player

    def _execute_pick(self, team_num, player_name):
        """Execute a draft pick."""
        self.all_teams[team_num].append(player_name)
        self.drafted_players.add(player_name)

        # Update your assistant's team if it's your pick
        if team_num == self.your_position:
            self.assistant.draft_player(player_name)

    def run_draft(self):
        """Run the full draft simulation."""
        print("=" * 80)
        print("FANTASY BASKETBALL DRAFT SIMULATION")
        print("=" * 80)
        print(f"\nSettings:")
        print(f"  - {self.num_teams} teams, {self.roster_size} roster spots")
        print(f"  - You are drafting at position {self.your_position}")
        print(f"  - Opponents use Balanced ADP strategy (value + position needs + randomness)")
        print(f"  - You use H-scoring strategy")
        print(f"\nStarting draft...\n")

        pick_num = 0

        for round_num in range(1, self.roster_size + 1):
            print("-" * 80)
            print(f"ROUND {round_num}")
            print("-" * 80)

            # Determine order for this round
            if round_num % 2 == 1:  # Odd round
                order = list(range(1, self.num_teams + 1))
            else:  # Even round (snake)
                order = list(range(self.num_teams, 0, -1))

            for team_num in order:
                pick_num += 1

                if team_num == self.your_position:
                    # Your pick using H-scoring
                    print(f"\nPick {pick_num} - YOUR TURN (Team {team_num})")
                    player = self._your_pick_h_score()

                    if player:
                        adp_name, adp_rank = self._find_player_in_adp(player)
                        adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                        print(f"  ✓ YOU DRAFTED: {player} {adp_str}")
                        self._execute_pick(team_num, player)
                    else:
                        print(f"  ✗ No available players!")

                else:
                    # Opponent pick using ADP
                    player = self._opponent_pick_adp(team_num)

                    if player:
                        adp_name, adp_rank = self._find_player_in_adp(player)
                        adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                        print(f"Pick {pick_num} - Team {team_num}: {player} {adp_str}")
                        self._execute_pick(team_num, player)
                    else:
                        print(f"Pick {pick_num} - Team {team_num}: No available players!")

        # Draft complete
        print("\n" + "=" * 80)
        print("DRAFT COMPLETE")
        print("=" * 80)

        self._show_final_results()

    def _show_final_results(self):
        """Show final draft results."""
        print(f"\nYOUR TEAM (Position {self.your_position}):")
        print("-" * 80)

        your_team = self.all_teams[self.your_position]
        for idx, player in enumerate(your_team, 1):
            adp_name, adp_rank = self._find_player_in_adp(player)
            adp_str = f"ADP: {adp_rank:.1f}" if adp_rank else "ADP: N/A"
            print(f"  {idx:2d}. {player:30s} | {adp_str}")

        # Show opponent teams summary
        print("\n\nOPPONENT TEAMS (Summary):")
        print("-" * 80)

        for team_num in range(1, self.num_teams + 1):
            if team_num == self.your_position:
                continue

            team = self.all_teams[team_num]
            print(f"\nTeam {team_num}: {len(team)} players")
            # Show first 3 picks
            for idx, player in enumerate(team[:3], 1):
                adp_name, adp_rank = self._find_player_in_adp(player)
                adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                print(f"  {idx}. {player} {adp_str}")
            if len(team) > 3:
                print(f"  ... and {len(team) - 3} more")

        # Save results
        self._save_results()

    def _save_results(self):
        """Save draft results to file."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results = {
            'settings': {
                'your_position': self.your_position,
                'num_teams': self.num_teams,
                'roster_size': self.roster_size
            },
            'your_team': self.all_teams[self.your_position],
            'all_teams': {f'Team_{i}': players for i, players in self.all_teams.items()},
            'strategy': 'H-scoring vs ADP'
        }

        output_file = f'draft_results_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n\n✓ Draft results saved to: {output_file}")


def main():
    """Run draft simulation."""

    # Find most recent data files
    data_dir = 'data'

    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        print("Please run: python test_data_collector.py or python collect_full_data.py")
        return

    # Use most recent files
    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print(f"Using data files:")
    print(f"  - {data_file}")
    print(f"  - {variance_file}")

    # ADP file
    adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    # Initialize and run simulation
    simulator = DraftSimulator(
        adp_file=adp_file,
        data_file=data_file,
        variance_file=variance_file,
        your_position=6,
        num_teams=12,
        roster_size=13
    )

    simulator.run_draft()


if __name__ == "__main__":
    main()