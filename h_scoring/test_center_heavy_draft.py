"""
Test H-scoring algorithm with forced center-heavy draft.

Purpose: Validate that H-scoring correctly adapts to team needs.
- Position 5
- 3 rounds only
- Force centers for rounds 1 and 2
- Analyze what H-scoring recommends for round 3

This tests whether the algorithm truly recognizes team weaknesses and adjusts.
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os
import json
from datetime import datetime


class CenterHeavyDraftTest:
    """Test H-scoring with forced center picks."""

    def __init__(self, adp_file, data_file, variance_file):
        """Initialize test."""
        self.your_position = 5
        self.num_teams = 12
        self.num_rounds = 3

        # Load ADP
        self.adp_df = pd.read_csv(adp_file)
        self.adp_df = self.adp_df.sort_values('ADP')

        # Initialize H-scoring
        self.assistant = DraftAssistant(
            data_file=data_file,
            variance_file=variance_file,
            format='each_category'
        )

        # Track draft
        self.all_teams = {i: [] for i in range(1, self.num_teams + 1)}
        self.drafted_players = set()

        # Categories for analysis
        self.categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M',
                          'FG_PCT', 'FT_PCT', 'FG3_PCT', 'DD']

        # Results storage
        self.test_results = {
            'draft_log': [],
            'team_composition_after_r2': {},
            'round_3_analysis': {},
            'h_score_recommendations': []
        }

    def _get_available_centers(self):
        """Get available center-eligible players by ADP."""
        centers = []
        for idx, row in self.adp_df.iterrows():
            if row['PLAYER'] in self.drafted_players:
                continue
            if 'C' in str(row['POS']):
                centers.append({
                    'PLAYER': row['PLAYER'],
                    'ADP': row['ADP'],
                    'POS': row['POS']
                })
        return centers

    def _get_available_players(self):
        """Get all available players."""
        available = []
        for idx, row in self.adp_df.iterrows():
            if row['PLAYER'] not in self.drafted_players:
                available.append({
                    'PLAYER': row['PLAYER'],
                    'ADP': row['ADP'],
                    'POS': row['POS']
                })
        return available

    def _opponent_pick(self, team_num):
        """Opponent picks best available by ADP."""
        available = self._get_available_players()
        if available:
            return available[0]['PLAYER']
        return None

    def _execute_pick(self, team_num, player_name, round_num, pick_type):
        """Execute and log a pick."""
        self.all_teams[team_num].append(player_name)
        self.drafted_players.add(player_name)

        if team_num == self.your_position:
            self.assistant.draft_player(player_name)

        # Log pick
        player_row = self.adp_df[self.adp_df['PLAYER'] == player_name].iloc[0]
        self.test_results['draft_log'].append({
            'round': round_num,
            'team': team_num,
            'player': player_name,
            'position': player_row['POS'],
            'adp': player_row['ADP'],
            'pick_type': pick_type
        })

    def _analyze_team_composition(self):
        """Analyze your team composition after 2 centers."""
        print("\n" + "=" * 80)
        print("TEAM COMPOSITION ANALYSIS (After 2 Centers)")
        print("=" * 80)

        your_team = self.all_teams[self.your_position]
        print(f"\nYour Roster:")
        for i, player in enumerate(your_team, 1):
            player_row = self.adp_df[self.adp_df['PLAYER'] == player].iloc[0]
            print(f"  {i}. {player:30s} | {player_row['POS']:5s} | ADP: {player_row['ADP']:.1f}")

        # Get X-scores for your team
        print(f"\n\nYour Team's Category Strengths (X-Scores):")
        print("-" * 80)

        team_x_scores = {}
        for cat in self.categories:
            total = 0.0
            for player in your_team:
                x_score = self.assistant.scoring.calculate_x_score(player, cat)
                total += x_score
            team_x_scores[cat] = total

        # Sort by strength
        sorted_cats = sorted(team_x_scores.items(), key=lambda x: x[1], reverse=True)

        for cat, score in sorted_cats:
            strength = ""
            if score > 4:
                strength = "★★★ VERY STRONG"
            elif score > 2:
                strength = "★★ STRONG"
            elif score > 0:
                strength = "★ POSITIVE"
            elif score > -2:
                strength = "✗ WEAK"
            else:
                strength = "✗✗ VERY WEAK"

            print(f"  {cat:10s}: {score:6.2f}  {strength}")

        self.test_results['team_composition_after_r2'] = {
            'roster': your_team,
            'category_scores': team_x_scores,
            'sorted_strengths': sorted_cats
        }

        return team_x_scores

    def _analyze_round3_recommendations(self):
        """Get detailed H-score recommendations for round 3."""
        print("\n\n" + "=" * 80)
        print("ROUND 3 H-SCORE ANALYSIS")
        print("=" * 80)

        # Get available players (top 60 by ADP)
        available = self._get_available_players()[:60]
        available_names = [p['PLAYER'] for p in available]

        print(f"\nEvaluating top {len(available_names)} available players...")

        # Update opponent rosters
        opponent_rosters = [
            self.all_teams[i] for i in range(1, self.num_teams + 1)
            if i != self.your_position
        ]
        self.assistant.update_opponent_rosters(opponent_rosters)

        # Get recommendations
        recommendations = self.assistant.recommend_pick(
            available_players=available_names,
            top_n=20
        )

        # Show top 20 recommendations
        print(f"\nTop 20 H-Score Recommendations:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Player':<30} {'Pos':<8} {'ADP':<8} {'H-Score':<10} {'G-Score':<10}")
        print("-" * 80)

        top20_analysis = []
        for idx, row in recommendations.head(20).iterrows():
            player = row['PLAYER_NAME']
            h_score = row['H_SCORE']
            g_score = row['G_SCORE']

            # Get position and ADP
            player_info = next((p for p in available if p['PLAYER'] == player), None)
            if player_info:
                pos = player_info['POS']
                adp = player_info['ADP']
            else:
                pos = "N/A"
                adp = 999

            # Get X-scores for this player
            player_x = {}
            for cat in self.categories:
                player_x[cat] = self.assistant.scoring.calculate_x_score(player, cat)

            print(f"{idx+1:<6} {player:<30} {pos:<8} {adp:<8.1f} {h_score:<10.4f} {g_score:<10.4f}")

            top20_analysis.append({
                'rank': idx + 1,
                'player': player,
                'position': pos,
                'adp': adp,
                'h_score': h_score,
                'g_score': g_score,
                'x_scores': player_x
            })

        # Detailed analysis of top 5
        print("\n\n" + "=" * 80)
        print("DETAILED ANALYSIS: Top 5 Recommendations")
        print("=" * 80)

        for i, player_data in enumerate(top20_analysis[:5], 1):
            print(f"\n{i}. {player_data['player']} ({player_data['position']}) - ADP {player_data['adp']:.1f}")
            print(f"   H-Score: {player_data['h_score']:.4f} | G-Score: {player_data['g_score']:.4f}")
            print(f"\n   Category Contributions (X-Scores):")

            # Sort by X-score
            sorted_x = sorted(player_data['x_scores'].items(), key=lambda x: x[1], reverse=True)
            for cat, x_score in sorted_x:
                if abs(x_score) > 0.1:  # Only show meaningful contributions
                    print(f"     {cat:10s}: {x_score:6.2f}")

        self.test_results['round_3_analysis'] = {
            'top_20': top20_analysis,
            'recommendation_count': len(recommendations)
        }

        return recommendations

    def _analyze_position_diversity(self, recommendations):
        """Analyze position diversity in recommendations."""
        print("\n\n" + "=" * 80)
        print("POSITION DIVERSITY ANALYSIS")
        print("=" * 80)

        # Count positions in top 10
        position_counts = {}
        for idx, row in recommendations.head(10).iterrows():
            player = row['PLAYER_NAME']
            player_info = self.adp_df[self.adp_df['PLAYER'] == player]
            if not player_info.empty:
                pos = player_info.iloc[0]['POS']
                if pos not in position_counts:
                    position_counts[pos] = 0
                position_counts[pos] += 1

        print("\nTop 10 Recommendations by Position:")
        for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pos:10s}: {count} players")

        # Check if guards/wings are prioritized (expected after 2 centers)
        guard_count = sum(c for p, c in position_counts.items() if 'G' in p)
        forward_count = sum(c for p, c in position_counts.items() if 'F' in p and 'C' not in p)
        center_count = sum(c for p, c in position_counts.items() if 'C' in p and 'G' not in p and 'F' not in p)

        print(f"\nAggregated Position Distribution:")
        print(f"  Guards (G, PG, SG):  {guard_count}")
        print(f"  Forwards (F, SF, PF): {forward_count}")
        print(f"  Centers (C):          {center_count}")

        return position_counts

    def run_test(self):
        """Run the 3-round draft test."""
        print("=" * 80)
        print("CENTER-HEAVY DRAFT TEST")
        print("=" * 80)
        print("\nTest Parameters:")
        print(f"  - Position: {self.your_position}")
        print(f"  - Rounds: {self.num_rounds}")
        print(f"  - Forced picks: Centers in rounds 1 & 2")
        print(f"  - Analysis: H-scoring recommendations for round 3")
        print("\n" + "=" * 80)

        # ROUND 1
        print("\n\nROUND 1")
        print("-" * 80)

        # Opponents before you
        for team in range(1, self.your_position):
            player = self._opponent_pick(team)
            if player:
                print(f"Pick {team} - Team {team}: {player}")
                self._execute_pick(team, player, 1, 'opponent_adp')

        # YOUR PICK - Force center
        print(f"\nPick {self.your_position} - YOUR TURN (Team {self.your_position})")
        centers = self._get_available_centers()
        if centers:
            forced_pick = centers[0]['PLAYER']
            print(f"  ★ FORCED CENTER PICK: {forced_pick} (ADP: {centers[0]['ADP']:.1f})")
            self._execute_pick(self.your_position, forced_pick, 1, 'forced_center')

        # Rest of round 1
        for team in range(self.your_position + 1, self.num_teams + 1):
            player = self._opponent_pick(team)
            if player:
                print(f"Pick {team} - Team {team}: {player}")
                self._execute_pick(team, player, 1, 'opponent_adp')

        # ROUND 2 (Snake - reverse order)
        print("\n\nROUND 2")
        print("-" * 80)

        # Opponents before you (reverse order)
        for team in range(self.num_teams, self.your_position, -1):
            player = self._opponent_pick(team)
            if player:
                print(f"Pick {team + 12 - 1} - Team {team}: {player}")
                self._execute_pick(team, player, 2, 'opponent_adp')

        # YOUR PICK - Force another center
        print(f"\nPick {12 + self.num_teams - self.your_position + 1} - YOUR TURN (Team {self.your_position})")
        centers = self._get_available_centers()
        if centers:
            forced_pick = centers[0]['PLAYER']
            print(f"  ★ FORCED CENTER PICK: {forced_pick} (ADP: {centers[0]['ADP']:.1f})")
            self._execute_pick(self.your_position, forced_pick, 2, 'forced_center')

        # Rest of round 2
        for team in range(self.your_position - 1, 0, -1):
            player = self._opponent_pick(team)
            if player:
                print(f"Pick {24 - team + 1} - Team {team}: {player}")
                self._execute_pick(team, player, 2, 'opponent_adp')

        # ANALYZE TEAM AFTER 2 CENTERS
        team_x_scores = self._analyze_team_composition()

        # ROUND 3 ANALYSIS
        recommendations = self._analyze_round3_recommendations()

        # Position diversity
        position_counts = self._analyze_position_diversity(recommendations)

        # Save results
        self._save_results()

        return self.test_results

    def _save_results(self):
        """Save test results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'center_heavy_test_{timestamp}.json'

        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\n\n{'=' * 80}")
        print(f"✓ Test results saved to: {output_file}")
        print("=" * 80)


def main():
    """Run the test."""
    # Find data files
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])
    adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    print(f"Using data files:")
    print(f"  - {data_file}")
    print(f"  - {variance_file}")
    print(f"  - {adp_file}\n")

    # Run test
    test = CenterHeavyDraftTest(adp_file, data_file, variance_file)
    test.run_test()


if __name__ == "__main__":
    main()
