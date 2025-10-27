"""
Simulate a fantasy basketball draft and season with AI opponents.

- 12 teams, snake draft
- You use H-scoring strategy
- Opponents use Claude AI to make picks
- Then simulate 100 seasons
"""

import pandas as pd
import numpy as np
from draft_assistant import DraftAssistant
import os
import json
import random
import time
from datetime import datetime
import argparse
import unicodedata
import anthropic


class SeasonSimulator:
    """Simulate fantasy basketball seasons."""

    def __init__(self, team_rosters, player_data, player_variances):
        """Initialize season simulator."""
        self.team_rosters = team_rosters
        self.player_data = player_data
        self.player_variances = player_variances

        # Create name mapping for unicode handling
        self.name_mapping = self._create_name_mapping()

        # Calculate player means and stds
        self.player_means = {}
        self.player_stds = {}

        for player_name in player_data['PLAYER_NAME'].unique():
            player_weekly = player_data[player_data['PLAYER_NAME'] == player_name]

            self.player_means[player_name] = {
                'PTS': player_weekly['PTS'].mean(),
                'REB': player_weekly['REB'].mean(),
                'AST': player_weekly['AST'].mean(),
                'STL': player_weekly['STL'].mean(),
                'BLK': player_weekly['BLK'].mean(),
                'TOV': player_weekly['TOV'].mean(),
                'FG3M': player_weekly['FG3M'].mean(),
                'FGM': player_weekly['FGM'].mean(),
                'FGA': player_weekly['FGA'].mean(),
                'FTM': player_weekly['FTM'].mean(),
                'FTA': player_weekly['FTA'].mean(),
                'DD': player_weekly['DD'].mean()
            }

            # Get per-game variance, convert to weekly std
            if player_name in player_variances:
                games_per_week = 3
                self.player_stds[player_name] = {
                    cat: np.sqrt(player_variances[player_name].get(cat, {}).get('var_per_game', 0) * games_per_week)
                    for cat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'FGM', 'FGA', 'FTM', 'FTA', 'DD']
                }
            else:
                self.player_stds[player_name] = {cat: 0 for cat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'FGM', 'FGA', 'FTM', 'FTA', 'DD']}

    def _create_name_mapping(self):
        """Create mapping from normalized names to actual names."""
        name_mapping = {}
        for player_name in self.player_data['PLAYER_NAME'].unique():
            normalized = normalize_name(player_name)
            name_mapping[normalized] = player_name
            name_mapping[player_name] = player_name
        return name_mapping

    def _lookup_player(self, player_name):
        """Look up player with unicode-safe matching."""
        if player_name in self.player_means:
            return player_name
        normalized = normalize_name(player_name)
        if normalized in self.name_mapping:
            return self.name_mapping[normalized]
        return None

    def simulate_matchup(self, team1_roster, team2_roster, games_per_week=3):
        """Simulate one weekly matchup between two teams."""
        team1_stats = self._simulate_team_week(team1_roster, games_per_week)
        team2_stats = self._simulate_team_week(team2_roster, games_per_week)

        categories = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'FG3M', 'FG3_PCT', 'DD']
        team1_wins = 0

        for cat in categories:
            if cat == 'TOV':
                if team1_stats[cat] < team2_stats[cat]:
                    team1_wins += 1
            else:
                if team1_stats[cat] > team2_stats[cat]:
                    team1_wins += 1

        return 1 if team1_wins > 5 else 0

    def _simulate_team_week(self, roster, games_per_week=3):
        """Simulate one week for a team."""
        totals = {
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0,
            'FG3M': 0, 'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'DD': 0
        }

        for player_name in roster:
            actual_name = self._lookup_player(player_name)

            if actual_name and actual_name in self.player_means:
                means = self.player_means[actual_name]
                stds = self.player_stds[actual_name]

                for cat in totals.keys():
                    mean_val = means.get(cat, 0)
                    std_val = stds.get(cat, 0)

                    if std_val > 0:
                        sample = np.random.normal(mean_val, std_val)
                    else:
                        sample = mean_val

                    totals[cat] += max(0, sample)

        stats = totals.copy()
        stats['FG_PCT'] = totals['FGM'] / totals['FGA'] if totals['FGA'] > 0 else 0
        stats['FT_PCT'] = totals['FTM'] / totals['FTA'] if totals['FTA'] > 0 else 0
        stats['FG3_PCT'] = totals['FG3M'] / (totals['FGM'] * 0.3) if totals['FGM'] > 0 else 0

        return stats

    def simulate_multiple_seasons(self, num_seasons=100):
        """Simulate multiple seasons."""
        num_teams = len(self.team_rosters)
        season_wins = {i: 0 for i in range(1, num_teams + 1)}
        season_losses = {i: 0 for i in range(1, num_teams + 1)}

        print(f"\nSimulating {num_seasons} seasons...")

        for season in range(num_seasons):
            if (season + 1) % 10 == 0:
                print(f"  Completed {season + 1}/{num_seasons} seasons")

            for team1 in range(1, num_teams + 1):
                for team2 in range(team1 + 1, num_teams + 1):
                    for _ in range(2):
                        result = self.simulate_matchup(
                            self.team_rosters[team1],
                            self.team_rosters[team2]
                        )

                        if result == 1:
                            season_wins[team1] += 1
                            season_losses[team2] += 1
                        else:
                            season_wins[team2] += 1
                            season_losses[team1] += 1

        results = []
        for team_num in range(1, num_teams + 1):
            wins = season_wins[team_num]
            losses = season_losses[team_num]
            win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0

            results.append({
                'Team': team_num,
                'Wins': wins,
                'Losses': losses,
                'Win_Pct': win_pct
            })

        results_df = pd.DataFrame(results).sort_values('Win_Pct', ascending=False)
        results_df['Rank'] = range(1, len(results_df) + 1)

        return results_df


def normalize_name(name):
    """Normalize player names by removing unicode characters."""
    nfd = unicodedata.normalize('NFD', name)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


class DraftSimulatorVsAI:
    """Simulate draft where opponents use Claude AI."""

    def __init__(self, adp_file, data_file, variance_file, your_position=6, num_teams=12, roster_size=13):
        """Initialize draft simulator."""
        random.seed(time.time() * os.getpid())

        self.your_position = your_position
        self.num_teams = num_teams
        self.roster_size = roster_size

        # Load ADP rankings
        self.adp_df = pd.read_csv(adp_file)
        self.adp_df = self.adp_df.sort_values('ADP')

        # Initialize H-scoring assistant
        self.assistant = DraftAssistant(
            data_file=data_file,
            variance_file=variance_file,
            format='each_category'
        )

        # Initialize Anthropic client
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

        # Track draft state
        self.all_teams = {i: [] for i in range(1, num_teams + 1)}
        self.drafted_players = set()
        self.current_round = 1

        # Cache for AI picks to avoid repetition
        self.ai_pick_history = {i: [] for i in range(1, num_teams + 1)}

    def _normalize_player_name(self, name):
        """Normalize player names for matching."""
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

    def _calculate_team_stats(self, roster):
        """Calculate average stats for a team."""
        if not roster:
            return {}

        stats = {
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0,
            'FG3M': 0, 'FG_PCT': 0, 'FT_PCT': 0, 'DD': 0
        }

        for player in roster:
            for cat in stats.keys():
                x_score = self.assistant.scoring.calculate_x_score(player, cat)
                stats[cat] += x_score

        return stats

    def _opponent_pick_ai(self, team_num):
        """Opponent picks using Claude AI."""
        available = self._get_available_adp_players()

        if not available:
            return None

        # Get current roster
        current_roster = self.all_teams[team_num]
        round_num = len(current_roster) + 1

        # Calculate current team stats
        team_stats = self._calculate_team_stats(current_roster)

        # Format available players (top 30 for token efficiency)
        available_list = []
        for i, p in enumerate(available[:30], 1):
            available_list.append(f"{i}. {p['PLAYER']} ({p['POS']}) - ADP {p['ADP']:.1f}")

        # Build prompt
        prompt = f"""You are an expert fantasy basketball drafter in a 12-team H2H 9-category league.

Categories: Points, Rebounds, Assists, Steals, Blocks, Turnovers, 3PM, FG%, FT%

CURRENT SITUATION:
- Round: {round_num}/13
- Your draft position: Team {team_num}
- Snake draft format

YOUR CURRENT ROSTER ({len(current_roster)} players):
{chr(10).join([f"  - {p}" for p in current_roster]) if current_roster else "  (Empty - first pick)"}

YOUR TEAM'S CATEGORY STRENGTHS (X-scores):
{chr(10).join([f"  {cat}: {score:.2f}" for cat, score in team_stats.items()]) if team_stats else "  (No players yet)"}

AVAILABLE PLAYERS (Top 30 by ADP):
{chr(10).join(available_list)}

INSTRUCTIONS:
1. Consider your team's strengths and weaknesses
2. Balance positional needs (need guards, forwards, centers)
3. Consider punting weak categories vs filling gaps
4. Pick ONE player from the available list

Respond with ONLY the player's name exactly as shown above, nothing else."""

        try:
            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract player name from response
            response_text = message.content[0].text.strip()

            # Try to match response to available players
            for p in available[:30]:
                player_name = p['PLAYER']
                # Check if player name appears in response
                if player_name.lower() in response_text.lower():
                    return player_name

            # Fallback: parse first line and try to extract
            first_line = response_text.split('\n')[0].strip()
            # Remove common prefixes
            for prefix in ['I pick', 'I choose', 'I select', 'My pick is', 'Pick:', 'Player:']:
                first_line = first_line.replace(prefix, '').strip()

            # Try matching again
            for p in available[:30]:
                if p['PLAYER'].lower() in first_line.lower():
                    return p['PLAYER']

            # Last resort: pick top available by ADP
            print(f"  ⚠️  Could not parse AI response: '{response_text[:50]}...', using ADP fallback")
            return available[0]['PLAYER']

        except Exception as e:
            print(f"  ⚠️  API error: {e}, using ADP fallback")
            return available[0]['PLAYER']

    def _your_pick_h_score(self):
        """Your pick using H-scoring."""
        available_adp = self._get_available_adp_players()

        if not available_adp:
            return None

        top_candidates = [p['PLAYER'] for p in available_adp[:50]]

        print(f"\n  Evaluating top {len(top_candidates)} candidates with H-scoring...")

        opponent_rosters = [
            self.all_teams[i] for i in range(1, self.num_teams + 1)
            if i != self.your_position
        ]
        self.assistant.update_opponent_rosters(opponent_rosters)

        recommendations = self.assistant.recommend_pick(
            available_players=top_candidates,
            top_n=10
        )

        if recommendations.empty:
            return available_adp[0]['PLAYER']

        best_player = recommendations.iloc[0]['PLAYER_NAME']

        print(f"\n  Top 5 H-Score Recommendations:")
        for idx, row in recommendations.head(5).iterrows():
            adp_name, adp_rank = self._find_player_in_adp(row['PLAYER_NAME'])
            adp_str = f"ADP: {adp_rank:.1f}" if adp_rank else "ADP: N/A"
            print(f"    {idx+1}. {row['PLAYER_NAME']:25s} | H-Score: {row['H_SCORE']:.4f} | {adp_str}")

        return best_player

    def _execute_pick(self, team_num, player_name):
        """Execute a draft pick."""
        self.all_teams[team_num].append(player_name)
        self.drafted_players.add(player_name)

        if team_num == self.your_position:
            self.assistant.draft_player(player_name)

    def run_draft(self):
        """Run the full draft simulation."""
        print("=" * 80)
        print("FANTASY BASKETBALL DRAFT SIMULATION (vs AI Opponents)")
        print("=" * 80)
        print(f"\nSettings:")
        print(f"  - {self.num_teams} teams, {self.roster_size} roster spots")
        print(f"  - You are drafting at position {self.your_position}")
        print(f"  - Opponents use Claude AI (Sonnet 3.5)")
        print(f"  - You use H-scoring strategy")
        print(f"\nStarting draft...\n")

        pick_num = 0

        for round_num in range(1, self.roster_size + 1):
            print("-" * 80)
            print(f"ROUND {round_num}")
            print("-" * 80)

            if round_num % 2 == 1:
                order = list(range(1, self.num_teams + 1))
            else:
                order = list(range(self.num_teams, 0, -1))

            for team_num in order:
                pick_num += 1

                if team_num == self.your_position:
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
                    print(f"\nPick {pick_num} - Team {team_num} (AI)")
                    player = self._opponent_pick_ai(team_num)

                    if player:
                        adp_name, adp_rank = self._find_player_in_adp(player)
                        adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                        print(f"  → AI DRAFTED: {player} {adp_str}")
                        self._execute_pick(team_num, player)
                    else:
                        print(f"  ✗ No available players!")

        print("\n" + "=" * 80)
        print("DRAFT COMPLETE")
        print("=" * 80)

        self._show_final_results()
        return self.all_teams

    def _show_final_results(self):
        """Show final draft results."""
        print(f"\nYOUR TEAM (Position {self.your_position}):")
        print("-" * 80)

        your_team = self.all_teams[self.your_position]
        for idx, player in enumerate(your_team, 1):
            adp_name, adp_rank = self._find_player_in_adp(player)
            adp_str = f"ADP: {adp_rank:.1f}" if adp_rank else "ADP: N/A"
            print(f"  {idx:2d}. {player:30s} | {adp_str}")

        print("\n\nALL TEAMS (Summary):")
        print("-" * 80)

        for team_num in range(1, self.num_teams + 1):
            team = self.all_teams[team_num]
            strategy = "H-Scoring" if team_num == self.your_position else "Claude AI"
            print(f"\nTeam {team_num} ({strategy}): {len(team)} players")
            for idx, player in enumerate(team[:3], 1):
                adp_name, adp_rank = self._find_player_in_adp(player)
                adp_str = f"(ADP: {adp_rank:.1f})" if adp_rank else ""
                print(f"  {idx}. {player} {adp_str}")
            if len(team) > 3:
                print(f"  ... and {len(team) - 3} more")

        self._save_results()

    def _save_results(self):
        """Save draft results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results = {
            'settings': {
                'your_position': self.your_position,
                'num_teams': self.num_teams,
                'roster_size': self.roster_size
            },
            'your_team': self.all_teams[self.your_position],
            'all_teams': {f'Team_{i}': players for i, players in self.all_teams.items()},
            'strategy': 'H-scoring vs Claude AI'
        }

        output_file = f'draft_results_vs_ai_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n\n✓ Draft results saved to: {output_file}")


def main():
    """Run draft + season simulation with AI opponents."""
    parser = argparse.ArgumentParser(description='Run H-scoring draft + season simulation vs AI')
    parser.add_argument('-p', '--position', type=int, default=6, choices=range(1, 13),
                        help='Your draft position (1-12). Default: 6')
    parser.add_argument('-n', '--num-seasons', type=int, default=100,
                        help='Number of seasons to simulate. Default: 100')
    parser.add_argument('-t', '--num-teams', type=int, default=12, choices=range(2, 21),
                        help='Number of teams in league. Default: 12')

    args = parser.parse_args()

    # Find data files
    data_dir = 'data'

    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    if not weekly_files or not variance_files:
        print("Error: No data files found!")
        print("Please run: python collect_full_data.py")
        return

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    print(f"Using data files:")
    print(f"  - {data_file}")
    print(f"  - {variance_file}")

    adp_file = '/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv'

    # Run draft
    simulator = DraftSimulatorVsAI(
        adp_file=adp_file,
        data_file=data_file,
        variance_file=variance_file,
        your_position=args.position,
        num_teams=args.num_teams,
        roster_size=13
    )

    team_rosters = simulator.run_draft()

    # Load data for season simulation
    print("\n" + "=" * 80)
    print("SEASON SIMULATION")
    print("=" * 80)

    player_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    season_sim = SeasonSimulator(team_rosters, player_data, player_variances)
    results = season_sim.simulate_multiple_seasons(num_seasons=args.num_seasons)

    # Display results
    print("\n" + "=" * 80)
    print(f"SEASON RESULTS ({args.num_seasons} seasons)")
    print("=" * 80)
    print(results.to_string(index=False))

    # Highlight your team
    your_result = results[results['Team'] == args.position].iloc[0]
    print("\n" + "=" * 80)
    print("YOUR TEAM (Team {}) PERFORMANCE:".format(args.position))
    print("=" * 80)
    print(f"  Rank: {your_result['Rank']}/{args.num_teams}")
    print(f"  Wins: {your_result['Wins']}")
    print(f"  Losses: {your_result['Losses']}")
    print(f"  Win %: {your_result['Win_Pct']:.3f}")

    # Save season results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'season_results_vs_ai_{timestamp}.csv'
    results.to_csv(output_file, index=False)
    print(f"\n✓ Season results saved to: {output_file}")


if __name__ == "__main__":
    main()
