"""
Weekly Fantasy Basketball Projection System

Uses adaptive Bayesian modeling to project weekly matchups.
Simulates 3 games per player, 500 matchup iterations.
Evaluates 11 categories: FG%, FT%, 3P%, 3PM, PTS, REB, AST, STL, BLK, TO, DD
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def parse_date(date_str: str) -> datetime:
    """Parse dates in format 'OCT 25, 2023' to datetime."""
    try:
        return datetime.strptime(date_str.strip(), "%b %d, %Y")
    except:
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except:
            return None


def adaptive_bayesian_update(prev_mean, prev_var, new_obs, obs_var, evolution_var):
    """Kalman-style Bayesian update for evolving stat rates."""
    K = (prev_var + evolution_var) / (prev_var + evolution_var + obs_var)
    new_mean = prev_mean + K * (new_obs - prev_mean)
    new_var = (1 - K) * (prev_var + evolution_var)
    return new_mean, new_var


class FantasyProjectionModel:
    """
    Adaptive Bayesian model for all fantasy basketball stats.
    Projects: FGM, FGA, FTM, FTA, 3PM, 3PA, PTS, REB, AST, STL, BLK, TOV
    """

    def __init__(self, evolution_rate: float = 0.5):
        # Attempt stats (sample from Poisson)
        self.attempt_stats = ['FGA', 'FTA', 'FG3A']
        # Counting stats (sample from Poisson)
        self.counting_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        # Shooting percentages (for conditional sampling)
        self.shooting_pcts = ['FG_PCT', 'FT_PCT', 'FG3_PCT']

        self.evolution_rate = evolution_rate
        self.distributions = {}
        self.percentages = {}

    def fit_player(self, historical_data: pd.DataFrame, player_nba_name: str):
        """Fit adaptive model for a player using all their historical data."""
        # Case-insensitive comparison to handle mapping inconsistencies
        player_data = historical_data[historical_data['PLAYER_NAME'].str.lower() == player_nba_name.lower()].copy()

        if len(player_data) == 0:
            return False

        # Parse and sort by date
        player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
        player_data = player_data.dropna(subset=['parsed_date'])
        player_data = player_data.sort_values('parsed_date')

        # Split into pre-2024-25 and first 10 games of 2024-25
        cutoff_date = datetime(2024, 10, 1)
        training_data = player_data[player_data['parsed_date'] < cutoff_date]
        season_2024 = player_data[player_data['parsed_date'] >= cutoff_date]

        # For rookies with no historical data, use all their 2024-25 games as training
        if len(training_data) < 5 and len(season_2024) >= 5:
            # Rookie - use all 2024-25 data
            training_data = season_2024
            season_2024 = pd.DataFrame()  # No update data

        if len(training_data) < 5:  # Still need minimum data
            return False

        # Initialize distributions for attempt stats and counting stats
        all_stats = self.attempt_stats + self.counting_stats
        for stat in all_stats:
            if stat not in training_data.columns:
                continue

            values = training_data[stat].values
            mean_val = np.mean(values)
            var_val = np.var(values)
            n_games = len(values)
            initial_uncertainty = var_val / n_games

            self.distributions[stat] = {
                'mean': mean_val,
                'var': var_val,
                'obs_var': var_val,
                'posterior_mean': mean_val,
                'posterior_var': initial_uncertainty
            }

        # Initialize shooting percentages
        for pct_stat in self.shooting_pcts:
            if pct_stat not in training_data.columns:
                continue

            values = training_data[pct_stat].values
            # Remove NaN and values outside [0, 1]
            values = values[(~np.isnan(values)) & (values >= 0) & (values <= 1)]

            if len(values) == 0:
                continue

            mean_val = np.mean(values)
            var_val = np.var(values)
            n_games = len(values)
            initial_uncertainty = var_val / n_games

            self.percentages[pct_stat] = {
                'mean': mean_val,
                'var': var_val,
                'obs_var': var_val,
                'posterior_mean': mean_val,
                'posterior_var': initial_uncertainty
            }

        # Adaptive update if 2024-25 data available (for veterans)
        if len(season_2024) > 0:
            n_update = min(10, len(season_2024))
            update_data = season_2024.head(n_update)

            # Update attempt stats and counting stats
            for stat in all_stats:
                if stat not in update_data.columns or stat not in self.distributions:
                    continue

                posterior_mean = self.distributions[stat]['posterior_mean']
                posterior_var = self.distributions[stat]['posterior_var']
                obs_var = self.distributions[stat]['obs_var']
                evolution_var = self.evolution_rate * obs_var / n_update

                for value in update_data[stat].values:
                    posterior_mean, posterior_var = adaptive_bayesian_update(
                        posterior_mean, posterior_var, value, obs_var, evolution_var
                    )

                self.distributions[stat]['posterior_mean'] = posterior_mean
                self.distributions[stat]['posterior_var'] = posterior_var

            # Update shooting percentages
            for pct_stat in self.shooting_pcts:
                if pct_stat not in update_data.columns or pct_stat not in self.percentages:
                    continue

                # Filter valid percentage values
                values = update_data[pct_stat].values
                values = values[(~np.isnan(values)) & (values >= 0) & (values <= 1)]

                if len(values) == 0:
                    continue

                posterior_mean = self.percentages[pct_stat]['posterior_mean']
                posterior_var = self.percentages[pct_stat]['posterior_var']
                obs_var = self.percentages[pct_stat]['obs_var']
                evolution_var = self.evolution_rate * obs_var / len(values)

                for value in values:
                    posterior_mean, posterior_var = adaptive_bayesian_update(
                        posterior_mean, posterior_var, value, obs_var, evolution_var
                    )

                self.percentages[pct_stat]['posterior_mean'] = posterior_mean
                self.percentages[pct_stat]['posterior_var'] = posterior_var

        return True

    def fit_from_espn_projection(self, espn_row):
        """Fit model using ESPN season projections (for rookies/players without historical data)."""
        # Mapping for attempt and counting stats
        espn_to_stat = {
            'FGA': 'FGA',
            'FTA': 'FTA',
            '3PA': 'FG3A',
            'PTS': 'PTS',
            'TREB': 'REB',
            'AST': 'AST',
            'STL': 'STL',
            'BLK': 'BLK',
            'TO': 'TOV'
        }

        # Mapping for shooting percentages
        espn_to_pct = {
            'FG%': 'FG_PCT',
            'FT%': 'FT_PCT',
            '3P%': 'FG3_PCT'
        }

        games_played = espn_row['GP']
        if games_played == 0:
            return False

        # Convert season totals to per-game averages for attempt/counting stats
        for espn_col, stat_name in espn_to_stat.items():
            if espn_col not in espn_row or pd.isna(espn_row[espn_col]):
                continue

            per_game = espn_row[espn_col] / games_played

            # Use a reasonable variance estimate (assume CV ~0.4 for counting stats)
            variance = (0.4 * per_game) ** 2 if per_game > 0 else 0.1

            self.distributions[stat_name] = {
                'mean': per_game,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': per_game,
                'posterior_var': variance / 10  # Lower uncertainty for ESPN projections
            }

        # Set shooting percentages
        for espn_col, pct_name in espn_to_pct.items():
            if espn_col not in espn_row or pd.isna(espn_row[espn_col]):
                continue

            pct_value = espn_row[espn_col]

            # Use a reasonable variance estimate for percentages
            variance = 0.02  # ~14% std for shooting percentages

            self.percentages[pct_name] = {
                'mean': pct_value,
                'var': variance,
                'obs_var': variance,
                'posterior_mean': pct_value,
                'posterior_var': variance / 10
            }

        return True

    def simulate_game(self) -> Dict:
        """Simulate a single game using correlated sampling for shooting stats."""
        stats = {}

        # Step 1: Sample attempt stats from Poisson
        for attempt_stat in self.attempt_stats:
            if attempt_stat in self.distributions:
                mean = self.distributions[attempt_stat]['posterior_mean']
                value = np.random.poisson(max(0, mean))
                stats[attempt_stat] = max(1, value)  # At least 1 attempt to avoid division by zero
            else:
                stats[attempt_stat] = 1

        # Step 2: Sample makes using Binomial(attempts, percentage)
        # FGM from FGA
        if 'FG_PCT' in self.percentages:
            fg_pct = max(0.0, min(1.0, self.percentages['FG_PCT']['posterior_mean']))
            stats['FGM'] = np.random.binomial(stats['FGA'], fg_pct)
        else:
            stats['FGM'] = 0

        # FTM from FTA
        if 'FT_PCT' in self.percentages:
            ft_pct = max(0.0, min(1.0, self.percentages['FT_PCT']['posterior_mean']))
            stats['FTM'] = np.random.binomial(stats['FTA'], ft_pct)
        else:
            stats['FTM'] = 0

        # 3PM from 3PA
        if 'FG3_PCT' in self.percentages:
            fg3_pct = max(0.0, min(1.0, self.percentages['FG3_PCT']['posterior_mean']))
            stats['FG3M'] = np.random.binomial(stats['FG3A'], fg3_pct)
        else:
            stats['FG3M'] = 0

        # Step 3: Enforce constraint that 3PM can't exceed FGM
        stats['FG3M'] = min(stats['FG3M'], stats['FGM'])

        # Step 4: Sample other counting stats from Poisson
        for counting_stat in self.counting_stats:
            if counting_stat in self.distributions:
                mean = self.distributions[counting_stat]['posterior_mean']
                value = np.random.poisson(max(0, mean))
                stats[counting_stat] = max(0, value)
            else:
                stats[counting_stat] = 0

        # Step 5: Calculate double-double
        dd_stats = [stats['PTS'], stats['REB'], stats['AST'], stats['STL'], stats['BLK']]
        stats['DD'] = 1 if sum(s >= 10 for s in dd_stats) >= 2 else 0

        return stats


def load_data():
    """Load all necessary data files."""
    print("Loading data files...")

    roster = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/roster_snapshots/roster_latest.csv')
    matchups = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/matchups_latest.csv')
    historical = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    mapping = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/mappings/player_mapping_latest.csv')
    espn_projections = pd.read_csv('/Users/rhu/fantasybasketball2/data/fantasy_basketball_clean2.csv')

    print(f"  Roster: {len(roster)} player-team entries")
    print(f"  Matchups: {len(matchups)} matchups")
    print(f"  Historical: {len(historical)} game logs")
    print(f"  Mapping: {len(mapping)} player mappings")
    print(f"  ESPN Projections: {len(espn_projections)} players")

    return roster, matchups, historical, mapping, espn_projections


def fit_player_models(roster: pd.DataFrame, historical: pd.DataFrame,
                     mapping: pd.DataFrame, espn_projections: pd.DataFrame) -> Dict:
    """Fit adaptive Bayesian models for all rostered players."""
    print("\nFitting player models...")

    # Create mapping dict from ESPN name to NBA API name
    name_map = dict(zip(mapping['espn_name'], mapping['nba_api_name']))

    # Create ESPN projections lookup by player name
    espn_proj_dict = {row['PLAYER']: row for _, row in espn_projections.iterrows()}

    player_models = {}
    active_players = roster[roster['currently_rostered'] == True]

    for idx, player_row in active_players.iterrows():
        espn_name = player_row['player_name']
        injury_status = player_row['injury_status']

        # Skip injured players
        if injury_status in ['OUT']:
            print(f"  Skipping {espn_name} (OUT)")
            continue

        # Get NBA API name
        nba_name = name_map.get(espn_name)
        if not nba_name:
            print(f"  Warning: No mapping for {espn_name}")
            continue

        # Fit model - try historical data first, then ESPN projections
        model = FantasyProjectionModel(evolution_rate=0.5)
        success = model.fit_player(historical, nba_name)

        if not success:
            # Try ESPN projections as fallback
            if espn_name in espn_proj_dict:
                success = model.fit_from_espn_projection(espn_proj_dict[espn_name])
                if success:
                    print(f"  Using ESPN projections for {espn_name}")
                else:
                    print(f"  Warning: Insufficient data for {espn_name}")
            else:
                print(f"  Warning: No historical or ESPN data for {espn_name}")

        if success:
            player_models[espn_name] = model

    print(f"\nSuccessfully fitted {len(player_models)} player models")
    return player_models


def simulate_team_week(team_roster: List[str], player_models: Dict,
                      n_games: int = 3) -> Dict:
    """Simulate a team's weekly performance (n games per player)."""
    team_stats = {
        'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
        'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
    }

    for player_name in team_roster:
        if player_name not in player_models:
            continue

        model = player_models[player_name]

        # Simulate n_games for this player
        for _ in range(n_games):
            game_stats = model.simulate_game()
            for stat in team_stats:
                if stat in game_stats:
                    team_stats[stat] += game_stats[stat]

    return team_stats


def calculate_category_winner(team_a_stats: Dict, team_b_stats: Dict) -> Dict:
    """
    Compare two teams across 11 categories.
    Returns dict with category results.
    """
    results = {}

    # Percentage categories (higher is better)
    for pct_stat, makes, attempts in [('FG%', 'FGM', 'FGA'),
                                       ('FT%', 'FTM', 'FTA'),
                                       ('3P%', 'FG3M', 'FG3A')]:
        a_pct = team_a_stats[makes] / team_a_stats[attempts] if team_a_stats[attempts] > 0 else 0
        b_pct = team_b_stats[makes] / team_b_stats[attempts] if team_b_stats[attempts] > 0 else 0

        if a_pct > b_pct:
            results[pct_stat] = 'A'
        elif b_pct > a_pct:
            results[pct_stat] = 'B'
        else:
            results[pct_stat] = 'TIE'

    # Counting stats (higher is better, except TO which is lower is better)
    counting_stats = ['FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'DD']
    for stat in counting_stats:
        if team_a_stats[stat] > team_b_stats[stat]:
            results[stat] = 'A'
        elif team_b_stats[stat] > team_a_stats[stat]:
            results[stat] = 'B'
        else:
            results[stat] = 'TIE'

    # Turnovers (lower is better)
    if team_a_stats['TOV'] < team_b_stats['TOV']:
        results['TO'] = 'A'
    elif team_b_stats['TOV'] < team_a_stats['TOV']:
        results['TO'] = 'B'
    else:
        results['TO'] = 'TIE'

    return results


def simulate_matchup(team_a_roster: List[str], team_b_roster: List[str],
                    player_models: Dict, n_simulations: int = 500) -> Dict:
    """
    Simulate a matchup n_simulations times and calculate win probability.
    """
    team_a_wins = 0
    team_b_wins = 0
    ties = 0

    category_wins_a = {cat: 0 for cat in ['FG%', 'FT%', '3P%', 'FG3M', 'PTS', 'REB',
                                           'AST', 'STL', 'BLK', 'TO', 'DD']}
    category_wins_b = {cat: 0 for cat in category_wins_a.keys()}

    for _ in range(n_simulations):
        # Simulate weekly stats for both teams
        team_a_stats = simulate_team_week(team_a_roster, player_models, n_games=3)
        team_b_stats = simulate_team_week(team_b_roster, player_models, n_games=3)

        # Compare categories
        category_results = calculate_category_winner(team_a_stats, team_b_stats)

        # Count category wins
        a_cats = sum(1 for v in category_results.values() if v == 'A')
        b_cats = sum(1 for v in category_results.values() if v == 'B')

        if a_cats > b_cats:
            team_a_wins += 1
        elif b_cats > a_cats:
            team_b_wins += 1
        else:
            ties += 1

        # Track category win rates
        for cat, winner in category_results.items():
            if winner == 'A':
                category_wins_a[cat] += 1
            elif winner == 'B':
                category_wins_b[cat] += 1

    return {
        'team_a_wins': team_a_wins,
        'team_b_wins': team_b_wins,
        'ties': ties,
        'team_a_win_pct': team_a_wins / n_simulations,
        'team_b_win_pct': team_b_wins / n_simulations,
        'category_win_rates_a': {k: v / n_simulations for k, v in category_wins_a.items()},
        'category_win_rates_b': {k: v / n_simulations for k, v in category_wins_b.items()},
    }


def main():
    """Main execution."""
    print("="*80)
    print("WEEKLY FANTASY BASKETBALL PROJECTION SYSTEM")
    print("="*80)

    # Load data
    roster, matchups, historical, mapping, espn_projections = load_data()

    # Fit player models
    player_models = fit_player_models(roster, historical, mapping, espn_projections)

    # Process each matchup
    print("\n" + "="*80)
    print("MATCHUP PROJECTIONS")
    print("="*80)

    results = []

    for idx, matchup_row in matchups.iterrows():
        home_team_id = matchup_row['home_team_id']
        away_team_id = matchup_row['away_team_id']
        home_team_name = matchup_row['home_team_name']
        away_team_name = matchup_row['away_team_name']

        print(f"\n{home_team_name} vs {away_team_name}")
        print("-" * 60)

        # Get rosters
        home_roster = roster[(roster['fantasy_team_id'] == home_team_id) &
                           (roster['currently_rostered'] == True)]['player_name'].tolist()
        away_roster = roster[(roster['fantasy_team_id'] == away_team_id) &
                           (roster['currently_rostered'] == True)]['player_name'].tolist()

        print(f"  {home_team_name} roster: {len(home_roster)} players")
        print(f"  {away_team_name} roster: {len(away_roster)} players")

        # Simulate matchup
        print(f"  Running 500 simulations...")
        matchup_result = simulate_matchup(home_roster, away_roster, player_models, n_simulations=500)

        # Print results
        print(f"\n  PROJECTED WINNER: ", end="")
        if matchup_result['team_a_win_pct'] > matchup_result['team_b_win_pct']:
            print(f"{home_team_name} ({matchup_result['team_a_win_pct']:.1%})")
        else:
            print(f"{away_team_name} ({matchup_result['team_b_win_pct']:.1%})")

        print(f"  Win Probability: {home_team_name} {matchup_result['team_a_win_pct']:.1%} | " +
              f"{away_team_name} {matchup_result['team_b_win_pct']:.1%}")
        print(f"  Projected Score: {home_team_name} {matchup_result['team_a_wins']}/500 | " +
              f"{away_team_name} {matchup_result['team_b_wins']}/500")

        results.append({
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_win_pct': matchup_result['team_a_win_pct'],
            'away_win_pct': matchup_result['team_b_win_pct'],
            'home_wins': matchup_result['team_a_wins'],
            'away_wins': matchup_result['team_b_wins'],
            'ties': matchup_result['ties']
        })

    # Summary table
    print("\n" + "="*80)
    print("WEEKLY PROJECTIONS SUMMARY")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("PROJECTIONS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
