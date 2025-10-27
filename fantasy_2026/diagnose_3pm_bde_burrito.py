"""
Diagnostic: Deep dive into BDE vs Burrito Barnes 3PM predictions

Examines player-by-player 3PM performance to understand why the model
predicted 31.3±7.8 for BDE (actual: 48) and 38.6±8.4 for Burrito Barnes (actual: 61).
"""

import pandas as pd
import numpy as np
import sys
import ast
from collections import defaultdict

sys.path.append('/Users/rhu/fantasybasketball2')
from weekly_projection_system import FantasyProjectionModel, load_data, fit_player_models

def main():
    print("="*80)
    print("3PM DEEP DIVE: BDE vs Burrito Barnes")
    print("="*80)

    # Load data
    roster, matchups, historical, mapping, espn_projections = load_data()
    player_models = fit_player_models(roster, historical, mapping, espn_projections)

    # Load box scores
    box_scores = pd.read_csv('/Users/rhu/fantasybasketball2/fantasy_2026/data/matchups/box_scores_latest.csv')
    week6 = box_scores[box_scores['week'] == 6]

    # Filter for BDE vs Burrito Barnes matchup
    matchup_data = week6[
        (week6['team_name'] == 'BDE') |
        (week6['team_name'] == 'Burrito Barnes')
    ]

    # Analyze each team
    for team_name in ['BDE', 'Burrito Barnes']:
        print(f"\n{'='*80}")
        print(f"{team_name}")
        print(f"{'='*80}\n")

        team_data = matchup_data[matchup_data['team_name'] == team_name]

        # Parse player stats
        player_stats = []

        for _, row in team_data.iterrows():
            player_name = row['player_name']
            try:
                stats = ast.literal_eval(row['stat_0'])

                # Extract total stats
                total = stats.get('total', {})
                # NOTE: box_scores uses '3PM' not 'FG3M'
                total_3pm = total.get('3PM', 0) or 0
                total_3pm = int(total_3pm) if total_3pm else 0

                total_3pa = total.get('3PA', 0) or 0
                total_3pa = int(total_3pa) if total_3pa else 0

                n_games = int(total.get('GP', 0) or 0)

                player_stats.append({
                    'player': player_name,
                    'total_3pm': total_3pm,
                    'total_3pa': total_3pa,
                    'games': n_games,
                    'per_game_3pm': total_3pm / n_games if n_games > 0 else 0
                })
            except Exception as e:
                print(f"  ERROR parsing {player_name}: {e}")
                continue

        # Sort by total 3PM
        player_stats.sort(key=lambda x: x['total_3pm'], reverse=True)

        print(f"Player-by-Player 3PM Breakdown:")
        print(f"{'-'*80}")
        print(f"{'Player':<30} {'Games':<6} {'Actual 3PM':<12} {'Predicted':<20} {'Z-score'}")
        print(f"{'-'*80}")

        team_total_3pm = 0
        team_predicted_3pm = 0
        team_predicted_var = 0
        player_z_scores = []

        for pdata in player_stats:
            player_name = pdata['player']
            actual_3pm = pdata['total_3pm']
            n_games = pdata['games']
            team_total_3pm += actual_3pm

            # Get model prediction
            model = player_models.get(player_name)

            if model:
                # Simulate this player's contribution
                simulated_3pm = []
                for _ in range(1000):
                    total = 0
                    for _ in range(n_games):
                        game_stats = model.simulate_game()
                        total += game_stats.get('FG3M', 0)
                    simulated_3pm.append(total)

                pred_mean = np.mean(simulated_3pm)
                pred_std = np.std(simulated_3pm)

                team_predicted_3pm += pred_mean
                team_predicted_var += pred_std ** 2

                z_score = (actual_3pm - pred_mean) / pred_std if pred_std > 0 else 0
                player_z_scores.append({
                    'player': player_name,
                    'z_score': z_score,
                    'actual': actual_3pm,
                    'predicted': pred_mean
                })

                # Format output
                z_indicator = ""
                if abs(z_score) > 2:
                    z_indicator = " 🔴"
                elif abs(z_score) > 1:
                    z_indicator = " 🟡"

                print(f"{player_name:<30} {n_games:<6} {actual_3pm:<12} "
                      f"{pred_mean:5.1f} ± {pred_std:4.1f}       "
                      f"{z_score:+5.2f}σ{z_indicator}")

            else:
                print(f"{player_name:<30} {n_games:<6} {actual_3pm:<12} "
                      f"{'NO MODEL':<20} {'N/A'}")

        # Team summary
        team_predicted_std = np.sqrt(team_predicted_var)
        team_z_score = (team_total_3pm - team_predicted_3pm) / team_predicted_std if team_predicted_std > 0 else 0

        print(f"{'-'*80}")
        print(f"{'TEAM TOTAL':<30} {'':<6} {team_total_3pm:<12} "
              f"{team_predicted_3pm:5.1f} ± {team_predicted_std:4.1f}       "
              f"{team_z_score:+5.2f}σ")
        print()

        # Analyze player z-score distribution
        if player_z_scores:
            print(f"\nPlayer Z-Score Distribution:")
            print(f"{'-'*80}")

            z_values = [p['z_score'] for p in player_z_scores]
            print(f"  Mean Z-score: {np.mean(z_values):+.2f}")
            print(f"  Median Z-score: {np.median(z_values):+.2f}")
            print(f"  Std of Z-scores: {np.std(z_values):.2f}")
            print()
            print(f"  Players > +2σ: {sum(1 for z in z_values if z > 2)}")
            print(f"  Players > +1σ: {sum(1 for z in z_values if z > 1)}")
            print(f"  Players within ±1σ: {sum(1 for z in z_values if abs(z) <= 1)}")
            print(f"  Players < -1σ: {sum(1 for z in z_values if z < -1)}")

            # Show top overperformers
            print(f"\n  Top Overperformers:")
            sorted_z = sorted(player_z_scores, key=lambda x: x['z_score'], reverse=True)
            for p in sorted_z[:5]:
                print(f"    {p['player']:<30} {p['z_score']:+5.2f}σ  "
                      f"(actual: {p['actual']:.0f}, pred: {p['predicted']:.1f})")

        # Detail for top 3PM scorers
        print(f"\n{'='*80}")
        print(f"Per-Game Analysis (Top 3 Three-Point Shooters)")
        print(f"{'='*80}\n")

        for pdata in player_stats[:3]:
            if pdata['games'] == 0:
                continue

            player_name = pdata['player']

            print(f"\n{player_name}:")
            print(f"  ACTUAL: {pdata['total_3pm']} 3PM in {pdata['games']} games "
                  f"= {pdata['per_game_3pm']:.1f} per game")
            print(f"  Attempts: {pdata['total_3pa']} 3PA "
                  f"= {pdata['total_3pa']/pdata['games']:.1f} per game")
            if pdata['total_3pa'] > 0:
                print(f"  Shooting: {pdata['total_3pm']/pdata['total_3pa']*100:.1f}%")

            # Show model's per-game distribution
            model = player_models.get(player_name)
            if model:
                sim_games_3pm = []
                sim_games_3pa = []
                for _ in range(1000):
                    game = model.simulate_game()
                    sim_games_3pm.append(game.get('FG3M', 0))
                    sim_games_3pa.append(game.get('FG3A', 0))

                print(f"\n  MODEL per-game predictions:")
                print(f"    3PM: {np.mean(sim_games_3pm):.2f} ± {np.std(sim_games_3pm):.2f}")
                print(f"    3PA: {np.mean(sim_games_3pa):.2f} ± {np.std(sim_games_3pa):.2f}")
                if np.mean(sim_games_3pa) > 0:
                    print(f"    3P%: {np.mean(sim_games_3pm)/np.mean(sim_games_3pa)*100:.1f}%")
                print(f"    90th percentile: {np.percentile(sim_games_3pm, 90):.1f} 3PM")
                print(f"    95th percentile: {np.percentile(sim_games_3pm, 95):.1f} 3PM")

    # Correlation analysis
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*80}\n")

    print("Did players who shot well tend to shoot MORE or just BETTER?")
    print("(i.e., is this a volume issue or efficiency issue?)\n")

    for team_name in ['BDE', 'Burrito Barnes']:
        team_data = matchup_data[matchup_data['team_name'] == team_name]

        player_3pm = []
        player_3pa = []
        player_3pct = []
        player_games = []

        for _, row in team_data.iterrows():
            try:
                stats = ast.literal_eval(row['stat_0'])
                total = stats.get('total', {})
                # NOTE: box_scores uses '3PM' not 'FG3M'
                total_3pm = int(total.get('3PM', 0) or 0)
                total_3pa = int(total.get('3PA', 0) or 0)
                n_games = int(total.get('GP', 0) or 0)

                if n_games > 0:
                    player_3pm.append(total_3pm)
                    player_3pa.append(total_3pa)
                    if total_3pa > 0:
                        player_3pct.append(total_3pm / total_3pa)
                    player_games.append(n_games)
            except:
                continue

        if len(player_3pm) > 0:
            print(f"{team_name}:")
            print(f"  Total 3PM: {sum(player_3pm)}")
            print(f"  Total 3PA: {sum(player_3pa)}")
            if sum(player_3pa) > 0:
                print(f"  Team 3P%: {sum(player_3pm)/sum(player_3pa):.3f}")
            print()

            # Check if model's 3PA prediction is off
            total_pred_3pm = 0
            total_pred_3pa = 0
            for _, row in team_data.iterrows():
                player_name = row['player_name']
                model = player_models.get(player_name)
                if model:
                    try:
                        stats = ast.literal_eval(row['stat_0'])
                        total = stats.get('total', {})
                        n_games = int(total.get('GP', 0) or 0)

                        if n_games > 0:
                            # Simulate 3PA and 3PM
                            sim_3pm = []
                            sim_3pa = []
                            for _ in range(100):
                                total_pm = 0
                                total_pa = 0
                                for _ in range(n_games):
                                    game_stats = model.simulate_game()
                                    total_pm += game_stats.get('FG3M', 0)
                                    total_pa += game_stats.get('FG3A', 0)
                                sim_3pm.append(total_pm)
                                sim_3pa.append(total_pa)
                            total_pred_3pm += np.mean(sim_3pm)
                            total_pred_3pa += np.mean(sim_3pa)
                    except:
                        continue

            print(f"  Predicted 3PM: {total_pred_3pm:.1f}")
            print(f"  Predicted 3PA: {total_pred_3pa:.1f}")
            if total_pred_3pa > 0:
                print(f"  Predicted 3P%: {total_pred_3pm/total_pred_3pa:.3f}")
            print(f"\n  GAPS:")
            print(f"    3PM: {sum(player_3pm) - total_pred_3pm:+.1f} ({(sum(player_3pm) - total_pred_3pm)/total_pred_3pm*100:+.1f}%)")
            print(f"    3PA: {sum(player_3pa) - total_pred_3pa:+.1f} ({(sum(player_3pa) - total_pred_3pa)/total_pred_3pa*100:+.1f}%)")
            print()

if __name__ == '__main__':
    main()
