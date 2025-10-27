"""
Test the user's critical insight:

In a SNAKE DRAFT at position 6:
- Pick #1 = position 6 (picks 1-5 already gone)
- Pick #2 = position 19 (picks 7-18 happen before you pick again!)

If DD is scarce and X_delta promises 5.57 DD from future picks,
will those DD players actually be AVAILABLE when you pick next?

Or will other teams have taken them all?
"""

import os
import json
import pandas as pd
import numpy as np
from modules.scoring import PlayerScoring
from modules.covariance import CovarianceCalculator
from modules.h_optimizer_paper_faithful import HScoreOptimizerPaperFaithful


def get_adp_rankings(league_data):
    """Get ADP rankings for draft simulation."""
    # Load ADP data
    try:
        adp_df = pd.read_csv('../data/fantasy_basketball_clean2.csv')
        # Create mapping of player name to ADP
        adp_map = {}
        for _, row in adp_df.iterrows():
            adp_map[row['Player']] = row['ADP']
        return adp_map
    except:
        # Fallback: use G-score as proxy for ADP
        return None


def simulate_snake_draft_picks_1_to_18(scoring, your_pick_6_player):
    """
    Simulate picks 1-18 in a snake draft (12 teams).
    You pick at position 6, so you miss picks 7-18 before your next turn.

    Returns: List of players drafted in picks 1-18
    """

    # Load ADP
    try:
        adp_df = pd.read_csv('../data/fantasy_basketball_clean2.csv')
        adp_map = dict(zip(adp_df['Player'], adp_df['ADP']))
    except:
        # Fallback: use G-score rankings
        g_rankings = scoring.rank_players_by_g_score(top_n=200)
        adp_map = {row['PLAYER_NAME']: idx for idx, row in g_rankings.iterrows()}

    # Get players sorted by ADP
    all_players = scoring.league_data['PLAYER_NAME'].unique()

    # Filter to players with ADP
    players_with_adp = []
    for player in all_players:
        if player in adp_map:
            players_with_adp.append((player, adp_map[player]))

    # Sort by ADP
    players_with_adp.sort(key=lambda x: x[1])
    adp_order = [p[0] for p in players_with_adp]

    # Simulate picks 1-18
    drafted = []

    # Picks 1-5 (before your turn)
    for i in range(5):
        if i < len(adp_order):
            drafted.append(adp_order[i])

    # Pick 6 (YOU)
    drafted.append(your_pick_6_player)

    # Picks 7-18 (12 picks before your next turn!)
    # In snake draft: picks 7-12 (rest of round 1), then picks 13-18 (round 2 going backwards)
    pick_num = 6
    for player in adp_order:
        if player not in drafted and pick_num < 18:
            drafted.append(player)
            pick_num += 1

    return drafted[:18]


def main():
    """Test the scarcity paradox in snake draft."""

    # Load data
    data_dir = 'data'
    weekly_files = sorted([f for f in os.listdir(data_dir) if f.startswith('league_weekly_data')])
    variance_files = sorted([f for f in os.listdir(data_dir) if f.startswith('player_variances')])

    data_file = os.path.join(data_dir, weekly_files[-1])
    variance_file = os.path.join(data_dir, variance_files[-1])

    league_data = pd.read_csv(data_file)
    with open(variance_file, 'r') as f:
        player_variances = json.load(f)

    # Initialize
    scoring = PlayerScoring(league_data, player_variances, roster_size=13)
    cov_calc = CovarianceCalculator(league_data, scoring)
    setup_params = cov_calc.get_setup_params()

    categories = setup_params['categories']
    dd_idx = categories.index('DD')

    # Create optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring, omega=0.7, gamma=0.25)

    print("=" * 80)
    print("SNAKE DRAFT SCARCITY TEST")
    print("=" * 80)

    print("\nSCENARIO:")
    print("  - 12-team snake draft")
    print("  - You draft at position 6")
    print("  - Pick #1 = overall pick 6")
    print("  - Pick #2 = overall pick 19 (picks 7-18 happen first!)")

    print("\n" + "=" * 80)
    print("WHAT X_DELTA PROMISES")
    print("=" * 80)

    baseline_weights = setup_params['baseline_weights']

    x_delta = optimizer._compute_xdelta_simplified(
        jC=baseline_weights,
        v=optimizer.v_vector,
        Sigma=optimizer.cov_matrix_original,
        gamma=optimizer.gamma,
        omega=optimizer.omega,
        N=13,
        K=0  # pick #1
    )

    dd_xdelta = x_delta[dd_idx]

    print(f"\nOn pick #1, X_delta promises: {dd_xdelta:.2f} DD from future picks")
    print(f"DD has {baseline_weights[dd_idx]*100:.1f}% weight (HIGHEST)")
    print(f"\nThis translates to: {dd_xdelta/12:.2f} DD per future pick")

    print("\n" + "=" * 80)
    print("SCENARIO A: You draft KEVIN DURANT at pick #6")
    print("=" * 80)

    kd_dd = scoring.calculate_x_score("Kevin Durant", 'DD')
    print(f"\nKD DD X-score: {kd_dd:.2f}")
    print(f"Team DD after KD: {kd_dd:.2f}")
    print(f"X_delta promises: {dd_xdelta:.2f}")
    print(f"Total projection: {kd_dd + dd_xdelta:.2f}")

    # Simulate what OTHER teams draft in picks 7-18
    drafted_picks_1_18 = simulate_snake_draft_picks_1_to_18(scoring, "Kevin Durant")

    print(f"\n{'='*80}")
    print("PICKS 1-18 (before your 2nd pick at #19)")
    print("=" * 80)

    print(f"\n{'Pick':<6} {'Player':<30} {'DD X-score':<12} {'G-score':<10}")
    print("-" * 80)

    dd_specialists_gone = []

    for i, player in enumerate(drafted_picks_1_18, 1):
        try:
            player_dd = scoring.calculate_x_score(player, 'DD')
            player_g = scoring.calculate_all_g_scores(player)['TOTAL']
        except:
            player_dd = 0
            player_g = 0

        marker = ""
        if player_dd > 2.0:  # DD specialist
            dd_specialists_gone.append(player)
            marker = "← DD SPECIALIST"

        if player == "Kevin Durant":
            marker = "← YOU"

        print(f"{i:<6} {player:<30} {player_dd:<12.2f} {player_g:<10.2f} {marker}")

    print("\n" + "=" * 80)
    print(f"DD SPECIALISTS DRAFTED IN PICKS 1-18")
    print("=" * 80)

    print(f"\nPlayers with DD X-score > 2.0:")
    for player in dd_specialists_gone:
        dd_x = scoring.calculate_x_score(player, 'DD')
        print(f"  {player:<30} DD X-score: {dd_x:.2f}")

    print(f"\nTotal: {len(dd_specialists_gone)} DD specialists gone before pick #19")

    print("\n" + "=" * 80)
    print("WHO'S LEFT AT PICK #19?")
    print("=" * 80)

    # Get all players NOT drafted yet
    all_players = scoring.league_data['PLAYER_NAME'].unique()
    available = [p for p in all_players if p not in drafted_picks_1_18]

    # Rank available players by DD
    available_dd = []
    for player in available:
        try:
            player_dd = scoring.calculate_x_score(player, 'DD')
            player_g = scoring.calculate_all_g_scores(player)['TOTAL']
            available_dd.append({
                'player': player,
                'dd_x': player_dd,
                'g_score': player_g
            })
        except:
            pass

    available_dd.sort(key=lambda x: x['dd_x'], reverse=True)

    print(f"\nTop 10 DD players still available at pick #19:")
    print(f"\n{'Rank':<6} {'Player':<30} {'DD X-score':<12} {'G-score':<10}")
    print("-" * 80)

    for i, player_info in enumerate(available_dd[:10], 1):
        print(f"{i:<6} {player_info['player']:<30} {player_info['dd_x']:<12.2f} {player_info['g_score']:<10.2f}")

    best_available_dd = available_dd[0]['dd_x'] if available_dd else 0

    print("\n" + "=" * 80)
    print("THE VERDICT")
    print("=" * 80)

    print(f"\nX_delta promised: {dd_xdelta:.2f} DD from 12 future picks")
    print(f"That's {dd_xdelta/12:.2f} DD per pick")

    print(f"\nBest DD player available at pick #19: {best_available_dd:.2f}")

    if best_available_dd < dd_xdelta/12:
        print(f"\n⚠️  SCARCITY PARADOX CONFIRMED!")
        print(f"\nX_delta assumed you'd average {dd_xdelta/12:.2f} DD per future pick")
        print(f"But the best available player only has {best_available_dd:.2f} DD!")
        print(f"\nBy passing on elite DD at pick #6, you CAN'T fulfill X_delta's promise")
        print(f"because all the elite DD players are GONE by pick #19!")

        print("\n" + "=" * 80)
        print("WHY THIS IS A BUG")
        print("=" * 80)

        print("\n1. X_delta assumes you can draft from the FULL player pool")
        print("   Reality: By pick #19, elite players in scarce categories are GONE")

        print("\n2. DD has high weight (22.6%) BECAUSE it's scarce")
        print("   But if it's scarce, OTHER teams also want DD players")
        print("   So they'll be drafted quickly!")

        print("\n3. The algorithm says:")
        print("   Pick #6: 'Pass on Sabonis, I'll get DD later'")
        print("   Pick #19: 'All DD specialists are gone! Can't fulfill the promise!'")

        print("\n4. Snake draft position matters!")
        print("   - Pick #6 → Pick #19: 12 picks away")
        print("   - In those 12 picks, OTHER teams take the scarce resources")
        print("   - X_delta doesn't account for this!")

        total_dd_specialists = len(dd_specialists_gone)

        print(f"\n5. Reality check:")
        print(f"   {total_dd_specialists} DD specialists drafted in picks 1-18")
        print(f"   That's {total_dd_specialists/18*100:.0f}% of picks!")
        print(f"   X_delta assumed you'd get your share of them")
        print(f"   But you only pick ONCE in those 18 picks!")

    else:
        print(f"\n✓ X_delta's assumption is reasonable")
        print(f"Best available DD ({best_available_dd:.2f}) >= expected ({dd_xdelta/12:.2f})")


if __name__ == "__main__":
    main()
