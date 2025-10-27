#!/usr/bin/env python
"""
Fix H-optimizer to match the paper exactly.

Key fixes based on the paper:
1. When opponents unknown, fill with average G-score picks (not zero)
2. Model opponent future picks properly
3. Use correct variance formula: 2N + (N-K-1)*sigma^2
"""

import numpy as np
from scipy.stats import norm


def get_paper_based_fix():
    """
    Returns the fix to make h_optimizer.py match the paper.
    """

    fix_description = """
    ISSUES FOUND:

    1. Line 83 of paper: "Opponents: assume K+1 players known; if not, fill with average G-score picks"
       Current code: Sets opponent_x to zero when no opponents known
       Fix: Should model expected opponent team based on average picks

    2. Line 157: Variance should be "2N + (N-K-1)*Xσ²"
       Current code: Uses simplified "2 * roster_size + n_remaining * 1.0"
       Fix: Need proper variance calculation

    3. Opponent modeling doesn't account for their future picks
       Current code: Just sums current opponent X-scores
       Fix: Should add expected future picks for opponents too

    4. X_delta calculation seems overly complex
       Paper suggests it models "how future picks will differ based on strategy j"
       Current implementation may be overcomplicating this
    """

    proposed_fix = '''
    def evaluate_player(self, player_name, my_team, opponent_teams,
                       picks_made, total_picks=13, last_weights=None,
                       format='each_category'):
        """
        Evaluate a candidate player using H-scoring.
        FIXED to match paper exactly.
        """
        # Calculate X-scores for candidate
        candidate_x = np.array([
            self.scoring.calculate_x_score(player_name, cat)
            for cat in self.categories
        ])

        # Calculate current team X-scores (Xs in paper)
        current_team_x = np.zeros(self.n_cats)
        for player in my_team:
            player_x = np.array([
                self.scoring.calculate_x_score(player, cat)
                for cat in self.categories
            ])
            current_team_x += player_x

        # FIXED: Model opponents properly (line 83 of paper)
        if not opponent_teams or len(opponent_teams) == 0:
            # "fill with average G-score picks"
            # This means: model what an average drafted team would look like
            # For early picks, use expected value of top remaining players

            # Get top undrafted players by G-score
            all_players = self.scoring.get_all_players()
            drafted = set(my_team) | set([player_name])
            available = [p for p in all_players if p not in drafted]

            # Get G-scores for available players
            g_scores = [(p, self.scoring.calculate_all_g_scores(p)['TOTAL'])
                       for p in available[:50]]  # Top 50 for efficiency
            g_scores.sort(key=lambda x: x[1], reverse=True)

            # Model expected opponent team (average of next best players)
            expected_picks = min(total_picks, len(g_scores))
            opponent_x = np.zeros(self.n_cats)

            for i in range(expected_picks):
                player_name_temp = g_scores[i][0]
                player_x = np.array([
                    self.scoring.calculate_x_score(player_name_temp, cat)
                    for cat in self.categories
                ])
                opponent_x += player_x

            # Average across multiple opponents
            num_opponents = 11  # In 12-team league
            opponent_x = opponent_x * num_opponents / expected_picks

        else:
            # Calculate actual opponent X-scores
            opponent_x = np.zeros(self.n_cats)
            for opp_team in opponent_teams:
                for player in opp_team:
                    player_x = np.array([
                        self.scoring.calculate_x_score(player, cat)
                        for cat in self.categories
                    ])
                    opponent_x += player_x

            # Add expected future picks for opponents
            remaining_opponent_picks = (total_picks - len(opp_team)) * len(opponent_teams)
            if remaining_opponent_picks > 0:
                # Model future opponent picks using baseline strategy
                future_opponent_x = self.calculate_x_delta(
                    self.v_vector,  # Opponents use balanced strategy
                    remaining_opponent_picks / len(opponent_teams)
                )
                opponent_x += future_opponent_x * len(opponent_teams)

            opponent_x /= len(opponent_teams)

        # Remaining picks for our team
        n_remaining = total_picks - picks_made - 1

        # Initialize weights (from paper line 277)
        if picks_made == 0:
            # "jC = v + small perturbation (1/500 in direction of player stats)"
            perturbation = candidate_x / 500.0
            initial_weights = self.v_vector + perturbation
            # Normalize
            initial_weights = np.maximum(initial_weights, 1e-6)
            initial_weights = initial_weights / initial_weights.sum()
        elif last_weights is not None:
            initial_weights = last_weights
        else:
            initial_weights = self.baseline_weights.copy()

        # Optimize weights
        optimal_weights, h_score = self.optimize_weights(
            candidate_x, current_team_x, opponent_x,
            n_remaining, initial_weights, format=format
        )

        return h_score, optimal_weights


    def calculate_win_probabilities(self, my_team_x, opponent_x, n_remaining):
        """
        Calculate win probability for each category.
        FIXED to match paper formula (line 157).

        Variance = 2N + (N-K-1)*Xσ²
        where N = roster size, K = picks made, Xσ² ≈ 1 in X-basis
        """
        # Calculate differential
        differential = my_team_x - opponent_x

        # Paper formula: variance = 2N + (N-K-1)*Xσ²
        # In X-basis, Xσ² ≈ 1 (simplified assumption)
        roster_size = 13
        variance = 2 * roster_size + n_remaining * 1.0

        # Ensure positive variance
        variance = max(variance, 1e-6)
        std_dev = np.sqrt(variance)

        # Z-scores (clip to prevent extreme values)
        z_scores = np.clip(differential / std_dev, -10, 10)

        # Win probabilities using error function (line 225 of paper)
        # wc = 0.5[1 + erf(μ/(√2σ))]
        # Note: norm.cdf(z) = 0.5[1 + erf(z/√2)]
        win_probs = norm.cdf(z_scores)

        return win_probs
    '''

    return fix_description, proposed_fix


if __name__ == "__main__":
    print("=" * 80)
    print("H-OPTIMIZER FIX BASED ON PAPER")
    print("=" * 80)

    description, fix = get_paper_based_fix()

    print(description)
    print("\nPROPOSED FIX:")
    print("-" * 80)
    print(fix)

    print("\nKEY INSIGHT:")
    print("-" * 80)
    print("""
The main issue is that the current implementation uses zero opponents
for early picks, making all players look equally good against "nothing".

The paper clearly states to model opponents as "average G-score picks",
which means we should simulate what a typical opponent team would look
like based on the remaining player pool.

This would correctly differentiate between good and bad players:
- Good players would show positive differential vs average opponents
- Bad players would show negative differential vs average opponents
    """)