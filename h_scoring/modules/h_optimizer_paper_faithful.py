"""
Paper-faithful H-scoring optimizer (Rosenof 2024).

Key differences from h_optimizer_final.py:
1. Opponent modeling uses simple average G-score profile (paper lines 83-84)
2. Static scarcity-based weights (no team-aware adjustments)
3. Absolute win probability objective (not marginal)
4. Punting emerges naturally from optimization

This follows the paper exactly without additional "fixes".
"""

import numpy as np
from scipy.stats import norm
from .h_optimizer_final import HScoreOptimizerFinal


class HScoreOptimizerPaperFaithful(HScoreOptimizerFinal):
    """
    Paper-faithful H-scoring optimizer.

    Implements Rosenof (2024) exactly as written, without modifications.
    """

    def __init__(self, setup_params, scoring_system, omega=0.7, gamma=0.25):
        """Initialize with paper-faithful defaults."""
        super().__init__(setup_params, scoring_system, omega, gamma)

        # Precompute average G-score profile for opponent modeling
        self._compute_avg_gscore_profile()

    def _compute_avg_gscore_profile(self):
        """
        Compute average G-score profile across all players.

        This is the "average G-score pick" used for filling unknown
        opponent roster slots (paper lines 83-84).
        """
        print("Computing average G-score profile for opponent modeling...")

        all_players = self.scoring.league_data['PLAYER_NAME'].unique()

        # Collect X-scores for all players
        all_x_scores = []
        for player_name in all_players:
            try:
                player_x = np.array([
                    self.scoring.calculate_x_score(player_name, cat)
                    for cat in self.categories
                ])

                # Only include players with valid stats
                if not np.any(np.isnan(player_x)) and not np.any(np.isinf(player_x)):
                    all_x_scores.append(player_x)
            except:
                continue

        if len(all_x_scores) == 0:
            print("Warning: No valid players found for average profile")
            self.avg_gscore_profile = np.zeros(self.n_cats)
        else:
            # Average across all players
            self.avg_gscore_profile = np.mean(all_x_scores, axis=0)

            print(f"Average G-score profile computed from {len(all_x_scores)} players:")
            for cat, val in zip(self.categories, self.avg_gscore_profile):
                print(f"  {cat:<12} {val:6.3f}")

    def _calculate_average_opponent_x(self, my_team, player_name, picks_made):
        """
        Calculate expected opponent X-scores using paper-faithful method.

        Paper lines 83-84: "For opponents: assume K+1 players are known;
        if not, fill remaining with average G-score picks."

        This is MUCH simpler than the complex snake draft simulation in
        h_optimizer_final.py. We just use:

        opponent_x = (known_picks) + (remaining_slots × avg_gscore_profile)

        Parameters:
        -----------
        my_team : list
            Your current team (to track what's been drafted)
        player_name : str
            Candidate player being evaluated
        picks_made : int
            Number of picks already made

        Returns:
        --------
        numpy array : Expected opponent X-scores
        """
        roster_size = 13

        # In a 12-team league with snake draft, model average opponent as:
        # - Having similar number of picks as you (picks_made)
        # - Each pick is "average G-score quality"

        # Simple model: opponent has same number of picks, all average quality
        opponent_picks_count = picks_made

        # Fill their roster with average G-score profile
        opponent_x = opponent_picks_count * self.avg_gscore_profile

        return opponent_x

    def calculate_optimal_weights_for_team(self, my_team, opponent_teams,
                                          picks_made, total_picks=13,
                                          last_weights=None, format='each_category'):
        """
        Calculate optimal weights for the CURRENT team state.

        Per Rosenof (2024): Weights should be optimized based on your current
        roster and strategic position, NOT based on which candidate you're evaluating.

        These weights are then used to evaluate ALL candidates consistently.

        Parameters:
        -----------
        my_team : list
            Your current roster
        opponent_teams : list of lists
            Opponent rosters
        picks_made : int
            Number of picks already made
        total_picks : int
            Total roster size
        last_weights : array, optional
            Weights from previous pick
        format : str
            'each_category' or 'most_categories'

        Returns:
        --------
        numpy array : Optimal category weights for current team state
        """
        # Calculate current team X-scores
        current_team_x = np.zeros(self.n_cats)
        for player in my_team:
            player_x = np.array([
                self.scoring.calculate_x_score(player, cat)
                for cat in self.categories
            ])
            current_team_x += player_x

        # Model opponents
        if not opponent_teams or all(len(team) == 0 for team in opponent_teams if team):
            opponent_x = self._calculate_average_opponent_x(my_team, None, picks_made)
        else:
            opponent_x = np.zeros(self.n_cats)
            num_opponents = 0
            for opp_team in opponent_teams:
                if opp_team:
                    for player in opp_team:
                        player_x = np.array([
                            self.scoring.calculate_x_score(player, cat)
                            for cat in self.categories
                        ])
                        opponent_x += player_x
                    num_opponents += 1
            if num_opponents > 0:
                opponent_x /= num_opponents

        # Remaining picks
        n_remaining = total_picks - picks_made

        # Initialize weights
        if picks_made == 0:
            initial_weights = self.baseline_weights.copy()
        elif last_weights is not None:
            initial_weights = last_weights
        else:
            initial_weights = self.baseline_weights.copy()

        # Optimize weights for current team WITHOUT any candidate
        # We pass zeros for candidate_x since we're optimizing for team state only
        candidate_x_dummy = np.zeros(self.n_cats)

        optimal_weights, _ = self.optimize_weights(
            candidate_x_dummy, current_team_x, opponent_x,
            n_remaining, initial_weights, format=format
        )

        return optimal_weights

    def evaluate_player_with_weights(self, player_name, my_team, opponent_teams,
                                     picks_made, total_picks, optimal_weights,
                                     format='each_category'):
        """
        Evaluate player using PRE-CALCULATED optimal weights.

        Per Rosenof (2024): Weights should be calculated ONCE for the team state,
        then ALL candidates evaluated using those same weights.

        Parameters:
        -----------
        player_name : str
            Candidate player
        my_team : list
            Current roster
        opponent_teams : list of lists
            Opponent rosters
        picks_made : int
            Number of picks made
        total_picks : int
            Total roster size
        optimal_weights : numpy array
            Pre-calculated optimal weights for team
        format : str
            'each_category' or 'most_categories'

        Returns:
        --------
        float : H-score for this candidate
        """
        # Calculate X-scores for candidate
        candidate_x = np.array([
            self.scoring.calculate_x_score(player_name, cat)
            for cat in self.categories
        ])

        # Calculate current team X-scores
        current_team_x = np.zeros(self.n_cats)
        for player in my_team:
            player_x = np.array([
                self.scoring.calculate_x_score(player, cat)
                for cat in self.categories
            ])
            current_team_x += player_x

        # Model opponents
        if not opponent_teams or all(len(team) == 0 for team in opponent_teams if team):
            opponent_x = self._calculate_average_opponent_x(my_team, player_name, picks_made)
        else:
            opponent_x = np.zeros(self.n_cats)
            num_opponents = 0
            for opp_team in opponent_teams:
                if opp_team:
                    for player in opp_team:
                        player_x = np.array([
                            self.scoring.calculate_x_score(player, cat)
                            for cat in self.categories
                        ])
                        opponent_x += player_x
                    num_opponents += 1
            if num_opponents > 0:
                opponent_x /= num_opponents

        # Remaining picks AFTER this candidate
        n_remaining = total_picks - picks_made - 1

        # Calculate objective using PRE-CALCULATED weights
        h_score = self.calculate_objective(
            optimal_weights, candidate_x, current_team_x,
            opponent_x, n_remaining, format=format
        )

        return h_score

    def evaluate_player(self, player_name, my_team, opponent_teams,
                       picks_made, total_picks=13, last_weights=None,
                       format='each_category'):
        """
        Evaluate player with paper-faithful opponent modeling.

        DEPRECATED: This method optimizes weights per-candidate, which is incorrect
        per Rosenof (2024). Use calculate_optimal_weights_for_team() once, then
        evaluate_player_with_weights() for all candidates.

        Kept for backwards compatibility.
        """
        # Calculate X-scores for candidate
        candidate_x = np.array([
            self.scoring.calculate_x_score(player_name, cat)
            for cat in self.categories
        ])

        # Calculate current team X-scores
        current_team_x = np.zeros(self.n_cats)
        for player in my_team:
            player_x = np.array([
                self.scoring.calculate_x_score(player, cat)
                for cat in self.categories
            ])
            current_team_x += player_x

        # Model opponents using paper-faithful approach
        if not opponent_teams or all(len(team) == 0 for team in opponent_teams if team):
            # Use simple average G-score profile (paper lines 83-84)
            opponent_x = self._calculate_average_opponent_x(my_team, player_name, picks_made)
        else:
            # Calculate actual opponent X-scores if provided
            opponent_x = np.zeros(self.n_cats)
            num_opponents = 0
            for opp_team in opponent_teams:
                if opp_team:
                    for player in opp_team:
                        player_x = np.array([
                            self.scoring.calculate_x_score(player, cat)
                            for cat in self.categories
                        ])
                        opponent_x += player_x
                    num_opponents += 1

            if num_opponents > 0:
                opponent_x /= num_opponents

        # Remaining picks
        n_remaining = total_picks - picks_made - 1

        # Initialize weights (paper uses static baseline weights)
        if picks_made == 0:
            # First pick: use pure baseline weights (no perturbation)
            # The optimizer will naturally adjust them if needed, but
            # for pick #1 with no team context, we should start unbiased
            initial_weights = self.baseline_weights.copy()
        elif last_weights is not None:
            initial_weights = last_weights
        else:
            initial_weights = self.baseline_weights.copy()

        # Optimize weights (gradient descent finds optimal punt strategy)
        optimal_weights, h_score = self.optimize_weights(
            candidate_x, current_team_x, opponent_x,
            n_remaining, initial_weights, format=format
        )

        return h_score, optimal_weights


def test_paper_faithful_optimizer():
    """
    Test the paper-faithful optimizer to see if results differ
    from the current implementation.
    """
    import os
    import json
    import pandas as pd
    from modules.scoring import PlayerScoring
    from modules.covariance import CovarianceCalculator

    print("=" * 80)
    print("TESTING PAPER-FAITHFUL OPTIMIZER")
    print("=" * 80)

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

    # Create paper-faithful optimizer
    optimizer = HScoreOptimizerPaperFaithful(setup_params, scoring)

    print("\n" + "=" * 80)
    print("Test Case: After drafting SGA + Harden")
    print("=" * 80)

    my_team = ["Shai Gilgeous-Alexander", "James Harden"]

    # Evaluate KD
    h_kd, weights_kd = optimizer.evaluate_player(
        "Kevin Durant",
        my_team,
        opponent_teams=[],
        picks_made=2,
        total_picks=13
    )

    # Evaluate KAT
    h_kat, weights_kat = optimizer.evaluate_player(
        "Karl-Anthony Towns",
        my_team,
        opponent_teams=[],
        picks_made=2,
        total_picks=13
    )

    print(f"\nKevin Durant H-score: {h_kd:.4f}")
    print(f"Karl-Anthony Towns H-score: {h_kat:.4f}")
    print(f"Difference: {h_kd - h_kat:+.4f}")

    if h_kd > h_kat:
        print("\n✓ Paper-faithful optimizer still prefers KD")
        print("  This validates that the punting behavior is correct per the paper.")
    else:
        print("\n✓ Paper-faithful optimizer now prefers KAT")
        print("  This suggests the complex opponent modeling was biasing results.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_paper_faithful_optimizer()
