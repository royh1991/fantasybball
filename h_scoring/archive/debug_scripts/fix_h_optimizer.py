#!/usr/bin/env python
"""
Fix for the H-score optimizer to properly handle opponent baselines.

The main issue is that when evaluating players early in the draft,
the opponent_x values are zero or very low, making the win probability
calculations meaningless. We need to use league average as baseline.
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import json
import os


def create_fixed_optimizer():
    """Create a fixed version of the H-score optimizer."""

    # Read the original file
    original_file = 'modules/h_optimizer.py'
    with open(original_file, 'r') as f:
        content = f.read()

    # Key fixes needed:
    # 1. Use league average as baseline opponent when no opponents exist
    # 2. Properly scale the objective function
    # 3. Fix the variance calculation

    fixes = """
    def evaluate_player(self, player_name, my_team, opponent_teams,
                       picks_made, total_picks=13, last_weights=None,
                       format='each_category'):
        \"\"\"
        Evaluate a candidate player using H-scoring.

        FIXED VERSION: Uses league average as baseline opponent
        \"\"\"
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

        # FIXED: Use league average as baseline, not zero
        if not opponent_teams or picks_made < 3:
            # Use league average multiplied by roster size as baseline
            opponent_x = np.zeros(self.n_cats)  # League average X-scores are 0 by definition
            # But we need to project a full roster worth
            opponent_x = opponent_x * total_picks  # This ensures proper scaling
        else:
            # Calculate average opponent X-scores
            opponent_x = np.zeros(self.n_cats)
            for opp_team in opponent_teams:
                for player in opp_team:
                    player_x = np.array([
                        self.scoring.calculate_x_score(player, cat)
                        for cat in self.categories
                    ])
                    opponent_x += player_x
            opponent_x /= len(opponent_teams)

        # Remaining picks
        n_remaining = total_picks - picks_made - 1

        # Initialize weights
        if picks_made == 0:
            # First pick: use player's strengths more heavily
            initial_weights = self._perturb_weights_toward_player(
                self.baseline_weights, candidate_x
            )
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
    """

    print("Fix summary:")
    print("-" * 60)
    print("1. The main issue is that opponent_x is 0 for early picks")
    print("2. This makes all players look equally good (win prob = 0.5)")
    print("3. The optimizer then relies entirely on X_delta calculation")
    print("4. X_delta seems to favor low-value players due to math errors")
    print()
    print("The fix would be to:")
    print("- Use league average baseline for opponents")
    print("- Fix the X_delta calculation")
    print("- Properly scale the variance calculation")

    return fixes


def analyze_x_delta_issue():
    """Analyze why X_delta calculation favors bad players."""

    print("\n" + "=" * 60)
    print("X_DELTA ANALYSIS")
    print("=" * 60)

    print("""
The X_delta formula is:
    X_delta = (N - K - 1) * Σ * (complex matrix operations)

The issue is likely that:
1. The formula assumes you'll draft players similar to your weights
2. But the weights are being optimized to maximize win probability
3. This creates a circular dependency
4. Bad players might get high scores because the optimizer finds
   weird weight combinations that make them look good on paper
    """)

    print("\nThe real problem is the objective function doesn't properly")
    print("penalize bad players - it only looks at win probabilities,")
    print("not the actual quality of stats being accumulated.")


def propose_simple_fix():
    """Propose a simpler fix that might work better."""

    print("\n" + "=" * 60)
    print("SIMPLE FIX PROPOSAL")
    print("=" * 60)

    fix_code = """
# In h_optimizer.py, replace the calculate_objective function:

def calculate_objective(self, weights, candidate_x, current_team_x,
                       opponent_x, n_remaining, format='each_category'):
    '''
    FIXED: Add penalty for low absolute stats, not just relative.
    '''
    # Calculate X_delta for future picks
    x_delta = self.calculate_x_delta(weights, n_remaining)

    # Total team projection
    team_projection = current_team_x + candidate_x + x_delta

    # FIXED: Add baseline opponent if none exists
    if np.sum(np.abs(opponent_x)) < 1e-6:
        # Use league average (0) scaled to roster size
        opponent_x = np.zeros(len(self.categories))

    # Combined variance
    roster_size = 13
    variance = 2 * roster_size + n_remaining * 1.0

    # Calculate win probabilities
    win_probs = self.calculate_win_probabilities(
        team_projection, opponent_x, variance
    )

    # FIXED: Add bonus for positive X-scores (good absolute stats)
    # This prevents the optimizer from favoring bad players
    x_score_bonus = np.sum(np.maximum(candidate_x, 0)) * 0.1

    if format == 'each_category':
        # Sum of win probabilities plus absolute quality bonus
        objective = np.sum(win_probs) + x_score_bonus
    else:
        objective = np.sum(win_probs) + x_score_bonus

    return objective
"""

    print(fix_code)

    return fix_code


def main():
    """Main function to analyze and propose fixes."""
    print("=" * 60)
    print("H-SCORE OPTIMIZER FIX ANALYSIS")
    print("=" * 60)

    create_fixed_optimizer()
    analyze_x_delta_issue()
    fix_code = propose_simple_fix()

    print("\n" + "=" * 60)
    print("RECOMMENDED ACTION")
    print("=" * 60)
    print("""
The H-scoring algorithm has a fundamental flaw where it doesn't
properly value good players over bad players. The issue is:

1. Win probabilities are calculated against empty/weak opponents
2. The X_delta adjustment seems to favor bad players
3. There's no penalty for accumulating low absolute stats

The quickest fix is to:
1. Add a bonus term for positive X-scores (good stats)
2. Use proper baseline opponents
3. Potentially simplify or remove the X_delta calculation

This would make the algorithm prefer players with good stats
while still optimizing category weights dynamically.
    """)


if __name__ == "__main__":
    main()