"""
Fixed H-scoring optimizer that addresses all identified issues.

Fixes:
1. Remove sigma scaling from b vector
2. Use sqrt(N-K-1) for sublinear multiplier growth
3. Properly model opponents with average G-score picks
4. Handle covariance matrix scaling issues
"""

import numpy as np
from scipy.stats import norm
import copy

from .h_optimizer import HScoreOptimizer


class HScoreOptimizerFixed(HScoreOptimizer):
    """Fixed H-scoring optimizer with all corrections."""

    def __init__(self, setup_params, scoring_system, omega=0.7, gamma=0.25):
        """Initialize with fixes."""
        super().__init__(setup_params, scoring_system, omega, gamma)

        # Fix flags
        self.use_normalized_b = True  # Don't scale by sigma
        self.use_sqrt_multiplier = True  # Use sqrt(N-K-1) instead of (N-K-1)
        self.use_proper_opponents = True  # Model opponents properly

    def _compute_xdelta_exact_fixed(self, jC, v, Sigma, gamma, omega, N, K, muC_P=None,
                                    eps_sigma_rel=1e-8, eps_Uinv=1e-8, ridge_frac=1e-8):
        """
        Fixed implementation of X_delta with proper scaling.

        Main changes:
        1. Don't scale b vector by sigma (removes magnitude explosion)
        2. Use sqrt(N-K-1) for sublinear growth
        """

        # --- basic sanity / forcing shapes ---
        jC = np.asarray(jC, dtype=float).reshape(-1)
        v  = np.asarray(v,  dtype=float).reshape(-1)
        m  = jC.size
        assert v.size == m
        Sigma = np.asarray(Sigma, dtype=float)
        assert Sigma.shape == (m, m)

        # 1) normalize jC and v to sum to 1 (paper does this after each gradient step)
        def normalized(x):
            s = x.sum()
            if abs(s) < 1e-10:
                # Return uniform weights if sum is near zero
                return np.ones_like(x) / len(x)
            return x / s
        jC = normalized(jC)
        v  = normalized(v)

        # 2) regularize Sigma if badly conditioned
        trace = np.trace(Sigma)
        ridge = max(ridge_frac * trace, 1e-12)
        Sigma_reg = Sigma + ridge * np.eye(m)

        # 3) compute the projection used in sigma^2:
        #    proj_coeff = (v^T Sigma jC) / (v^T Sigma v)
        denom_vSv = float(v.T @ Sigma_reg @ v)
        if denom_vSv <= 0:
            # Bad denominator, return zero adjustment
            return np.zeros(m), {'error': 'v^T Sigma v <= 0'}
        proj_coeff = float((v.T @ Sigma_reg @ jC) / denom_vSv)
        jC_perp = jC - v * proj_coeff   # (jC - projection of jC onto v in Sigma-metric)

        # 4) sigma^2 = jC_perp^T Sigma jC_perp  (see paper). Protect against negative rounding.
        sigma2 = float(jC_perp.T @ Sigma_reg @ jC_perp)
        sigma = float(np.sqrt(max(sigma2, eps_sigma_rel * (trace + 1e-12))))

        # 5) Build U and b (constraints vector). U should be 2 x m (rows [v; jC])
        U = np.vstack([v, jC])            # shape (2, m)

        # FIX 1: Don't scale b by sigma to avoid magnitude explosion
        if self.use_normalized_b:
            b = np.array([-gamma, omega], dtype=float)  # Normalized version
        else:
            b = np.array([-gamma * sigma, omega * sigma], dtype=float)  # Original

        # 6) Compute middle matrix M = U Sigma U^T (2x2) and invert robustly
        M = U @ Sigma_reg @ U.T          # shape (2,2)
        # regularize M for inversion if condition number high / near-singular
        cond_M = np.linalg.cond(M)
        if cond_M > 1e12:
            # add small jitter proportional to trace
            jitter = eps_Uinv * (np.trace(M) + 1e-12)
            M = M + jitter * np.eye(2)
        # solve rather than invert directly
        try:
            z = np.linalg.solve(M, b)    # shape (2,)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inverse
            z = np.linalg.pinv(M) @ b

        # 7) final x_delta (before multiplying by remaining picks):
        xdelta_unit = Sigma_reg @ U.T @ z   # shape (m,)

        # 8) FIX 2: Use sqrt multiplier for sublinear growth
        remaining = float(max(0, N - K - 1))
        if self.use_sqrt_multiplier:
            multiplier = np.sqrt(remaining)  # Sublinear growth
        else:
            multiplier = remaining  # Original linear growth

        xdelta = multiplier * xdelta_unit

        if muC_P is not None:
            xdelta = xdelta + np.asarray(muC_P, dtype=float).reshape(-1)

        # diagnostics for debugging
        diag = {
            'proj_coeff': proj_coeff,
            'sigma': sigma,
            'sigma2': sigma2,
            'cond_M': cond_M,
            'ridge_added_to_Sigma': ridge,
            'multiplier': multiplier,
            'norm_xdelta_unit': float(np.linalg.norm(xdelta_unit)),
            'norm_xdelta': float(np.linalg.norm(xdelta)),
            'jC_dot': float(jC.T @ xdelta),
            'v_dot': float(v.T @ xdelta),
            'normalized_b': self.use_normalized_b,
            'sqrt_multiplier': self.use_sqrt_multiplier
        }

        return xdelta, diag

    def calculate_x_delta(self, weights, n_remaining, candidate_player=None):
        """
        Calculate expected future pick adjustment using fixed formula.
        """
        if n_remaining <= 0:
            return np.zeros(self.n_cats)

        # Calculate roster size info
        N = 13  # total roster size
        K = 13 - n_remaining - 1  # picks already made

        # Use the fixed implementation
        x_delta, diagnostics = self._compute_xdelta_exact_fixed(
            jC=weights,
            v=self.v_vector,
            Sigma=self.cov_matrix,
            gamma=self.gamma,
            omega=self.omega,
            N=N,
            K=K
        )

        return x_delta

    def _calculate_average_opponent_x(self, my_team, player_name):
        """
        Calculate expected opponent X-scores based on average G-score picks.

        This implements the paper's directive (line 83-84):
        "fill with average G-score picks"
        """
        # Get all players and their G-scores
        all_players = self.scoring.league_data['PLAYER_NAME'].unique()

        # Remove already drafted players
        drafted = set(my_team) | {player_name}
        available = [p for p in all_players if p not in drafted]

        # Get G-scores for available players
        player_g_scores = []
        for p in available[:100]:  # Limit for efficiency
            try:
                g_score = self.scoring.calculate_all_g_scores(p)['TOTAL']
                if not np.isnan(g_score):
                    player_g_scores.append((p, g_score))
            except:
                continue

        # Sort by G-score
        player_g_scores.sort(key=lambda x: x[1], reverse=True)

        # Take average of top 13 players (roster size) for each opponent
        # Assume 11 opponents in 12-team league
        num_opponents = 11
        picks_per_opponent = 13

        opponent_x = np.zeros(self.n_cats)

        # Model: each opponent gets players from the top of the pool
        # Distribute top 13*11 = 143 players among opponents
        total_picks_needed = min(picks_per_opponent * num_opponents, len(player_g_scores))

        for i in range(min(total_picks_needed, len(player_g_scores))):
            player, _ = player_g_scores[i]
            player_x = np.array([
                self.scoring.calculate_x_score(player, cat)
                for cat in self.categories
            ])
            opponent_x += player_x

        # Average across all opponent picks
        if total_picks_needed > 0:
            opponent_x /= total_picks_needed
            # Scale up to represent full roster
            opponent_x *= picks_per_opponent

        return opponent_x

    def evaluate_player(self, player_name, my_team, opponent_teams,
                       picks_made, total_picks=13, last_weights=None,
                       format='each_category'):
        """
        Evaluate a candidate player using fixed H-scoring.

        FIX 3: Properly model opponents when unknown.
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

        # FIX 3: Calculate average opponent X-scores properly
        if self.use_proper_opponents and (not opponent_teams or all(len(team) == 0 for team in opponent_teams)):
            # Use average G-score picks as per paper
            opponent_x = self._calculate_average_opponent_x(my_team, player_name)
        else:
            # Original logic for when opponents are known
            opponent_x = np.zeros(self.n_cats)
            if opponent_teams:
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
            # First pick: perturb baseline toward candidate
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