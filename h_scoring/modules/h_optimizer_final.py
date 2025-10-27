"""
Final fixed H-scoring optimizer with proper covariance normalization.

Key fixes:
1. Normalize covariance matrix to unit diagonal (fixes massive eigenvalue issue)
2. Use much smaller multiplier for X_delta
3. Properly model opponents with average G-score picks
"""

import numpy as np
from scipy.stats import norm
import copy

from .h_optimizer import HScoreOptimizer


class HScoreOptimizerFinal(HScoreOptimizer):
    """Final fixed H-scoring optimizer."""

    def __init__(self, setup_params, scoring_system, omega=0.7, gamma=0.25):
        """Initialize with fixes."""
        super().__init__(setup_params, scoring_system, omega, gamma)

        # Store original covariance for reference
        self.cov_matrix_original = self.cov_matrix.copy()

        # Use better multiplier scale (from h_scoringchatgpt.py which worked well)
        self.xdelta_multiplier_scale = 0.25  # This gave good results in h_scoringchatgpt.py

        # Regularization parameters from h_scoringchatgpt.py
        self.ridge_frac = 1e-8  # For regularizing Sigma
        self.use_sigma_in_b = True  # Whether to multiply b vector by sigma

        # Store category-specific variances for proper win probability calculation
        self.category_variances = self._calculate_category_variances()

    def _normalize_covariance(self, Sigma):
        """
        Normalize covariance matrix to have unit diagonal.
        This prevents massive eigenvalues from amplifying X_delta.
        """
        # Get diagonal elements
        diag = np.diag(Sigma)

        # Create scaling matrix (1/sqrt(diag))
        # Protect against zero/negative diagonal
        diag_safe = np.maximum(diag, 1e-10)
        D_inv = np.diag(1.0 / np.sqrt(diag_safe))

        # Normalize: D^(-1) @ Sigma @ D^(-1)
        Sigma_normalized = D_inv @ Sigma @ D_inv

        return Sigma_normalized

    def _calculate_category_variances(self):
        """
        Calculate category-specific variances for win probability calculations.
        Uses actual within-player variances from the data, with reasonable caps.
        """
        category_variances = {}

        for cat in self.categories:
            # Get league stats which include variance information
            # Use the diagonal of the original covariance matrix as a proxy
            cat_idx = self.categories.index(cat)

            # The variance for a differential between two teams
            # Each team has ~13 players, so variance scales with roster size
            base_variance = self.cov_matrix_original[cat_idx, cat_idx]

            # Scale by roster size (13 players per team, 2 teams)
            roster_size = 13
            scaled_variance = base_variance * roster_size * 2

            # Cap variances to reasonable ranges
            # FG3M in particular has massive variance that needs limiting
            # Use a cap based on the median variance to keep relative differences
            if cat == 'FG3M':
                # FG3M has extreme variance, cap it more aggressively
                category_variances[cat] = min(300.0, max(50.0, scaled_variance))
            elif cat in ['FG_PCT', 'FT_PCT', 'FG3_PCT']:
                # Percentage stats should have lower variance
                category_variances[cat] = min(25.0, max(5.0, scaled_variance))
            else:
                # Other counting stats - cap at reasonable levels
                category_variances[cat] = min(400.0, max(20.0, scaled_variance))

        return category_variances

    def calculate_win_probabilities(self, my_team_x, opponent_x, variance=None):
        """
        Calculate win probabilities with category-specific variances.

        Overrides parent method to use proper category-specific variances.
        """
        # Calculate differential
        differential = my_team_x - opponent_x

        # Use category-specific variances
        win_probs = np.zeros(self.n_cats)

        for i, cat in enumerate(self.categories):
            # Get category-specific variance
            cat_variance = self.category_variances.get(cat, 26.0)  # Default fallback

            # Ensure positive variance
            cat_variance = max(cat_variance, 1e-6)

            # Standard deviation for this category
            std_dev = np.sqrt(cat_variance)

            # Z-score (clip to prevent extreme values)
            z_score = np.clip(differential[i] / std_dev, -10, 10)

            # Win probability (CDF of normal distribution)
            win_probs[i] = norm.cdf(z_score)

        return win_probs

    def calculate_x_delta(self, weights, n_remaining, candidate_x=None, current_team_x=None):
        """
        Calculate X_delta with proper scaling and candidate adaptation.

        Key fix: X_delta should adapt based on the candidate being evaluated.
        If you draft an elite DD player, future picks shouldn't focus on DD.
        If you draft a weak DD player, future picks should compensate.
        """
        if n_remaining <= 0:
            return np.zeros(self.n_cats)

        # Calculate roster size info
        N = 13  # total roster size
        K = 13 - n_remaining - 1  # picks already made

        # CRITICAL FIX: Adjust weights based on team composition after adding candidate
        adjusted_weights = weights.copy()

        if candidate_x is not None and current_team_x is not None:
            # Calculate team state after adding this candidate
            team_with_candidate = current_team_x + candidate_x

            # Adjust weights to de-emphasize categories where we're already strong
            # and emphasize categories where we're weak
            # NOTE: Adjustments are conservative since candidate is only 1/13 of final roster
            for i in range(self.n_cats):
                team_strength = team_with_candidate[i]

                # More conservative thresholds and smaller adjustments
                if team_strength > 5.0:
                    # Very strong - modestly reduce weight
                    # Adjustment strength: 0.15 (down from 0.3-0.7)
                    adjustment = max(0.85, 1.0 - (team_strength - 5.0) / 30.0)
                    adjusted_weights[i] *= adjustment
                elif team_strength < -2.0:
                    # Very weak - modestly increase weight
                    # Adjustment strength: 0.10 (down from 0.5)
                    adjustment = min(1.10, 1.0 + abs(team_strength + 2.0) / 30.0)
                    adjusted_weights[i] *= adjustment

            # Renormalize weights to sum to 1
            adjusted_weights = adjusted_weights / adjusted_weights.sum()

        # Use adjusted weights for X_delta calculation
        x_delta = self._compute_xdelta_simplified(
            jC=adjusted_weights,
            v=self.v_vector,
            Sigma=self.cov_matrix,  # Already normalized
            gamma=self.gamma,
            omega=self.omega,
            N=N,
            K=K
        )

        return x_delta

    def _compute_xdelta_simplified(self, jC, v, Sigma, gamma, omega, N, K):
        """
        Exact X_delta calculation from h_scoringchatgpt.py with better numerical stability.
        """
        # Normalize weights to sum to 1 (safe normalization)
        jC = np.asarray(jC, dtype=float).reshape(-1)
        v = np.asarray(v, dtype=float).reshape(-1)

        jC_sum = np.sum(jC)
        if abs(jC_sum) < 1e-12:
            jC = np.ones_like(jC) / float(len(jC))
        else:
            jC = jC / float(jC_sum)

        v_sum = np.sum(v)
        if abs(v_sum) < 1e-12:
            v = np.ones_like(v) / float(len(v))
        else:
            v = v / float(v_sum)

        m = len(jC)

        # Regularize Sigma (key insight from h_scoringchatgpt.py)
        trace = np.trace(Sigma)
        ridge = max(self.ridge_frac * (trace + 1e-12), 1e-12)
        Sigma_reg = Sigma + ridge * np.eye(m)

        # Compute projection of jC onto v in Sigma-metric
        denom_vSv = float(v.T @ Sigma_reg @ v)
        if denom_vSv <= 0 or not np.isfinite(denom_vSv):
            denom_vSv = 1e-8
        proj_coeff = float((v.T @ Sigma_reg @ jC) / denom_vSv)
        jC_perp = jC - v * proj_coeff

        # Compute sigma (standard deviation of jC_perp in Sigma metric)
        sigma2 = float(max(jC_perp.T @ Sigma_reg @ jC_perp, 0.0))
        sigma = np.sqrt(sigma2) if sigma2 > 0 else 0.0

        # Build constraint matrix U and target vector b
        U = np.vstack([v, jC])  # shape (2, m)

        if self.use_sigma_in_b and sigma > 1e-12:
            b = np.array([-gamma * sigma, omega * sigma], dtype=float)
        else:
            # Fallback if sigma is too small
            b = np.array([-gamma, omega], dtype=float)

        # Solve for z using regularized system
        M = U @ Sigma_reg @ U.T  # shape (2, 2)

        # Check condition number and add regularization if needed
        cond_M = np.linalg.cond(M) if np.all(np.isfinite(M)) else 1e16
        if cond_M > 1e12 or not np.isfinite(cond_M):
            jitter = 1e-8 * (np.trace(M) + 1e-12)
            M = M + jitter * np.eye(2)

        # Solve the system
        try:
            z = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            z = np.linalg.pinv(M) @ b

        # Compute xdelta_unit
        xdelta_unit = Sigma_reg @ U.T @ z

        # Apply multiplier
        remaining_picks = float(max(0, N - K - 1))
        effective_multiplier = remaining_picks * self.xdelta_multiplier_scale

        x_delta = effective_multiplier * xdelta_unit

        return x_delta

    def optimize_weights_with_regularization(self, candidate_x, current_team_x, opponent_x,
                                            n_remaining, initial_weights, picks_made, total_picks,
                                            max_iterations=100, learning_rate=0.01,
                                            format='each_category'):
        """
        Optimize category weights with regularization to prevent punting on early picks.

        Key fix: On early picks (pick #1-3), strongly penalize deviation from baseline weights.
        This prevents the optimizer from "discovering" that punting scarce categories works,
        when the whole point is to draft elite players in those scarce categories.

        Regularization strength decreases as draft progresses (more team context).
        """
        # Initialize weights
        if initial_weights is None:
            weights = self.baseline_weights.copy()
        else:
            weights = initial_weights.copy()

        # Normalize
        weights = weights / weights.sum()

        # Regularization strength: Strong on pick #1, decreases to 0 by pick #13
        # Formula: lambda_reg * (1 - picks_made / total_picks)^2
        # REDUCED: Was 2.0, now 0.5 to allow more aggressive weight shifts
        draft_progress = picks_made / total_picks
        reg_strength = 0.5 * (1.0 - draft_progress) ** 2  # Weaker to allow adaptation

        # Adam optimizer parameters
        m = np.zeros(self.n_cats)
        v = np.zeros(self.n_cats)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        def regularized_objective(w):
            """Objective with L2 penalty for deviating from baseline."""
            # Base objective (win probability)
            base_obj = self.calculate_objective(
                w, candidate_x, current_team_x,
                opponent_x, n_remaining, format
            )

            # Regularization penalty (L2 distance from baseline)
            deviation = np.sum((w - self.baseline_weights) ** 2)
            penalty = reg_strength * deviation

            # Return penalized objective
            return base_obj - penalty

        # Calculate initial objective value
        initial_value = regularized_objective(weights)

        best_value = initial_value
        best_weights = weights.copy()

        for iteration in range(max_iterations):
            # Calculate gradient numerically (including regularization)
            gradient = np.zeros(self.n_cats)
            eps = 1e-5

            for i in range(self.n_cats):
                # Perturb weight i
                weights_plus = weights.copy()
                weights_plus[i] += eps
                weights_plus = weights_plus / weights_plus.sum()

                weights_minus = weights.copy()
                weights_minus[i] -= eps
                weights_minus = np.maximum(weights_minus, 1e-6)
                weights_minus = weights_minus / weights_minus.sum()

                # Finite difference
                obj_plus = regularized_objective(weights_plus)
                obj_minus = regularized_objective(weights_minus)
                gradient[i] = (obj_plus - obj_minus) / (2 * eps)

            # Check for NaN in gradient
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                break

            # Adam update
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))

            # Update weights
            weights = weights + learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Ensure positive weights
            weights = np.maximum(weights, 1e-4)

            # Normalize to sum to 1
            weights = weights / weights.sum()

            # Calculate current objective value
            current_value = regularized_objective(weights)

            # Check for NaN/inf in objective
            if np.isnan(current_value) or np.isinf(current_value):
                break

            # Track best
            if current_value > best_value:
                best_value = current_value
                best_weights = weights.copy()

            # Early stopping if converged
            if iteration > 10 and abs(current_value - best_value) < 1e-6:
                break

        # Return unpenalized H-score (actual win probability)
        final_h_score = self.calculate_objective(
            best_weights, candidate_x, current_team_x,
            opponent_x, n_remaining, format
        )

        return best_weights, final_h_score

    def evaluate_player(self, player_name, my_team, opponent_teams,
                       picks_made, total_picks=13, last_weights=None,
                       format='each_category'):
        """
        Evaluate player with all fixes applied.
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

        # Model opponents properly
        if not opponent_teams or all(len(team) == 0 for team in opponent_teams if team):
            # Use expected value based on average picks
            opponent_x = self._calculate_average_opponent_x(my_team, player_name, picks_made)
        else:
            # Calculate actual opponent X-scores
            opponent_x = np.zeros(self.n_cats)
            num_opponents = 0
            for opp_team in opponent_teams:
                if opp_team:  # Skip empty teams
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

        # Initialize weights
        if picks_made == 0:
            # First pick: use baseline weights with small perturbation
            initial_weights = self._perturb_weights_toward_player(
                self.baseline_weights, candidate_x
            )
        elif last_weights is not None:
            initial_weights = last_weights
        else:
            initial_weights = self.baseline_weights.copy()

        # CRITICAL FIX: Don't optimize on first pick - use baseline weights
        # On pick 1, there's no team context yet, and optimization tends to
        # punt scarce categories (the opposite of what you want early!)
        if picks_made == 0:
            # Use baseline weights without optimization for first pick only
            optimal_weights = self.baseline_weights.copy()
            h_score = self.calculate_objective(
                optimal_weights, candidate_x, current_team_x,
                opponent_x, n_remaining, format
            )
        else:
            # After pick 1, optimize weights with regularization
            optimal_weights, h_score = self.optimize_weights_with_regularization(
                candidate_x, current_team_x, opponent_x,
                n_remaining, initial_weights, picks_made, total_picks, format=format
            )

        return h_score, optimal_weights

    def _calculate_average_opponent_x(self, my_team, player_name, picks_made):
        """
        Calculate expected opponent X-scores based on remaining player pool.
        """
        # Get all players and their G-scores
        all_players = self.scoring.league_data['PLAYER_NAME'].unique()

        # Remove already drafted
        drafted = set(my_team) | {player_name}
        available = [p for p in all_players if p not in drafted]

        # Get G-scores for top available players
        player_scores = []
        for p in available[:150]:  # Look at more players for better estimate
            try:
                g_score = self.scoring.calculate_all_g_scores(p)['TOTAL']
                if not np.isnan(g_score):
                    player_scores.append((p, g_score))
            except:
                continue

        # Sort by G-score
        player_scores.sort(key=lambda x: x[1], reverse=True)

        # Model opponent strength based on draft position
        # In a snake draft, opponents get picks at various positions
        # For first pick evaluation, model average opponent roster

        # Calculate expected opponent X-scores
        opponent_x = np.zeros(self.n_cats)

        # In a 12-team league, opponents on average get picks at positions:
        # 6th, 18th, 30th, 42nd, etc. (assuming we're team 6)
        # Model this by taking players at those approximate positions

        if len(player_scores) == 0:
            return opponent_x

        # Sample players that opponents would likely get
        # Start from position based on picks_made
        roster_size = 13

        # For each opponent roster spot, estimate the player they'd get
        for roster_spot in range(roster_size):
            # Calculate approximate draft position for this roster spot
            # In a snake draft with 12 teams
            if roster_spot % 2 == 0:
                # Even rounds: picks 6-17 available to opponents
                avg_position = 6 + roster_spot * 12
            else:
                # Odd rounds: picks 7-18 available to opponents
                avg_position = 7 + roster_spot * 12

            # Adjust for already made picks
            avg_position = avg_position - picks_made

            # Get player at this position (with some averaging)
            if avg_position < len(player_scores):
                # Average a few players around this position for smoothing
                start_idx = max(0, avg_position - 2)
                end_idx = min(len(player_scores), avg_position + 3)

                for idx in range(start_idx, end_idx):
                    if idx < len(player_scores):
                        player, _ = player_scores[idx]
                        player_x = np.array([
                            self.scoring.calculate_x_score(player, cat)
                            for cat in self.categories
                        ])
                        # Weight by distance from target position
                        weight = 1.0 / (1.0 + abs(idx - avg_position))
                        opponent_x += player_x * weight / 5.0  # Normalize by window size

        return opponent_x

    def _apply_diminishing_returns(self, win_probs):
        """
        Apply diminishing returns to win probabilities.

        Categories near 0% or 100% win probability have diminishing marginal value.
        Categories near 50% have maximum marginal value (swing categories).

        This prevents stacking already-dominant categories.
        """
        # Transform win probabilities to emphasize close matchups
        # Use a formula that peaks at 0.5 and decreases toward 0 and 1

        # Option 1: Entropy-like (peaks at 0.5)
        # value = -p*log(p) - (1-p)*log(1-p)

        # Option 2: Simpler quadratic (peaks at 0.5)
        # value = 4 * p * (1 - p)
        # At p=0.5: value = 1.0
        # At p=0.0 or p=1.0: value = 0.0

        # We'll use option 2 but add base value to avoid completely ignoring dominant cats
        # marginal_value = 0.3 + 0.7 * (4 * p * (1-p))
        # This gives: p=0.5 → 1.0, p=0.0 → 0.3, p=1.0 → 0.3

        p = np.clip(win_probs, 0.01, 0.99)  # Avoid log(0)

        # Marginal value is higher for close matchups
        swing_factor = 4 * p * (1 - p)  # Peaks at 0.5
        marginal_value = 0.3 + 0.7 * swing_factor

        # Apply transformation: weighted win probability
        return win_probs * marginal_value

    def calculate_objective(self, weights, candidate_x, current_team_x,
                           opponent_x, n_remaining, format='each_category'):
        """
        Calculate objective function with proper category-specific variances.

        Overrides parent to use better variance modeling.
        """
        # Calculate X_delta for future picks (with candidate adaptation)
        x_delta = self.calculate_x_delta(weights, n_remaining,
                                         candidate_x=candidate_x,
                                         current_team_x=current_team_x)

        # Total team projection
        team_projection = current_team_x + candidate_x + x_delta

        # Use our improved win probability calculation with category-specific variances
        win_probs = self.calculate_win_probabilities(team_projection, opponent_x)

        # Apply diminishing returns to prevent stacking saturated categories
        adjusted_win_probs = self._apply_diminishing_returns(win_probs)

        if format == 'each_category':
            # CRITICAL FIX: Weight win probabilities by baseline weights (scarcity)
            # A 90% chance to win DD (22.6% weight) should be worth MORE than
            # a 90% chance to win FG_PCT (2.5% weight)
            objective = np.sum(self.baseline_weights * adjusted_win_probs)
        else:
            # Most categories format
            objective = np.sum(adjusted_win_probs)  # Simplified for now

        return objective