"""
H-scoring optimizer with gradient descent.

Implements the core H-scoring algorithm from Rosenof (2024) including:
- X_delta calculation for future pick adjustment
- Win probability calculation
- Gradient descent optimization with Adam
- Player valuation for draft decisions
"""

import numpy as np
from scipy.stats import norm
import copy


class HScoreOptimizer:
    """H-scoring optimizer using gradient descent."""

    def __init__(self, setup_params, scoring_system, omega=0.7, gamma=0.25):
        """
        Initialize H-score optimizer.

        Parameters:
        -----------
        setup_params : dict
            Setup parameters from CovarianceCalculator
        scoring_system : PlayerScoring
            Scoring system for calculating X-scores
        omega : float
            Parameter for weighted category strength (default 0.7)
        gamma : float
            Parameter for generic value penalty (default 0.25)
        """
        self.setup_params = setup_params
        self.scoring = scoring_system
        self.omega = omega
        self.gamma = gamma

        self.categories = setup_params['categories']
        self.n_cats = len(self.categories)

        # Extract key matrices
        self.cov_matrix = setup_params['covariance_matrix']
        self.inv_cov = setup_params['inverse_covariance']
        self.baseline_weights = setup_params['baseline_weights']
        self.v_vector = setup_params['v_vector']

    def calculate_x_delta(self, weights, n_remaining, candidate_player=None):
        """
        Calculate expected future pick adjustment (X_delta) using exact formula.

        This implementation is numerically stable and follows the paper exactly.
        """
        if n_remaining <= 0:
            return np.zeros(self.n_cats)

        # Calculate roster size info
        N = 13  # total roster size
        K = 13 - n_remaining - 1  # picks already made

        # Use the exact implementation
        x_delta, diagnostics = self._compute_xdelta_exact(
            jC=weights,
            v=self.v_vector,
            Sigma=self.cov_matrix,
            gamma=self.gamma,
            omega=self.omega,
            N=N,
            K=K
        )

        return x_delta

    def _compute_xdelta_exact(self, jC, v, Sigma, gamma, omega, N, K, muC_P=None,
                             eps_sigma_rel=1e-8, eps_Uinv=1e-8, ridge_frac=1e-8):
        """
        Exact implementation of X_delta with numerical stability.

        jC, v : 1D arrays (length m) -- ensure both sum to 1 (paper's convention)
        Sigma : (m,m) covariance matrix (must be symmetric)
        gamma, omega : scalars (paper used omega=0.7, gamma=0.25 as starting values)
        N, K : integers; picks remaining multiplier will be (N - K - 1)
        muC_P: optional vector to add at the end (positional adjustment μ_C P)
        Returns x_delta (length m) and diagnostics dict
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
        # paper defines constraints v^T x = -gamma*sigma, jC^T x = omega*sigma
        b = np.array([-gamma * sigma, omega * sigma], dtype=float)  # shape (2,)

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

        # 8) multiply by number of picks remaining and add positional adj (muC_P)
        multiplier = float(max(0, N - K - 1))   # per paper
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
        }

        return xdelta, diag

    def calculate_sigma_squared(self, weights):
        """
        Calculate sigma^2 for weight standard deviation.

        sigma^2 = jC^T * A * Σ * A^T * jC
        where A = I - (Σ*v*v^T)/(v^T*Σ*v)

        Parameters:
        -----------
        weights : numpy array
            Category weights

        Returns:
        --------
        float : sigma^2
        """
        A = self.setup_params['A_matrix']
        Sigma = self.cov_matrix

        # sigma^2 = jC^T * A * Σ * A^T * jC
        sigma_sq = weights.T @ A @ Sigma @ A.T @ weights

        return max(sigma_sq, 1e-10)  # Ensure positive

    def calculate_win_probabilities(self, my_team_x, opponent_x, variance):
        """
        Calculate win probability for each category.

        Win_prob = Φ((my_team - opponent) / sqrt(variance))

        Parameters:
        -----------
        my_team_x : numpy array
            My team's X-scores by category
        opponent_x : numpy array
            Opponent's X-scores by category
        variance : float
            Combined variance

        Returns:
        --------
        numpy array : Win probabilities by category
        """
        # Calculate differential
        differential = my_team_x - opponent_x

        # Ensure positive variance
        variance = max(variance, 1e-6)

        # Standard deviation for differential
        std_dev = np.sqrt(variance)

        # Z-scores (clip to prevent extreme values)
        z_scores = np.clip(differential / std_dev, -10, 10)

        # Win probabilities (CDF of normal distribution)
        win_probs = norm.cdf(z_scores)

        return win_probs

    def calculate_objective(self, weights, candidate_x, current_team_x,
                           opponent_x, n_remaining, format='each_category'):
        """
        Calculate objective function value.

        For Each Category: V = sum of win probabilities
        For Most Categories: V = P(win majority of categories)

        Parameters:
        -----------
        weights : numpy array
            Category weights
        candidate_x : numpy array
            Candidate player X-scores
        current_team_x : numpy array
            Current team aggregate X-scores
        opponent_x : numpy array
            Opponent team aggregate X-scores
        n_remaining : int
            Remaining picks
        format : str
            'each_category' or 'most_categories'

        Returns:
        --------
        float : Objective value
        """
        # Calculate X_delta for future picks
        x_delta = self.calculate_x_delta(weights, n_remaining)

        # Total team projection
        team_projection = current_team_x + candidate_x + x_delta

        # Combined variance (simplified - using constant variance per player)
        # In full implementation, would model opponent variance more carefully
        roster_size = 13
        variance = 2 * roster_size + n_remaining * 1.0

        # Calculate win probabilities
        win_probs = self.calculate_win_probabilities(
            team_projection, opponent_x, variance
        )

        if format == 'each_category':
            # Sum of win probabilities
            objective = np.sum(win_probs)
        else:
            # Most categories: need to calculate P(win > n_cats/2)
            # Simplified implementation - use binomial approximation
            # TODO: Implement full tree-based calculation for most categories
            objective = np.sum(win_probs)  # Placeholder

        return objective

    def calculate_gradient(self, weights, candidate_x, current_team_x,
                          opponent_x, n_remaining, format='each_category'):
        """
        Calculate gradient of objective function w.r.t. weights.

        Gradient = sum over categories of: PDF(z) * d(X_delta)/d(jC) / sigma

        Parameters:
        -----------
        weights : numpy array
            Category weights
        candidate_x : numpy array
            Candidate player X-scores
        current_team_x : numpy array
            Current team X-scores
        opponent_x : numpy array
            Opponent X-scores
        n_remaining : int
            Remaining picks
        format : str
            Format type

        Returns:
        --------
        numpy array : Gradient vector
        """
        # Calculate X_delta
        x_delta = self.calculate_x_delta(weights, n_remaining)

        # Team projection
        team_projection = current_team_x + candidate_x + x_delta

        # Variance
        roster_size = 13
        variance = 2 * roster_size + n_remaining * 1.0
        std_dev = np.sqrt(variance)

        # Calculate differentials and z-scores
        differential = team_projection - opponent_x
        z_scores = differential / std_dev

        # PDF values at z-scores
        pdf_values = norm.pdf(z_scores)

        # Numerical gradient calculation
        # In full implementation, would derive analytical gradient
        epsilon = 1e-5
        gradient = np.zeros(self.n_cats)

        for i in range(self.n_cats):
            weights_plus = weights.copy()
            weights_plus[i] += epsilon

            # Normalize
            weights_plus = weights_plus / weights_plus.sum()

            x_delta_plus = self.calculate_x_delta(weights_plus, n_remaining)
            team_plus = current_team_x + candidate_x + x_delta_plus

            diff_plus = team_plus - opponent_x
            z_plus = diff_plus / std_dev
            win_prob_plus = norm.cdf(z_plus).sum()

            # Current objective
            win_prob_current = norm.cdf(z_scores).sum()

            # Finite difference
            gradient[i] = (win_prob_plus - win_prob_current) / epsilon

        return gradient

    def optimize_weights(self, candidate_x, current_team_x, opponent_x,
                        n_remaining, initial_weights=None,
                        max_iterations=100, learning_rate=0.01,
                        format='each_category'):
        """
        Optimize category weights using gradient descent (Adam optimizer).

        Parameters:
        -----------
        candidate_x : numpy array
            Candidate player X-scores
        current_team_x : numpy array
            Current team X-scores
        opponent_x : numpy array
            Opponent X-scores
        n_remaining : int
            Remaining picks
        initial_weights : numpy array, optional
            Initial weights (if None, uses baseline)
        max_iterations : int
            Maximum gradient descent iterations
        learning_rate : float
            Learning rate for Adam
        format : str
            League format

        Returns:
        --------
        tuple : (optimal_weights, optimal_value)
        """
        # Initialize weights
        if initial_weights is None:
            weights = self.baseline_weights.copy()
        else:
            weights = initial_weights.copy()

        # Normalize
        weights = weights / weights.sum()

        # Adam optimizer parameters
        m = np.zeros(self.n_cats)  # First moment
        v = np.zeros(self.n_cats)  # Second moment
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        # Calculate initial objective value
        initial_value = self.calculate_objective(
            weights, candidate_x, current_team_x,
            opponent_x, n_remaining, format
        )

        best_value = initial_value
        best_weights = weights.copy()

        for iteration in range(max_iterations):
            # Calculate gradient
            gradient = self.calculate_gradient(
                weights, candidate_x, current_team_x,
                opponent_x, n_remaining, format
            )

            # Check for NaN in gradient
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                # Gradient calculation failed, return current best
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
            current_value = self.calculate_objective(
                weights, candidate_x, current_team_x,
                opponent_x, n_remaining, format
            )

            # Check for NaN/inf in objective
            if np.isnan(current_value) or np.isinf(current_value):
                # Optimization failed, return best so far
                break

            # Track best
            if current_value > best_value:
                best_value = current_value
                best_weights = weights.copy()

            # Check convergence
            if iteration > 10 and abs(current_value - best_value) < 1e-6:
                break

        return best_weights, best_value

    def evaluate_player(self, player_name, my_team, opponent_teams,
                       picks_made, total_picks=13, last_weights=None,
                       format='each_category'):
        """
        Evaluate a candidate player using H-scoring.

        Parameters:
        -----------
        player_name : str
            Candidate player name
        my_team : list of str
            Current team player names
        opponent_teams : list of lists
            Opponent rosters
        picks_made : int
            Number of picks made
        total_picks : int
            Total roster size
        last_weights : numpy array, optional
            Weights from last optimization
        format : str
            League format

        Returns:
        --------
        tuple : (h_score, optimal_weights)
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

        # Calculate average opponent X-scores
        # FIXED: Per paper line 83, when opponents unknown, fill with average G-score picks
        if not opponent_teams or all(len(team) == 0 for team in opponent_teams):
            # Model expected opponent teams based on top G-score picks
            # Get list of all drafted players
            drafted_players = set(my_team) | {player_name}

            # For first pick, model what average opponent team would look like
            # with same number of picks. Since X-scores are centered at 0,
            # a team of average players has X-score sum near 0
            # But top drafted players should have positive G-scores/X-scores on average

            # Model: opponents have made picks_made selections from top of pool
            # Their average X-score per player would be positive (good players)
            # Estimate based on typical draft position value

            if picks_made == 0:
                # First pick - opponents would also be picking top players
                # Model them as having similar quality to candidates being evaluated
                # Use baseline weights to estimate typical opponent strength
                opponent_x = np.ones(self.n_cats) * 0.5  # Slightly positive
            else:
                # Opponents have made picks - model accumulated value
                # Top picks have declining value, roughly linear in X-score space
                avg_pick_quality = max(0, 2.0 - 0.15 * picks_made)  # Declining quality
                opponent_x = np.ones(self.n_cats) * avg_pick_quality * picks_made
        else:
            opponent_x = np.zeros(self.n_cats)
            for opp_team in opponent_teams:
                for player in opp_team:
                    player_x = np.array([
                        self.scoring.calculate_x_score(player, cat)
                        for cat in self.categories
                    ])
                    opponent_x += player_x

            opponent_x /= len(opponent_teams) if opponent_teams else 1

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

    def _perturb_weights_toward_player(self, baseline, player_x):
        """
        Perturb baseline weights toward player's strengths.

        Uses 70% baseline + 30% player strength weighting.
        """
        # Normalize player X-scores to get relative strengths
        player_strengths = np.maximum(player_x, 0)

        if player_strengths.sum() > 0:
            player_strengths = player_strengths / player_strengths.sum()
        else:
            player_strengths = np.ones(self.n_cats) / self.n_cats

        # Mix baseline with player strengths
        perturbed = 0.7 * baseline + 0.3 * player_strengths

        # Normalize
        perturbed = perturbed / perturbed.sum()

        return perturbed


if __name__ == "__main__":
    # Example usage would require full setup
    print("H-Score Optimizer module")