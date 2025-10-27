"""
Covariance matrix and baseline weight calculation for H-scoring.

Implements the statistical foundation for modeling player correlations
and calculating baseline category weights.
"""

import pandas as pd
import numpy as np
from scipy import stats


class CovarianceCalculator:
    """Calculate covariance matrix and baseline weights for H-scoring."""

    def __init__(self, league_data, scoring_system):
        """
        Initialize covariance calculator.

        Parameters:
        -----------
        league_data : DataFrame
            Weekly player statistics
        scoring_system : PlayerScoring
            Initialized scoring system for X-scores
        """
        self.league_data = league_data
        self.scoring = scoring_system
        self.categories = scoring_system.all_cats

        # Calculate season averages for covariance
        self.season_averages = self._calculate_season_averages()

    def _calculate_season_averages(self):
        """Calculate season averages per player for covariance calculation."""
        # Group by player and calculate mean for each category
        season_avgs = self.league_data.groupby('PLAYER_NAME').agg({
            **{cat: 'mean' for cat in self.categories}
        }).reset_index()

        return season_avgs

    def calculate_covariance_matrix(self):
        """
        Calculate covariance matrix between categories using X-scores.

        The covariance matrix captures how different statistical categories
        correlate across players (e.g., high REB correlates with BLK).

        Returns:
        --------
        numpy array : Covariance matrix (n_categories x n_categories)
        """
        # Calculate X-scores for all players
        x_score_matrix = []

        for player_name in self.season_averages['PLAYER_NAME']:
            x_scores = self.scoring.calculate_all_x_scores(player_name)
            x_score_vector = [x_scores[cat] for cat in self.categories]
            x_score_matrix.append(x_score_vector)

        x_score_matrix = np.array(x_score_matrix)

        # Check for NaN values
        if np.any(np.isnan(x_score_matrix)):
            print(f"⚠️  Warning: Found NaN values in X-score matrix")
            # Replace NaN with 0
            x_score_matrix = np.nan_to_num(x_score_matrix, nan=0.0)

        # Calculate covariance matrix
        # Use ddof=1 for sample covariance
        cov_matrix = np.cov(x_score_matrix, rowvar=False, ddof=1)

        # Check for NaN in covariance matrix
        if np.any(np.isnan(cov_matrix)):
            print(f"⚠️  Warning: NaN in covariance matrix - replacing with small values")
            # Replace NaN with small positive values on diagonal, 0 off-diagonal
            for i in range(cov_matrix.shape[0]):
                for j in range(cov_matrix.shape[1]):
                    if np.isnan(cov_matrix[i, j]):
                        if i == j:
                            cov_matrix[i, j] = 1e-6  # Small positive variance
                        else:
                            cov_matrix[i, j] = 0.0  # No covariance

        return cov_matrix

    def calculate_covariance_by_position(self):
        """
        Calculate separate covariance matrices for each position.

        Returns:
        --------
        dict : Covariance matrices by position
        """
        # Note: Position data would need to be added to league_data
        # For now, calculate overall covariance
        # This is a placeholder for position-specific covariance

        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        cov_by_position = {}

        # TODO: Implement position-specific covariance when position data available
        # For now, use overall covariance for all positions
        overall_cov = self.calculate_covariance_matrix()

        for pos in positions:
            cov_by_position[pos] = overall_cov

        return cov_by_position

    def calculate_baseline_weights(self, method='scarcity'):
        """
        Calculate baseline category weights.

        Parameters:
        -----------
        method : str
            Method for calculating baseline weights:
            - 'uniform': Equal weight to all categories (1/n)
            - 'scarcity': Weight by coefficient of variation
            - 'v_vector': Use v vector (X to G conversion)

        Returns:
        --------
        numpy array : Baseline weights (sum to 1)
        """
        n_cats = len(self.categories)

        if method == 'uniform':
            # Equal weights
            weights = np.ones(n_cats) / n_cats

        elif method == 'scarcity':
            # Weight by coefficient of variation (higher CV = more scarce = more valuable)
            cv_values = []

            for cat in self.categories:
                cat_values = self.season_averages[cat].values
                mean_val = np.mean(cat_values)
                std_val = np.std(cat_values)

                if mean_val > 0:
                    cv = std_val / mean_val
                else:
                    cv = 0

                cv_values.append(cv)

            cv_values = np.array(cv_values)

            # Normalize to sum to 1
            if cv_values.sum() > 0:
                weights = cv_values / cv_values.sum()
            else:
                weights = np.ones(n_cats) / n_cats

        elif method == 'v_vector':
            # Use v vector from scoring system
            weights = self.scoring.calculate_v_vector()

        else:
            raise ValueError(f"Unknown method: {method}")

        return weights

    def calculate_inverse_covariance(self, cov_matrix=None, regularization=1e-6):
        """
        Calculate inverse of covariance matrix (precision matrix).

        Parameters:
        -----------
        cov_matrix : numpy array, optional
            Covariance matrix (if None, calculates it)
        regularization : float
            Regularization term to ensure invertibility

        Returns:
        --------
        numpy array : Inverse covariance matrix
        """
        if cov_matrix is None:
            cov_matrix = self.calculate_covariance_matrix()

        # Add regularization to diagonal for numerical stability
        n = cov_matrix.shape[0]
        regularized_cov = cov_matrix + regularization * np.eye(n)

        # Calculate inverse
        try:
            inv_cov = np.linalg.inv(regularized_cov)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            print("Warning: Singular matrix, using pseudo-inverse")
            inv_cov = np.linalg.pinv(regularized_cov)

        return inv_cov

    def get_category_correlations(self):
        """
        Get pairwise correlations between categories.

        Returns:
        --------
        DataFrame : Correlation matrix
        """
        # Calculate correlations from season averages
        corr_matrix = self.season_averages[self.categories].corr()

        return corr_matrix

    def analyze_correlations(self):
        """
        Analyze and report key category correlations.

        Returns:
        --------
        dict : Notable correlations
        """
        corr_matrix = self.get_category_correlations()

        # Find strongest positive and negative correlations
        correlations = []

        for i, cat1 in enumerate(self.categories):
            for j, cat2 in enumerate(self.categories):
                if i < j:  # Upper triangle only
                    corr_value = corr_matrix.loc[cat1, cat2]
                    correlations.append({
                        'cat1': cat1,
                        'cat2': cat2,
                        'correlation': corr_value
                    })

        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.sort_values('correlation', ascending=False)

        return {
            'strongest_positive': correlations_df.head(10),
            'strongest_negative': correlations_df.tail(10),
            'all_correlations': correlations_df
        }

    def calculate_A_matrix(self, v_vector, cov_matrix):
        """
        Calculate A matrix used in X_delta calculation.

        A = I - (Σ * v * v^T) / (v^T * Σ * v)

        Parameters:
        -----------
        v_vector : numpy array
            V vector for X to G conversion
        cov_matrix : numpy array
            Covariance matrix

        Returns:
        --------
        numpy array : A matrix
        """
        n = len(v_vector)
        I = np.eye(n)

        # Calculate v^T * Σ * v
        denominator = v_vector.T @ cov_matrix @ v_vector

        if denominator == 0:
            return I

        # Calculate Σ * v * v^T
        numerator = cov_matrix @ v_vector @ v_vector.T

        # A = I - numerator / denominator
        A = I - (numerator / denominator)

        return A

    def get_setup_params(self):
        """
        Get all setup parameters needed for H-scoring.

        Returns:
        --------
        dict : Setup parameters including covariance, baseline weights, etc.
        """
        # Calculate covariance matrix
        cov_matrix = self.calculate_covariance_matrix()

        # Calculate inverse
        inv_cov = self.calculate_inverse_covariance(cov_matrix)

        # Calculate baseline weights (using scarcity method)
        baseline_weights = self.calculate_baseline_weights(method='scarcity')

        # Calculate v vector
        v_vector = self.scoring.calculate_v_vector()

        # Calculate A matrix
        A_matrix = self.calculate_A_matrix(v_vector, cov_matrix)

        # Analyze correlations
        correlations = self.analyze_correlations()

        return {
            'covariance_matrix': cov_matrix,
            'inverse_covariance': inv_cov,
            'baseline_weights': baseline_weights,
            'v_vector': v_vector,
            'A_matrix': A_matrix,
            'categories': self.categories,
            'correlations': correlations['all_correlations'],
            'top_positive_corr': correlations['strongest_positive'],
            'top_negative_corr': correlations['strongest_negative']
        }


if __name__ == "__main__":
    # Example usage
    import json
    from scoring import PlayerScoring

    # Load data
    league_data = pd.read_csv('../data/league_weekly_data_20240101_120000.csv')

    with open('../data/player_variances_20240101_120000.json', 'r') as f:
        player_variances = json.load(f)

    # Initialize scoring
    scoring = PlayerScoring(league_data, player_variances)

    # Initialize covariance calculator
    cov_calc = CovarianceCalculator(league_data, scoring)

    # Get setup parameters
    setup_params = cov_calc.get_setup_params()

    print("\nCovariance Matrix shape:", setup_params['covariance_matrix'].shape)
    print("\nBaseline Weights:")
    for idx, cat in enumerate(setup_params['categories']):
        print(f"{cat}: {setup_params['baseline_weights'][idx]:.4f}")

    print("\nTop 5 Positive Correlations:")
    print(setup_params['top_positive_corr'].head())

    print("\nTop 5 Negative Correlations:")
    print(setup_params['top_negative_corr'].tail())