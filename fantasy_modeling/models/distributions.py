"""
Statistical distributions for basketball modeling.

Implements Beta-Binomial, Poisson, and Negative Binomial distributions
for modeling various basketball statistics.
"""

import numpy as np
from scipy import stats, special
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BetaBinomial:
    """
    Beta-Binomial distribution for modeling shooting percentages.

    The Beta-Binomial is a compound distribution where the success
    probability p follows a Beta distribution, and given p, the
    number of successes follows a Binomial distribution.
    """

    def __init__(self, alpha: float, beta: float):
        """
        Initialize Beta-Binomial distribution.

        Args:
            alpha: Alpha parameter of Beta distribution (prior successes)
            beta: Beta parameter of Beta distribution (prior failures)
        """
        self.alpha = alpha
        self.beta = beta

    def pmf(self, k: Union[int, np.ndarray], n: int) -> Union[float, np.ndarray]:
        """
        Probability mass function.

        Args:
            k: Number of successes
            n: Number of trials

        Returns:
            Probability of k successes in n trials
        """
        # Use log-space for numerical stability
        log_pmf = (
            special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1) +
            special.gammaln(k + self.alpha) + special.gammaln(n - k + self.beta) -
            special.gammaln(n + self.alpha + self.beta) +
            special.gammaln(self.alpha + self.beta) -
            special.gammaln(self.alpha) - special.gammaln(self.beta)
        )
        return np.exp(log_pmf)

    def sample(self, n: int, size: int = 1) -> np.ndarray:
        """
        Sample from Beta-Binomial distribution.

        Args:
            n: Number of trials
            size: Number of samples to generate

        Returns:
            Array of sampled successes
        """
        # First sample p from Beta
        p = np.random.beta(self.alpha, self.beta, size)

        # Then sample successes from Binomial with sampled p
        samples = np.random.binomial(n, p)

        return samples if size > 1 else samples[0]

    def mean(self, n: int) -> float:
        """Expected number of successes in n trials."""
        return n * self.alpha / (self.alpha + self.beta)

    def variance(self, n: int) -> float:
        """Variance of successes in n trials."""
        p = self.alpha / (self.alpha + self.beta)
        dispersion = (self.alpha + self.beta + n) / (self.alpha + self.beta + 1)
        return n * p * (1 - p) * dispersion

    def update_posterior(self, successes: int, trials: int) -> 'BetaBinomial':
        """
        Update distribution with observed data (Bayesian update).

        Args:
            successes: Observed number of successes
            trials: Observed number of trials

        Returns:
            New BetaBinomial with updated parameters
        """
        new_alpha = self.alpha + successes
        new_beta = self.beta + (trials - successes)
        return BetaBinomial(new_alpha, new_beta)

    def credible_interval(self, n: int, alpha_level: float = 0.05) -> Tuple[int, int]:
        """
        Compute credible interval for number of successes.

        Args:
            n: Number of trials
            alpha_level: Significance level (default 0.05 for 95% CI)

        Returns:
            Tuple of (lower, upper) bounds
        """
        # Sample many times and compute quantiles
        samples = self.sample(n, size=10000)
        lower = np.quantile(samples, alpha_level / 2)
        upper = np.quantile(samples, 1 - alpha_level / 2)
        return int(lower), int(upper)


class PoissonDistribution:
    """
    Poisson distribution for modeling count statistics.

    Used for modeling counting stats like points, rebounds, assists, etc.
    """

    def __init__(self, lambda_param: float, prior_strength: float = 0):
        """
        Initialize Poisson distribution.

        Args:
            lambda_param: Rate parameter (expected value)
            prior_strength: Strength of prior (for Bayesian updates)
        """
        self.lambda_param = lambda_param
        self.prior_strength = prior_strength
        self.n_observations = 0

    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability mass function."""
        return stats.poisson.pmf(k, self.lambda_param)

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from distribution."""
        samples = np.random.poisson(self.lambda_param, size)
        return samples if size > 1 else samples[0]

    def mean(self) -> float:
        """Expected value."""
        return self.lambda_param

    def variance(self) -> float:
        """Variance (equal to mean for Poisson)."""
        return self.lambda_param

    def update_with_gamma_prior(self, observations: np.ndarray,
                               shape: float = 1.0, rate: float = 1.0) -> 'PoissonDistribution':
        """
        Bayesian update with Gamma prior (conjugate).

        Args:
            observations: Array of observed counts
            shape: Shape parameter of Gamma prior
            rate: Rate parameter of Gamma prior

        Returns:
            Updated Poisson distribution
        """
        # Gamma posterior parameters
        posterior_shape = shape + np.sum(observations)
        posterior_rate = rate + len(observations)

        # Posterior mean is shape/rate
        new_lambda = posterior_shape / posterior_rate

        new_dist = PoissonDistribution(new_lambda)
        new_dist.n_observations = len(observations)
        return new_dist

    def credible_interval(self, alpha_level: float = 0.05) -> Tuple[int, int]:
        """
        Compute credible interval.

        Args:
            alpha_level: Significance level

        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = stats.poisson.ppf(alpha_level / 2, self.lambda_param)
        upper = stats.poisson.ppf(1 - alpha_level / 2, self.lambda_param)
        return int(lower), int(upper)


class NegativeBinomial:
    """
    Negative Binomial distribution for overdispersed count data.

    Used when count data shows more variance than expected from Poisson.
    Common for high-variance stats like points for star players.
    """

    def __init__(self, mean: float, dispersion: float):
        """
        Initialize Negative Binomial distribution.

        Args:
            mean: Expected value
            dispersion: Dispersion parameter (higher = more variance)
        """
        self.mean = mean
        self.dispersion = dispersion

        # Convert to scipy parameterization
        # scipy uses n (number of failures) and p (success probability)
        self.n = mean / (dispersion - 1) if dispersion > 1 else mean
        self.p = 1 / dispersion if dispersion > 0 else 0.5

    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability mass function."""
        return stats.nbinom.pmf(k, self.n, self.p)

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from distribution."""
        samples = np.random.negative_binomial(self.n, self.p, size)
        return samples if size > 1 else samples[0]

    def mean_value(self) -> float:
        """Expected value."""
        return self.mean

    def variance(self) -> float:
        """Variance (greater than mean due to overdispersion)."""
        return self.mean * self.dispersion

    def fit_from_data(self, data: np.ndarray) -> 'NegativeBinomial':
        """
        Fit distribution parameters from observed data.

        Args:
            data: Array of observed counts

        Returns:
            Fitted NegativeBinomial distribution
        """
        mean_obs = np.mean(data)
        var_obs = np.var(data)

        # Method of moments estimation
        if var_obs > mean_obs:
            dispersion = var_obs / mean_obs
        else:
            # If variance <= mean, use Poisson (dispersion = 1)
            dispersion = 1.0

        return NegativeBinomial(mean_obs, dispersion)

    def credible_interval(self, alpha_level: float = 0.05) -> Tuple[int, int]:
        """
        Compute credible interval.

        Args:
            alpha_level: Significance level

        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = stats.nbinom.ppf(alpha_level / 2, self.n, self.p)
        upper = stats.nbinom.ppf(1 - alpha_level / 2, self.n, self.p)
        return int(lower), int(upper)


class MultivariateNormalResiduals:
    """
    Multivariate Normal distribution for modeling correlated residuals.

    Used to capture correlations between different statistics
    (e.g., assists and turnovers tend to be correlated).
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Initialize Multivariate Normal distribution.

        Args:
            mean: Mean vector
            cov: Covariance matrix
        """
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.n_dims = len(mean)

        # Ensure covariance is positive semi-definite
        self._regularize_covariance()

    def _regularize_covariance(self, epsilon: float = 1e-6):
        """Regularize covariance matrix to ensure positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)

        # Replace negative eigenvalues with small positive value
        eigenvalues = np.maximum(eigenvalues, epsilon)

        # Reconstruct covariance
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def sample(self, size: int = 1) -> np.ndarray:
        """
        Sample from distribution.

        Args:
            size: Number of samples

        Returns:
            Array of shape (size, n_dims) or (n_dims,) if size=1
        """
        samples = np.random.multivariate_normal(self.mean, self.cov, size)
        return samples if size > 1 else samples[0]

    def fit_from_residuals(self, residuals: np.ndarray) -> 'MultivariateNormalResiduals':
        """
        Fit distribution from observed residuals.

        Args:
            residuals: Array of shape (n_samples, n_dims)

        Returns:
            Fitted distribution
        """
        mean = np.mean(residuals, axis=0)
        cov = np.cov(residuals, rowvar=False)
        return MultivariateNormalResiduals(mean, cov)

    def correlation_matrix(self) -> np.ndarray:
        """Get correlation matrix from covariance."""
        std_dev = np.sqrt(np.diag(self.cov))
        corr = self.cov / np.outer(std_dev, std_dev)
        return corr