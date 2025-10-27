"""
Fantasy Basketball Modeling Package

Core statistical models for player performance projection.
"""

from .empirical_bayes import EmpiricalBayes
from .distributions import BetaBinomial, PoissonDistribution, NegativeBinomial
from .bayesian_model import BayesianPlayerModel
from .correlation_model import CorrelationModel

__all__ = [
    'EmpiricalBayes',
    'BetaBinomial',
    'PoissonDistribution',
    'NegativeBinomial',
    'BayesianPlayerModel',
    'CorrelationModel'
]