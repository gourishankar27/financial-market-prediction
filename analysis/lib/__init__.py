"""
Analysis library for TFT experiment comparison.

Usage:
    from lib.loaders import load_experiments
    from lib.ablation import ablate, compare_groups, correlation_matrix
"""

from .loaders import load_experiments, get_regime_experiments, get_config_groups
from .ablation import (
    ablate, 
    compare_groups, 
    correlation_matrix,
    rank_experiments,
    DEFAULT_PERFORMANCE_METRICS,
    DEFAULT_BEHAVIOR_METRICS
)

__all__ = [
    'load_experiments',
    'get_regime_experiments', 
    'get_config_groups',
    'ablate',
    'compare_groups',
    'correlation_matrix',
    'rank_experiments',
    'DEFAULT_PERFORMANCE_METRICS',
    'DEFAULT_BEHAVIOR_METRICS',
]