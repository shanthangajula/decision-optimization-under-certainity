"""Decision Optimization Under Uncertainty.

Adaptive allocation buys reward. It also destroys the sampling assumptions
that ordinary treatment-effect estimators rely on. This package quantifies
both sides of that trade on a synthetic marketplace where the ground truth
is known by construction.
"""

from .estimators import (
    EffectEstimate,
    adaptively_weighted_aipw,
    aipw,
    aipw_scores,
    difference_in_means,
    ipw,
    sequential_outcome_predictions,
)
from .marketplace import Marketplace, MarketplaceConfig, Sessions
from .policies import (
    UCB1,
    EpsilonGreedy,
    FixedAssignment,
    Policy,
    ThompsonSampling,
    build_policy,
)
from .simulate import RunLog, estimate_from_log, run_once, run_replications

__version__ = "0.1.0"

__all__ = [
    "EffectEstimate",
    "EpsilonGreedy",
    "FixedAssignment",
    "Marketplace",
    "MarketplaceConfig",
    "Policy",
    "RunLog",
    "Sessions",
    "ThompsonSampling",
    "UCB1",
    "adaptively_weighted_aipw",
    "aipw",
    "aipw_scores",
    "build_policy",
    "difference_in_means",
    "estimate_from_log",
    "ipw",
    "run_once",
    "run_replications",
    "sequential_outcome_predictions",
]
