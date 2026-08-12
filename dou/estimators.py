"""Treatment-effect estimators for adaptively collected data.

Why this module exists
----------------------
When a bandit allocates traffic, the resulting log is not an i.i.d. sample.
The assignment probability at step ``t`` depends on outcomes realised before
``t``, which means arms that got lucky early are sampled more often
afterwards. The sample mean of a bandit-collected arm is therefore biased,
and the direction is predictable: it is biased *downward* for arms the
algorithm favours, because an arm is preferentially sampled exactly when its
running estimate is high, and subsequent draws regress toward the truth.

Four estimators are implemented, in increasing order of what they repair:

1. ``difference_in_means`` — the naive estimator. Ignores the problem.
2. ``ipw`` — inverse propensity weighting. Unbiased by a martingale
   argument, but its variance explodes as propensities anneal toward zero,
   and its normal-approximation intervals under-cover.
3. ``aipw`` — augmented IPW. Adds an outcome regression fit only on past
   data, cutting variance. Still relies on a central limit theorem that the
   adaptive design does not deliver.
4. ``adaptively_weighted_aipw`` — reweights each score by a
   variance-stabilising factor so the studentised statistic is asymptotically
   normal again, restoring nominal coverage. This follows the construction
   of Hadad, Hirshberg, Zhan, Wager and Athey (PNAS, 2021).

The AIPW score for arm ``a`` at time ``t`` is::

    G_t(a) = m_a(X_t) + 1{A_t = a} / pi_t(a) * (Y_t - m_a(X_t))

where ``m_a`` is fit strictly on data prior to ``t`` so the score remains a
martingale difference sequence with respect to the observation filtration.
Fitting the nuisance model on the full sample would break exactly the
property the estimator depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class EffectEstimate:
    """A point estimate with an interval and the diagnostics behind it."""

    estimator: str
    arm: int
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n_effective: float
    valid: bool = True
    note: str = ""

    def covers(self, truth: float) -> bool:
        return bool(self.valid and self.ci_low <= truth <= self.ci_high)


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _finalise(
    estimator: str,
    arm: int,
    estimate: float,
    variance: float,
    n_effective: float,
    alpha: float,
    note: str = "",
) -> EffectEstimate:
    variance = max(float(variance), 0.0)
    se = float(np.sqrt(variance))
    z = _z(alpha)
    return EffectEstimate(
        estimator=estimator,
        arm=arm,
        estimate=float(estimate),
        std_error=se,
        ci_low=float(estimate - z * se),
        ci_high=float(estimate + z * se),
        n_effective=float(n_effective),
        valid=np.isfinite(estimate) and np.isfinite(se) and se > 0,
        note=note,
    )


# --------------------------------------------------------------- nuisances


def sequential_outcome_predictions(
    features: np.ndarray,
    arms: np.ndarray,
    rewards: np.ndarray,
    n_arms: int,
    refit_every: int = 50,
    ridge: float = 1.0,
) -> np.ndarray:
    """Per-arm outcome predictions ``m_a(X_t)`` fit only on the past.

    A ridge regression per arm is refit at block boundaries using all data
    observed strictly before the block starts, and used to predict every
    step inside the block. Refitting in blocks rather than at every step is
    a compute concession, not a statistical one: predictions at step ``t``
    remain measurable with respect to the history before ``t``, which is
    the condition the martingale argument actually requires.

    Returns an array of shape ``(T, n_arms)``.
    """
    t_total, d = features.shape
    preds = np.full((t_total, n_arms), 0.5)

    for start in range(0, t_total, refit_every):
        stop = min(start + refit_every, t_total)
        if start == 0:
            continue  # nothing observed yet; keep the 0.5 prior

        past_x = features[:start]
        past_a = arms[:start]
        past_y = rewards[:start]
        fallback = float(past_y.mean()) if past_y.size else 0.5

        for a in range(n_arms):
            mask = past_a == a
            if mask.sum() < d + 1:
                preds[start:stop, a] = fallback
                continue
            xa = past_x[mask]
            ya = past_y[mask]
            gram = xa.T @ xa + ridge * np.eye(d)
            try:
                beta = np.linalg.solve(gram, xa.T @ ya)
            except np.linalg.LinAlgError:
                preds[start:stop, a] = fallback
                continue
            preds[start:stop, a] = np.clip(features[start:stop] @ beta, 0.0, 1.0)

    return preds


def aipw_scores(
    arms: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    outcome_preds: np.ndarray,
) -> np.ndarray:
    """AIPW score matrix of shape ``(T, n_arms)``."""
    t_total, n_arms = propensities.shape
    indicator = np.zeros((t_total, n_arms))
    indicator[np.arange(t_total), arms] = 1.0
    residual = rewards[:, None] - outcome_preds
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(
            indicator > 0, indicator / np.maximum(propensities, 1e-12) * residual, 0.0
        )
    return outcome_preds + correction


# -------------------------------------------------------------- estimators


def difference_in_means(
    arms: np.ndarray,
    rewards: np.ndarray,
    arm: int,
    control: int = 0,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Naive contrast of sample means. Biased under adaptive collection."""
    treat = rewards[arms == arm]
    ctrl = rewards[arms == control]
    if treat.size < 2 or ctrl.size < 2:
        return EffectEstimate(
            "difference_in_means", arm, np.nan, np.nan, np.nan, np.nan, 0.0,
            valid=False, note="insufficient samples in one cell",
        )
    est = treat.mean() - ctrl.mean()
    var = treat.var(ddof=1) / treat.size + ctrl.var(ddof=1) / ctrl.size
    return _finalise(
        "difference_in_means", arm, est, var, treat.size + ctrl.size, alpha
    )


def ipw(
    arms: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    arm: int,
    control: int = 0,
    alpha: float = 0.05,
    clip: float = 0.0,
) -> EffectEstimate:
    """Horvitz-Thompson contrast.

    Unbiased in expectation but high variance. ``clip`` floors the
    propensities before inversion, which tames the variance at the cost of
    reintroducing bias — the trade is exposed as a knob rather than buried.
    """
    if _degenerate(propensities):
        return _positivity_failure("ipw", arm)

    pi = propensities if clip <= 0 else np.maximum(propensities, clip)
    t_total = rewards.shape[0]
    score = np.zeros(t_total)
    treat_mask = arms == arm
    ctrl_mask = arms == control
    score[treat_mask] = rewards[treat_mask] / pi[treat_mask, arm]
    score[ctrl_mask] = -rewards[ctrl_mask] / pi[ctrl_mask, control]

    est = score.mean()
    var = score.var(ddof=1) / t_total
    return _finalise("ipw", arm, est, var, t_total, alpha)


def aipw(
    scores: np.ndarray,
    propensities: np.ndarray,
    arm: int,
    control: int = 0,
    alpha: float = 0.05,
    subset: np.ndarray | None = None,
) -> EffectEstimate:
    """Unweighted augmented IPW contrast.

    Doubly robust and lower variance than plain IPW, but its interval still
    leans on a central limit theorem that adaptive sampling does not supply.
    """
    if _degenerate(propensities):
        return _positivity_failure("aipw", arm)

    gamma = scores[:, arm] - scores[:, control]
    if subset is not None:
        gamma = gamma[subset]
    n = gamma.shape[0]
    if n < 2:
        return EffectEstimate(
            "aipw", arm, np.nan, np.nan, np.nan, np.nan, 0.0,
            valid=False, note="empty subset",
        )
    est = gamma.mean()
    var = gamma.var(ddof=1) / n
    return _finalise("aipw", arm, est, var, n, alpha)


def adaptively_weighted_aipw(
    scores: np.ndarray,
    propensities: np.ndarray,
    arm: int,
    control: int = 0,
    alpha: float = 0.05,
    subset: np.ndarray | None = None,
) -> EffectEstimate:
    """Variance-stabilised AIPW with asymptotically valid intervals.

    Each score is weighted by ``h_t``, chosen so that ``h_t`` times the
    conditional standard deviation of the score is roughly constant across
    the horizon. The conditional variance of the contrast score scales as
    ``1/pi_t(a) + 1/pi_t(0)``, so::

        h_t = (1 / pi_t(a) + 1 / pi_t(0)) ** -0.5

    which reduces to ``sqrt(pi_t(a))`` in the single-arm case — the constant
    allocation rate weights of Hadad et al. (2021).

    The intuition: late observations from a heavily exploited arm are
    individually precise but arrive under a propensity that has drifted, and
    early observations are noisy. Equalising their contributions is what
    restores a stable limiting distribution, at the price of a modest
    increase in variance relative to the unweighted estimator.
    """
    if _degenerate(propensities):
        return _positivity_failure("aw_aipw", arm)

    pi_a = np.maximum(propensities[:, arm], 1e-12)
    pi_c = np.maximum(propensities[:, control], 1e-12)
    h = (1.0 / pi_a + 1.0 / pi_c) ** -0.5
    gamma = scores[:, arm] - scores[:, control]

    if subset is not None:
        h = h[subset]
        gamma = gamma[subset]
    if h.shape[0] < 2 or h.sum() <= 0:
        return EffectEstimate(
            "aw_aipw", arm, np.nan, np.nan, np.nan, np.nan, 0.0,
            valid=False, note="empty subset",
        )

    n = h.shape[0]
    weight_sum = h.sum()
    est = float(h @ gamma / weight_sum)

    # The asymptotic variance is sum(h^2 (G - est)^2) / sum(h)^2. The
    # n / (n - 1) factor is the usual finite-sample correction; it also makes
    # this estimator collapse *exactly* onto the unweighted AIPW variance when
    # the weights are constant, which is asserted in the test suite.
    var = float((h**2 @ (gamma - est) ** 2) / weight_sum**2) * n / (n - 1)
    n_eff = float(weight_sum**2 / (h**2).sum())
    return _finalise("aw_aipw", arm, est, var, n_eff, alpha)


# ------------------------------------------------------------- diagnostics


def _degenerate(propensities: np.ndarray, tol: float = 1e-9) -> bool:
    """True when any step gave some arm essentially zero probability."""
    return bool(np.any(propensities.min(axis=1) < tol))


def _positivity_failure(estimator: str, arm: int) -> EffectEstimate:
    return EffectEstimate(
        estimator=estimator,
        arm=arm,
        estimate=np.nan,
        std_error=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        n_effective=0.0,
        valid=False,
        note="positivity violated: some arm had zero assignment probability",
    )


def effective_sample_fraction(propensities: np.ndarray, arm: int) -> float:
    """Kish effective sample size for an arm, as a fraction of the horizon.

    A useful companion to the point estimate: it says how much of the
    nominal sample the inverse-propensity weights actually bought.
    """
    w = 1.0 / np.maximum(propensities[:, arm], 1e-12)
    return float(w.sum() ** 2 / (w.shape[0] * (w**2).sum()))
