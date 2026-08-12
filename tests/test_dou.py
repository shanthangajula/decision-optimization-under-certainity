"""Property tests for the marketplace, policies and estimators.

The tests that matter most here are the ones checking *statistical* claims
rather than plumbing: that IPW is unbiased under a known design, that AIPW
survives a deliberately wrong outcome model, and above all that the outcome
model never sees the future. That last one is the failure mode most likely
to silently invalidate every number in the repository, because a leaky
nuisance model produces results that look better, not worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from dou.estimators import (
    adaptively_weighted_aipw,
    aipw,
    aipw_scores,
    difference_in_means,
    effective_sample_fraction,
    ipw,
    sequential_outcome_predictions,
)
from dou.marketplace import Marketplace, MarketplaceConfig
from dou.policies import (
    UCB1,
    EpsilonGreedy,
    FixedAssignment,
    ThompsonSampling,
    build_policy,
)
from dou.simulate import run_once

# --------------------------------------------------------------- DGP tests


def test_true_ate_matches_sampled_potential_outcomes():
    """Ground truth from probabilities agrees with realised draws."""
    market = Marketplace()
    rng = np.random.default_rng(0)
    sessions = market.draw_sessions(400_000, rng)
    empirical = sessions.potential_outcomes.mean(axis=0)
    empirical_ate = empirical - empirical[0]
    assert np.allclose(empirical_ate, market.true_ate(), atol=0.005)


def test_control_arm_has_zero_effect_by_construction():
    assert market_ate()[0] == pytest.approx(0.0, abs=1e-12)


def market_ate():
    return Marketplace().true_ate(n_mc=200_000, seed=5)


def test_segment_effects_are_heterogeneous():
    """The design deliberately has effect modification; assert it is there."""
    seg = Marketplace().true_segment_ate(n_mc=200_000)
    spread = seg[:, 1:].max(axis=0) - seg[:, 1:].min(axis=0)
    assert np.all(spread > 0.02)


def test_best_marginal_arm_is_not_the_largest_base_effect():
    """Heterogeneity flips the ranking; a homogeneous DGP would be a weaker test bed."""
    market = Marketplace()
    assert market.best_arm != market.config.n_arms - 1


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        MarketplaceConfig(arm_base_effect=(0.1, 0.2, 0.3, 0.4))
    with pytest.raises(ValueError):
        MarketplaceConfig(segment_probs=(0.5, 0.4, 0.2))


# ------------------------------------------------------------ policy tests


@pytest.mark.parametrize("name", ["fixed", "epsilon_greedy", "thompson", "ucb1"])
def test_probabilities_are_valid_distributions(name):
    rng = np.random.default_rng(1)
    policy = build_policy(name, 4, rng)
    for t in range(50):
        probs = policy.action_probabilities(t)
        assert probs.shape == (4,)
        assert np.all(probs >= 0)
        assert probs.sum() == pytest.approx(1.0)
        arm, _ = policy.select(t)
        policy.update(arm, float(rng.random() < 0.3))


def test_floor_is_respected_even_under_extreme_posteriors():
    rng = np.random.default_rng(2)
    policy = ThompsonSampling(4, rng, floor=0.05)
    for _ in range(400):  # hammer arm 1 so its posterior dominates
        policy.update(1, 1.0)
        policy.update(0, 0.0)
    probs = policy.action_probabilities(400)
    assert probs.min() >= 0.05 - 1e-12


def test_ucb1_violates_positivity_and_says_so():
    rng = np.random.default_rng(3)
    policy = UCB1(4, rng)
    assert policy.supports_inference is False
    for t in range(30):
        arm, probs = policy.select(t)
        policy.update(arm, 1.0)
    assert probs.min() == 0.0


def test_fixed_assignment_is_not_adaptive():
    rng = np.random.default_rng(4)
    policy = FixedAssignment(4, rng)
    first = policy.action_probabilities(0).copy()
    for t in range(100):
        policy.update(t % 4, 1.0)
    assert np.allclose(first, policy.action_probabilities(100))


def test_fixed_assignment_rejects_zero_weights():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        FixedAssignment(3, rng, weights=np.array([0.0, 0.5, 0.5]))


def test_epsilon_greedy_tries_every_arm_before_exploiting():
    """Unplayed arms carry an optimistic mean of 1.0, so they are tried first."""
    rng = np.random.default_rng(6)
    policy = EpsilonGreedy(3, rng, epsilon=0.1, floor=0.0)
    for _ in range(200):
        policy.update(2, 1.0)
        policy.update(0, 0.0)
    # Arm 1 is still untouched, so optimism must keep it on top.
    assert int(np.argmax(policy.action_probabilities(200))) == 1

    for _ in range(200):  # now give arm 1 a poor track record
        policy.update(1, 0.0)
    assert int(np.argmax(policy.action_probabilities(400))) == 2


# --------------------------------------------------------- estimator tests


def test_outcome_model_never_sees_the_future():
    """The single most important correctness property in the repository.

    Predictions for the first block must not change when data *after* that
    block is altered. If they do, the nuisance model has leaked future
    information into the past, the AIPW scores stop being a martingale
    difference sequence, and every interval in the project is invalid.
    """
    rng = np.random.default_rng(7)
    n, d, k = 400, 6, 3
    features = rng.normal(size=(n, d))
    arms = rng.integers(0, k, size=n)
    rewards = (rng.random(n) < 0.4).astype(float)

    baseline = sequential_outcome_predictions(features, arms, rewards, k, refit_every=50)

    tampered = rewards.copy()
    tampered[200:] = 1.0 - tampered[200:]  # flip every outcome in the second half
    perturbed = sequential_outcome_predictions(features, arms, tampered, k, refit_every=50)

    assert np.allclose(baseline[:200], perturbed[:200])
    assert not np.allclose(baseline[200:], perturbed[200:])


def test_ipw_is_unbiased_under_fixed_assignment():
    """Known constant propensities, so IPW must hit the truth on average."""
    market = Marketplace()
    truth = market.true_ate()
    estimates = []
    for rep in range(120):
        log = run_once("fixed", 1500, seed=500 + rep)
        est = ipw(log.arms, log.rewards, log.propensities, arm=1)
        estimates.append(est.estimate)
    mean = float(np.mean(estimates))
    mcse = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    assert abs(mean - truth[1]) < 4 * mcse


def test_aipw_survives_a_deliberately_wrong_outcome_model():
    """Double robustness: with correct propensities, bad nuisances are tolerable.

    The outcome model is replaced with a constant 0.9, which is badly wrong
    for every arm. Because the propensities are known and correct, the
    inverse-propensity correction must still recover the truth.
    """
    market = Marketplace()
    truth = market.true_ate()
    estimates = []
    for rep in range(120):
        log = run_once("fixed", 1500, seed=900 + rep)
        junk = np.full((log.horizon, market.config.n_arms), 0.9)
        scores = aipw_scores(log.arms, log.rewards, log.propensities, junk)
        estimates.append(aipw(scores, log.propensities, arm=1).estimate)
    mean = float(np.mean(estimates))
    mcse = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    assert abs(mean - truth[1]) < 4 * mcse


def test_aw_aipw_equals_aipw_under_constant_propensities():
    """With non-adaptive assignment the stabilising weights are constant,
    so the weighted estimator must collapse exactly onto the unweighted one."""
    log = run_once("fixed", 1200, seed=42)
    preds = sequential_outcome_predictions(
        log.features, log.arms, log.rewards, 4, refit_every=50
    )
    scores = aipw_scores(log.arms, log.rewards, log.propensities, preds)
    plain = aipw(scores, log.propensities, arm=2)
    weighted = adaptively_weighted_aipw(scores, log.propensities, arm=2)
    assert weighted.estimate == pytest.approx(plain.estimate, rel=1e-9)
    assert weighted.std_error == pytest.approx(plain.std_error, rel=1e-6)


def test_aw_aipw_is_more_precise_than_aipw_under_adaptive_allocation():
    """The whole point of the weighting: tighter intervals when propensities move."""
    plain_widths, weighted_widths = [], []
    for rep in range(40):
        log = run_once("thompson", 2500, seed=300 + rep, policy_kwargs={"floor": 0.005})
        preds = sequential_outcome_predictions(
            log.features, log.arms, log.rewards, 4, refit_every=50
        )
        scores = aipw_scores(log.arms, log.rewards, log.propensities, preds)
        plain_widths.append(aipw(scores, log.propensities, arm=1).std_error)
        weighted_widths.append(
            adaptively_weighted_aipw(scores, log.propensities, arm=1).std_error
        )
    assert np.mean(weighted_widths) < np.mean(plain_widths)


def test_estimators_refuse_degenerate_propensities():
    log = run_once("ucb1", 600, seed=11)
    assert log.propensities.min() == 0.0
    for est in (
        ipw(log.arms, log.rewards, log.propensities, arm=1),
        aipw(np.zeros((log.horizon, 4)), log.propensities, arm=1),
        adaptively_weighted_aipw(np.zeros((log.horizon, 4)), log.propensities, arm=1),
    ):
        assert est.valid is False
        assert "positivity" in est.note


def test_difference_in_means_is_biased_under_adaptive_allocation():
    """Documents the failure the project exists to measure.

    Thompson sampling abandons the control arm, so the control mean is
    computed from a small, adversely selected sample. The contrast is
    therefore biased away from zero — here, upward.
    """
    market = Marketplace()
    truth = market.true_ate()[1]
    errors = []
    for rep in range(150):
        log = run_once("thompson", 3000, seed=700 + rep, policy_kwargs={"floor": 0.002})
        errors.append(difference_in_means(log.arms, log.rewards, 1).estimate - truth)
    mean_err = float(np.mean(errors))
    mcse = float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
    assert mean_err > 2 * mcse, "expected detectable upward bias"


def test_effective_sample_fraction_is_one_under_uniform_assignment():
    log = run_once("fixed", 800, seed=13)
    assert effective_sample_fraction(log.propensities, arm=1) == pytest.approx(1.0)


def test_effective_sample_fraction_drops_under_adaptivity():
    log = run_once("thompson", 3000, seed=14, policy_kwargs={"floor": 0.002})
    assert effective_sample_fraction(log.propensities, arm=0) < 0.9


def test_estimate_covers_truth_reasonably_often_under_fixed_assignment():
    market = Marketplace()
    truth = market.true_ate()
    hits = 0
    reps = 120
    for rep in range(reps):
        log = run_once("fixed", 2000, seed=2000 + rep)
        preds = sequential_outcome_predictions(
            log.features, log.arms, log.rewards, 4, refit_every=50
        )
        scores = aipw_scores(log.arms, log.rewards, log.propensities, preds)
        hits += adaptively_weighted_aipw(scores, log.propensities, arm=1).covers(truth[1])
    coverage = hits / reps
    assert 0.88 < coverage < 1.0
