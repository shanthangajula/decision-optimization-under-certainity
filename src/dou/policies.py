"""Allocation policies, each of which logs the propensity it acted under.

Every policy exposes ``action_probabilities`` *before* it draws an arm, and
the simulator logs that full probability vector. This is the single most
important design decision in the repository: causal estimation on adaptively
collected data is only possible when the assignment probability at each step
is known and bounded away from zero. A logging pipeline that records the
chosen arm but not the probability it was chosen with has thrown away the
information needed to debias the resulting dataset.

Two properties are tracked on every policy:

``is_adaptive``
    Whether the assignment probability at time ``t`` depends on outcomes
    observed before ``t``. Adaptive collection is what breaks the naive
    sample-mean estimator.

``supports_inference``
    Whether the policy satisfies positivity (every arm retains non-zero
    assignment probability at every step). UCB1 does not: it is a
    deterministic rule, so its propensities are degenerate 0/1 and no
    inverse-propensity estimator is defined. It is included precisely to
    make that failure visible rather than to hide it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """Base class for allocation rules over a fixed set of arms."""

    is_adaptive: bool = True
    supports_inference: bool = True
    name: str = "policy"

    def __init__(self, n_arms: int, rng: np.random.Generator) -> None:
        self.n_arms = n_arms
        self.rng = rng

    @abstractmethod
    def action_probabilities(self, t: int) -> np.ndarray:
        """Assignment probabilities at step ``t``, measurable wrt history."""

    def select(self, t: int) -> tuple[int, np.ndarray]:
        """Return ``(chosen_arm, probability_vector)``."""
        probs = self.action_probabilities(t)
        arm = int(self.rng.choice(self.n_arms, p=probs))
        return arm, probs

    def update(self, arm: int, reward: float) -> None:  # noqa: B027
        """Absorb the realised reward for the chosen arm.

        Deliberately concrete and empty rather than abstract: a
        non-adaptive policy such as :class:`FixedAssignment` has nothing to
        absorb, and forcing it to implement a no-op override would obscure
        that it ignores outcomes by design.
        """


class FixedAssignment(Policy):
    """Non-adaptive assignment: the classic randomised experiment.

    Propensities are constant and known, so every standard estimator is
    valid. This is the inferential baseline the adaptive policies are
    measured against.
    """

    is_adaptive = False
    name = "fixed"

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        weights: np.ndarray | None = None,
    ) -> None:
        super().__init__(n_arms, rng)
        if weights is None:
            weights = np.ones(n_arms) / n_arms
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_arms,):
            raise ValueError("weights must have shape (n_arms,)")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        if np.any(weights <= 0):
            raise ValueError("fixed assignment requires strictly positive weights")
        self.weights = weights

    def action_probabilities(self, t: int) -> np.ndarray:
        return self.weights


class EpsilonGreedy(Policy):
    """Explore uniformly with probability ``epsilon``, else exploit the mean.

    ``epsilon`` decays as ``epsilon_0 / (1 + decay * t)`` when ``decay > 0``.
    The floor keeps propensities bounded away from zero even as exploration
    anneals, which is what keeps inverse-propensity weights finite.
    """

    name = "epsilon_greedy"

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        epsilon: float = 0.15,
        decay: float = 0.0,
        floor: float = 0.02,
    ) -> None:
        super().__init__(n_arms, rng)
        self.epsilon0 = epsilon
        self.decay = decay
        self.floor = floor
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def _means(self) -> np.ndarray:
        # Unseen arms get an optimistic 1.0 so each is tried early.
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 1.0)
        return means

    def action_probabilities(self, t: int) -> np.ndarray:
        eps = self.epsilon0 / (1.0 + self.decay * t)
        probs = np.full(self.n_arms, eps / self.n_arms)
        best = int(np.argmax(self._means()))
        probs[best] += 1.0 - eps
        return _apply_floor(probs, self.floor)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class ThompsonSampling(Policy):
    """Beta-Bernoulli Thompson sampling with explicit propensity computation.

    Thompson sampling has no closed-form selection probability, so it is
    computed by posterior Monte Carlo: draw ``n_posterior_samples`` vectors
    from the arm posteriors and record how often each arm is the argmax.
    The arm is then drawn from *that* estimated distribution rather than by
    the usual draw-once-and-argmax shortcut.

    The two procedures are equivalent in distribution, but only this one
    yields an exactly known propensity. Sampling from the same vector that
    was used to log the probability is what makes the logged propensity the
    true conditional assignment probability.
    """

    name = "thompson"

    def __init__(
        self,
        n_arms: int,
        rng: np.random.Generator,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_posterior_samples: int = 200,
        floor: float = 0.01,
    ) -> None:
        super().__init__(n_arms, rng)
        self.alpha = np.full(n_arms, float(prior_alpha))
        self.beta = np.full(n_arms, float(prior_beta))
        self.n_posterior_samples = n_posterior_samples
        self.floor = floor

    def action_probabilities(self, t: int) -> np.ndarray:
        draws = self.rng.beta(
            self.alpha[None, :],
            self.beta[None, :],
            size=(self.n_posterior_samples, self.n_arms),
        )
        winners = np.argmax(draws, axis=1)
        probs = np.bincount(winners, minlength=self.n_arms) / self.n_posterior_samples
        return _apply_floor(probs, self.floor)

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


class UCB1(Policy):
    """Deterministic upper-confidence-bound allocation.

    Included as a negative control. UCB1 maximises reward well but assigns
    probability 1 to a single arm at every step, violating positivity. Any
    inverse-propensity estimator applied to UCB1 logs is undefined, and the
    outcome-regression path has no randomisation to lean on. The repository
    flags this rather than silently producing a number.
    """

    name = "ucb1"
    supports_inference = False

    def __init__(
        self, n_arms: int, rng: np.random.Generator, c: float = 1.0
    ) -> None:
        super().__init__(n_arms, rng)
        self.c = c
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def action_probabilities(self, t: int) -> np.ndarray:
        probs = np.zeros(self.n_arms)
        unseen = np.flatnonzero(self.counts == 0)
        if unseen.size > 0:
            probs[unseen[0]] = 1.0
            return probs
        means = self.sums / self.counts
        bonus = self.c * np.sqrt(2.0 * np.log(max(t, 2)) / self.counts)
        probs[int(np.argmax(means + bonus))] = 1.0
        return probs

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


def _apply_floor(probs: np.ndarray, floor: float) -> np.ndarray:
    """Mix the distribution toward uniform so no arm drops below ``floor``.

    Implemented as an explicit mixture with the uniform distribution rather
    than clip-and-renormalise, because the mixture keeps the result a valid
    probability vector in one step and makes the exploration cost legible:
    a floor of ``f`` with ``K`` arms spends ``f * K`` of the assignment mass
    on guaranteed exploration.
    """
    if floor <= 0:
        return probs
    k = probs.shape[0]
    if floor * k >= 1.0:
        return np.ones(k) / k
    mix = floor * k
    return (1.0 - mix) * probs + mix / k


POLICY_REGISTRY: dict[str, type[Policy]] = {
    "fixed": FixedAssignment,
    "epsilon_greedy": EpsilonGreedy,
    "thompson": ThompsonSampling,
    "ucb1": UCB1,
}


def build_policy(name: str, n_arms: int, rng: np.random.Generator, **kwargs) -> Policy:
    if name not in POLICY_REGISTRY:
        raise KeyError(f"unknown policy {name!r}; choose from {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name](n_arms, rng, **kwargs)
