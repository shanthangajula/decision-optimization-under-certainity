"""Synthetic two-sided marketplace with known ground-truth treatment effects.

The point of a synthetic DGP is that the estimand is known by construction.
That is what makes it possible to measure *estimator bias* and *confidence
interval coverage* rather than just reporting a point estimate and hoping.

Design
------
Each timestep is one session (an impression on the demand side of the
marketplace). A session carries a context ``X``; the platform must choose one
of ``K`` allocation policies (arms); a binary conversion outcome is realised.

Outcome model (logit link on conversion probability)::

    logit P(Y=1 | X, A=a) = b0 + f(X) + tau_a(X)

with ``tau_0(X) == 0`` so arm 0 is the control. Treatment effects are
*heterogeneous*: ``tau_a(X)`` depends on the buyer segment, which is what
gives the regression-adjusted estimators (AIPW) something to exploit.

All potential outcomes ``Y(0), ..., Y(K-1)`` are drawn for every session, so
the simulator can report oracle regret alongside the observed data. The
estimators never see the counterfactuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Buyer segments: 0 = new, 1 = casual, 2 = power buyer.
SEGMENT_NAMES = ("new", "casual", "power")
DEVICE_NAMES = ("mobile", "desktop")

# Arm 0 is always the incumbent / control ranking.
ARM_NAMES = (
    "control_ranking",
    "boost_new_sellers",
    "promo_discount",
    "aggressive_promo",
)


@dataclass
class MarketplaceConfig:
    """Parameters of the synthetic marketplace."""

    n_arms: int = 4
    segment_probs: tuple[float, ...] = (0.45, 0.35, 0.20)
    desktop_prob: float = 0.40

    # Baseline conversion propensity on the logit scale.
    intercept: float = -1.35

    # Context main effects.
    segment_coef: tuple[float, ...] = (-0.30, 0.15, 0.65)
    desktop_coef: float = 0.22
    price_sensitivity_coef: float = -0.55

    # Homogeneous component of each arm's effect (arm 0 pinned to zero).
    arm_base_effect: tuple[float, ...] = (0.0, 0.50, 0.80, 1.00)

    # Effect modification by segment: added effect per unit of segment index.
    # Positive means the arm works better for higher-value segments.
    arm_segment_interaction: tuple[float, ...] = (0.0, 0.35, -0.20, -0.55)

    # Optional non-stationarity: linear drift in the intercept across the
    # horizon. Zero by default; used by the drift experiment.
    intercept_drift: float = 0.0

    def __post_init__(self) -> None:
        if len(self.arm_base_effect) != self.n_arms:
            raise ValueError("arm_base_effect must have length n_arms")
        if len(self.arm_segment_interaction) != self.n_arms:
            raise ValueError("arm_segment_interaction must have length n_arms")
        if self.arm_base_effect[0] != 0.0:
            raise ValueError("arm 0 is the control; its base effect must be 0")
        if self.arm_segment_interaction[0] != 0.0:
            raise ValueError("arm 0 is the control; its interaction must be 0")
        if not np.isclose(sum(self.segment_probs), 1.0):
            raise ValueError("segment_probs must sum to 1")


@dataclass
class Sessions:
    """A batch of marketplace sessions with all potential outcomes drawn."""

    segment: np.ndarray  # (T,) int
    desktop: np.ndarray  # (T,) int
    price_sensitivity: np.ndarray  # (T,) float
    features: np.ndarray  # (T, d) float design matrix
    outcome_probs: np.ndarray  # (T, K) P(Y=1 | X, a)
    potential_outcomes: np.ndarray  # (T, K) realised Y(a) for every a
    feature_names: tuple[str, ...] = field(default=())

    def __len__(self) -> int:
        return self.segment.shape[0]


FEATURE_NAMES = (
    "intercept",
    "seg_casual",
    "seg_power",
    "desktop",
    "price_sensitivity",
    "segment_index",
)


def _design_matrix(
    segment: np.ndarray, desktop: np.ndarray, price_sensitivity: np.ndarray
) -> np.ndarray:
    """Build the design matrix handed to the outcome models.

    Deliberately includes the raw ``segment_index`` alongside the one-hot
    columns: the estimators are given a *correctly specified enough* basis so
    that any residual bias is attributable to the adaptive sampling rather
    than to a misspecified nuisance model.
    """
    n = segment.shape[0]
    return np.column_stack(
        [
            np.ones(n),
            (segment == 1).astype(float),
            (segment == 2).astype(float),
            desktop.astype(float),
            price_sensitivity,
            segment.astype(float),
        ]
    )


class Marketplace:
    """Generates sessions and, on request, the exact ground-truth estimands."""

    def __init__(self, config: MarketplaceConfig | None = None) -> None:
        self.config = config or MarketplaceConfig()

    # ---------------------------------------------------------------- draws

    def draw_sessions(self, n: int, rng: np.random.Generator) -> Sessions:
        """Draw ``n`` sessions, including every counterfactual outcome."""
        cfg = self.config

        segment = rng.choice(len(cfg.segment_probs), size=n, p=list(cfg.segment_probs))
        desktop = (rng.random(n) < cfg.desktop_prob).astype(int)
        price_sensitivity = rng.normal(0.0, 1.0, size=n)

        features = _design_matrix(segment, desktop, price_sensitivity)

        drift = np.zeros(n)
        if cfg.intercept_drift != 0.0 and n > 1:
            drift = np.linspace(0.0, cfg.intercept_drift, n)

        seg_coef = np.asarray(cfg.segment_coef)
        base = (
            cfg.intercept
            + drift
            + seg_coef[segment]
            + cfg.desktop_coef * desktop
            + cfg.price_sensitivity_coef * price_sensitivity
        )

        tau = self._tau(segment)  # (n, K)
        logits = base[:, None] + tau
        probs = _sigmoid(logits)

        potential = (rng.random((n, cfg.n_arms)) < probs).astype(int)

        return Sessions(
            segment=segment,
            desktop=desktop,
            price_sensitivity=price_sensitivity,
            features=features,
            outcome_probs=probs,
            potential_outcomes=potential,
            feature_names=FEATURE_NAMES,
        )

    def _tau(self, segment: np.ndarray) -> np.ndarray:
        cfg = self.config
        base = np.asarray(cfg.arm_base_effect)[None, :]
        inter = np.asarray(cfg.arm_segment_interaction)[None, :]
        return base + inter * segment[:, None].astype(float)

    # ------------------------------------------------------------ estimands

    def true_arm_values(
        self, n_mc: int = 4_000_000, seed: int = 20240101
    ) -> np.ndarray:
        """``E[Y(a)]`` for every arm, by high-precision Monte Carlo.

        Uses the outcome *probabilities* rather than sampled outcomes, so the
        only error is from integrating over the context distribution. With
        4e6 draws the Monte Carlo standard error is on the order of 2e-4,
        an order of magnitude below the estimator differences we care about.
        """
        rng = np.random.default_rng(seed)
        cfg = self.config

        # Drift is a within-run effect; the population estimand is defined at
        # the mid-point of the horizon so it stays well defined either way.
        segment = rng.choice(len(cfg.segment_probs), size=n_mc, p=list(cfg.segment_probs))
        desktop = (rng.random(n_mc) < cfg.desktop_prob).astype(int)
        price_sensitivity = rng.normal(0.0, 1.0, size=n_mc)

        seg_coef = np.asarray(cfg.segment_coef)
        base = (
            cfg.intercept
            + 0.5 * cfg.intercept_drift
            + seg_coef[segment]
            + cfg.desktop_coef * desktop
            + cfg.price_sensitivity_coef * price_sensitivity
        )
        probs = _sigmoid(base[:, None] + self._tau(segment))
        return probs.mean(axis=0)

    def true_ate(self, n_mc: int = 4_000_000, seed: int = 20240101) -> np.ndarray:
        """``E[Y(a) - Y(0)]`` for every arm (entry 0 is identically zero)."""
        values = self.true_arm_values(n_mc=n_mc, seed=seed)
        return values - values[0]

    def true_segment_ate(
        self, n_mc: int = 1_000_000, seed: int = 20240102
    ) -> np.ndarray:
        """Group ATE by buyer segment: shape ``(n_segments, n_arms)``."""
        rng = np.random.default_rng(seed)
        cfg = self.config
        n_seg = len(cfg.segment_probs)
        out = np.zeros((n_seg, cfg.n_arms))

        for s in range(n_seg):
            desktop = (rng.random(n_mc) < cfg.desktop_prob).astype(int)
            price_sensitivity = rng.normal(0.0, 1.0, size=n_mc)
            segment = np.full(n_mc, s)
            base = (
                cfg.intercept
                + 0.5 * cfg.intercept_drift
                + np.asarray(cfg.segment_coef)[segment]
                + cfg.desktop_coef * desktop
                + cfg.price_sensitivity_coef * price_sensitivity
            )
            probs = _sigmoid(base[:, None] + self._tau(segment))
            values = probs.mean(axis=0)
            out[s] = values - values[0]
        return out

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.true_arm_values()))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
