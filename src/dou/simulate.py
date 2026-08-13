"""Run allocation policies against the synthetic marketplace and log results.

A single replication produces a ``RunLog``: the observed data an analyst
would actually have (context, chosen arm, logged propensity, realised
outcome) plus oracle quantities the estimators never see, used only to score
the estimators after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators import (
    adaptively_weighted_aipw,
    aipw,
    aipw_scores,
    difference_in_means,
    effective_sample_fraction,
    ipw,
    sequential_outcome_predictions,
)
from .marketplace import Marketplace, MarketplaceConfig
from .policies import build_policy


@dataclass
class RunLog:
    """One replication of an allocation experiment."""

    features: np.ndarray  # (T, d)
    segment: np.ndarray  # (T,)
    arms: np.ndarray  # (T,) chosen arm
    rewards: np.ndarray  # (T,) realised outcome
    propensities: np.ndarray  # (T, K) logged assignment probabilities
    expected_rewards: np.ndarray  # (T, K) oracle P(Y=1|X,a) — never used by estimators
    policy_name: str
    supports_inference: bool

    @property
    def horizon(self) -> int:
        return self.arms.shape[0]

    def cumulative_reward(self) -> float:
        return float(self.rewards.sum())

    def expected_cumulative_reward(self) -> float:
        """Lower-variance reward measure using oracle probabilities."""
        return float(self.expected_rewards[np.arange(self.horizon), self.arms].sum())

    def oracle_regret(self) -> float:
        """Regret against the single best arm in hindsight, in expectation."""
        best_per_step = self.expected_rewards.max(axis=1)
        realised = self.expected_rewards[np.arange(self.horizon), self.arms]
        return float((best_per_step - realised).sum())

    def arm_share(self, n_arms: int) -> np.ndarray:
        return np.bincount(self.arms, minlength=n_arms) / self.horizon


def run_once(
    policy_name: str,
    horizon: int,
    seed: int,
    config: MarketplaceConfig | None = None,
    policy_kwargs: dict | None = None,
) -> RunLog:
    """Run one replication of ``policy_name`` over ``horizon`` sessions."""
    market = Marketplace(config)
    rng = np.random.default_rng(seed)
    sessions = market.draw_sessions(horizon, rng)

    policy = build_policy(
        policy_name, market.config.n_arms, rng, **(policy_kwargs or {})
    )

    arms = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon)
    propensities = np.zeros((horizon, market.config.n_arms))

    for t in range(horizon):
        arm, probs = policy.select(t)
        reward = float(sessions.potential_outcomes[t, arm])
        policy.update(arm, reward)
        arms[t] = arm
        rewards[t] = reward
        propensities[t] = probs

    return RunLog(
        features=sessions.features,
        segment=sessions.segment,
        arms=arms,
        rewards=rewards,
        propensities=propensities,
        expected_rewards=sessions.outcome_probs,
        policy_name=policy_name,
        supports_inference=policy.supports_inference,
    )


def estimate_from_log(
    log: RunLog,
    n_arms: int,
    control: int = 0,
    alpha: float = 0.05,
    refit_every: int = 50,
    ipw_clip: float = 0.0,
) -> list[dict]:
    """Apply every estimator to one replication; return tidy records."""
    preds = sequential_outcome_predictions(
        log.features, log.arms, log.rewards, n_arms, refit_every=refit_every
    )
    scores = aipw_scores(log.arms, log.rewards, log.propensities, preds)

    records: list[dict] = []
    for arm in range(n_arms):
        if arm == control:
            continue
        estimates = [
            difference_in_means(log.arms, log.rewards, arm, control, alpha),
            ipw(log.arms, log.rewards, log.propensities, arm, control, alpha, ipw_clip),
            aipw(scores, log.propensities, arm, control, alpha),
            adaptively_weighted_aipw(scores, log.propensities, arm, control, alpha),
        ]
        ess = effective_sample_fraction(log.propensities, arm)
        for est in estimates:
            records.append(
                {
                    "policy": log.policy_name,
                    "estimator": est.estimator,
                    "arm": arm,
                    "estimate": est.estimate,
                    "std_error": est.std_error,
                    "ci_low": est.ci_low,
                    "ci_high": est.ci_high,
                    "valid": est.valid,
                    "note": est.note,
                    "effective_sample_fraction": ess,
                }
            )
    return records


def estimate_segment_effects(
    log: RunLog,
    n_arms: int,
    n_segments: int = 3,
    control: int = 0,
    alpha: float = 0.05,
    refit_every: int = 50,
) -> list[dict]:
    """Group ATE by buyer segment, via the adaptively weighted estimator."""
    preds = sequential_outcome_predictions(
        log.features, log.arms, log.rewards, n_arms, refit_every=refit_every
    )
    scores = aipw_scores(log.arms, log.rewards, log.propensities, preds)

    records: list[dict] = []
    for seg in range(n_segments):
        mask = log.segment == seg
        for arm in range(n_arms):
            if arm == control:
                continue
            est = adaptively_weighted_aipw(
                scores, log.propensities, arm, control, alpha, subset=mask
            )
            records.append(
                {
                    "policy": log.policy_name,
                    "segment": seg,
                    "arm": arm,
                    "estimate": est.estimate,
                    "ci_low": est.ci_low,
                    "ci_high": est.ci_high,
                    "valid": est.valid,
                }
            )
    return records


def run_replications(
    policy_name: str,
    horizon: int,
    n_reps: int,
    base_seed: int = 7,
    config: MarketplaceConfig | None = None,
    policy_kwargs: dict | None = None,
    alpha: float = 0.05,
    refit_every: int = 50,
    ipw_clip: float = 0.0,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run many replications; return ``(effect_records, run_summaries)``.

    Replications share the marketplace but use disjoint seeds, so the spread
    across replications is exactly the sampling distribution of each
    estimator under the design.
    """
    config = config or MarketplaceConfig()
    n_arms = config.n_arms

    effect_rows: list[dict] = []
    run_rows: list[dict] = []

    for rep in range(n_reps):
        log = run_once(
            policy_name,
            horizon,
            seed=base_seed + 1000 * rep,
            config=config,
            policy_kwargs=policy_kwargs,
        )
        if log.supports_inference:
            rows = estimate_from_log(
                log, n_arms, alpha=alpha, refit_every=refit_every, ipw_clip=ipw_clip
            )
        else:
            rows = [
                {
                    "policy": log.policy_name,
                    "estimator": name,
                    "arm": arm,
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "valid": False,
                    "note": "positivity violated by deterministic policy",
                    "effective_sample_fraction": np.nan,
                }
                for arm in range(1, n_arms)
                for name in ("difference_in_means", "ipw", "aipw", "aw_aipw")
            ]
            # The naive estimator is still computable — and still reported,
            # because a practitioner would compute it without noticing.
            for row in rows:
                if row["estimator"] == "difference_in_means":
                    est = difference_in_means(
                        log.arms, log.rewards, row["arm"], 0, alpha
                    )
                    row.update(
                        estimate=est.estimate,
                        std_error=est.std_error,
                        ci_low=est.ci_low,
                        ci_high=est.ci_high,
                        valid=est.valid,
                        note="computable but design has no valid propensities",
                    )

        for row in rows:
            row["rep"] = rep
        effect_rows.extend(rows)

        shares = log.arm_share(n_arms)
        run_rows.append(
            {
                "policy": policy_name,
                "rep": rep,
                "cumulative_reward": log.cumulative_reward(),
                "expected_cumulative_reward": log.expected_cumulative_reward(),
                "oracle_regret": log.oracle_regret(),
                **{f"share_arm{a}": shares[a] for a in range(n_arms)},
            }
        )

        if progress and (rep + 1) % 25 == 0:
            print(f"  {policy_name}: {rep + 1}/{n_reps} replications", flush=True)

    return pd.DataFrame(effect_rows), pd.DataFrame(run_rows)
