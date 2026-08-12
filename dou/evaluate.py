"""Score estimators against known ground truth across Monte Carlo replications.

Point estimates are not the interesting output here. What separates a usable
estimator from an unusable one on adaptively collected data is whether its
*intervals* cover at the nominal rate. An estimator can be nearly unbiased
and still be worthless if its 95% intervals only contain the truth 70% of
the time, because every downstream decision built on it will be overconfident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _nan_safe_rmse(errors: pd.Series) -> float:
    """RMSE that returns NaN rather than warning when nothing is valid.

    An estimator can be invalid for every replication — UCB1 has no valid
    propensities at all — and that should read as a blank cell, not as a
    numerical warning.
    """
    values = errors.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def summarise_effects(
    effects: pd.DataFrame, true_ate: np.ndarray, alpha: float = 0.05
) -> pd.DataFrame:
    """Collapse per-replication estimates into estimator quality metrics.

    Returns one row per (policy, estimator, arm) with bias, RMSE, empirical
    coverage of the nominal ``1 - alpha`` interval, and mean interval width.
    """
    df = effects.copy()
    df["truth"] = df["arm"].map(lambda a: float(true_ate[a]))

    valid = df["valid"].fillna(False)
    df["error"] = np.where(valid, df["estimate"] - df["truth"], np.nan)
    df["covered"] = np.where(
        valid, (df["ci_low"] <= df["truth"]) & (df["truth"] <= df["ci_high"]), np.nan
    )
    df["width"] = np.where(valid, df["ci_high"] - df["ci_low"], np.nan)

    grouped = df.groupby(["policy", "estimator", "arm"], as_index=False).agg(
        truth=("truth", "first"),
        mean_estimate=("estimate", "mean"),
        bias=("error", "mean"),
        rmse=("error", _nan_safe_rmse),
        coverage=("covered", "mean"),
        mean_ci_width=("width", "mean"),
        valid_fraction=("valid", "mean"),
        n_reps=("estimate", "size"),
    )
    grouped["nominal_coverage"] = 1.0 - alpha
    grouped["coverage_gap"] = grouped["coverage"] - grouped["nominal_coverage"]

    # Monte Carlo standard error on the coverage estimate, so readers can see
    # whether an apparent under-coverage is real or replication noise.
    p = grouped["coverage"].to_numpy(dtype=float)
    n = grouped["n_reps"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        grouped["coverage_mcse"] = np.sqrt(np.clip(p * (1 - p), 0, None) / n)

    return grouped


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-policy reward and regret, averaged over replications."""
    share_cols = [c for c in runs.columns if c.startswith("share_arm")]
    agg = {
        "mean_reward": ("cumulative_reward", "mean"),
        "mean_expected_reward": ("expected_cumulative_reward", "mean"),
        "mean_regret": ("oracle_regret", "mean"),
        "sd_regret": ("oracle_regret", "std"),
        "n_reps": ("rep", "size"),
    }
    out = runs.groupby("policy", as_index=False).agg(**agg)
    shares = runs.groupby("policy", as_index=False)[share_cols].mean()
    return out.merge(shares, on="policy")


def coverage_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Policy x estimator coverage matrix, averaged over treatment arms."""
    return (
        summary.pivot_table(
            index="estimator", columns="policy", values="coverage", aggfunc="mean"
        )
        .reindex(["difference_in_means", "ipw", "aipw", "aw_aipw"])
        .round(3)
    )


def bias_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Policy x estimator bias matrix, averaged over treatment arms."""
    return (
        summary.pivot_table(
            index="estimator", columns="policy", values="bias", aggfunc="mean"
        )
        .reindex(["difference_in_means", "ipw", "aipw", "aw_aipw"])
        .round(4)
    )


def summarise_segment_effects(
    segments: pd.DataFrame, true_segment_ate: np.ndarray
) -> pd.DataFrame:
    """Score the group-ATE estimates against per-segment ground truth."""
    df = segments.copy()
    df["truth"] = [
        float(true_segment_ate[int(s), int(a)])
        for s, a in zip(df["segment"], df["arm"], strict=True)
    ]
    valid = df["valid"].fillna(False)
    df["error"] = np.where(valid, df["estimate"] - df["truth"], np.nan)
    df["covered"] = np.where(
        valid, (df["ci_low"] <= df["truth"]) & (df["truth"] <= df["ci_high"]), np.nan
    )
    return df.groupby(["policy", "segment", "arm"], as_index=False).agg(
        truth=("truth", "first"),
        mean_estimate=("estimate", "mean"),
        bias=("error", "mean"),
        coverage=("covered", "mean"),
        n_reps=("estimate", "size"),
    )


def format_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a Markdown table without extra dependencies."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else floatfmt.format(v)
            )
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    divider = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in out.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])
