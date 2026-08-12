"""Reproduce every result in the README.

The full experiment simulates a few million marketplace sessions and takes
roughly 15 minutes single-threaded, so it is split into resumable stages
that checkpoint to ``results/``::

    python experiments/run_all.py                    # all stages
    python experiments/run_all.py --quick            # ~1 minute smoke test

Stages can also be driven individually, which is useful on constrained
machines, in CI, and when only one part of the analysis needs refreshing::

    python experiments/run_all.py --stage policies --policies fixed thompson
    python experiments/run_all.py --stage sweep --floors 0.25 0.1 0.05
    python experiments/run_all.py --stage sweep --floors 0.02 0.005 0.001
    python experiments/run_all.py --stage segments
    python experiments/run_all.py --stage report

Each stage appends to a checkpoint CSV and is safe to re-run: rows are
de-duplicated on their key columns with the most recent run winning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dou.evaluate import (  # noqa: E402
    bias_table,
    coverage_table,
    format_markdown,
    summarise_effects,
    summarise_runs,
    summarise_segment_effects,
)
from dou.marketplace import ARM_NAMES, Marketplace, MarketplaceConfig  # noqa: E402
from dou.simulate import (  # noqa: E402
    estimate_segment_effects,
    run_once,
    run_replications,
)

ESTIMATOR_ORDER = ["difference_in_means", "ipw", "aipw", "aw_aipw"]
ESTIMATOR_LABELS = {
    "difference_in_means": "Difference in means",
    "ipw": "IPW",
    "aipw": "AIPW",
    "aw_aipw": "Adaptively weighted AIPW",
}
DEFAULT_POLICIES = ["fixed", "epsilon_greedy", "thompson", "ucb1"]
DEFAULT_FLOORS = [0.25, 0.10, 0.05, 0.02, 0.005, 0.001]

EFFECTS_CKPT = "checkpoint_effects.csv"
RUNS_CKPT = "checkpoint_runs.csv"
SWEEP_CKPT = "checkpoint_sweep.csv"
SEGMENT_CKPT = "checkpoint_segments.csv"


def policy_kwargs_for(name: str, floor: float) -> dict:
    if name == "epsilon_greedy":
        return {"epsilon": 0.20, "decay": 0.01, "floor": floor}
    if name == "thompson":
        return {"floor": floor}
    if name == "ucb1":
        return {"c": 0.6}
    return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "policies", "sweep", "segments", "report"],
    )
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    ap.add_argument("--floors", nargs="+", type=float, default=DEFAULT_FLOORS)
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--sweep-reps", type=int, default=250)
    ap.add_argument("--segment-reps", type=int, default=150)
    ap.add_argument("--floor", type=float, default=0.02, help="default exploration floor")
    ap.add_argument("--seed", type=int, default=20240501)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args()


def checkpoint(
    outdir: Path, filename: str, new: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    """Append ``new`` to a checkpoint file, keeping the latest row per key."""
    path = outdir / filename
    combined = new
    if path.exists():
        combined = pd.concat([pd.read_csv(path), new], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def load_checkpoint(outdir: Path, filename: str) -> pd.DataFrame:
    path = outdir / filename
    if not path.exists():
        raise SystemExit(f"missing {path}; run the earlier stages first")
    return pd.read_csv(path)


# ----------------------------------------------------------------- stages


def stage_policies(args, config, market) -> None:
    print(f"[policies] {args.policies}")
    for name in args.policies:
        t0 = time.time()
        eff, runs = run_replications(
            name, args.horizon, args.reps, base_seed=args.seed,
            config=config, policy_kwargs=policy_kwargs_for(name, args.floor),
        )
        checkpoint(args.outdir, EFFECTS_CKPT, eff, ["policy", "estimator", "arm", "rep"])
        checkpoint(args.outdir, RUNS_CKPT, runs, ["policy", "rep"])
        print(f"  {name:16s} {time.time() - t0:6.1f}s  ({args.reps} reps)")


def stage_sweep(args, config, market) -> None:
    print(f"[sweep] floors {args.floors}")
    true_ate = market.true_ate()
    n_treated = config.n_arms - 1
    rows = []
    for floor in args.floors:
        t0 = time.time()
        eff, runs = run_replications(
            "thompson", args.horizon, args.sweep_reps, base_seed=args.seed + 99,
            config=config, policy_kwargs={"floor": floor},
        )
        s = summarise_effects(eff, true_ate)
        rs = summarise_runs(runs)
        for est in ESTIMATOR_ORDER:
            sub = s[s.estimator == est]
            rows.append(
                {
                    "floor": floor,
                    "estimator": est,
                    "coverage": sub.coverage.mean(),
                    "coverage_mcse": float(
                        np.sqrt(0.95 * 0.05 / (args.sweep_reps * n_treated))
                    ),
                    "mean_ci_width": sub.mean_ci_width.mean(),
                    "bias": sub.bias.mean(),
                    "rmse": sub.rmse.mean(),
                    "mean_regret": float(rs.mean_regret.iloc[0]),
                    "mean_reward": float(rs.mean_expected_reward.iloc[0]),
                    "control_share": float(runs["share_arm0"].mean()),
                    "reps": args.sweep_reps,
                }
            )
        print(f"  floor={floor:<7} {time.time() - t0:6.1f}s")
    checkpoint(args.outdir, SWEEP_CKPT, pd.DataFrame(rows), ["floor", "estimator"])


def stage_segments(args, config, market) -> None:
    print("[segments]")
    t0 = time.time()
    rows = []
    for rep in range(args.segment_reps):
        log = run_once(
            "thompson", args.horizon, seed=args.seed + 7 * rep,
            config=config, policy_kwargs={"floor": args.floor},
        )
        for row in estimate_segment_effects(log, config.n_arms):
            row["rep"] = rep
            rows.append(row)
    checkpoint(
        args.outdir, SEGMENT_CKPT, pd.DataFrame(rows),
        ["policy", "segment", "arm", "rep"],
    )
    print(f"  {args.segment_reps} replications in {time.time() - t0:.1f}s")


def stage_report(args, config, market) -> None:
    print("[report]")
    true_ate = market.true_ate()
    true_seg = market.true_segment_ate()

    effects = load_checkpoint(args.outdir, EFFECTS_CKPT)
    runs = load_checkpoint(args.outdir, RUNS_CKPT)
    sweep = load_checkpoint(args.outdir, SWEEP_CKPT)
    segments = load_checkpoint(args.outdir, SEGMENT_CKPT)

    effect_summary = summarise_effects(effects, true_ate)
    run_summary = summarise_runs(runs)
    seg_summary = summarise_segment_effects(segments, true_seg)

    effect_summary.to_csv(args.outdir / "estimator_quality.csv", index=False)
    run_summary.to_csv(args.outdir / "policy_performance.csv", index=False)
    seg_summary.to_csv(args.outdir / "segment_effects.csv", index=False)

    print("\n" + coverage_table(effect_summary).to_string())
    print("\n" + run_summary.round(3).to_string(index=False))

    _plot_reward_precision_frontier(sweep, args.outdir)
    _plot_coverage(sweep, args.outdir)
    _plot_estimator_comparison(effect_summary, args.outdir)
    _write_summary_markdown(
        args.outdir, effect_summary, run_summary, sweep, seg_summary, true_ate, args
    )

    (args.outdir / "run_metadata.json").write_text(
        json.dumps(
            {
                "horizon": args.horizon,
                "reps": args.reps,
                "sweep_reps": args.sweep_reps,
                "segment_reps": args.segment_reps,
                "default_floor": args.floor,
                "seed": args.seed,
                "true_ate": true_ate.tolist(),
            },
            indent=2,
        )
    )
    print(f"\nwrote tables and figures to {args.outdir}")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.horizon, args.reps = 1500, 40
        args.sweep_reps, args.segment_reps = 30, 20
        args.floors = [0.25, 0.02, 0.001]

    args.outdir.mkdir(parents=True, exist_ok=True)
    config = MarketplaceConfig()
    market = Marketplace(config)

    print(f"true ATE vs control: {np.round(market.true_ate(), 4).tolist()}")
    print(f"best arm: {ARM_NAMES[market.best_arm]}\n")

    stages = {
        "policies": stage_policies,
        "sweep": stage_sweep,
        "segments": stage_segments,
        "report": stage_report,
    }
    order = (
        ["policies", "sweep", "segments", "report"]
        if args.stage == "all"
        else [args.stage]
    )
    started = time.time()
    for name in order:
        stages[name](args, config, market)
    print(f"\ntotal {time.time() - started:.1f}s")


# ------------------------------------------------------------------ plots


def _plot_reward_precision_frontier(sweep: pd.DataFrame, outdir: Path) -> None:
    aw = sweep[sweep.estimator == "aw_aipw"].sort_values("floor", ascending=False)
    fig, ax1 = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(len(aw))

    ax1.plot(x, aw.mean_regret, "o-", color="#1f77b4", lw=2)
    ax1.set_ylabel("Cumulative regret (lower is better)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{f:g}" for f in aw.floor])
    ax1.set_xlabel("Exploration floor  (guaranteed assignment probability per arm)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, aw.mean_ci_width, "s--", color="#d62728", lw=2)
    ax2.set_ylabel("Mean 95% CI width on the ATE", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title(
        "Adaptivity buys reward and sells precision\n"
        "Thompson sampling, adaptively weighted AIPW"
    )
    fig.tight_layout()
    fig.savefig(outdir / "reward_precision_frontier.png", dpi=160)
    plt.close(fig)


def _plot_coverage(sweep: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    order = sorted(sweep.floor.unique(), reverse=True)
    for est in ESTIMATOR_ORDER:
        sub = sweep[sweep.estimator == est].set_index("floor").reindex(order)
        ax.plot(
            np.arange(len(order)), sub.coverage, "o-", lw=2, label=ESTIMATOR_LABELS[est]
        )
    ax.axhline(0.95, color="black", ls=":", lw=1.2, label="Nominal 95%")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{f:g}" for f in order])
    ax.set_xlabel("Exploration floor")
    ax.set_ylabel("Empirical coverage of the 95% interval")
    ax.set_title("Throttling exploration breaks the intervals, unevenly")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "coverage_by_floor.png", dpi=160)
    plt.close(fig)


def _plot_estimator_comparison(summary: pd.DataFrame, outdir: Path) -> None:
    policies = [
        p for p in ["fixed", "epsilon_greedy", "thompson"] if p in set(summary.policy)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    width = 0.2
    for ax, metric, label, ref in [
        (axes[0], "coverage", "Empirical coverage", 0.95),
        (axes[1], "rmse", "RMSE of the ATE estimate", None),
    ]:
        for i, est in enumerate(ESTIMATOR_ORDER):
            vals = [
                summary[(summary.policy == p) & (summary.estimator == est)][metric].mean()
                for p in policies
            ]
            ax.bar(
                np.arange(len(policies)) + i * width, vals, width,
                label=ESTIMATOR_LABELS[est],
            )
        if ref is not None:
            ax.axhline(ref, color="black", ls=":", lw=1.2)
            ax.set_ylim(0.85, 1.0)
        ax.set_xticks(np.arange(len(policies)) + 1.5 * width)
        ax.set_xticklabels(policies)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Estimator behaviour under fixed versus adaptive allocation")
    fig.tight_layout()
    fig.savefig(outdir / "estimator_comparison.png", dpi=160)
    plt.close(fig)


def _write_summary_markdown(
    outdir: Path,
    effect_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    sweep: pd.DataFrame,
    seg: pd.DataFrame,
    true_ate: np.ndarray,
    args: argparse.Namespace,
) -> None:
    aw = sweep[sweep.estimator == "aw_aipw"].sort_values("floor", ascending=False)
    parts = [
        "# Results",
        "",
        "Generated by `python experiments/run_all.py`. "
        f"Horizon {args.horizon}, {args.reps} replications per policy, "
        f"{args.sweep_reps} per sweep point, default exploration floor {args.floor}.",
        "",
        f"True ATE against control: {np.round(true_ate[1:], 4).tolist()}",
        "",
        "## Policy performance",
        "",
        format_markdown(run_summary.round(3)),
        "",
        "## Interval coverage, nominal 0.95",
        "",
        format_markdown(coverage_table(effect_summary).reset_index(), "{:.3f}"),
        "",
        "## Bias",
        "",
        format_markdown(bias_table(effect_summary).reset_index()),
        "",
        "## Exploration floor sweep, adaptively weighted AIPW",
        "",
        format_markdown(
            aw[
                [
                    "floor", "control_share", "mean_regret",
                    "mean_ci_width", "coverage", "rmse",
                ]
            ].round(4)
        ),
        "",
        "## Segment-level effects",
        "",
        format_markdown(seg.round(4)),
        "",
    ]
    (outdir / "RESULTS.md").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
