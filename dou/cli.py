"""Command line entry point: ``python -m dou.cli --help``."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .evaluate import coverage_table, summarise_effects, summarise_runs
from .marketplace import ARM_NAMES, Marketplace, MarketplaceConfig
from .policies import POLICY_REGISTRY
from .simulate import run_once, run_replications


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dou", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    truth = sub.add_parser("truth", help="print the ground-truth estimands")
    truth.add_argument("--mc", type=int, default=2_000_000)

    single = sub.add_parser("run", help="run one replication of one policy")
    single.add_argument("policy", choices=sorted(POLICY_REGISTRY))
    single.add_argument("--horizon", type=int, default=4000)
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--floor", type=float, default=0.02)

    comp = sub.add_parser("compare", help="Monte Carlo comparison of policies")
    comp.add_argument(
        "--policies", nargs="+", default=["fixed", "thompson"],
        choices=sorted(POLICY_REGISTRY),
    )
    comp.add_argument("--horizon", type=int, default=4000)
    comp.add_argument("--reps", type=int, default=200)
    comp.add_argument("--seed", type=int, default=17)
    comp.add_argument("--floor", type=float, default=0.02)
    comp.add_argument("--csv", type=str, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MarketplaceConfig()
    market = Marketplace(config)

    if args.command == "truth":
        values = market.true_arm_values(n_mc=args.mc)
        payload = {
            "arm_values": {ARM_NAMES[i]: round(float(v), 5) for i, v in enumerate(values)},
            "ate_vs_control": {
                ARM_NAMES[i]: round(float(v - values[0]), 5)
                for i, v in enumerate(values)
            },
            "best_arm": ARM_NAMES[int(np.argmax(values))],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "run":
        kwargs = {} if args.policy in {"fixed", "ucb1"} else {"floor": args.floor}
        log = run_once(
            args.policy, args.horizon, seed=args.seed, config=config, policy_kwargs=kwargs
        )
        shares = log.arm_share(config.n_arms)
        print(f"policy               {args.policy}")
        print(f"cumulative reward    {log.cumulative_reward():.0f}")
        print(f"oracle regret        {log.oracle_regret():.2f}")
        print(f"min logged propensity {log.propensities.min():.5f}")
        print("arm shares:")
        for i, name in enumerate(ARM_NAMES[: config.n_arms]):
            print(f"  {name:20s} {shares[i]:.3f}")
        if not log.supports_inference:
            print("\nWARNING: this policy violates positivity; no valid causal estimate.")
        return 0

    effects, runs = [], []
    for name in args.policies:
        kwargs = {} if name in {"fixed", "ucb1"} else {"floor": args.floor}
        e, r = run_replications(
            name, args.horizon, args.reps, base_seed=args.seed,
            config=config, policy_kwargs=kwargs, progress=True,
        )
        effects.append(e)
        runs.append(r)

    effects = pd.concat(effects, ignore_index=True)
    runs = pd.concat(runs, ignore_index=True)
    summary = summarise_effects(effects, market.true_ate())

    pd.set_option("display.width", 200)
    print("\nPolicy performance")
    print(summarise_runs(runs).round(3).to_string(index=False))
    print("\nCoverage of nominal 95% intervals")
    print(coverage_table(summary).to_string())
    print("\nEstimator quality")
    print(
        summary[
            [
                "policy", "estimator", "arm", "truth",
                "bias", "rmse", "coverage", "mean_ci_width",
            ]
        ].round(4).to_string(index=False)
    )
    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
