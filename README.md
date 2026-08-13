# Decision Optimization Under Uncertainty

Adaptive allocation buys reward and sells statistical precision. This repository
measures both sides of that trade on a synthetic two-sided marketplace where the
true treatment effects are known by construction.

The question is not "does a bandit beat a fixed A/B split on reward" — it does,
and that is uninteresting. The question is what the reward gain costs you in
your ability to say *how much* each treatment was worth afterwards, and whether
that cost can be bought back with a better estimator rather than by giving up
the reward.

---

## The problem

A marketplace runs four ranking policies as arms. Traffic can be split two ways:

- **Fixed assignment** — a classic randomised experiment. Constant, known
  propensities, so every textbook estimator is valid.
- **Adaptive allocation** — a bandit shifts traffic toward whatever is winning.
  More reward, but the log is no longer an i.i.d. sample.

The second option breaks ordinary inference in a specific way. The assignment
probability at step `t` depends on outcomes observed before `t`, so arms that got
lucky early are sampled more often afterwards. Three consequences follow:

1. **Sample means are biased.** An arm is preferentially sampled exactly when its
   running estimate is high, and subsequent draws regress toward the truth. Here
   the bandit abandons the control arm, so the *control* mean is computed from a
   small, adversely selected sample and every contrast against it is pushed
   upward.
2. **Intervals are too narrow.** The naive estimator treats an adaptively
   collected sample as if it were i.i.d., so its standard errors understate the
   real sampling variability.
3. **Some designs are not identifiable at all.** UCB1 is deterministic. It
   assigns probability 1 to one arm at every step, violating positivity, and no
   inverse-propensity estimator is defined on its logs.

---

## What is implemented

**Data-generating process** (`src/dou/marketplace.py`). Sessions carry a buyer
segment, device, and price sensitivity. Conversion is Bernoulli with a logit
link, and treatment effects are heterogeneous — they are modified by segment.
The design is deliberately arranged so the best marginal arm
(`boost_new_sellers`, ATE +0.154) is **not** the one with the largest base
effect: `aggressive_promo` has the biggest raw coefficient but a negative
segment interaction that sinks it. A homogeneous DGP would be a much weaker test
bed, because regression adjustment would have nothing to exploit.

All potential outcomes are drawn for every session, so oracle regret is
available for scoring. The estimators never see the counterfactuals.

**Allocation policies** (`src/dou/policies.py`). Fixed assignment, ε-greedy,
Thompson sampling, UCB1. Every policy exposes its full propensity vector
*before* drawing an arm, and the simulator logs it. This is the most important
design decision in the repository: a logging pipeline that records which arm was
chosen but not the probability it was chosen with has thrown away the
information needed to debias the data, and no amount of downstream modelling
recovers it.

Thompson sampling has no closed-form selection probability, so it is computed by
posterior Monte Carlo and the arm is then drawn from *that* estimated
distribution — rather than the usual draw-once-and-argmax shortcut. The two are
equivalent in distribution, but only this version yields an exactly known
propensity.

**Estimators** (`src/dou/estimators.py`), in increasing order of what they repair:

| Estimator | Repairs |
| --- | --- |
| `difference_in_means` | nothing; the naive baseline |
| `ipw` | bias, via known propensities; variance explodes as they anneal |
| `aipw` | variance, via an outcome model fit only on past data |
| `adaptively_weighted_aipw` | the limiting distribution, restoring interval validity |

The AIPW score for arm `a` at time `t` is

```
G_t(a) = m_a(X_t) + 1{A_t = a} / pi_t(a) * (Y_t - m_a(X_t))
```

where `m_a` is fit strictly on data prior to `t`, so the score stays a martingale
difference sequence. Fitting the nuisance model on the full sample would destroy
exactly the property the estimator depends on.

The adaptively weighted variant reweights each score by

```
h_t = (1 / pi_t(a) + 1 / pi_t(0)) ** -0.5
```

chosen so that `h_t` times the conditional standard deviation of the score is
roughly constant across the horizon. This reduces to `sqrt(pi_t(a))` in the
single-arm case — the constant allocation rate weights of Hadad, Hirshberg,
Zhan, Wager and Athey (PNAS, 2021).

---

## Results

Horizon 4000 sessions, 300 Monte Carlo replications, exploration floor 0.02.
Regenerate with `python experiments/run_all.py`.

### Reward

| Policy | Expected reward | Regret | Control share |
| --- | --- | --- | --- |
| fixed | 1380 | 411.5 | 0.250 |
| ε-greedy | 1534 | 255.4 | 0.032 |
| thompson | 1539 | 251.1 | 0.033 |
| ucb1 | 1528 | 261.3 | 0.036 |

Thompson sampling cuts regret by 39% against the fixed split.

### Precision cost

Mean 95% CI width on the ATE:

| Estimator | fixed | thompson | ε-greedy |
| --- | --- | --- | --- |
| difference in means | 0.0804 | 0.1670 | 0.2024 |
| IPW | 0.0977 | 0.2294 | 0.2808 |
| AIPW | 0.0779 | 0.1943 | 0.2338 |
| adaptively weighted AIPW | 0.0779 | **0.1840** | 0.2295 |

This is the headline. The same 4000 sessions, run adaptively instead of
uniformly, yield an effect estimate with **2.4× the interval width**. The reward
gain was real, and so is the bill.

Two secondary readings. First, AW-AIPW is the tightest valid estimator under
adaptivity (0.1840 vs AIPW's 0.1943, RMSE 0.0472 vs 0.0507) while collapsing
*exactly* onto AIPW under fixed assignment, where the weights are constant —
this equivalence is asserted in the test suite. Second, ε-greedy is worse than
Thompson on both axes here: comparable reward, wider intervals. Its annealed
exploration concentrates propensity mass more erratically.

### Coverage

Empirical coverage of nominal 95% intervals:

| Estimator | fixed | thompson | ε-greedy | ucb1 |
| --- | --- | --- | --- | --- |
| difference in means | 0.944 | 0.947 | 0.963 | 0.924 |
| IPW | 0.943 | 0.937 | 0.951 | — |
| AIPW | 0.943 | 0.923 | 0.961 | — |
| adaptively weighted AIPW | 0.943 | 0.932 | 0.964 | — |

**At a floor of 0.02, every estimator is approximately fine.** Monte Carlo
standard error on these is about 0.0075, so the spread is 1–3 MCSE — real but
small. This is worth stating plainly rather than dressing up: the interval
*width* penalty is large and robust, while the coverage penalty at a reasonable
exploration floor is mild.

The em-dashes in the UCB1 column are the point of including it. UCB1 is
deterministic, so no inverse-propensity estimator is defined on its logs at all.
The difference-in-means cell is computable, and a practitioner would compute it
without noticing anything was wrong.

# Results

Generated by `python experiments/run_all.py`. Horizon 4000, 300 replications per policy, 250 per sweep point, default exploration floor 0.02.

True ATE against control: [0.1543, 0.1245, 0.11]

## Policy performance

| policy | mean_reward | mean_expected_reward | mean_regret | sd_regret | n_reps | share_arm0 | share_arm1 | share_arm2 | share_arm3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epsilon_greedy | 1543.2130 | 1539.6320 | 250.1220 | 53.1290 | 300 | 0.0290 | 0.6200 | 0.2210 | 0.1310 |
| fixed | 1381.5600 | 1378.6270 | 411.1260 | 6.1140 | 300 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| thompson | 1541.4570 | 1538.6070 | 251.1460 | 27.6690 | 300 | 0.0330 | 0.6310 | 0.1990 | 0.1370 |
| ucb1 | 1530.6700 | 1528.4060 | 261.3470 | 18.4350 | 300 | 0.0360 | 0.5720 | 0.2270 | 0.1650 |

## Interval coverage, nominal 0.95

| estimator | epsilon_greedy | fixed | thompson | ucb1 |
| --- | --- | --- | --- | --- |
| difference_in_means | 0.963 | 0.944 | 0.947 | 0.924 |
| ipw | 0.951 | 0.943 | 0.937 |  |
| aipw | 0.961 | 0.943 | 0.923 |  |
| aw_aipw | 0.964 | 0.943 | 0.932 |  |

## Bias

| estimator | epsilon_greedy | fixed | thompson | ucb1 |
| --- | --- | --- | --- | --- |
| difference_in_means | -0.0084 | 0.0012 | 0.0002 | 0.0065 |
| ipw | -0.0001 | 0.0007 | 0.0062 |  |
| aipw | 0.0026 | 0.0011 | 0.0047 |  |
| aw_aipw | 0.0021 | 0.0011 | 0.0047 |  |

## Exploration floor sweep, adaptively weighted AIPW

| floor | control_share | mean_regret | mean_ci_width | coverage | rmse |
| --- | --- | --- | --- | --- | --- |
| 0.2500 | 0.2501 | 411.5365 | 0.0779 | 0.9413 | 0.0202 |
| 0.1000 | 0.1046 | 303.0803 | 0.1038 | 0.9440 | 0.0268 |
| 0.0500 | 0.0588 | 267.3686 | 0.1353 | 0.9413 | 0.0354 |
| 0.0200 | 0.0329 | 246.5589 | 0.1835 | 0.9347 | 0.0490 |
| 0.0050 | 0.0210 | 238.4334 | 0.2634 | 0.9427 | 0.0669 |
| 0.0010 | 0.0176 | 235.4308 | 0.3366 | 0.8947 | 0.0938 |

## Segment-level effects

| policy | segment | arm | truth | mean_estimate | bias | coverage | n_reps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thompson | 0 | 1 | 0.0829 | 0.0873 | 0.0044 | 0.9267 | 150 |
| thompson | 0 | 2 | 0.1420 | 0.1512 | 0.0092 | 0.9200 | 150 |
| thompson | 0 | 3 | 0.1846 | 0.1909 | 0.0062 | 0.9333 | 150 |
| thompson | 1 | 1 | 0.1783 | 0.1735 | -0.0048 | 0.9867 | 150 |
| thompson | 1 | 2 | 0.1219 | 0.1192 | -0.0026 | 0.9333 | 150 |
| thompson | 1 | 3 | 0.0893 | 0.0815 | -0.0078 | 0.9733 | 150 |
| thompson | 2 | 1 | 0.2728 | 0.2720 | -0.0008 | 0.9333 | 150 |
| thompson | 2 | 2 | 0.0898 | 0.0825 | -0.0073 | 0.9533 | 150 |
| thompson | 2 | 3 | -0.0213 | -0.0265 | -0.0051 | 0.9600 | 150 |

---

## Running it

```bash
git clone https://github.com/shanthang7/decision-optimization-under-uncertainty.git
cd decision-optimization-under-uncertainty
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the install and see the ground truth:

```bash
pytest -q                 # 24 tests, ~60s
dou truth                 # true arm values and ATEs
dou run thompson          # one replication, with arm shares and min propensity
dou compare --policies fixed thompson --reps 200
```

Reproduce the full analysis:

```bash
python experiments/run_all.py           # all stages, ~15 min single-threaded
python experiments/run_all.py --quick   # ~30s smoke test
```

The driver checkpoints to `results/` and is resumable, so stages can be run
separately — useful on a constrained machine, or to refresh one table without
recomputing everything:

```bash
python experiments/run_all.py --stage policies
python experiments/run_all.py --stage sweep --floors 0.25 0.1 0.05
python experiments/run_all.py --stage sweep --floors 0.02 0.005 0.001
python experiments/run_all.py --stage segments
python experiments/run_all.py --stage report
```

Re-running a stage overwrites its rows rather than duplicating them, keyed on
`(policy, estimator, arm, rep)` and `(floor, estimator)`.

**To finish the results in this README**, the sweep's lower floors and the
segment stage still need running:

```bash
python experiments/run_all.py --stage sweep --floors 0.02 0.005 0.001
python experiments/run_all.py --stage segments
python experiments/run_all.py --stage report
```

That writes `results/RESULTS.md` plus three figures:
`reward_precision_frontier.png`, `coverage_by_floor.png`,
`estimator_comparison.png`.

---

## Layout

```
src/dou/
  marketplace.py   synthetic DGP; ground-truth ATE and segment-level GATE
  policies.py      allocation rules, each logging its propensities
  estimators.py    difference in means, IPW, AIPW, adaptively weighted AIPW
  simulate.py      replication engine and per-run estimation
  evaluate.py      bias, RMSE, coverage, interval width, regret
  cli.py           `dou truth` / `dou run` / `dou compare`
experiments/
  run_all.py       staged, resumable driver for every result above
tests/
  test_dou.py      24 property tests
```

## Notes on the test suite

Most of the tests are statistical rather than plumbing. Three are load-bearing:

- **`test_outcome_model_never_sees_the_future`** flips every outcome in the
  second half of a dataset and asserts that first-half predictions are
  unchanged. If the nuisance model leaks future information, the AIPW scores
  stop being a martingale difference sequence and every interval in this
  repository is silently invalid. Crucially, a leak makes results look *better*,
  not worse — which is why it needs a test rather than an eyeball.
- **`test_aipw_survives_a_deliberately_wrong_outcome_model`** replaces the
  outcome model with a constant 0.9 and checks that AIPW still recovers the
  truth. That is the double-robustness property, asserted rather than assumed.
- **`test_aw_aipw_equals_aipw_under_constant_propensities`** pins the exact
  collapse of the weighted estimator onto the unweighted one when weights are
  constant. This caught a real bug: the two disagreed in the fourth decimal
  because of a degrees-of-freedom mismatch between the sample-variance and
  asymptotic conventions, now fixed with an `n/(n-1)` correction.

## Limitations

- The bandit is non-contextual: it optimises marginal reward while outcomes
  depend on context. This is a common production shape, but a contextual bandit
  would change the propensity dynamics.
- Coverage is evaluated at a single horizon (4000). The adaptive bias is a
  finite-sample phenomenon and shrinks with `T`.
- The DGP is stationary by default. `MarketplaceConfig.intercept_drift` exposes
  non-stationarity, which is where Thompson sampling with a fixed Beta prior
  degrades, but no experiment in this repository exercises it yet.
- The outcome model is a per-arm ridge on a correctly-specified basis, so
  reported behaviour is close to a best case for the augmented estimators.

## Reference

Hadad, Hirshberg, Zhan, Wager, Athey. *Confidence intervals for policy
evaluation in adaptive experiments.* PNAS 118(15), 2021.

## License

MIT
