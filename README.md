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

### Where it actually breaks

Coverage degrades sharply once the exploration floor is throttled below ~0.02.
Exploratory runs at 150 replications showed IPW coverage collapsing to 0.73 at a
floor of 0.001, with AW-AIPW holding at 0.90 on intervals 42% narrower — but the
full-scale sweep over the lower floors has **not** been run in this repository
yet, so those numbers are not reported as final. The sweep stage regenerates
them; see below.

The completed portion of the sweep (250 replications, AW-AIPW) already shows the
shape of the frontier:

| Floor | Control share | Regret | CI width | Coverage |
| --- | --- | --- | --- | --- |
| 0.25 (uniform) | 0.250 | 411.5 | 0.0779 | 0.941 |
| 0.10 | 0.105 | 303.1 | 0.1038 | 0.944 |
| 0.05 | 0.059 | 267.4 | 0.1353 | 0.941 |

Regret falls steeply from 0.25 to 0.10 and then flattens, while CI width grows
throughout. That shape is the design question the project is named for: the
exploration floor is a dial between reward and inferential power, and the
returns to pushing it down are not linear.

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
