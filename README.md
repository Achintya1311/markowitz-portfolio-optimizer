# Markowitz portfolio optimizer

Educational research on historical data. Not investment advice. Not a SEBI-registered research analyst.

**Question:** does mean-variance optimization actually beat a naive equal-weight portfolio out-of-sample, and does that answer hold once the universe is bigger than 3 names?
**Method:** walk-forward (non-overlapping rolling-window) long-only max-Sharpe tangency on a Ledoit-Wolf shrunk covariance, scored against equal-weight, rerun on both a 3-ticker and a 29-name large-cap universe.
**Finding (Day 9, cost-adjusted Day 40, sensitivity-checked Day 43):** widening the universe fixes the 3-ticker single-name corner solution (9/11 folds → 1/126), but the walk-forward bottom line flips the other way — equal-weight now beats the optimizer (+426.96% vs. +309.00% compounded, 126 folds) — and neither result is statistically significant (t=-0.50). Charging the same transaction-cost schedule the rest of this portfolio already treats as decisive (R9) widens the gap, not narrows it: the optimizer's 57.6% average monthly turnover (equal-weight's is ~0, since it is reset to exactly 1/n every fold) pulls its compounded return down to +252.20% after costs — a much bigger drop than the 15.06-point sum of per-fold cost rates would suggest, because costs compound against a smaller capital base in every later fold. Day 43 swept the risk-free rate (0/4/7%) and three more train/test window pairings: equal-weight wins at every one — robust, not a one-setting artifact — though a shorter 126-day training window widens the gap enough to cross conventional significance (t=-2.09).
**Chart:** `outputs/` frontier chart (tangency point + capital market line) from `optimizer.report`.
**Reproduce:** `python -m optimizer.report --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS`

Mean-variance optimization over an NSE universe, with the fragility of the inputs treated as the main finding rather than a footnote.

**Status:** Complete on a 3-stock universe (Sep 2026), reran on a 29-name large-cap universe (17 Oct 2026, Day 9), cost-adjusted (18 Oct 2026, Day 40), and sensitivity-checked for risk-free rate and window pairing (19 Oct 2026, Day 43). Part of a nine-repo equity research pipeline; [STOCKSTALKER](https://github.com/Achintya1311/STOCKSTALKER) is the front door.

*Educational research on historical data. Not investment advice. Not a SEBI-registered research analyst.*

## What this is

Efficient frontier, tangency portfolio and capital market line, under realistic constraints - sector caps, position limits, turnover.

The centrepiece is the comparison between a sample covariance matrix and a Ledoit-Wolf shrinkage estimate, showing how far the optimal weights move for the same data. An optimizer presented without that comparison is a sales pitch.

## Correctness gate

Frontier is convex, the minimum-variance point matches the analytic two-asset solution, and the out-of-sample comparison reruns from committed fixtures.

This is the test that decides whether the repo is finished. A result that has not passed it is a draft.

## Data sources

Every source is free. Nothing in this project requires a paid tier, a subscription, or a funded account.

- yfinance - NSE daily prices
- FRED / RBI - risk-free rate

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m optimizer.stats --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS
python -m optimizer.frontier --tickers TICKER1.NS,TICKER2.NS --target-return 0.12
python -m optimizer.tangency --tickers TICKER1.NS,TICKER2.NS --risk-free-rate 0.065
python -m optimizer.shrinkage --tickers TICKER1.NS,TICKER2.NS --risk-free-rate 0.065
python -m optimizer.constraints --tickers TICKER1.NS,TICKER2.NS,TICKER3.NS --target-return -0.15 \
    --max-position 0.6 --turnover-lambda 50 --prev-weights 0.2,0.3,0.5
python -m optimizer.oos --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS \
    --train-window 252 --test-window 21 --turnover-lambda 5
python -m optimizer.tangency --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --risk-free-rate-sweep 0.0,0.04,0.07
python -m optimizer.oos --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --risk-free-rate-sweep 0.0,0.04,0.07
python -m optimizer.report --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --risk-free-rate 0.065
python -m optimizer.audit --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS
python -m optimizer.sizing --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --sector-cap 0.5 \
    --contract outputs/sizing_contract_RELIANCE.json --contract-ticker RELIANCE.NS
```

`optimizer.constraints` (Day 5) adds sector caps, a per-asset position limit,
and a turnover penalty on top of Day 2's minimum-variance-for-a-target-return
problem, and reports the constrained solution next to the unconstrained
baseline so the effect of each constraint is visible. `--sector-cap
Energy=0.5,...` caps sector exposure (default sector map: the shared
3-ticker fixture universe's real NSE sectors); `--max-position` caps any
single name; `--turnover-lambda` penalizes squared distance from
`--prev-weights` (default: equal-weight).

`optimizer.shrinkage` (Day 4) is the centrepiece: compares the plain sample
covariance against a Ledoit-Wolf shrinkage estimate, and reports how far the
global minimum-variance and long-only tangency weights move between the two.

`optimizer.stats` (Day 1) is the returns/covariance diagnostic below; `optimizer.frontier` (Day 2) solves the minimum-variance portfolio for one target return, long-only and long/short; `optimizer.tangency` (Day 3) solves the maximum-Sharpe tangency portfolio and reports the capital market line. `--risk-free-rate` defaults to `0.0` - there is no committed risk-free-rate fixture and no live FRED/RBI fetch available offline, so pass a real annualized rate explicitly if you have one. `--risk-free-rate-sweep` (Day 43, R10) reruns the long-only tangency portfolio at several comma-separated rates instead and reports whether the capital market line's slope sign holds across all of them.

`optimizer.oos` (Day 6) walks a fixed-length training window forward through the fixture history, solving the long-only max-Sharpe/Ledoit-Wolf tangency portfolio on each window and holding it (no rebalancing) through the following test window, scored against equal-weight held the same way. `--train-window`/`--test-window` are in trading days (default 252/21, roughly one year training, one month holding). Day 40 adds a transaction-cost term (STOCKSTALKER's own STT + stamp duty + slippage schedule, R9) on each fold's L1 turnover from the previous fold's held weights, reporting gross and cost-adjusted compounded return/t-stat side by side; `--turnover-lambda` reuses `optimizer.constraints`'s turnover-penalized solve to re-target the same expected return with less rebalancing. `--risk-free-rate-sweep` (Day 43, R10) reruns the whole walk-forward at several rates and reports whether the after-cost optimizer-vs-equal-weight sign holds across all of them.

`optimizer.report` (Day 7) writes a long-only frontier chart (with the tangency point and capital market line) to `outputs/` and prints the min-variance and max-Sharpe weights as an aligned terminal table; pass `--no-chart` to skip the PNG. `optimizer.audit` (Day 7) is the `ml-pipeline-audit` pass: checks `optimizer.oos`'s walk-forward folds for boundary gaps/overlaps and for a fold's decision depending on data after its own test window, printing a summary and writing `outputs/audit_<date>.md`; exits non-zero if either check finds a violation.

`optimizer.sizing` (integration day) is the v0.4 contract to the spine: the same Ledoit-Wolf shrunk covariance Day 4 built, fed into a long-only max-Sharpe tangency solve with Day 5's sector-cap constraint added on top, because the raw unconstrained tangency portfolio Days 3/4/6/7 already found (100% RELIANCE.NS) is not a position size any real book would take. `--sector-cap` (default 50%) is a uniform per-sector weight cap; `--contract`/`--contract-ticker` write the `{"sizing": {"weight", "method", "constraint_binding"}}` block STOCKSTALKER reads.

This repo is offline-only, against committed fixtures. No code here reads an API key or performs any live fetch; there is no live-data path. The one real live path in this repo is `scripts/fetch_largecaps.py` (bare `yfinance`, no key), which runs through `.github/workflows/fetch-largecaps.yml` and commits the refreshed fixture back - it is not invoked by any `optimizer.*` command above.

## Findings

**Day 1 - returns and covariance, and how noisy each one is.** Fixtures: the
same 501-day daily-close series (2024-09-04 .. 2026-09-04) for RELIANCE.NS,
TATACHEM.NS and CROMPTON.NS already committed by `STOCKSTALKER` and
`fama-french-factor-model`, so a real cross-repo match is possible once the
optimizer ships a contract, not just a synthetic one.

Annualized sample mean, volatility, and the standard error of that mean
estimate (`optimizer.stats`):

| Ticker | Mean (ann.) | Vol (ann.) | SE(mean) | t-stat |
|---|---|---|---|---|
| RELIANCE.NS | -6.42% | 20.75% | 14.73% | -0.44 |
| TATACHEM.NS | -26.27% | 28.71% | 20.38% | -1.29 |
| CROMPTON.NS | -34.07% | 28.48% | 20.22% | -1.69 |

Every one of those t-stats is below 2 in magnitude. None of the three mean
estimates is statistically distinguishable from zero at this sample size -
this is the textbook problem with plugging sample means into Markowitz
(Ch.12.9's risk-adjusted-return framing assumes a usable expected-return
input; two years of daily data is not enough to produce one for single
names). The optimizer built in Days 2-4 will still rank these assets against
each other by their point estimates, because that is what mean-variance does
- but the ranking is standing on numbers this noisy, and the frontier
chapters say so rather than presenting the weights as precise.

Covariance is comparatively well behaved: pairwise correlations sit around
0.28-0.29 (all three are momentum-screened NSE names from the same Stock
Stalker universe, so some co-movement is expected) and the covariance
matrix's condition number is 3.2 - not ill-conditioned yet with only 3
assets, though this is exactly the number Day 4's Ledoit-Wolf comparison
will watch as the universe grows.

A split-half check (first 250 days vs last 250) makes the mean's instability
concrete instead of asserting it: the covariance matrix moved 19.1%
(relative Frobenius norm) between halves, and TATACHEM.NS's annualized mean
swung by -25.2 percentage points between the two windows - a bigger move
than the full-sample point estimate itself. A single full-window mean is
papering over that swing, not resolving it.

**Day 2 - minimum-variance portfolio for a target return, long-only and
long/short (`optimizer.frontier`).** Solves, for a fixed target annualized
return, the weights minimizing portfolio variance subject to
`sum(w) == 1` and `w @ mu == target`, via `scipy.optimize.minimize`
(SLSQP) - once with `w >= 0` and once unconstrained.

The honest headline finding is immediate, before any optimization: **all
three of this universe's Day 1 sample means are negative**
(RELIANCE.NS -6.42%, TATACHEM.NS -26.27%, CROMPTON.NS -34.07%). A
long-only portfolio's return is a convex combination of its constituents'
means, so the entire long-only-feasible target-return range is
`[-34.07%, -6.42%]` - **no long-only portfolio built from this 3-ticker
universe can target a positive expected return**, however the weights are
chosen. Asking for one (e.g. `--target-return 0.10`) correctly comes back
infeasible; only the long/short variant can reach it, and only by
shorting RELIANCE.NS - +162.9%/RELIANCE.NS, -12.5%/TATACHEM.NS,
-50.4%/CROMPTON.NS - which trades the return for leverage and short risk
this universe's fixtures cannot back-test the funding cost of.

For a target inside the feasible range (-15.00%), long-only and
long/short agree on the same interior solution
(RELIANCE.NS +63.59%, TATACHEM.NS +19.09%, CROMPTON.NS +17.32%, vol
17.93%) - the non-negativity bound isn't binding there, so both variants
find the same unconstrained minimum.

Both "done when" checks from `NEXT_STEPS.md` are covered by tests, not
just asserted: `tests/test_frontier.py` checks that for exactly 2 assets
the numerical minimizer's weights match the closed-form solution (with 2
assets and 2 equality constraints - budget and target return - the
weights are pinned down algebraically, independent of the covariance
matrix), on both a synthetic pair and a real fixture pair
(RELIANCE.NS/TATACHEM.NS); and that a 15-point sweep of variance against
target return has non-negative second differences (convex), both on the
real 3-ticker fixture and as an explicit negative check that a concave
sequence is correctly rejected.

**Day 3 - maximum-Sharpe tangency portfolio and the capital market line
(`optimizer.tangency`).** Given a risk-free rate, the tangency portfolio
maximizes `(w @ mu - rf) / sqrt(w @ cov @ w)`. Unconstrained (any sign,
full investment) it has the standard closed form `w = inv(cov)(mu - rf) /
sum(inv(cov)(mu - rf))`; long-only (`w >= 0`) it is solved numerically
with `scipy.optimize` (SLSQP), same pattern as Day 2's frontier.

**A real bug the closed form hides if you don't check it:** the textbook
formula only maximizes Sharpe when `sum(inv(cov)(mu - rf)) > 0`. This
repo's universe has every asset's annualized sample mean negative (Day
1's finding), so at any risk-free rate near zero that sum is *negative*
- dividing by it flips the sign of every weight, landing on the
Sharpe-*minimizing* portfolio while looking exactly like a normal
answer. The first version of `optimizer.tangency` written today did
this silently (reported Sharpe -1.35, the theoretical minimum, dressed
up as "the tangency portfolio"). It is now detected and reported as
undefined instead: `max_sharpe_weights_unconstrained` returns
`success=False` whenever `sum(inv(cov)(mu-rf)) <= 0`, and a numerical
search (200 random SLSQP restarts against the real fixtures) confirms
the underlying reason - the achievable Sharpe climbs toward the
theoretical bound (`sqrt(excess' inv(cov) excess) = 1.3462` at rf=0)
only as position sizes diverge into the millions, never converging. No
finite full-investment portfolio attains it. This is recorded as a
genuine finding, not a numerical footnote: with this universe, "the"
unconstrained max-Sharpe portfolio does not exist.

The long-only tangency portfolio is well-behaved by contrast (the
long-only simplex is compact, so a maximum is always attained): at rf=0
it is a corner solution, 100% RELIANCE.NS (the least-negative mean),
Sharpe -0.3094. Still negative - every long-only portfolio from this
universe underperforms holding cash on a risk-adjusted basis, which
means the "capital market line" built from it is **downward-sloping**:
expected return falls as more is allocated away from cash, all the way
out through 100% tangency (-6.42% return at 20.75% vol) and beyond. A
downward-sloping CML is not a chart error, it is what a negative-Sharpe
tangency portfolio looks like - reported straight rather than dressed up
as an efficient allocation line.

`tests/test_tangency.py` checks the closed form against a direct
numerical Sharpe-maximization (SLSQP, no closed form used) on a
synthetic well-behaved (positive-excess-return) universe, checks it
against the theoretical `sqrt(excess' inv(cov) excess)` bound, and has an
explicit regression test that the degenerate branch never silently
returns the minimizing portfolio - the exact bug found and fixed today.
It also checks the long-only solver against the same numerical
optimizer on the real fixtures, and checks the long-only Sharpe never
exceeds the unconstrained theoretical bound (a subset of a feasible set
can't beat the full set's optimum, on any input).

**Day 4 - Ledoit-Wolf shrinkage vs sample covariance
(`optimizer.shrinkage`), the centrepiece.** Shrinks the sample covariance
toward the scaled-identity target `mu*I` (`mu = trace(S)/n`) using the
closed-form Ledoit & Wolf (2004) asymptotically-optimal intensity, then
compares optimal weights under the sample covariance vs the shrunk one -
deliberately on the **global minimum-variance portfolio**, which depends
on the covariance matrix alone, not on the noisy mean vector Day 1 already
flagged, so the comparison isolates exactly one input. (The formula was
cross-checked during development against `sklearn.covariance.LedoitWolf`
on this repo's own fixtures and matched to float precision;
`scikit-learn` is not added as a runtime dependency for one formula, so
that check lives as a pinned regression value in `tests/test_shrinkage.py`
rather than an importable test dependency.)

The honest result on this fixture universe: shrinkage intensity comes out
small, **7.06%**, and the sample covariance was already reasonably
well-conditioned (condition number 3.23 -> 2.94 after shrinkage). The
minimum-variance weights move by single-digit percentage points
(RELIANCE.NS 57.75% -> 55.02%, TATACHEM.NS 20.95% -> 22.31%, CROMPTON.NS
21.30% -> 22.67%; L1 weight movement 5.46%, portfolio vol basically
unchanged at 17.87% -> 17.78%), and the long-only tangency portfolio does
not move at all (it is a 100%-RELIANCE.NS corner solution either way).
This is the correct outcome for this specific universe, not a weak
result glossed over: with only 3 assets and 500 days (T >> n), this is
exactly the regime where shrinkage has the least noise to fix - two
synthetic stress tests in `tests/test_shrinkage.py` confirm the formula
behaves as theory predicts at the extremes (shrinkage saturates above
99% when the true covariance genuinely is the scaled-identity target, and
falls from 36% to 0.4% as observations grow from 100 to 5,000 when the
true covariance has real, estimable off-diagonal structure), so the small
number here is a property of this 3-ticker universe, not evidence the
implementation doesn't work. The dramatic weight swings the hub's
`knowaboutit.md` describes as the headline of this day should be expected
once the universe grows toward the ~10-20 names later projects target,
where `n` gets materially closer to `T`.

**Day 5 - sector caps, position limits, and a turnover penalty
(`optimizer.constraints`).** Adds three constraints on top of Day 2's
minimum-variance-for-a-target-return problem: a per-sector weight cap
(inequality constraint), a per-asset position limit (a tighter bound than
plain long-only `w >= 0`), and a turnover penalty - a quadratic
`turnover_lambda * sum((w - w_prev) ** 2)` term added to the objective,
discouraging drift from a reference portfolio. Every solve reports the
constrained weights next to the unconstrained Day-2 baseline so the effect
of each constraint is visible, not just asserted.

Two honest findings, both surfaced by testing against the real fixtures
rather than glossed over:

1. **This 3-ticker universe can't demonstrate sector caps as a distinct
   constraint from position limits.** RELIANCE.NS, TATACHEM.NS and
   CROMPTON.NS sit in three different NSE sectors (Energy, Chemicals,
   Consumer Durables - `DEFAULT_SECTOR_MAP`), so capping every sector at X
   is mathematically identical to capping every position at X: at
   `--target-return -0.15`, `--max-position 0.6` and an equal 60% cap on
   all three sectors produce *the exact same weights*
   (RELIANCE.NS 60.00%, TATACHEM.NS 31.82%, CROMPTON.NS 8.18%, both runs
   binding only on RELIANCE.NS). Sector caps are genuinely exercised
   instead on a synthetic 4-asset/2-sector universe in
   `tests/test_constraints.py`, where a 40% sector cap forces two assets
   in the stronger sector down from a combined 42.9% to exactly 40.0% while
   redistributing into the other sector - a case this fixture universe
   cannot produce on its own. The CLI prints this degeneracy as a note
   whenever `--sector-cap` is used against the real 3-ticker universe
   rather than presenting the (real but unexercising) result as more than
   it is.

2. **The turnover penalty does not monotonically reduce L1 turnover**, and
   that surprised the implementation before it was checked against the
   real fixtures. `turnover_lambda * sum((w - w_prev) ** 2)` is a
   *quadratic* (L2) penalty, not the more standard L1 turnover cost
   (`sum(|w - w_prev|)`) - L1 is non-differentiable at zero, which SLSQP is
   not built for. Squared-L2 distance to the reference *is* provably
   non-increasing as `turnover_lambda` grows (a standard parametric-QP
   argument: summing the two optimality conditions for any lambda1 <
   lambda2 gives `dist(w_lambda1) >= dist(w_lambda2)` directly), and
   `tests/test_constraints.py` checks exactly that. But L1 distance is a
   different metric with a differently-shaped unit ball, and is *not*
   guaranteed to move the same way - concretely, at
   `--target-return -0.2226` (this universe's mean target) with reference
   weights `[20%, 30%, 50%]`, turning the penalty on (`--turnover-lambda
   50`) moves the solution from 29.50% L1 turnover at lambda=0 to **30.72%**
   at lambda=50 - slightly *farther* from the reference, not closer, even
   though the L2 distance did shrink as guaranteed. The CLI reports both
   the baseline's and the constrained solution's L1 turnover so this is
   visible rather than hidden behind a metric that always looks good.

Verified: 113/113 tests pass (28 new), including the synthetic sector-cap
and position-limit binding cases, the two guaranteed monotonicity
properties above, and validation errors (mismatched `--prev-weights`
length, unknown sector in `--sector-cap`, non-positive `--max-position`).
The CLI was run by hand with no constraints, with a binding position limit,
with a binding sector cap, and with the turnover penalty on and off.

**Day 6 - walk-forward out-of-sample test (`optimizer.oos`).** Every prior
day reported how a portfolio looks on the same window it was estimated
from - not a result, just curve-fitting, per `NEXT_STEPS.md`'s own "traps"
section. This module walks a fixed 252-day (~1 year) training window
forward through the fixture history in non-overlapping 21-day (~1 month)
steps: solve the long-only max-Sharpe tangency portfolio on Ledoit-Wolf
shrunk covariance (Day 4's centrepiece, and exactly the `max_sharpe_ledoit_wolf`
method the v0.4 contract to the spine names) using only data up to the end
of the training window, hold those weights fixed with no rebalancing
through the following test window, and score the return the optimizer
never saw when it picked the weights. Equal weight is held the same way
over the same test windows as the benchmark.

On the real 3-ticker universe this produces 11 non-overlapping folds
(2025-09-10 through 2026-08-12, 17 trailing days unused). The honest
result: **the optimizer edges out equal-weight, but not by a margin that
means anything.** Compounded return over the 11 folds was -16.73%
(optimizer) vs -17.33% (equal-weight); the optimizer won 6 of 11 folds
(54.5%); mean per-fold excess return was +0.13% with a standard deviation
of 5.69%, giving a t-stat of **+0.07** - nowhere close to distinguishable
from zero at conventional confidence. This is the expected outcome given
Day 1's finding that none of this universe's mean estimates clears |t| >
2: a walk-forward test built on noisy inputs should not, and does not,
show a reliable edge. A second, unglamorous finding: because this is a
3-asset long-only tangency solve, almost every fold's optimizer weights
are a 100% corner solution in a single name (RELIANCE.NS in 9 of 11
folds) - "optimization" here mostly means picking the best-looking single
name each period, not genuine diversification, a direct consequence of
having only 3 assets to allocate across.

**No NSE index/benchmark fixture is committed anywhere in this sandbox**
(no network access, zero spend, and none was ever fetched for this
pipeline) - `optimizer.oos` compares the optimizer against equal-weight
only, not against the index `NEXT_STEPS.md` also asks for. The CLI prints
this as a limitation on every run rather than silently dropping the index
leg of the comparison; see the Limitations section below.

Verified: 138/138 tests pass (25 new: fold-slicing edge cases, exact
buy-and-hold return reconciliation against manual price-ratio math, the
equal-weight fallback path for an infeasible tangency solve, and full
aggregation checks against the real fixtures and a synthetic two-fold
case). The CLI was run by hand at the default 252/21-day windows and
again at 200/50-day windows to confirm the fold count and date ranges
respond correctly to both.

**Day 7 - frontier chart, a weights table, and a pipeline audit
(`optimizer.report`, `optimizer.audit`).** `optimizer.report` computes
nothing new - it calls Day 2's `build_frontier` and Day 3's
`max_sharpe_weights_long_only`/`capital_market_line` and lays the same
numbers out as a one-page summary: a mean-vol chart (long-only frontier,
individual assets, the tangency point, and the capital market line,
written to `outputs/`) plus an aligned terminal table of the min-variance
and max-Sharpe weights. On the real 3-ticker universe both portfolios are
corner solutions again - the min-variance point is 56/21/22 across
RELIANCE.NS/TATACHEM.NS/CROMPTON.NS, but the long-only tangency portfolio
is 100% RELIANCE.NS, same single-name concentration Day 6's walk-forward
folds already showed - a direct consequence of having only 3 assets to
allocate across, not a new finding, just a second view of the same one.

`optimizer.audit` is the `ml-pipeline-audit` pass, scoped to the one
module in this repo with real look-ahead risk: `optimizer.oos`'s
walk-forward test. Every other day's module (`frontier`, `tangency`,
`shrinkage`, `constraints`) solves a single static window with no time
axis to leak across, so there is nothing to audit there; the walk-forward
test is the only place a silent bug could let a fold's decision see data
it would not have had yet. Two mechanical checks, not by-inspection ones:
(1) fold boundaries - every fold's train/test slice must abut exactly
with no gap or overlap, and no two folds' test windows may overlap (the
same day scored twice); (2) decision causality - a fold's chosen weights
on the full fixture must be unchanged when every trading day after that
fold's own test window is deleted. Both are true by construction today
given how `run_fold` slices its input, but "by construction" is exactly
the kind of claim a later refactor (an expanding window, a shared cache
keyed by ticker) could quietly break without any existing test catching
it - this checks the observable behaviour, not the code. On the real
3-ticker/11-fold walk-forward run: **clean**, no boundary or causality
violations. A negative-control test (`tests/test_audit.py`) confirms the
causality check actually catches a leaking training window when one is
injected, so a clean result here is not just "the check never fires."

Verified: 155/155 tests pass (17 new). Both CLIs were run by hand against
the real fixtures: `optimizer.report` wrote a real PNG to `outputs/` and
printed the weights table above; `optimizer.audit` printed and wrote a
clean `outputs/audit_<date>.md` report, exit code 0.

**Integration day - position sizing contract for the spine (`optimizer.sizing`,
v0.4).** Day 3/4/6/7 all found the same thing: this 3-asset universe's raw
long-only max-Sharpe tangency portfolio is a 100% RELIANCE.NS corner
solution, not a position size a real book would take. This module combines
Day 4's Ledoit-Wolf shrunk covariance (the method `NEXT_STEPS.md` committed
to, `max_sharpe_ledoit_wolf`) with Day 5's sector-cap constraint - reusing
both pieces verbatim, no new optimization theory - to solve the same
max-Sharpe objective subject to a uniform per-sector weight cap (50% by
default). On the real universe RELIANCE.NS's sector cap binds immediately
(it wants 100%, gets capped at 50%), and the freed-up 50% goes entirely to
TATACHEM.NS, not split across both remaining names - CROMPTON.NS gets 0%
even under the cap. A second, less flattering finding: the capped
portfolio's Sharpe ratio (-0.8251) is *worse* than the unconstrained one
(-0.3034) - forcing diversification here does not improve risk-adjusted
return, because both stand-in names' negative sample means (Day 1) make
"more names" mean "more exposure to noise", not less. Reported straight,
not smoothed into a diversification success story. The v0.4 contract this
repo ships to STOCKSTALKER is `{"sizing": {"weight": 0.5, "method":
"max_sharpe_ledoit_wolf", "constraint_binding": "Energy"}}` for RELIANCE.NS
- a real match, since RELIANCE.NS is one of STOCKSTALKER's own three
universe tickers, same as Day 4/6's monte-carlo-risk-lab and
fama-french-factor-model integrations. Because every ticker in this fixture
universe sits in its own sector (Day 5's finding, restated here), the
sector cap is mathematically identical to a plain position limit on
RELIANCE.NS alone - not exercised as a distinct multi-name constraint on
this universe, same degeneracy Day 5 already reported, though
`tests/test_sizing.py` covers a synthetic multi-sector universe where a
sector cap genuinely redistributes weight differently from a position
limit. Verified: 171/171 tests pass (16 new: the sector-capped solver
against a synthetic multi-sector case, the real-universe degenerate case,
and `to_contract()`'s shape/value/error paths), CLI run by hand at the
default 50% cap and confirmed byte-for-byte deterministic across five runs
(a real bug was caught and fixed here: building the per-sector cap dict
from a plain `set()` of sector names made the CLI's own column order
non-reproducible between runs, since Python randomizes string-hash-based
set iteration order per process - fixed by building it from `dict.fromkeys`
instead, which preserves insertion order).

**Day 9 - rerun on the large-cap universe (D4, hub CRITIQUES.md).** D4's
critique of Days 1-8: with only 3 assets, the long-only max-Sharpe tangency
solve is a 100% single-name corner in 9 of 11 walk-forward folds, and a
sector cap degenerates into a plain position limit - both artifacts of too
few names to choose from, not necessarily of the method. `scripts/fetch_largecaps.py`
(Day 8) widens the universe to 25-30 Nifty 50 large caps; this day reruns
`optimizer.stats`, `optimizer.shrinkage`, `optimizer.frontier` and
`optimizer.oos` on it once the fixture existed, and reports the (different)
honest result rather than assuming D4's hypothesis was confirmed.

The fixture landed as 29 names, not 30: TATAMOTORS.NS returns a genuine
yfinance 404 ("Quote not found for symbol"), not a rate limit - the same
error pattern as a Tata Motors Yahoo-side symbol change, not something a
retry fixes. `fetch_and_write`'s dict comprehension had no per-ticker error
isolation, so that one permanent failure silently aborted the whole
30-ticker batch on both of this workflow's first two runs - the Actions
workflow reported `success` while writing zero fixture files either time.
Fixed today: a ticker that still fails after `download_one`'s own retries
is now skipped and logged, not allowed to abort the other 29 (2 new tests,
`tests/test_fetch_largecaps.py`). The surviving universe is RELIANCE.NS,
TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, HINDUNILVR.NS, ITC.NS, SBIN.NS,
BHARTIARTL.NS, KOTAKBANK.NS, LT.NS, AXISBANK.NS, BAJFINANCE.NS,
ASIANPAINT.NS, MARUTI.NS, SUNPHARMA.NS, TITAN.NS, ULTRACEMCO.NS, WIPRO.NS,
NESTLEIND.NS, POWERGRID.NS, NTPC.NS, ONGC.NS, COALINDIA.NS, TATASTEEL.NS,
HCLTECH.NS, DRREDDY.NS, GRASIM.NS, HEROMOTOCO.NS - 2,910 trading days,
2015-01-02 to 2026-10-15, under `fixtures/ohlcv_largecap/`, a **separate
directory** from the original `fixtures/ohlcv/`.

A second, more serious bug surfaced the moment the fetch actually
succeeded: the script originally wrote straight into `fixtures/ohlcv/`,
the same directory Days 1-8's 3-ticker fixture lives in - and this
universe also includes RELIANCE.NS. The first successful run silently
overwrote `fixtures/ohlcv/RELIANCE_NS.csv` with this universe's longer
2015-2026 history, which no longer shared a trading-day calendar with the
untouched `TATACHEM_NS.csv`/`CROMPTON_NS.csv`, and broke every Day 1-8
test and the v0.4 sizing contract's own fixture with a calendar-mismatch
`ValueError` - caught by running the full suite after that commit (78 of
183 tests failed) rather than before pushing it further. Fixed by moving
all 29 large-cap fixtures to `fixtures/ohlcv_largecap/`, restoring the
original `fixtures/ohlcv/RELIANCE_NS.csv` from before the overwrite, and
pointing `scripts/fetch_largecaps.py`'s `FIXTURES_DIR` default and the
workflow's commit step at the new directory. 183/183 tests pass again
after the fix, and the numbers below were re-confirmed against the
relocated fixtures (`--fixtures-dir fixtures/ohlcv_largecap`) to make sure
the move itself changed nothing.

`optimizer.stats`: unlike the 3-ticker universe (every mean negative), all
29 annualized sample means are positive, from ITC.NS (+4.50%) to
BAJFINANCE.NS (+29.09%). Still mostly not distinguishable from zero at this
sample size (only BAJFINANCE.NS clears |t|=2, at 2.79) - the same
estimation-error problem Day 1 flagged, same conclusion, opposite sign. The
covariance condition number is 33.7 (vs 3.2 at 3 names - expected with
more assets estimated from the same window, not itself a red flag), and
the split-half check is less stable in relative terms (covariance moved
46.1% between halves, vs 19.1% at 3 names).

`optimizer.shrinkage`: intensity is smaller still at this size, 1.6% (vs
7.06% at 3 names) - 2,910 days is comfortably T>>n for 29 assets too, so
there is correspondingly little for shrinkage to fix, consistent with Day
4's own stress-test logic rather than contradicting it. The result Day 9
was actually run to see: the long-only tangency portfolio (Ledoit-Wolf
shrunk covariance, the v0.4 `max_sharpe_ledoit_wolf` method) is **no
longer a 100%-single-name corner** - it is a genuine 7-name mix
(RELIANCE.NS 6.05%, BHARTIARTL.NS 9.58%, BAJFINANCE.NS 20.58%, TITAN.NS
19.16%, NESTLEIND.NS 16.54%, POWERGRID.NS 16.97%, HCLTECH.NS 11.13%), with
Sharpe **+1.11** (vs -0.31 at 3 names) - the first positive-Sharpe
long-only tangency portfolio this repo has reported on any universe.

`optimizer.frontier` (target 12%): the long-only feasible range widens to
[+4.50%, +29.09%] (vs [-34.07%, -6.42%] at 3 names, where no long-only
portfolio could even target a positive return); all 25/25 swept target
points are feasible and the variance-vs-return sweep is still convex. The
minimum-variance point sits at +9.62% return / 13.33% vol, spread across
15+ of the 29 names rather than concentrated in 2-3.

`optimizer.oos` (same 252-day train / 21-day test windows as Day 6, now
126 folds instead of 11): D4's corner-solution finding is directly
addressed - only **1 of 126 folds** is a single-name solution now, down
from 9 of 11. That part of D4 is confirmed as an artifact of a too-small
universe, not of the max-Sharpe method itself. But the bottom-line result
flips, not improves: compounded return over all 126 folds is **+309.00%**
for the optimizer vs **+426.96%** for equal-weight - equal-weight wins
this time, where it narrowly lost at 3 names. Win rate drops to 51.6%
(from 54.5%), and the mean per-fold excess return's t-stat is **-0.50**
(vs +0.07 before) - still nowhere near significant at any conventional
confidence, same "no real signal" conclusion as Day 6, just with the sign
of which side nominally comes out ahead flipped by which 11-vs-126-fold
sample happened to be drawn. `optimizer.audit`'s two mechanical checks
(fold boundaries, causality) are clean on this 126-fold run, same as Day
7's 11-fold one.

**Honest conclusion:** widening the universe resolves the single-name
concentration D4 flagged, but does not surface a usable edge - equal-weight
is still a result this optimizer does not clear with any statistical
confidence, on either universe size. This rerun does not replace or
invalidate Days 1-8's 3-ticker findings, which remain the accurate record
for that universe and are what the v0.4 sizing contract to STOCKSTALKER
(whose own screen universe is RELIANCE.NS/TATACHEM.NS/CROMPTON.NS, not this
29-name one) is built from; this day is a separate, wider analysis that
answers D4's specific critique rather than supersedes the earlier one. The
same D8/Day 6 limitation still applies: no NSE index/benchmark fixture
exists in this sandbox, so this is optimizer-vs-equal-weight only, not the
optimizer-vs-index comparison `NEXT_STEPS.md` also asks for.

Verified: 183/183 tests pass (`PYTHONPATH=. pytest`; bare `pytest` still
can't collect this repo at all, pre-existing P17, unrelated, scoped to Day
13). `optimizer.stats`, `optimizer.shrinkage`, `optimizer.frontier`,
`optimizer.oos` and `optimizer.audit` were all run by hand against the real
29-name fixture at the numbers quoted above.

**Day 40 - cost-adjust the walk-forward, and a turnover-penalty escape
hatch (R9, hub CRITIQUES.md).** R9's critique of Days 6 and 9: `optimizer.oos`
charges no transaction cost anywhere, even though this portfolio's own
backtest (STOCKSTALKER, R8) already treats the same cost schedule as
decisive enough to flip a result. Measured directly from the committed
29-name run's fold-to-fold weights, the optimizer's turnover averages 57.6%
of the book every month - equal-weight's rebalance-back-to-1/n has nothing
to turn over in this walk-forward design (see below), so reporting the
gross number as "equal-weight wins" was actually generous to the optimizer,
not neutral.

`optimizer.oos` now costs each fold's L1 turnover from the *previous*
fold's realized weights at a blended rate derived from STOCKSTALKER's own
STT/stamp-duty/slippage schedule (0.10% STT + 0.015% stamp duty, buy leg
only + 10bps slippage, both legs → 0.2075% per unit of L1 turnover; see
`optimizer.oos`'s module docstring for the half-stamp-duty derivation - buy
and sell legs are each exactly half of L1 turnover because weights sum to 1
before and after). Equal-weight's own turnover is always exactly 0: this
walk-forward resets it to precisely 1/n every fold with no within-period
price drift tracked, so there is no drift to rebalance away from - a real
book would pay something to pull drifted weights back to 1/n, but this
model does not simulate that drift, and this is now stated plainly as a
limitation rather than fabricated as a number.

Re-run on the real 29-name large-cap fixture (same 252-day train/21-day
test windows as Day 9, `--fixtures-dir fixtures/ohlcv_largecap`):

| | gross (Day 9) | after costs (Day 40) |
|---|---|---|
| optimizer compounded return | +309.00% | **+252.20%** |
| equal-weight compounded return | +426.96% | +426.96% (cost ≈ 0) |
| mean per-fold excess t-stat | -0.50 | **-0.86** |

Costs widen equal-weight's margin, exactly as R9 expected rather than
narrowing it: the optimizer's average 57.6% monthly turnover sums to 15.06
points of per-fold cost rates over the 126 folds (equal-weight pays ~0),
but because each fold's cost compounds against a capital base already
reduced by every earlier fold's cost, the actual compounded-return effect
is far larger than that simple sum - +308.996% gross pulled down to
+252.204% net, a 56.79-point drop, not 15.06. The t-stat on the mean excess
return moves from -0.50 to -0.86 - still nowhere near conventional
significance, same "no real signal" conclusion, now on a realistic-cost
basis instead of a cost-free one. On the original 3-ticker
universe the effect is sharper: the optimizer's nominal pre-cost win
(-16.73% vs -17.33%, Day 6) flips sign once costed (-18.02% vs -17.33%),
because that universe's near-corner-solution tangency weights swing hard
fold to fold.

Day 5's own turnover-penalty machinery (`optimizer.constraints.constrained_min_variance`,
a quadratic penalty on squared distance from a reference portfolio) was
built but never wired into the walk-forward. `optimizer.oos --turnover-lambda`
now reuses it directly: each fold, if the plain tangency solve succeeds,
it re-solves for *the same expected return the tangency portfolio
targeted* but penalized for moving away from the previous fold's realized
weights, falling back to the plain tangency weights if that penalized
solve is infeasible (mirroring the existing equal-weight fallback for an
infeasible tangency solve). On the 29-name universe, `--turnover-lambda 5`
cuts realized average turnover from 57.6% to **27.4%** and the summed
per-fold cost rate from 15.06 to **7.17** points, narrowing (not closing)
the after-cost gap to equal-weight (+385.98% vs +426.96%, t=+0.13 on the
after-cost excess,
up from -0.86 unpenalized - the penalty also changes which weights are
held, so the gross number moves too, not just the cost: +422.06% gross at
lambda=5 vs +309.00% unpenalized). Equal-weight still wins after costs at
this lambda; Day 5's own caveat still applies here - the penalty is a quadratic
(L2) proxy for the non-differentiable L1 turnover cost that actually
matters, so a larger `--turnover-lambda` is not guaranteed to keep reducing
realized L1 turnover monotonically, only checked empirically at this one
setting.

Verified: 199/199 tests pass (15 new: the blended cost-rate derivation,
per-fold turnover/cost wiring against an explicit previous-weights vector,
the equal-weight leg's turnover staying exactly zero, the
`--turnover-lambda` requiring `tickers`, a spy test confirming
`constrained_min_variance` is actually called with the right
`turnover_lambda`, determinism, and the net-vs-gross aggregation) -
`uv run --with-editable . --with pytest pytest` (bare `pytest` on this
sandbox still can't import `numpy` at all, the same pre-existing local-only
quirk Day 13/LESSONS.md already documented, unrelated to this repo's own
packaging). `python -m optimizer.oos` run by hand on both the 3-ticker and
29-name fixtures, with and without `--turnover-lambda`, at the numbers
quoted above; both runs confirmed byte-for-byte deterministic across repeat
invocations.

**Day 43 - risk-free-rate and train/test-window sensitivity (R10, hub
CRITIQUES.md).** R10's critique of every earlier day: the risk-free rate
(`DEFAULT_RISK_FREE_RATE = 0.0`) and the 252/21-day train/test pairing
behind the Day 9/40 headline were both asserted once and never
sensitivity-swept, unlike every other assumption this README stress-tests
(shrinkage at the extremes - Day 4, split-half mean stability - Day 1,
universe size - Day 9). `optimizer.tangency` and `optimizer.oos` both gain
a `--risk-free-rate-sweep` flag (a comma-separated list of rates, reusing
the existing single-rate code path once per rate rather than approximating,
since `mu - rf` enters both the closed-form and the numerical solve
nonlinearly) that reruns the whole pipeline at each rate and reports
whether the qualitative sign holds.

*Risk-free rate.* On the original 3-ticker universe (every sample mean
negative, Day 1), the long-only tangency portfolio's Sharpe gets *more*
negative as the assumed rate rises - Sharpe -0.3094 at rf=0%, -0.5022 at
4%, -0.6468 at 7% - so the downward-sloping capital market line (Day 3) is
robust across every plausible rate, as expected once the mechanism is
understood (a positive rf only makes every excess return `mu - rf` more
negative here). On the 29-name universe (Day 9, mostly positive means) the
sign runs the other way but is equally robust: Sharpe +1.1007 at rf=0%,
+0.8737 at 4%, +0.7193 at 7% - upward-sloping at every rate, though the
magnitude nearly halves from 0% to 7%, which matters if this portfolio's
Sharpe is ever compared against another asset priced off a different rate
assumption. Feeding the same sweep through `optimizer.oos` on the 29-name
universe (default 252/21-day windows) shows the headline optimizer-vs-equal-weight
sign (Day 40: equal-weight wins after costs) is unchanged at every swept
rate - equal-weight's compounded return is identical at all three rates
(**+426.9614%**, since its 1/n weights never depend on rf at all), while
the optimizer's net compounded return rises with rf (**+252.20%** at 0% to
**+284.92%** at 7%, since a higher rf tilts every fold's tangency solve away
from the lowest-mean names) without ever closing the gap - the after-cost
t-stat moves from -0.86 to -0.56, narrowing but staying far short of
conventional significance.

*Train/test window pairing.* Day 9/40's headline numbers all come from one
pairing (252-day train, 21-day test). Three additional pairings on the same
29-name universe, cost-adjusted the same way as Day 40:

| train/test (days) | folds | optimizer (net) | equal-weight (net) | gap (pp) | t-stat (net) |
|---|---|---|---|---|---|
| 252/21 (Day 40, headline) | 126 | +252.20% | +426.96% | 174.8 | -0.86 |
| 126/21 | 132 | +116.32% | +382.37% | 266.1 | **-2.09** |
| 252/63 | 42 | +242.55% | +417.38% | 174.8 | -0.89 |
| 378/21 | 120 | +174.10% | +349.24% | 175.1 | -1.05 |

Equal-weight wins at every pairing tested - the sign is robust, same
conclusion as the risk-free-rate sweep. The gap's *magnitude* is also
fairly stable at three of the four pairings (~175 percentage points), which
is itself informative: lengthening the test window (63d) or the training
window (378d) barely moves the result. The one pairing that moves is
shortening the training window to 126 days, which widens the gap to 266pp
and is the only one of the four to cross conventional significance
(t=-2.09) - half the training data means noisier mean/covariance estimates
feeding the tangency solve each fold, which (per Day 40's turnover finding)
means more corner-solution weight swings, more realized turnover, and more
cost drag, not less. This is reported as a genuine finding about estimation
windows, not evidence the headline 252/21 pairing was cherry-picked to make
the optimizer look better than a shorter window would - if anything, 252/21
is a *more* favorable pairing for the optimizer than 126/21.

Verified: 211/211 tests pass (12 new: `sweep_risk_free_rates` in both
`optimizer.tangency` and `optimizer.oos` matches a plain per-rate `run()`
call exactly, passes `turnover_lambda` through, and the real-universe sign
checks above; CLI tests for `--risk-free-rate-sweep` on both modules,
including that it overrides a simultaneously-passed `--risk-free-rate` and
is deterministic). Both CLIs were run by hand at the rates and window pairs
quoted above against the real 3-ticker and 29-name fixtures.

## Limitations and what would make me wrong

- Mean-variance is famously sensitive to expected returns, which are estimated with large error. Small input changes move weights a lot. Day 1 measured this directly: all three tickers' annualized mean estimates have |t| < 2, i.e. none is statistically distinguishable from zero over this sample.
- Covariance estimated from a trailing window assumes a stability that does not survive regime changes. Day 1's split-half check found the covariance matrix moved 19% (relative Frobenius norm) and TATACHEM.NS's mean swung 25 points between the first and second half of the same 501-day window - the window itself is not obviously one regime.
- Days 1-8 ran on only 3 tickers (the shared Stock Stalker universe). Day 9 (D4) reran `stats`/`shrinkage`/`frontier`/`oos` on a 29-name large-cap universe once `scripts/fetch_largecaps.py`'s fixture landed - see the Findings section for that rerun. It is a separate, wider analysis, not a replacement: the v0.4 sizing contract STOCKSTALKER reads still comes from this repo's original 3-ticker fixture, because that is STOCKSTALKER's own screen universe.
- Day 6's walk-forward test found the optimizer nominally beats equal-weight on the 3-ticker universe (-16.73% vs -17.33% compounded over 11 out-of-sample folds, 54.5% fold win rate) but the mean per-fold excess return's t-stat is +0.07 - not distinguishable from zero at any conventional confidence. Day 9's 126-fold rerun on 29 names flips which side nominally wins (equal-weight +426.96% vs optimizer +309.00%, 51.6% win rate, t-stat -0.50) while reaching the same "not significant" conclusion - out-of-sample, equal-weight is a hard benchmark to beat on either universe size, and neither result should be read as the optimizer (or equal-weight) having a real edge.
- Day 6's 3-asset long-only max-Sharpe solve was a corner solution (100% in one name) in 9 of its 11 out-of-sample folds - "optimization" there was mostly picking the best-looking single name each period, not diversification, purely a consequence of having only 3 assets to choose from. Day 9 confirms this was indeed a too-small-universe artifact: on the 29-name rerun, only 1 of 126 folds is a single-name corner solution.
- There is no NSE index/benchmark fixture committed anywhere in this sandbox (no network access here, and none was ever fetched for this pipeline), so both Day 6's and Day 9's out-of-sample comparisons are optimizer-vs-equal-weight only - neither includes the vs-index leg `NEXT_STEPS.md` asks for. `optimizer.oos` states this on every run rather than substituting a fabricated benchmark.
- Day 9's large-cap fixture is 29 names, not the 30 `scripts/fetch_largecaps.py` targets: TATAMOTORS.NS returns a genuine yfinance 404 ("Quote not found for symbol"), not a transient rate limit, so it is skipped rather than retried into existing. `fetch_and_write` now treats a permanently-failing ticker as droppable rather than fatal to the whole batch - before this fix, that one ticker silently aborted the Actions workflow's fixture write on both of its first two runs (it reported `success` while writing zero files).
- Day 2's frontier ranks portfolios by the same sample mean vector Day 1 showed has |t| < 2 on every ticker. Concretely: all three means are negative, so every long-only frontier point here targets a *loss*, and a positive-return target is only reachable long/short, by shorting the least-negative name and going long the more-negative ones - an artifact of noisy point estimates as much as a real edge, not a strategy this repo is recommending.
- The long/short variant (`optimizer.frontier`, `optimizer.constraints --long-short`) still has no *leverage* limit - `--max-position` caps each name's magnitude but says nothing about gross exposure, so a long/short target far from the feasible long-only range can still imply large aggregate short exposure across several names even with every individual position capped.
- Day 5's sector caps are implemented and tested (a synthetic 4-asset/2-sector universe in `tests/test_constraints.py` shows one genuinely binding differently from a position limit), but on this repo's real 3-ticker universe every ticker is in its own sector, so a sector cap here is mathematically identical to a plain position limit - not a distinct constraint on this fixture set, and the CLI says so.
- Day 5's turnover penalty is quadratic (L2), not the more standard L1 turnover cost, because L1 is non-differentiable at zero and SLSQP needs a smooth objective. Squared-L2 distance to the reference portfolio provably shrinks as the penalty coefficient grows; L1 distance - the number anyone actually managing turnover cares about - does not always move the same way. A concrete real-fixture case moves L1 turnover from 29.50% to 30.72% when the penalty is turned on, i.e. slightly worse by the metric that matters, while L2 distance improved as promised. Report L1, not the penalty value, as "the turnover number" if this module is ever used for real.
- Day 3's unconstrained tangency portfolio does not exist as a finite full-investment portfolio for this universe (every asset's sample mean is negative, so `sum(inv(cov)(mu-rf))` is negative at any realistic risk-free rate) - `optimizer.tangency` reports this as undefined rather than returning a number. The long-only tangency portfolio is well-defined but has negative Sharpe (-0.31 at rf=0), so the capital market line built from it is downward-sloping: this universe offers no long-only allocation that beats holding cash on a risk-adjusted basis.
- The risk-free rate has no committed fixture and no live FRED/RBI fetch is available in this sandbox, so `optimizer.tangency` defaults `--risk-free-rate` to 0.0 rather than a fabricated "real" figure. Every Sharpe ratio and CML number above is only as meaningful as that assumption - pass a real rate via the flag if one is available. Day 43 swept 0/4/7% and found the qualitative CML-slope and optimizer-vs-equal-weight signs unchanged at every rate on both universes - but the magnitude moves (the 29-name tangency Sharpe nearly halves from rf=0% to rf=7%), so a rate sweep is still worth rerunning if a materially different real rate becomes available, not a one-time check.
- Day 9/40's headline walk-forward numbers come from one train/test window pairing (252/21 days). Day 43 reran three more pairings on the same 29-name universe and found equal-weight wins at every one, but not by the same margin: shortening the training window to 126 days (vs the headline 252) widens the after-cost gap from 175 to 266 percentage points and is the only pairing of the four that reaches conventional statistical significance (t=-2.09 vs -0.86 to -1.05 for the others) - a shorter estimation window means noisier inputs feeding the tangency solve, not a more favorable test for the optimizer. Four pairings is still a small, hand-picked sample of the pairing space, not an exhaustive grid search.
- Day 4's shrinkage comparison is genuine but undramatic on this specific universe: with only 3 assets and 500 days of history (T >> n), the sample covariance was already well-conditioned, so shrinkage intensity (7.1%) and the resulting weight moves are small. That is the honest result for this fixture set, not a demonstration that shrinkage doesn't matter - the synthetic stress tests in `tests/test_shrinkage.py` show the same formula shrinking to >99% or falling toward 0% at the extremes the theory predicts. A wider universe (Day 5+ and later projects target ~10-20 names) is where this day's centrepiece finding would actually bite.
- Day 7's `optimizer.audit` is scoped to `optimizer.oos` only - the one module in this repo with a time axis to leak across. It is not a general correctness audit of `frontier`/`tangency`/`shrinkage`/`constraints` (those are single-window static solves, checked instead by their own convexity/closed-form/monotonicity tests), and it says nothing about whether the walk-forward *result* is good, only that it is honestly out-of-sample. Day 7's frontier chart and weights table are a different view of numbers Days 2-3 already reported, not a new finding.
- The v0.4 sizing contract's sector cap is, on this repo's real 3-ticker universe, mathematically identical to a plain position limit on RELIANCE.NS alone (every ticker sits in its own sector, the same degeneracy Day 5 already found) - it does not exercise sector caps as a genuine multi-name diversification constraint here, only in `tests/test_sizing.py`'s synthetic multi-sector case. And the sector-capped portfolio's Sharpe ratio (-0.83) is worse, not better, than the unconstrained corner solution's (-0.30): forcing weight into TATACHEM.NS to satisfy the cap adds exposure to a second noisy, negative-mean estimate rather than diversifying away risk. The `weight: 0.5` this repo ships to STOCKSTALKER is an honest output of that constrained optimization, not a claim that 50/50 RELIANCE/TATACHEM is a good portfolio.
- Day 40's equal-weight leg always has zero modeled turnover cost: this walk-forward resets equal-weight to exactly 1/n every fold rather than tracking within-period price drift, so there is never a drifted position to rebalance back from. A real book holding equal-weight would pay something to pull drifted weights back to 1/n each period - smaller than the optimizer's turnover, per the original R9 critique, but not literally zero the way this model reports it. Simulating that drift would need a day-by-day (not fold-level) weight path, out of scope for this day's fix.
- Day 40's `--turnover-lambda` reuses Day 5's quadratic (L2) turnover penalty, which Day 5 already found does not monotonically reduce the L1 turnover anyone managing a real book actually cares about. It helped at the one setting tested here (`--turnover-lambda 5` on the 29-name universe: realized turnover 57.6% → 27.4%), but that is an empirical result at that lambda and universe, not a proof it always helps - check the realized L1 turnover printed by the CLI rather than assuming a larger lambda is always better.
- Day 40's transaction-cost schedule (STT + stamp duty + slippage, reused from STOCKSTALKER's R8) assumes every fold's rebalance executes at that flat percentage of traded value, the same simplification R8 already carries in STOCKSTALKER's own backtest - no bid-ask spread, no market-impact cost that would grow with the 57.6%-of-book average trade size, and no distinction between a liquid large-cap and a thinner name within the 29-name universe.
- **This repo does not independently verify whether its 29-name large-cap universe is survivorship-bias-free or point-in-time.** `scripts/fetch_largecaps.py`'s `DEFAULT_TICKERS` is today's (2026) Nifty 50 constituent list, explicitly picked for "a continuous NSE listing well before 2015" and replayed back to 2015-01-02 - a name that delisted, merged or dropped out of the index between 2015 and today would not appear here even if it was a real large-cap constituent for part of that window. No point-in-time Nifty 50 constituent history is available in this sandbox (zero spend, no network), so this is a documentation limitation, not a fixable one from here. Both the Day 9/40/43 optimizer and equal-weight legs run on this same 29-name list and inherit whatever survivorship bias, if any, that implies - in the direction that would make the past look easier than it was, since names that failed and vanished from today's index are the ones missing.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-10-17 | `ded4e49` | Day 9 (hub CRITIQUES.md D4): reran `optimizer.stats`, `optimizer.shrinkage`, `optimizer.frontier` and `optimizer.oos` on the 29-name large-cap universe Day 8's `scripts/fetch_largecaps.py` fetches, now that its Actions workflow had actually landed a fixture. Found and fixed, first: `fetch_and_write` had no per-ticker error isolation, so TATAMOTORS.NS's genuine yfinance 404 ("Quote not found") silently aborted the whole 30-ticker batch on both of the workflow's first two runs -- it reported `success` while writing zero fixture files either time. Fixed to skip a permanently-failing ticker instead (2 new tests); the fixture then landed as 29 names, 2,910 days (2015-01-02..2026-10-15) -- but into `fixtures/ohlcv/`, the same directory the original 3-ticker fixture lives in, and this universe includes RELIANCE.NS: the successful run silently overwrote `fixtures/ohlcv/RELIANCE_NS.csv` with a longer, differently-calendared history, breaking every Day 1-8 test with a calendar-mismatch error (78/183 tests failed). Fixed second: moved all 29 large-cap fixtures to a new `fixtures/ohlcv_largecap/` directory, restored the original `RELIANCE_NS.csv`, and repointed the script's default and the workflow's commit step there; 183/183 tests pass again, and the rerun below was re-confirmed against the relocated fixtures to prove the move changed nothing. D4's own hypothesis confirmed on one count, not the other: the single-name tangency corner solution drops from 9/11 folds to 1/126, but the bottom line flips rather than improves -- equal-weight now beats the optimizer (+426.96% vs +309.00% compounded over 126 folds, 51.6% win rate, t=-0.50), where it narrowly lost at 3 names (t=+0.07) -- same "not significant" conclusion either way. All other Day 1-8 findings and the v0.4 sizing contract (built from the original 3-ticker fixture, STOCKSTALKER's own screen universe) are untouched; this is a separate, wider rerun, not a replacement. Verified: 183/183 tests pass (`PYTHONPATH=. pytest`), all five CLIs (`stats`, `shrinkage`, `frontier`, `oos`, `audit`) run by hand against the real 29-name fixture. | Day 10 -- NSE option chains on real data (implied-vol-surface), per hub NEXT_STEPS.md |
| 2026-09-25 | `f098f44` | Integration day: shipped Stock Stalker v0.4 (optimizer-driven position sizing). optimizer.sizing combines Day 4's Ledoit-Wolf shrunk covariance with Day 5's sector-cap constraints to solve a sector-capped long-only max-Sharpe tangency portfolio, since Days 3/4/6/7 already found the raw unconstrained tangency is a 100% RELIANCE.NS corner solution -- not a position size any real book would take. --contract/--contract-ticker write the {sizing: {weight, method, constraint_binding}} JSON block as a file, never a Python import. On the spine side, STOCKSTALKER's new stockstalker.gates.sizing reads that file and attaches it to the one screen candidate whose ticker matches; report.py's new --sizing-contract/--sizing-ticker flags apply the gate and add two columns (Weight, Constraint) to the screen table when populated. Like v0.3's and v0.2's integrations, this is a genuine match: markowitz-portfolio-optimizer's fixtures come from STOCKSTALKER's own committed OHLCV history, so RELIANCE.NS is a real shared ticker. Honest result: RELIANCE.NS's sector cap (50%) binds immediately, and the freed weight goes entirely to TATACHEM.NS (CROMPTON.NS gets 0%) -- but the resulting Sharpe ratio (-0.83) is worse than the unconstrained corner solution's (-0.30), so forcing this diversification does not improve risk-adjusted return on this fixture set, recorded plainly rather than dressed up. Because every ticker in this 3-name universe sits in its own sector, the sector cap is mathematically identical to a plain position limit on RELIANCE.NS alone -- the same degeneracy Day 5 already found -- though a synthetic multi-sector test in tests/test_sizing.py exercises the distinct case. STOCKSTALKER bumped 0.7.2 -> 0.7.3, not 0.4.0, following the same out-of-pipeline-order version deviation v0.2's and v0.3's changelog entries already recorded. Verified honestly: 171/171 markowitz-portfolio-optimizer tests pass (16 new), 197/198 STOCKSTALKER tests pass (the one failure is the pre-existing live-network smoke test hitting a rate-limited yfinance response with a NaN close for today's date -- unrelated to this change), and both CLIs were run by hand end to end including the real RELIANCE.NS match. Also found and fixed both repos' and the hub's local main branches stuck in stale detached HEAD (the same recurring issue every prior integration day has hit) before committing. A real bug was caught and fixed during development: building the per-sector cap dict from a plain set() of sector names made optimizer.sizing's own CLI output non-deterministic between runs (Python randomizes string-hash-based set iteration order per process) -- fixed with dict.fromkeys instead. | headline-sentiment-vs-price Day 1 - RSS scraper respecting robots.txt, timestamped headlines |
| 2026-09-25 | `bdb3739` | Day 7: frontier chart + weights terminal table (optimizer.report) and the ml-pipeline-audit pass (optimizer.audit), scoped to optimizer.oos - the only module in this repo with a time axis to leak across. optimizer.report reuses Day 2/3's build_frontier and max_sharpe_weights_long_only/capital_market_line verbatim, adding no new math: a long-only mean-vol frontier chart (tangency point + CML) to outputs/, and an aligned terminal table of min-variance vs max-Sharpe weights - both corner solutions on the real 3-ticker universe (min-var 56/21/22 split, tangency 100% RELIANCE.NS), consistent with Day 6's single-name concentration finding. optimizer.audit checks two mechanical invariants of optimizer.oos's walk-forward folds rather than trusting the slicing by inspection: fold train/test boundaries have no gap or overlap and no two folds' test windows overlap; and a fold's chosen weights are unchanged when every trading day after that fold's own test window is deleted, i.e. the decision never depends on data the strategy would not have had yet. Both checks pass clean on the real 11-fold run; a negative-control test injects a leaking training window and confirms the causality check actually fires, so the clean result isn't just an unfired check. Verified: 155/155 tests pass (17 new), both CLIs run by hand against the real fixtures - report wrote a real PNG and printed the weights table, audit printed and wrote a clean outputs/audit_<date>.md at exit 0. | markowitz-portfolio-optimizer is now 7/7 days done - next up is the integration day: read this satellite's v0.4 sizing contract ({"sizing": {"weight": 0.08, "method": "max_sharpe_ledoit_wolf"}}) and ship the matching Stock Stalker release from projects.json's releases list (position sizing on STOCKSTALKER's screen output). |
| 2026-09-25 | `cb99616` | Day 6: walk-forward out-of-sample test (optimizer.oos) - fixed 252-day training window rolled forward through the fixture history in non-overlapping 21-day steps, solving the long-only max-Sharpe/Ledoit-Wolf tangency portfolio on each window and holding those weights (no rebalancing) through the following test window, scored against equal-weight held the same way. Honest result: the optimizer nominally beats equal-weight (-16.73% vs -17.33% compounded over 11 folds, 54.5% fold win rate) but mean per-fold excess return's t-stat is +0.07 - not distinguishable from zero, consistent with Day 1's finding that this universe's means carry no real signal. Second finding: with only 3 assets, the long-only tangency solve is a 100%-single-name corner solution in 9 of 11 folds, so this is mostly picking one name per period rather than diversifying. No NSE index/benchmark fixture exists anywhere in this sandbox (no network, zero spend), so this is optimizer-vs-equal-weight only, not the vs-index comparison NEXT_STEPS.md also asks for - stated on every CLI run and recorded in the README's limitations section rather than hidden. Verified: 138/138 tests pass (25 new), CLI run by hand at the default 252/21-day windows and again at 200/50-day windows. Also found and fixed both this repo's and the hub's local main branches stuck in stale detached HEAD (the same recurring issue every prior integration day has hit) before committing. | markowitz-portfolio-optimizer Day 7 - frontier chart to outputs/, weights as a terminal table, ml-pipeline-audit pass |
| 2026-09-24 | `4ad5b25` | Day 5: constraints on top of Day 2's minimum-variance frontier (optimizer.constraints) - a per-sector weight cap, a per-asset position limit, and a turnover penalty, each reported next to the unconstrained baseline. Two honest findings: (1) this 3-ticker universe (RELIANCE.NS/TATACHEM.NS/CROMPTON.NS) sits in three different NSE sectors, so a sector cap here is mathematically identical to a position limit - tested for real instead on a synthetic 4-asset/2-sector universe where a 40% sector cap genuinely redistributes weight differently from a position limit would; the CLI prints this degeneracy as a note on the real universe. (2) the turnover penalty is quadratic (L2), not the standard L1 turnover cost, because L1 is non-differentiable and SLSQP needs a smooth objective - squared-L2 distance to the reference portfolio is provably non-increasing as the penalty grows (proved via the standard parametric-QP argument and checked in tests), but L1 distance is not: a real-fixture case moves L1 turnover from 29.50% to 30.72% when the penalty is turned on, slightly worse by the metric that actually matters for a real book, recorded as a limitation rather than hidden. Verified: 113/113 tests pass (28 new: synthetic position-limit and sector-cap binding cases, both guaranteed monotonicity properties, and input validation), CLI run by hand with no constraints, a binding position limit, a binding sector cap, and the turnover penalty on/off. | markowitz-portfolio-optimizer Day 6 - out-of-sample test: optimize on window t, hold t+1, compare against equal-weight and the index, report it straight even if equal-weight wins |
| 2026-09-24 | `a206dfc` | Day 4: Ledoit-Wolf shrinkage vs sample covariance (optimizer.shrinkage), the centrepiece day. Implemented the closed-form Ledoit & Wolf (2004) shrinkage-to-scaled-identity estimator directly (no scikit-learn runtime dependency), validated during development against sklearn.covariance.LedoitWolf on this repo's own fixtures (matched to float precision, pinned as regression values in tests). Compared optimal weights under sample vs shrunk covariance on the global minimum-variance portfolio (covariance only, no mean vector - isolates the comparison from Day 1's already-flagged noisy mu), plus the long-only tangency portfolio for completeness. Honest result: on this 3-ticker/500-day universe shrinkage intensity is small (7.06%), the sample covariance was already fairly well-conditioned (cond 3.23 -> 2.94), and weights move by single-digit points (GMV) or not at all (tangency, a corner solution either way). Two synthetic stress tests confirm the formula behaves correctly at the extremes (>99% shrinkage when true covariance is the target, falling toward 0% as observations grow given real correlation structure) - the small number here is this universe's T>>n regime, not a bug; the dramatic swing knowaboutit.md describes should show up once the universe grows past ~10 names. Verified: 85/85 tests pass (21 new), CLI run by hand at rf=0.0 and rf=0.065. | markowitz-portfolio-optimizer Day 5 - constraints: sector caps, position limits, turnover penalty |
| 2026-09-23 | `3721bfc` | Day 3: maximum-Sharpe tangency portfolio and the capital market line (optimizer.tangency). Closed-form unconstrained weights (w = inv(cov)(mu-rf)/sum(...)) plus a numerically solved long-only variant via scipy SLSQP. Found and fixed a real bug before committing: the closed form only maximizes Sharpe when sum(inv(cov)(mu-rf)) > 0, and this universe's sample means are all negative (Day 1), so that sum is negative at rf=0 - dividing by it silently returns the Sharpe-minimizing portfolio dressed up as the answer. Caught by cross-checking against 200 random-restart SLSQP runs (which never converge in this branch, confirming no finite full-investment portfolio attains the theoretical Sharpe bound of 1.3462) and fixed to report success=False instead. The long-only tangency portfolio is well-defined (100% RELIANCE.NS, Sharpe -0.3094) but negative, so the capital market line built from it is downward-sloping - no long-only allocation from this universe beats cash on a risk-adjusted basis. No risk-free-rate fixture or live FRED/RBI fetch is available offline, so --risk-free-rate defaults to 0.0 as a stated assumption, documented in the README rather than a fabricated real rate. Verified: 64/64 tests pass (21 new), CLI run by hand at rf=0.0 and rf=0.065, both confirming identical qualitative findings. | markowitz-portfolio-optimizer Day 4 - Ledoit-Wolf shrinkage vs sample covariance, showing how far the weights swing between them |
| 2026-09-22 | `4b784d5` | Day 2: efficient frontier via scipy.optimize (SLSQP) - minimum-variance weights for a target return, long-only (w>=0) and long/short (unconstrained). Both NEXT_STEPS.md 'done when' gates are covered by tests: numerical weights match the closed-form 2-asset solution (2 equality constraints pin down 2 unknowns independent of covariance) on a synthetic pair and a real RELIANCE.NS/TATACHEM.NS fixture pair, and a 15-point sweep's variance is convex in target return (checked on the real 3-ticker fixture, plus a negative control that a concave sequence is correctly rejected). Honest finding surfaced immediately: all three Day 1 sample means are negative, so the entire long-only-feasible target-return range is [-34.07%, -6.42%] - no long-only portfolio from this universe can target positive expected return; long/short can only reach a positive target by shorting into >250% gross exposure. Recorded in the README rather than glossed over. Verified: 43/43 tests pass (18 new), CLI run by hand for a feasible interior target, an infeasible-long-only-but-feasible-long-short target, and an even-more-infeasible target requiring more shorting. | markowitz-portfolio-optimizer Day 3 - maximum-Sharpe tangency portfolio and the capital market line |
| 2026-09-22 | `205df86` | Day 1: annualized mean/covariance estimation (optimizer.returns, optimizer.covariance) plus an estimation-error report CLI (optimizer.stats) against the shared 3-ticker Stock Stalker OHLCV fixtures (RELIANCE/TATACHEM/CROMPTON). Reused STOCKSTALKER's committed fixtures rather than synthetic ones, so a future v0.4 contract can match a real ticker. Findings are genuinely unflattering: all three annualized mean-return estimates have |t| < 2 (none distinguishable from zero at this sample size), and a first-half/second-half split shows the covariance matrix moving 19% (relative Frobenius norm) and TATACHEM.NS's mean swinging 25pp between windows -- recorded in the README as the reason later days' frontier weights should not be read as precise. Verified honestly: 25/25 new tests pass, and optimizer.stats was run by hand against the fixtures (no live network here, fixture path is the default). Also fixed this repo's and the hub's local main branches stuck in stale detached HEAD (the same recurring issue every prior integration day has hit) before committing. | markowitz-portfolio-optimizer Day 2 - efficient frontier via scipy.optimize (minimum variance for a target return, long-only and long/short variants) |
<!-- CHECKPOINTS:END -->

## Where this sits

Part of a nine-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back:

```json
{
  "sizing": {
    "weight": 0.5,
    "method": "max_sharpe_ledoit_wolf",
    "constraint_binding": "Energy"
  }
}
```

Communication is by file contract, not imports, so either side can be refactored without breaking the other.

## Exam mapping

Series XV ch.12 (risk and return), ch.12.9 (risk-adjusted returns)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.
