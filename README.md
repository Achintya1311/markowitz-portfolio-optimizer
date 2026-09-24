# Markowitz portfolio optimizer

Mean-variance optimization over an NSE universe, with the fragility of the inputs treated as the main finding rather than a footnote.

**Status:** Last checkpoint 2026-09-24 · Next: markowitz-portfolio-optimizer Day 5 - constraints: sector caps, position limits, turnover penalty

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

`optimizer.stats` (Day 1) is the returns/covariance diagnostic below; `optimizer.frontier` (Day 2) solves the minimum-variance portfolio for one target return, long-only and long/short; `optimizer.tangency` (Day 3) solves the maximum-Sharpe tangency portfolio and reports the capital market line. `--risk-free-rate` defaults to `0.0` - there is no committed risk-free-rate fixture and no live FRED/RBI fetch available offline, so pass a real annualized rate explicitly if you have one.

Runs offline against committed fixtures by default. Live data needs a key in `.env` (see `.env.example`); the fixture path is the default so nothing blocks on network access.

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

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-24 | `a206dfc` | Day 4: Ledoit-Wolf shrinkage vs sample covariance (optimizer.shrinkage), the centrepiece day. Implemented the closed-form Ledoit & Wolf (2004) shrinkage-to-scaled-identity estimator directly (no scikit-learn runtime dependency), validated during development against sklearn.covariance.LedoitWolf on this repo's own fixtures (matched to float precision, pinned as regression values in tests). Compared optimal weights under sample vs shrunk covariance on the global minimum-variance portfolio (covariance only, no mean vector - isolates the comparison from Day 1's already-flagged noisy mu), plus the long-only tangency portfolio for completeness. Honest result: on this 3-ticker/500-day universe shrinkage intensity is small (7.06%), the sample covariance was already fairly well-conditioned (cond 3.23 -> 2.94), and weights move by single-digit points (GMV) or not at all (tangency, a corner solution either way). Two synthetic stress tests confirm the formula behaves correctly at the extremes (>99% shrinkage when true covariance is the target, falling toward 0% as observations grow given real correlation structure) - the small number here is this universe's T>>n regime, not a bug; the dramatic swing knowaboutit.md describes should show up once the universe grows past ~10 names. Verified: 85/85 tests pass (21 new), CLI run by hand at rf=0.0 and rf=0.065. | markowitz-portfolio-optimizer Day 5 - constraints: sector caps, position limits, turnover penalty |
| 2026-09-23 | `3721bfc` | Day 3: maximum-Sharpe tangency portfolio and the capital market line (optimizer.tangency). Closed-form unconstrained weights (w = inv(cov)(mu-rf)/sum(...)) plus a numerically solved long-only variant via scipy SLSQP. Found and fixed a real bug before committing: the closed form only maximizes Sharpe when sum(inv(cov)(mu-rf)) > 0, and this universe's sample means are all negative (Day 1), so that sum is negative at rf=0 - dividing by it silently returns the Sharpe-minimizing portfolio dressed up as the answer. Caught by cross-checking against 200 random-restart SLSQP runs (which never converge in this branch, confirming no finite full-investment portfolio attains the theoretical Sharpe bound of 1.3462) and fixed to report success=False instead. The long-only tangency portfolio is well-defined (100% RELIANCE.NS, Sharpe -0.3094) but negative, so the capital market line built from it is downward-sloping - no long-only allocation from this universe beats cash on a risk-adjusted basis. No risk-free-rate fixture or live FRED/RBI fetch is available offline, so --risk-free-rate defaults to 0.0 as a stated assumption, documented in the README rather than a fabricated real rate. Verified: 64/64 tests pass (21 new), CLI run by hand at rf=0.0 and rf=0.065, both confirming identical qualitative findings. | markowitz-portfolio-optimizer Day 4 - Ledoit-Wolf shrinkage vs sample covariance, showing how far the weights swing between them |
| 2026-09-22 | `4b784d5` | Day 2: efficient frontier via scipy.optimize (SLSQP) - minimum-variance weights for a target return, long-only (w>=0) and long/short (unconstrained). Both NEXT_STEPS.md 'done when' gates are covered by tests: numerical weights match the closed-form 2-asset solution (2 equality constraints pin down 2 unknowns independent of covariance) on a synthetic pair and a real RELIANCE.NS/TATACHEM.NS fixture pair, and a 15-point sweep's variance is convex in target return (checked on the real 3-ticker fixture, plus a negative control that a concave sequence is correctly rejected). Honest finding surfaced immediately: all three Day 1 sample means are negative, so the entire long-only-feasible target-return range is [-34.07%, -6.42%] - no long-only portfolio from this universe can target positive expected return; long/short can only reach a positive target by shorting into >250% gross exposure. Recorded in the README rather than glossed over. Verified: 43/43 tests pass (18 new), CLI run by hand for a feasible interior target, an infeasible-long-only-but-feasible-long-short target, and an even-more-infeasible target requiring more shorting. | markowitz-portfolio-optimizer Day 3 - maximum-Sharpe tangency portfolio and the capital market line |
| 2026-09-22 | `205df86` | Day 1: annualized mean/covariance estimation (optimizer.returns, optimizer.covariance) plus an estimation-error report CLI (optimizer.stats) against the shared 3-ticker Stock Stalker OHLCV fixtures (RELIANCE/TATACHEM/CROMPTON). Reused STOCKSTALKER's committed fixtures rather than synthetic ones, so a future v0.4 contract can match a real ticker. Findings are genuinely unflattering: all three annualized mean-return estimates have |t| < 2 (none distinguishable from zero at this sample size), and a first-half/second-half split shows the covariance matrix moving 19% (relative Frobenius norm) and TATACHEM.NS's mean swinging 25pp between windows -- recorded in the README as the reason later days' frontier weights should not be read as precise. Verified honestly: 25/25 new tests pass, and optimizer.stats was run by hand against the fixtures (no live network here, fixture path is the default). Also fixed this repo's and the hub's local main branches stuck in stale detached HEAD (the same recurring issue every prior integration day has hit) before committing. | markowitz-portfolio-optimizer Day 2 - efficient frontier via scipy.optimize (minimum variance for a target return, long-only and long/short variants) |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Mean-variance is famously sensitive to expected returns, which are estimated with large error. Small input changes move weights a lot. Day 1 measured this directly: all three tickers' annualized mean estimates have |t| < 2, i.e. none is statistically distinguishable from zero over this sample.
- Covariance estimated from a trailing window assumes a stability that does not survive regime changes. Day 1's split-half check found the covariance matrix moved 19% (relative Frobenius norm) and TATACHEM.NS's mean swung 25 points between the first and second half of the same 501-day window - the window itself is not obviously one regime.
- Only 3 tickers so far (the shared Stock Stalker universe). A 3-asset covariance matrix's condition number (3.2) says little about how the estimator will behave once the universe grows to the ~10-20 names later days will need.
- Out-of-sample, equal-weight is a hard benchmark to beat. Where it wins, the README says so.
- Day 2's frontier ranks portfolios by the same sample mean vector Day 1 showed has |t| < 2 on every ticker. Concretely: all three means are negative, so every long-only frontier point here targets a *loss*, and a positive-return target is only reachable long/short, by shorting the least-negative name and going long the more-negative ones - an artifact of noisy point estimates as much as a real edge, not a strategy this repo is recommending.
- The long/short variant (`optimizer.frontier`, `optimizer.constraints --long-short`) still has no *leverage* limit - `--max-position` caps each name's magnitude but says nothing about gross exposure, so a long/short target far from the feasible long-only range can still imply large aggregate short exposure across several names even with every individual position capped.
- Day 5's sector caps are implemented and tested (a synthetic 4-asset/2-sector universe in `tests/test_constraints.py` shows one genuinely binding differently from a position limit), but on this repo's real 3-ticker universe every ticker is in its own sector, so a sector cap here is mathematically identical to a plain position limit - not a distinct constraint on this fixture set, and the CLI says so.
- Day 5's turnover penalty is quadratic (L2), not the more standard L1 turnover cost, because L1 is non-differentiable at zero and SLSQP needs a smooth objective. Squared-L2 distance to the reference portfolio provably shrinks as the penalty coefficient grows; L1 distance - the number anyone actually managing turnover cares about - does not always move the same way. A concrete real-fixture case moves L1 turnover from 29.50% to 30.72% when the penalty is turned on, i.e. slightly worse by the metric that matters, while L2 distance improved as promised. Report L1, not the penalty value, as "the turnover number" if this module is ever used for real.
- Day 3's unconstrained tangency portfolio does not exist as a finite full-investment portfolio for this universe (every asset's sample mean is negative, so `sum(inv(cov)(mu-rf))` is negative at any realistic risk-free rate) - `optimizer.tangency` reports this as undefined rather than returning a number. The long-only tangency portfolio is well-defined but has negative Sharpe (-0.31 at rf=0), so the capital market line built from it is downward-sloping: this universe offers no long-only allocation that beats holding cash on a risk-adjusted basis.
- The risk-free rate has no committed fixture and no live FRED/RBI fetch is available in this sandbox, so `optimizer.tangency` defaults `--risk-free-rate` to 0.0 rather than a fabricated "real" figure. Every Sharpe ratio and CML number above is only as meaningful as that assumption - pass a real rate via the flag if one is available.
- Day 4's shrinkage comparison is genuine but undramatic on this specific universe: with only 3 assets and 500 days of history (T >> n), the sample covariance was already well-conditioned, so shrinkage intensity (7.1%) and the resulting weight moves are small. That is the honest result for this fixture set, not a demonstration that shrinkage doesn't matter - the synthetic stress tests in `tests/test_shrinkage.py` show the same formula shrinking to >99% or falling toward 0% at the extremes the theory predicts. A wider universe (Day 5+ and later projects target ~10-20 names) is where this day's centrepiece finding would actually bite.

## Where this sits

Part of a nine-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back:

```json
{
  "sizing": {
    "weight": 0.08,
    "method": "max_sharpe_ledoit_wolf"
  }
}
```

Communication is by file contract, not imports, so either side can be refactored without breaking the other.

## Exam mapping

Series XV ch.12 (risk and return), ch.12.9 (risk-adjusted returns)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.
