# Markowitz portfolio optimizer

Mean-variance optimization over an NSE universe, with the fragility of the inputs treated as the main finding rather than a footnote.

**Status:** Last checkpoint 2026-09-22 · Next: markowitz-portfolio-optimizer Day 3 - maximum-Sharpe tangency portfolio and the capital market line

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
```

`optimizer.stats` (Day 1) is the returns/covariance diagnostic below; `optimizer.frontier` (Day 2) solves the minimum-variance portfolio for one target return, long-only and long/short.

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

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-22 | `4b784d5` | Day 2: efficient frontier via scipy.optimize (SLSQP) - minimum-variance weights for a target return, long-only (w>=0) and long/short (unconstrained). Both NEXT_STEPS.md 'done when' gates are covered by tests: numerical weights match the closed-form 2-asset solution (2 equality constraints pin down 2 unknowns independent of covariance) on a synthetic pair and a real RELIANCE.NS/TATACHEM.NS fixture pair, and a 15-point sweep's variance is convex in target return (checked on the real 3-ticker fixture, plus a negative control that a concave sequence is correctly rejected). Honest finding surfaced immediately: all three Day 1 sample means are negative, so the entire long-only-feasible target-return range is [-34.07%, -6.42%] - no long-only portfolio from this universe can target positive expected return; long/short can only reach a positive target by shorting into >250% gross exposure. Recorded in the README rather than glossed over. Verified: 43/43 tests pass (18 new), CLI run by hand for a feasible interior target, an infeasible-long-only-but-feasible-long-short target, and an even-more-infeasible target requiring more shorting. | markowitz-portfolio-optimizer Day 3 - maximum-Sharpe tangency portfolio and the capital market line |
| 2026-09-22 | `205df86` | Day 1: annualized mean/covariance estimation (optimizer.returns, optimizer.covariance) plus an estimation-error report CLI (optimizer.stats) against the shared 3-ticker Stock Stalker OHLCV fixtures (RELIANCE/TATACHEM/CROMPTON). Reused STOCKSTALKER's committed fixtures rather than synthetic ones, so a future v0.4 contract can match a real ticker. Findings are genuinely unflattering: all three annualized mean-return estimates have |t| < 2 (none distinguishable from zero at this sample size), and a first-half/second-half split shows the covariance matrix moving 19% (relative Frobenius norm) and TATACHEM.NS's mean swinging 25pp between windows -- recorded in the README as the reason later days' frontier weights should not be read as precise. Verified honestly: 25/25 new tests pass, and optimizer.stats was run by hand against the fixtures (no live network here, fixture path is the default). Also fixed this repo's and the hub's local main branches stuck in stale detached HEAD (the same recurring issue every prior integration day has hit) before committing. | markowitz-portfolio-optimizer Day 2 - efficient frontier via scipy.optimize (minimum variance for a target return, long-only and long/short variants) |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Mean-variance is famously sensitive to expected returns, which are estimated with large error. Small input changes move weights a lot. Day 1 measured this directly: all three tickers' annualized mean estimates have |t| < 2, i.e. none is statistically distinguishable from zero over this sample.
- Covariance estimated from a trailing window assumes a stability that does not survive regime changes. Day 1's split-half check found the covariance matrix moved 19% (relative Frobenius norm) and TATACHEM.NS's mean swung 25 points between the first and second half of the same 501-day window - the window itself is not obviously one regime.
- Only 3 tickers so far (the shared Stock Stalker universe). A 3-asset covariance matrix's condition number (3.2) says little about how the estimator will behave once the universe grows to the ~10-20 names later days will need.
- Out-of-sample, equal-weight is a hard benchmark to beat. Where it wins, the README says so.
- Day 2's frontier ranks portfolios by the same sample mean vector Day 1 showed has |t| < 2 on every ticker. Concretely: all three means are negative, so every long-only frontier point here targets a *loss*, and a positive-return target is only reachable long/short, by shorting the least-negative name and going long the more-negative ones - an artifact of noisy point estimates as much as a real edge, not a strategy this repo is recommending.
- The long/short variant has no leverage or position-size limit yet (Day 5 adds constraints), so its weights for extreme target returns (e.g. +10%) imply >250% gross exposure - directionally correct for what unconstrained mean-variance does, but not a portfolio anyone should actually hold.

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
