# Markowitz portfolio optimizer

Mean-variance optimization over an NSE universe, with the fragility of the inputs treated as the main finding rather than a footnote.

**Status:** Not started · Next: Day 1 - returns and covariance estimation with estimation-error note

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

`optimizer.stats` (Day 1) is the returns/covariance diagnostic below; `optimizer.frontier` is the Day 2+ optimizer.

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

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Mean-variance is famously sensitive to expected returns, which are estimated with large error. Small input changes move weights a lot. Day 1 measured this directly: all three tickers' annualized mean estimates have |t| < 2, i.e. none is statistically distinguishable from zero over this sample.
- Covariance estimated from a trailing window assumes a stability that does not survive regime changes. Day 1's split-half check found the covariance matrix moved 19% (relative Frobenius norm) and TATACHEM.NS's mean swung 25 points between the first and second half of the same 501-day window - the window itself is not obviously one regime.
- Only 3 tickers so far (the shared Stock Stalker universe). A 3-asset covariance matrix's condition number (3.2) says little about how the estimator will behave once the universe grows to the ~10-20 names later days will need.
- Out-of-sample, equal-weight is a hard benchmark to beat. Where it wins, the README says so.

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
