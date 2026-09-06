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
python -m optimizer.frontier --tickers TICKER1.NS,TICKER2.NS --target-return 0.12
```

Runs offline against committed fixtures by default. Live data needs a key in `.env` (see `.env.example`); the fixture path is the default so nothing blocks on network access.

## Findings

Nothing yet. This section fills in as the work lands, including the results that do not flatter the method.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Mean-variance is famously sensitive to expected returns, which are estimated with large error. Small input changes move weights a lot.
- Covariance estimated from a trailing window assumes a stability that does not survive regime changes.
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
