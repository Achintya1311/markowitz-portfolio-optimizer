"""Walk-forward out-of-sample test: optimize on window t, hold t+1 (Day 6).

    python -m optimizer.oos --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS \\
        --train-window 252 --test-window 21

Every earlier day in this project reports how a portfolio looks on the same
window it was estimated from. That is not a result - it is curve-fitting
with extra steps, and NEXT_STEPS.md's own "traps" section says so. This
module instead walks a fixed-length training window forward through the
fixture history in non-overlapping steps: solve the long-only maximum-
Sharpe tangency portfolio (:func:`optimizer.tangency.max_sharpe_weights_long_only`)
on Ledoit-Wolf shrunk covariance (Day 4's centrepiece, and the method the
v0.4 contract to the spine names: ``max_sharpe_ledoit_wolf``) using only
data up to the end of the training window, then hold those weights fixed
(no rebalancing) through the following test window and see what they
actually returned - numbers the optimizer never saw when it picked the
weights.

The benchmark is equal weight, held the same way over the same test
windows. NEXT_STEPS.md asks for a comparison against equal-weight *and*
the index; there is no committed NSE index fixture anywhere in this
sandbox (no network, zero spend, and no benchmark series was ever fetched
for this pipeline) and this module does not fabricate one, so the index
leg of that comparison is reported here as a limitation, not silently
dropped - see the README.

Equal-weight is a genuinely hard benchmark for a short, noisy sample to
beat, and this project's Day 1 already established that the mean vector
feeding the optimizer is not distinguishable from zero at conventional
confidence. Whichever side wins here is reported as found, per the hub's
ground rules on honest results.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from optimizer.covariance import sample_mean_cov
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix
from optimizer.shrinkage import ledoit_wolf_shrinkage
from optimizer.tangency import DEFAULT_RISK_FREE_RATE, max_sharpe_weights_long_only

DEFAULT_TRAIN_WINDOW = 252
DEFAULT_TEST_WINDOW = 21


@dataclass
class FoldResult:
    """One walk-forward step: a training window's weights, held over the next test window."""

    fold: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    shrinkage: float
    optimizer_weights: np.ndarray
    optimizer_success: bool
    used_fallback: bool
    optimizer_return: float
    equal_weight_return: float

    @property
    def excess_return(self) -> float:
        return self.optimizer_return - self.equal_weight_return


@dataclass
class OOSResult:
    """Aggregate walk-forward result across all folds."""

    tickers: list[str]
    train_window: int
    test_window: int
    risk_free: float
    folds: list[FoldResult]
    unused_days: int
    optimizer_compounded_return: float = field(init=False)
    equal_weight_compounded_return: float = field(init=False)
    win_rate: float = field(init=False)
    mean_excess: float = field(init=False)
    std_excess: float = field(init=False)
    t_stat_excess: float | None = field(init=False)
    n_fallback: int = field(init=False)

    def __post_init__(self):
        opt_returns = np.array([f.optimizer_return for f in self.folds])
        eq_returns = np.array([f.equal_weight_return for f in self.folds])
        excess = opt_returns - eq_returns

        self.optimizer_compounded_return = float(np.prod(1.0 + opt_returns) - 1.0)
        self.equal_weight_compounded_return = float(np.prod(1.0 + eq_returns) - 1.0)
        self.win_rate = float(np.mean(excess > 0)) if len(excess) else 0.0
        self.mean_excess = float(np.mean(excess)) if len(excess) else 0.0
        self.std_excess = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
        if len(excess) > 1 and self.std_excess > 0:
            self.t_stat_excess = self.mean_excess / (self.std_excess / np.sqrt(len(excess)))
        else:
            self.t_stat_excess = None
        self.n_fallback = sum(1 for f in self.folds if f.used_fallback)


def walk_forward_folds(n_days: int, train_window: int, test_window: int) -> list[tuple[slice, slice]]:
    """Non-overlapping (train, test) row slices, stepping forward by ``test_window``.

    The training window has fixed length and slides forward each fold (a
    rolling window, not an expanding one), so every fold's optimizer sees
    the same amount of history - comparable estimation noise fold to fold,
    at the cost of forgetting data older than ``train_window`` days. Folds
    that would run past the end of the data are dropped rather than
    truncated, so every reported fold has a full test window to score
    against.
    """
    if train_window < 2:
        raise ValueError("train_window must be at least 2")
    if test_window < 1:
        raise ValueError("test_window must be at least 1")

    folds = []
    start = 0
    while start + train_window + test_window <= n_days:
        folds.append((slice(start, start + train_window), slice(start + train_window, start + train_window + test_window)))
        start += test_window
    return folds


def total_simple_returns(log_returns_window: np.ndarray) -> np.ndarray:
    """Per-asset total simple return over a window of daily log returns.

    ``exp(sum(log returns)) - 1`` is the exact compounded price return over
    the window - what a buy-and-hold position in that asset actually earned,
    unlike a sum of simple returns.
    """
    return np.exp(log_returns_window.sum(axis=0)) - 1.0


def run_fold(
    returns_matrix: np.ndarray,
    dates: list[str],
    train_slice: slice,
    test_slice: slice,
    fold_index: int,
    risk_free: float,
) -> FoldResult:
    n_assets = returns_matrix.shape[1]
    train_returns = returns_matrix[train_slice]

    mu_daily, _ = sample_mean_cov(train_returns)
    shrinkage, cov_shrunk_daily = ledoit_wolf_shrinkage(train_returns)
    mu = annualize_mean(mu_daily)
    cov_shrunk = annualize_cov(cov_shrunk_daily)

    tangency = max_sharpe_weights_long_only(mu, cov_shrunk, risk_free)
    used_fallback = not tangency.success
    weights = tangency.weights if tangency.success else np.full(n_assets, 1.0 / n_assets)

    test_returns = total_simple_returns(returns_matrix[test_slice])
    equal_weight = np.full(n_assets, 1.0 / n_assets)

    return FoldResult(
        fold=fold_index,
        train_start_date=dates[train_slice][0],
        train_end_date=dates[train_slice][-1],
        test_start_date=dates[test_slice][0],
        test_end_date=dates[test_slice][-1],
        shrinkage=shrinkage,
        optimizer_weights=weights,
        optimizer_success=tangency.success,
        used_fallback=used_fallback,
        optimizer_return=float(weights @ test_returns),
        equal_weight_return=float(equal_weight @ test_returns),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward out-of-sample test: optimize on window t, hold t+1, vs equal-weight (Day 6)."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--train-window", type=int, default=DEFAULT_TRAIN_WINDOW,
        help=f"trading days in each estimation window (default: {DEFAULT_TRAIN_WINDOW})",
    )
    parser.add_argument(
        "--test-window", type=int, default=DEFAULT_TEST_WINDOW,
        help=f"trading days each fold's weights are held for (default: {DEFAULT_TEST_WINDOW})",
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE,
        help="annualized risk-free rate for the tangency portfolio (default: 0.0, see optimizer.tangency)",
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    return parser


def run(tickers: list[str], fixtures_dir: str, train_window: int, test_window: int, risk_free: float) -> OOSResult:
    dates, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    n_days = returns_matrix.shape[0]

    fold_slices = walk_forward_folds(n_days, train_window, test_window)
    if not fold_slices:
        raise ValueError(
            f"only {n_days} days of returns available, not enough for one "
            f"train_window={train_window} + test_window={test_window} fold"
        )

    folds = [
        run_fold(returns_matrix, dates, train_slice, test_slice, i, risk_free)
        for i, (train_slice, test_slice) in enumerate(fold_slices)
    ]
    used_days = fold_slices[-1][1].stop
    return OOSResult(
        tickers=tickers,
        train_window=train_window,
        test_window=test_window,
        risk_free=risk_free,
        folds=folds,
        unused_days=n_days - used_days,
    )


def _format_report(result: OOSResult) -> str:
    lines = []
    lines.append(f"tickers: {', '.join(result.tickers)}")
    lines.append(
        f"walk-forward: train_window={result.train_window}d   test_window={result.test_window}d   "
        f"folds={len(result.folds)}   unused trailing days={result.unused_days}"
    )
    lines.append("method: max-Sharpe tangency portfolio, long-only, on Ledoit-Wolf shrunk covariance (v0.4's max_sharpe_ledoit_wolf)")
    lines.append("benchmark: equal weight, held the same way over the same test windows")
    lines.append("")

    lines.append(f"{'fold':>4}  {'test window':<23}  {'optimizer':>10}  {'equal-wt':>10}  {'excess':>8}  weights")
    for f in result.folds:
        weight_str = ", ".join(f"{t}={w:+.3f}" for t, w in zip(result.tickers, f.optimizer_weights))
        flag = " (fallback: equal-weight, optimizer infeasible)" if f.used_fallback else ""
        lines.append(
            f"{f.fold:>4}  {f.test_start_date}..{f.test_end_date}  {f.optimizer_return:>+10.4%}  "
            f"{f.equal_weight_return:>+10.4%}  {f.excess_return:>+8.4%}  {weight_str}{flag}"
        )
    lines.append("")

    lines.append(f"compounded return over all {len(result.folds)} test windows:")
    lines.append(f"  optimizer:    {result.optimizer_compounded_return:+.4%}")
    lines.append(f"  equal-weight: {result.equal_weight_compounded_return:+.4%}")
    lines.append("")
    lines.append(f"win rate (optimizer beats equal-weight): {result.win_rate:.1%} of {len(result.folds)} folds")
    lines.append(f"mean per-fold excess return: {result.mean_excess:+.4%}   std: {result.std_excess:.4%}")
    if result.t_stat_excess is None:
        lines.append("t-stat of mean excess: not computable (fewer than 2 folds, or zero variance)")
    else:
        verdict = "distinguishable from zero at conventional confidence" if abs(result.t_stat_excess) >= 2 else "NOT distinguishable from zero at conventional confidence"
        lines.append(f"t-stat of mean excess: {result.t_stat_excess:+.2f} ({verdict})")
    if result.n_fallback:
        lines.append(f"note: {result.n_fallback}/{len(result.folds)} folds fell back to equal-weight - the tangency solve was infeasible on that training window")
    lines.append("")

    if result.equal_weight_compounded_return > result.optimizer_compounded_return:
        lines.append(
            "honest result: equal-weight beat the optimizer over this walk-forward run. Reported as found, "
            "not as a footnote - see the README's limitations section."
        )
    else:
        lines.append(
            "the optimizer beat equal-weight over this walk-forward run, but see the t-stat above and the "
            "README's limitations section before reading that as a durable edge."
        )
    lines.append(
        "limitation: no NSE index/benchmark fixture is committed anywhere in this sandbox (no network, zero "
        "spend) - this is optimizer-vs-equal-weight only, not the optimizer-vs-index comparison NEXT_STEPS.md "
        "asks for."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    result = run(tickers, args.fixtures_dir, args.train_window, args.test_window, args.risk_free_rate)
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
