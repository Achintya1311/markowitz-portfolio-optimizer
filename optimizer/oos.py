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
the index. This module stays optimizer-vs-equal-weight; the index leg
(a price-only Nifty 50 fixture fetched in Actions, P129) is in
``optimizer.methods`` on the large-cap universe, and this CLI says so on
every run - see the README.

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

from optimizer import DISCLAIMER
from optimizer.constraints import constrained_min_variance
from optimizer.covariance import sample_mean_cov
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix
from optimizer.shrinkage import ledoit_wolf_shrinkage
from optimizer.tangency import DEFAULT_RISK_FREE_RATE, max_sharpe_weights_long_only

DEFAULT_TRAIN_WINDOW = 252
DEFAULT_TEST_WINDOW = 21

# Indian delivery-equity cost schedule, reused from STOCKSTALKER's own
# backtest (stockstalker/backtest.py: STT_RATE=0.001 both legs,
# STAMP_DUTY_RATE=0.00015 buy leg only, DEFAULT_SLIPPAGE_BPS=10 both legs -
# the same schedule R8 already made decisive for that repo's backtest).
# A rebalance from w_prev to w_new buys sum(max(w_new - w_prev, 0)) and
# sells sum(max(w_prev - w_new, 0)) of book value; because weights sum to 1
# both before and after, those two amounts are always equal - each exactly
# half of the L1 turnover sum(|w_new - w_prev|). So the blended cost per
# unit of L1 turnover is the average of the two legs' per-unit rates: STT
# (charged on both legs) + slippage (both legs) + half the stamp duty (buy
# leg only) = 0.001 + 0.001 + 0.00015 / 2 = 0.2075%.
STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
SLIPPAGE_BPS = 10.0
TURNOVER_COST_RATE = STT_RATE + STAMP_DUTY_RATE / 2 + SLIPPAGE_BPS / 10_000


@dataclass
class FoldResult:
    """One walk-forward step: a training window's weights, held over the next test window.

    ``optimizer_turnover_l1``/``optimizer_cost`` are this fold's rebalance
    away from the *previous* fold's realized optimizer weights (equal-weight
    for fold 0), costed at ``TURNOVER_COST_RATE`` (R9, hub CRITIQUES.md).
    ``equal_weight_turnover_l1``/``equal_weight_cost`` are always 0.0: this
    walk-forward resets the equal-weight leg to exactly 1/n every fold with
    no within-period price drift tracked (see module docstring), so there is
    no drift to rebalance away from - the README's limitations section says
    so plainly rather than fabricating a nonzero drift-rebalancing cost this
    model does not simulate. ``baseline_turnover_l1`` is what the optimizer's
    turnover would have been from the plain (unpenalized) tangency weights;
    it equals ``optimizer_turnover_l1`` unless ``used_turnover_penalty`` is
    True.
    """

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
    optimizer_turnover_l1: float = 0.0
    optimizer_cost: float = 0.0
    equal_weight_turnover_l1: float = 0.0
    equal_weight_cost: float = 0.0
    baseline_turnover_l1: float = 0.0
    used_turnover_penalty: bool = False

    @property
    def excess_return(self) -> float:
        return self.optimizer_return - self.equal_weight_return

    @property
    def optimizer_return_net(self) -> float:
        return self.optimizer_return - self.optimizer_cost

    @property
    def equal_weight_return_net(self) -> float:
        return self.equal_weight_return - self.equal_weight_cost

    @property
    def excess_return_net(self) -> float:
        return self.optimizer_return_net - self.equal_weight_return_net


@dataclass
class OOSResult:
    """Aggregate walk-forward result across all folds."""

    tickers: list[str]
    train_window: int
    test_window: int
    risk_free: float
    folds: list[FoldResult]
    unused_days: int
    turnover_lambda: float = 0.0
    optimizer_compounded_return: float = field(init=False)
    equal_weight_compounded_return: float = field(init=False)
    win_rate: float = field(init=False)
    mean_excess: float = field(init=False)
    std_excess: float = field(init=False)
    t_stat_excess: float | None = field(init=False)
    n_fallback: int = field(init=False)
    optimizer_compounded_return_net: float = field(init=False)
    equal_weight_compounded_return_net: float = field(init=False)
    mean_excess_net: float = field(init=False)
    std_excess_net: float = field(init=False)
    t_stat_excess_net: float | None = field(init=False)
    avg_optimizer_turnover_l1: float = field(init=False)
    avg_equal_weight_turnover_l1: float = field(init=False)
    avg_baseline_turnover_l1: float = field(init=False)
    total_optimizer_cost: float = field(init=False)
    total_equal_weight_cost: float = field(init=False)
    n_turnover_penalty_used: int = field(init=False)

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

        opt_returns_net = np.array([f.optimizer_return_net for f in self.folds])
        eq_returns_net = np.array([f.equal_weight_return_net for f in self.folds])
        excess_net = opt_returns_net - eq_returns_net

        self.optimizer_compounded_return_net = float(np.prod(1.0 + opt_returns_net) - 1.0)
        self.equal_weight_compounded_return_net = float(np.prod(1.0 + eq_returns_net) - 1.0)
        self.mean_excess_net = float(np.mean(excess_net)) if len(excess_net) else 0.0
        self.std_excess_net = float(np.std(excess_net, ddof=1)) if len(excess_net) > 1 else 0.0
        if len(excess_net) > 1 and self.std_excess_net > 0:
            self.t_stat_excess_net = self.mean_excess_net / (self.std_excess_net / np.sqrt(len(excess_net)))
        else:
            self.t_stat_excess_net = None

        self.avg_optimizer_turnover_l1 = (
            float(np.mean([f.optimizer_turnover_l1 for f in self.folds])) if self.folds else 0.0
        )
        self.avg_equal_weight_turnover_l1 = (
            float(np.mean([f.equal_weight_turnover_l1 for f in self.folds])) if self.folds else 0.0
        )
        self.avg_baseline_turnover_l1 = (
            float(np.mean([f.baseline_turnover_l1 for f in self.folds])) if self.folds else 0.0
        )
        self.total_optimizer_cost = float(np.sum([f.optimizer_cost for f in self.folds]))
        self.total_equal_weight_cost = float(np.sum([f.equal_weight_cost for f in self.folds]))
        self.n_turnover_penalty_used = sum(1 for f in self.folds if f.used_turnover_penalty)


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
    turnover_lambda: float = 0.0,
    prev_optimizer_weights: np.ndarray | None = None,
    tickers: list[str] | None = None,
) -> FoldResult:
    """Solve one fold's weights and cost its rebalance from the previous fold (R9).

    ``turnover_lambda`` > 0 reuses :func:`optimizer.constraints.constrained_min_variance`
    (Day 5's turnover-penalized solve) to re-solve for the same expected
    return the plain tangency weights targeted, but penalized for moving
    away from ``prev_optimizer_weights`` - requires ``tickers`` (one per
    asset) since that function's sector-cap machinery expects them, even
    though no sector cap is applied here. If the penalized solve is
    infeasible, this fold falls back to the plain tangency weights, same as
    an infeasible tangency solve falls back to equal-weight.
    """
    n_assets = returns_matrix.shape[1]
    train_returns = returns_matrix[train_slice]

    mu_daily, _ = sample_mean_cov(train_returns)
    shrinkage, cov_shrunk_daily = ledoit_wolf_shrinkage(train_returns)
    mu = annualize_mean(mu_daily)
    cov_shrunk = annualize_cov(cov_shrunk_daily)

    equal_weight = np.full(n_assets, 1.0 / n_assets)
    if prev_optimizer_weights is None:
        prev_optimizer_weights = equal_weight

    tangency = max_sharpe_weights_long_only(mu, cov_shrunk, risk_free)
    used_fallback = not tangency.success
    baseline_weights = tangency.weights if tangency.success else equal_weight

    weights = baseline_weights
    used_turnover_penalty = False
    if turnover_lambda > 0 and tangency.success:
        if tickers is None:
            raise ValueError("tickers is required when turnover_lambda > 0")
        target_return = float(baseline_weights @ mu)
        penalized = constrained_min_variance(
            mu,
            cov_shrunk,
            target_return,
            tickers,
            long_only=True,
            turnover_lambda=turnover_lambda,
            prev_weights=prev_optimizer_weights,
        )
        if penalized.success:
            weights = penalized.weights
            used_turnover_penalty = True

    test_returns = total_simple_returns(returns_matrix[test_slice])

    optimizer_turnover_l1 = float(np.sum(np.abs(weights - prev_optimizer_weights)))
    baseline_turnover_l1 = float(np.sum(np.abs(baseline_weights - prev_optimizer_weights)))
    optimizer_cost = TURNOVER_COST_RATE * optimizer_turnover_l1

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
        optimizer_turnover_l1=optimizer_turnover_l1,
        optimizer_cost=optimizer_cost,
        baseline_turnover_l1=baseline_turnover_l1,
        used_turnover_penalty=used_turnover_penalty,
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
    parser.add_argument(
        "--turnover-lambda",
        type=float,
        default=0.0,
        help="quadratic turnover-penalty coefficient, reused from optimizer.constraints "
        "(default: 0.0, plain unconstrained tangency weights every fold)",
    )
    parser.add_argument(
        "--risk-free-rate-sweep",
        default=None,
        help=(
            "comma-separated annualized risk-free rates, e.g. 0.0,0.04,0.07 (R10, hub CRITIQUES.md). "
            "When given, overrides --risk-free-rate: reruns the whole walk-forward at each rate and "
            "reports whether the after-cost optimizer-vs-equal-weight sign is robust to the choice, "
            "instead of a single-rate report"
        ),
    )
    return parser


def run(
    tickers: list[str],
    fixtures_dir: str,
    train_window: int,
    test_window: int,
    risk_free: float,
    turnover_lambda: float = 0.0,
) -> OOSResult:
    dates, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    n_days = returns_matrix.shape[0]
    n_assets = returns_matrix.shape[1]

    fold_slices = walk_forward_folds(n_days, train_window, test_window)
    if not fold_slices:
        raise ValueError(
            f"only {n_days} days of returns available, not enough for one "
            f"train_window={train_window} + test_window={test_window} fold"
        )

    folds = []
    prev_optimizer_weights = np.full(n_assets, 1.0 / n_assets)
    for i, (train_slice, test_slice) in enumerate(fold_slices):
        fold = run_fold(
            returns_matrix,
            dates,
            train_slice,
            test_slice,
            i,
            risk_free,
            turnover_lambda=turnover_lambda,
            prev_optimizer_weights=prev_optimizer_weights,
            tickers=tickers,
        )
        folds.append(fold)
        prev_optimizer_weights = fold.optimizer_weights

    used_days = fold_slices[-1][1].stop
    return OOSResult(
        tickers=tickers,
        train_window=train_window,
        test_window=test_window,
        risk_free=risk_free,
        folds=folds,
        unused_days=n_days - used_days,
        turnover_lambda=turnover_lambda,
    )


def sweep_risk_free_rates(
    tickers: list[str],
    fixtures_dir: str,
    train_window: int,
    test_window: int,
    risk_free_rates: list[float],
    turnover_lambda: float = 0.0,
) -> list[OOSResult]:
    """Rerun the walk-forward test at each of several risk-free rates (R10, hub CRITIQUES.md).

    The risk-free rate enters every fold's tangency solve (``mu - rf``), so a
    different assumed rate can pick different weights each fold, not just
    rescale the Sharpe ratio reported in :mod:`optimizer.tangency`. This
    checks whether the headline optimizer-vs-equal-weight sign survives that,
    the same way Day 9 checked whether it survived a wider universe.
    """
    return [
        run(tickers, fixtures_dir, train_window, test_window, rf, turnover_lambda)
        for rf in risk_free_rates
    ]


def _format_sweep_report(results: list[OOSResult]) -> str:
    first = results[0]
    lines = [
        f"tickers: {', '.join(first.tickers)}",
        f"risk-free-rate sweep (R10, hub CRITIQUES.md): train_window={first.train_window}d   "
        f"test_window={first.test_window}d   folds={len(first.folds)}",
        "",
        f"  {'rf':>8}  {'optimizer (net)':>16}  {'equal-weight (net)':>19}  {'sign':>14}  {'t-stat (net)':>12}",
    ]
    signs = []
    for result in results:
        opt = result.optimizer_compounded_return_net
        eq = result.equal_weight_compounded_return_net
        sign = "optimizer" if opt > eq else "equal-weight"
        signs.append(opt > eq)
        t_stat = result.t_stat_excess_net
        t_str = f"{t_stat:+.2f}" if t_stat is not None else "n/a"
        lines.append(f"  {result.risk_free:>+8.4%}  {opt:>+16.4%}  {eq:>+19.4%}  {sign:>14}  {t_str:>12}")
    lines.append("")

    if all(signs) or not any(signs):
        winner = "optimizer" if signs[0] else "equal-weight"
        lines.append(f"verdict: {winner} wins the after-cost compounded-return comparison at every rate swept")
    else:
        lines.append(
            "verdict: the after-cost optimizer-vs-equal-weight sign is NOT robust - it flips somewhere in "
            "this sweep, so a single assumed risk-free rate can change which side the headline finding favors"
        )
    return "\n".join(lines)


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

    lines.append(f"compounded return over all {len(result.folds)} test windows, before transaction costs:")
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
            "honest result, before costs: equal-weight beat the optimizer over this walk-forward run. Reported "
            "as found, not as a footnote - see the README's limitations section."
        )
    else:
        lines.append(
            "before costs, the optimizer beat equal-weight over this walk-forward run, but see the t-stat above "
            "and the README's limitations section before reading that as a durable edge."
        )
    lines.append(
        "limitation: this is optimizer-vs-equal-weight only; the index leg (price-only Nifty 50, no "
        "dividends, P129) is in python -m optimizer.methods on the large-cap universe."
    )
    lines.append("")

    lines.append(
        f"transaction costs (R9, hub CRITIQUES.md): STT {STT_RATE:.2%}/leg + stamp duty "
        f"{STAMP_DUTY_RATE:.3%} (buy leg only) + {SLIPPAGE_BPS:.0f}bps slippage/leg - STOCKSTALKER's own "
        f"backtest schedule - charged on each fold's L1 turnover from the previous fold's held weights, "
        f"blended to {TURNOVER_COST_RATE:.4%} per unit of L1 turnover (see optimizer.oos module docstring)."
    )
    lines.append(
        f"  average per-fold L1 turnover: optimizer {result.avg_optimizer_turnover_l1:.2%}   "
        f"equal-weight {result.avg_equal_weight_turnover_l1:.2%} (always ~0 - see FoldResult docstring: "
        f"this walk-forward resets equal-weight to exactly 1/n every fold with no drift tracked)"
    )
    if result.turnover_lambda > 0:
        lines.append(
            f"  --turnover-lambda {result.turnover_lambda:g}: the turnover-penalized solve "
            f"(optimizer.constraints.constrained_min_variance) was used in "
            f"{result.n_turnover_penalty_used}/{len(result.folds)} folds; average per-fold L1 turnover "
            f"without it would have been {result.avg_baseline_turnover_l1:.2%} (plain tangency weights)"
        )
    lines.append(
        f"  sum of per-fold cost rates (not compounded - see the before/after compounded "
        f"returns below for the actual effect): optimizer {result.total_optimizer_cost:.4%}   "
        f"equal-weight {result.total_equal_weight_cost:.4%}"
    )
    lines.append("")

    lines.append(f"compounded return over all {len(result.folds)} test windows, after transaction costs:")
    lines.append(
        f"  optimizer:    {result.optimizer_compounded_return_net:+.4%}  "
        f"(before costs: {result.optimizer_compounded_return:+.4%})"
    )
    lines.append(
        f"  equal-weight: {result.equal_weight_compounded_return_net:+.4%}  "
        f"(before costs: {result.equal_weight_compounded_return:+.4%})"
    )
    lines.append("")
    lines.append(f"mean per-fold excess return, after costs: {result.mean_excess_net:+.4%}   std: {result.std_excess_net:.4%}")
    if result.t_stat_excess_net is None:
        lines.append("t-stat of mean excess, after costs: not computable (fewer than 2 folds, or zero variance)")
    else:
        verdict_net = (
            "distinguishable from zero at conventional confidence"
            if abs(result.t_stat_excess_net) >= 2
            else "NOT distinguishable from zero at conventional confidence"
        )
        lines.append(f"t-stat of mean excess, after costs: {result.t_stat_excess_net:+.2f} ({verdict_net})")
    lines.append("")

    if result.equal_weight_compounded_return_net > result.optimizer_compounded_return_net:
        lines.append(
            "honest result, after costs: equal-weight still beats the optimizer once transaction costs are "
            "charged - R9's expectation that costs would widen, not narrow, equal-weight's margin."
        )
    else:
        lines.append(
            "honest result, after costs: the optimizer still beats equal-weight after transaction costs, but "
            "see the after-cost t-stat above before reading that as a durable edge."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    print(DISCLAIMER)
    if args.risk_free_rate_sweep:
        rates = [float(r.strip()) for r in args.risk_free_rate_sweep.split(",") if r.strip()]
        results = sweep_risk_free_rates(
            tickers, args.fixtures_dir, args.train_window, args.test_window, rates, args.turnover_lambda
        )
        print(_format_sweep_report(results))
        return 0
    result = run(
        tickers, args.fixtures_dir, args.train_window, args.test_window, args.risk_free_rate, args.turnover_lambda
    )
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
