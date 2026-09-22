"""Minimum-variance portfolio for a target return, long-only and long/short
(Day 2).

    python -m optimizer.frontier --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --target-return -0.10

For a fixed annualized target return, solves for the weights that minimize
portfolio variance subject to ``sum(w) == 1`` and ``w @ mu == target``, once
with ``w >= 0`` (long-only) and once unconstrained (long/short). This is the
frontier Day 3's tangency portfolio and Day 4's Ledoit-Wolf comparison sit on
top of.

The mean vector feeding this is the same one Day 1 measured as noisy (every
ticker's annualized mean had |t| < 2) - this module ranks portfolios by that
noisy vector, same as any Markowitz optimizer does, and says so rather than
presenting the frontier as precise.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from optimizer.covariance import sample_mean_cov
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix

_SLSQP_OPTIONS = {"maxiter": 500, "ftol": 1e-12}
_FEASIBILITY_TOL = 1e-6


@dataclass
class FrontierPoint:
    """One solved point on the minimum-variance-for-target-return curve."""

    target_return: float
    achieved_return: float
    variance: float
    weights: np.ndarray
    long_only: bool
    success: bool


def _variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def min_variance_weights(
    mu: np.ndarray, cov: np.ndarray, target_return: float, long_only: bool
) -> FrontierPoint:
    """Weights minimizing variance for an exact target return.

    Two equality constraints (budget, target return) leave a 2-asset
    portfolio with zero degrees of freedom, so for ``n == 2`` this has a
    closed form independent of the covariance matrix - see
    :func:`analytic_two_asset_weights`, which the tests check this
    function against for the Day 2 "done when" gate.

    ``success`` is computed from the constraint residuals directly rather
    than trusted from ``scipy``'s exit code alone: SLSQP can report
    convergence on a target return that is infeasible under the
    ``long_only`` bound (e.g. above every asset's own mean), leaving a
    residual constraint violation that its own ``success`` flag misses.
    """
    n = len(mu)
    if n < 2:
        raise ValueError("need at least 2 assets for a frontier")

    x0 = np.full(n, 1.0 / n)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0.0, None) if long_only else (None, None) for _ in range(n)]

    result = minimize(
        _variance,
        x0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SLSQP_OPTIONS,
    )

    weights = result.x
    budget_ok = abs(np.sum(weights) - 1.0) < _FEASIBILITY_TOL
    return_ok = abs(float(weights @ mu) - target_return) < _FEASIBILITY_TOL
    bounds_ok = (not long_only) or bool(np.all(weights >= -_FEASIBILITY_TOL))
    success = bool(result.success) and budget_ok and return_ok and bounds_ok

    return FrontierPoint(
        target_return=target_return,
        achieved_return=float(weights @ mu),
        variance=float(weights @ cov @ weights),
        weights=weights,
        long_only=long_only,
        success=success,
    )


def analytic_two_asset_weights(mu: np.ndarray, target_return: float) -> np.ndarray:
    """Closed-form minimum-variance weights for exactly 2 assets.

    ``w1 + w2 = 1`` and ``w1*mu1 + w2*mu2 = target`` are two linear
    equations in two unknowns - the weights are pinned down by the mean
    vector and target return alone, with no reference to the covariance
    matrix at all (unlike ``n >= 3``, where the covariance shapes which
    of the many feasible weight vectors is chosen). This is the analytic
    check NEXT_STEPS.md's "done when" gate asks for.
    """
    if len(mu) != 2:
        raise ValueError("analytic solution only applies to exactly 2 assets")
    mu1, mu2 = mu
    if np.isclose(mu1, mu2):
        raise ValueError("analytic solution requires mu1 != mu2 (target return is then unconstrained)")
    w1 = (target_return - mu2) / (mu1 - mu2)
    return np.array([w1, 1.0 - w1])


def feasible_target_range(mu: np.ndarray, long_only: bool) -> tuple[float, float]:
    """Range of target returns some portfolio can actually hit.

    Long-only weights are a convex combination of the asset means, so the
    feasible range is exactly ``[min(mu), max(mu)]`` - nothing outside it
    is reachable no matter the covariance. Long/short has no such bound in
    theory (leverage can push the portfolio return arbitrarily far), so
    the range is widened around the same span for reporting purposes,
    not because it is a hard limit.
    """
    lo, hi = float(np.min(mu)), float(np.max(mu))
    if long_only:
        return lo, hi
    span = hi - lo
    pad = span if span > 0 else (abs(hi) * 0.5 or 0.05)
    return lo - pad, hi + pad


def build_frontier(
    mu: np.ndarray, cov: np.ndarray, long_only: bool, n_points: int = 25
) -> list[FrontierPoint]:
    """Minimum-variance portfolio at each of ``n_points`` target returns spanning the feasible range."""
    lo, hi = feasible_target_range(mu, long_only)
    targets = np.linspace(lo, hi, n_points)
    return [min_variance_weights(mu, cov, t, long_only) for t in targets]


def is_convex(variances: np.ndarray, tol: float = 1e-8) -> bool:
    """Whether variance is a convex function of target return along the frontier.

    Checked via non-negative second differences on the (evenly spaced)
    target-return grid :func:`build_frontier` uses - the minimum-variance
    value of a QP is convex in the right-hand side of a linear equality
    constraint, so this should hold for both the long-only and long/short
    variants. Part of Day 2's "done when" gate.
    """
    second_diff = np.diff(variances, n=2)
    return bool(np.all(second_diff >= -tol))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimum-variance portfolio for a target return, long-only and long/short (Day 2)."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--target-return", type=float, required=True, help="annualized target portfolio return, e.g. 0.10 for 10%%"
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    parser.add_argument(
        "--n-points", type=int, default=25, help="number of target returns to sweep for the convexity check (default: 25)"
    )
    return parser


def run(tickers: list[str], target_return: float, fixtures_dir: str, n_points: int = 25) -> dict:
    _, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    mu = annualize_mean(mu_daily)
    cov = annualize_cov(cov_daily)

    long_only = min_variance_weights(mu, cov, target_return, long_only=True)
    long_short = min_variance_weights(mu, cov, target_return, long_only=False)

    frontier = build_frontier(mu, cov, long_only=True, n_points=n_points)
    solved = [p for p in frontier if p.success]
    variances = np.array([p.variance for p in solved])
    frontier_convex = is_convex(variances) if len(variances) >= 3 else None
    min_var_point = min(solved, key=lambda p: p.variance) if solved else None

    return {
        "tickers": tickers,
        "mu_annual": mu,
        "target_return": target_return,
        "long_only": long_only,
        "long_short": long_short,
        "feasible_range_long_only": feasible_target_range(mu, long_only=True),
        "frontier_n_points": n_points,
        "frontier_n_feasible": len(solved),
        "frontier_convex": frontier_convex,
        "min_var_point": min_var_point,
    }


def _format_weights(tickers: list[str], point: FrontierPoint) -> str:
    if not point.success:
        return "  infeasible - target return is not reachable under this variant's constraints"
    parts = [f"{t}={w:+.4f}" for t, w in zip(tickers, point.weights)]
    return f"  weights: {', '.join(parts)}\n  achieved return: {point.achieved_return:>+.4%}   variance: {point.variance:.6f}   vol: {point.variance ** 0.5:.4%}"


def _format_report(result: dict) -> str:
    tickers = result["tickers"]
    lines = []
    lines.append(f"tickers: {', '.join(tickers)}")
    lines.append("annualized mean (sample): " + ", ".join(f"{t}={m:+.4%}" for t, m in zip(tickers, result["mu_annual"])))
    lines.append(f"target return: {result['target_return']:+.4%}")
    lines.append("")

    lo, hi = result["feasible_range_long_only"]
    lines.append(f"long-only feasible target range (convex combination of asset means): [{lo:+.4%}, {hi:+.4%}]")
    lines.append("")

    lines.append("long-only (w >= 0):")
    lines.append(_format_weights(tickers, result["long_only"]))
    lines.append("")
    lines.append("long/short (unconstrained):")
    lines.append(_format_weights(tickers, result["long_short"]))
    lines.append("")

    n_points = result["frontier_n_points"]
    n_feasible = result["frontier_n_feasible"]
    convex = result["frontier_convex"]
    lines.append(f"long-only frontier sweep: {n_feasible}/{n_points} target returns feasible")
    if convex is None:
        lines.append("  not enough feasible points to check convexity")
    else:
        lines.append(f"  variance is convex in target return: {convex}")
    mv = result["min_var_point"]
    if mv is not None:
        lines.append(
            f"  minimum-variance point on the sweep: return={mv.achieved_return:+.4%}   vol={mv.variance ** 0.5:.4%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    result = run(tickers, args.target_return, args.fixtures_dir, args.n_points)
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
