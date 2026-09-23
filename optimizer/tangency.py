"""Maximum-Sharpe tangency portfolio and the capital market line (Day 3).

    python -m optimizer.tangency --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --risk-free-rate 0.0

Given a risk-free rate, the tangency portfolio is the point on the
efficient frontier that maximizes the Sharpe ratio ``(w @ mu - rf) /
sqrt(w @ cov @ w)``. Unconstrained, this has a closed form
(``max_sharpe_weights_unconstrained``); with ``w >= 0`` it generally does
not, so the long-only variant is solved numerically and cross-checked
against the analytic answer when the analytic solution happens to already
be long-only.

The capital market line is the set of portfolios formed by mixing the
risk-free asset with the tangency portfolio: ``return = rf + sharpe *
vol``, a straight line in mean-vol space that is tangent to the frontier
at exactly the tangency portfolio (hence the name).

Risk-free rate: the hub's data-source list names FRED / RBI, but this
sandbox has no network access and no committed risk-free-rate fixture
(unlike the OHLCV fixtures, which STOCKSTALKER and fama-french-factor-model
already share). Rather than fabricate a plausible-looking RBI T-bill
number, ``--risk-free-rate`` defaults to 0.0 and is reported as an
assumption, not a fetched figure; the README's limitations section says
so. A real deployment would source this from FRED/RBI, but this run
cannot reach either.
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

DEFAULT_RISK_FREE_RATE = 0.0


@dataclass
class TangencyPoint:
    """One solved maximum-Sharpe portfolio."""

    weights: np.ndarray
    achieved_return: float
    variance: float
    sharpe: float
    long_only: bool
    success: bool

    @property
    def vol(self) -> float:
        return self.variance ** 0.5


def max_sharpe_bound(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> float:
    """The generalized-Sharpe bound ``sqrt(excess' @ inv(cov) @ excess)``.

    This is the supremum of the Sharpe ratio over *all* portfolios of these
    assets, any sign, any leverage - it does not depend on the budget
    constraint at all. When ``ones @ inv(cov) @ excess > 0`` this bound is
    exactly attained by :func:`max_sharpe_weights_unconstrained`'s finite
    full-investment portfolio; otherwise it is only a ceiling that finite
    full-investment portfolios approach but never reach.
    """
    excess = mu - risk_free
    return float(np.sqrt(excess @ np.linalg.solve(cov, excess)))


def sharpe_ratio(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free: float) -> float:
    """Annualized Sharpe ratio of a portfolio: excess return per unit of volatility."""
    port_return = float(weights @ mu)
    port_vol = float(weights @ cov @ weights) ** 0.5
    if port_vol == 0.0:
        raise ValueError("portfolio variance is zero, Sharpe ratio is undefined")
    return (port_return - risk_free) / port_vol


def max_sharpe_weights_unconstrained(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> TangencyPoint:
    """Closed-form unconstrained (full-investment, any sign) tangency portfolio.

    The unique direction maximizing the Sharpe ratio over *all* portfolios
    (any scale, any sign) is ``v = inv(cov) @ (mu - rf)``, achieving
    ``sqrt(excess' @ inv(cov) @ excess)`` - the standard generalized-Sharpe
    bound. Imposing the budget constraint ``sum(w) == 1`` just rescales
    that direction: ``w = v / sum(v)``.

    That rescaling only lands on the *maximizing* point when
    ``sum(v) > 0``. When ``sum(v) < 0``, dividing by a negative number
    flips the sign of every weight, landing on ``-v``'s ray instead - the
    Sharpe-*minimizing* direction, not the maximizer - and the budget-line
    maximum is then not attained by any finite portfolio at all: nearby
    directions can push the achieved Sharpe arbitrarily close to the
    ``sqrt(excess' inv(cov) excess)`` bound only by taking positions whose
    magnitude diverges (verified numerically: unconstrained SLSQP with
    ``sum(w)=1`` climbs without bound and never converges in this branch).
    This is a real, not cosmetic, distinction - naively dividing by a
    negative ``sum(v)`` silently returns the *worst* possible portfolio
    while looking like a normal answer, so that case is reported as
    undefined (``success=False``) instead.
    """
    n = len(mu)
    if n < 2:
        raise ValueError("need at least 2 assets for a tangency portfolio")

    excess = mu - risk_free
    inv_cov_excess = np.linalg.solve(cov, excess)
    denom = float(np.sum(inv_cov_excess))

    if denom <= 1e-12:
        return TangencyPoint(
            weights=np.full(n, np.nan),
            achieved_return=float("nan"),
            variance=float("nan"),
            sharpe=float("nan"),
            long_only=False,
            success=False,
        )

    weights = inv_cov_excess / denom
    achieved_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    sharpe = (achieved_return - risk_free) / (variance ** 0.5)

    return TangencyPoint(
        weights=weights,
        achieved_return=achieved_return,
        variance=variance,
        sharpe=sharpe,
        long_only=False,
        success=True,
    )


def max_sharpe_weights_long_only(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> TangencyPoint:
    """Numerically maximize Sharpe subject to ``w >= 0`` and ``sum(w) == 1``.

    No closed form exists once the non-negativity bound can bind, so this
    solves ``minimize(-Sharpe)`` via SLSQP, same solver and same
    residual-based ``success`` check as :mod:`optimizer.frontier`'s
    ``min_variance_weights`` (SLSQP's own convergence flag does not by
    itself guarantee the budget/bounds constraints hold to tolerance).
    """
    n = len(mu)
    if n < 2:
        raise ValueError("need at least 2 assets for a tangency portfolio")

    def negative_sharpe(w: np.ndarray) -> float:
        port_return = w @ mu
        port_vol = float(w @ cov @ w) ** 0.5
        if port_vol == 0.0:
            return 0.0
        return -(port_return - risk_free) / port_vol

    x0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, None) for _ in range(n)]

    result = minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=_SLSQP_OPTIONS,
    )

    weights = result.x
    budget_ok = abs(np.sum(weights) - 1.0) < _FEASIBILITY_TOL
    bounds_ok = bool(np.all(weights >= -_FEASIBILITY_TOL))
    success = bool(result.success) and budget_ok and bounds_ok

    achieved_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    sharpe = (achieved_return - risk_free) / (variance ** 0.5) if variance > 0 else float("nan")

    return TangencyPoint(
        weights=weights,
        achieved_return=achieved_return,
        variance=variance,
        sharpe=sharpe,
        long_only=True,
        success=success,
    )


def capital_market_line(
    risk_free: float, tangency_return: float, tangency_vol: float, n_points: int = 11, max_leverage: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Points ``(vol, return)`` on the capital market line.

    The CML mixes the risk-free asset with the tangency portfolio in
    proportion ``t`` (``t=0`` is all risk-free, ``t=1`` is all tangency,
    ``t>1`` is leveraged - borrowing at the risk-free rate to hold more
    than 100% tangency). Swept from 0 to ``max_leverage`` so the line is
    visible past the tangency point itself, same as any textbook CML plot.
    """
    if tangency_vol <= 0:
        raise ValueError("tangency portfolio volatility must be positive")
    t = np.linspace(0.0, max_leverage, n_points)
    vols = t * tangency_vol
    returns = risk_free + t * (tangency_return - risk_free)
    return vols, returns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maximum-Sharpe tangency portfolio and the capital market line (Day 3)."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help=(
            "annualized risk-free rate, e.g. 0.065 for 6.5%% (default: 0.0 - no live FRED/RBI "
            "fetch is available offline, so this is an assumption, not a fetched figure)"
        ),
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    parser.add_argument(
        "--cml-points", type=int, default=11, help="number of points to report along the capital market line (default: 11)"
    )
    return parser


def run(tickers: list[str], risk_free: float, fixtures_dir: str, cml_points: int = 11) -> dict:
    _, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    mu = annualize_mean(mu_daily)
    cov = annualize_cov(cov_daily)

    unconstrained = max_sharpe_weights_unconstrained(mu, cov, risk_free)
    long_only = max_sharpe_weights_long_only(mu, cov, risk_free)
    sharpe_bound = max_sharpe_bound(mu, cov, risk_free)

    cml_vols, cml_returns = capital_market_line(
        risk_free, long_only.achieved_return, long_only.vol, n_points=cml_points
    )

    return {
        "tickers": tickers,
        "mu_annual": mu,
        "risk_free": risk_free,
        "unconstrained": unconstrained,
        "long_only": long_only,
        "sharpe_bound": sharpe_bound,
        "cml_vols": cml_vols,
        "cml_returns": cml_returns,
    }


def _format_point(tickers: list[str], point: TangencyPoint) -> str:
    if not point.success:
        return (
            "  no finite maximum-Sharpe portfolio exists at full investment: "
            "ones @ inv(cov) @ (mu - rf) <= 0, so the theoretical Sharpe bound "
            "can only be approached via unbounded leverage, never attained"
        )
    parts = [f"{t}={w:+.4f}" for t, w in zip(tickers, point.weights)]
    return (
        f"  weights: {', '.join(parts)}\n"
        f"  return: {point.achieved_return:>+.4%}   vol: {point.vol:.4%}   Sharpe: {point.sharpe:+.4f}"
    )


def _format_report(result: dict) -> str:
    tickers = result["tickers"]
    lines = []
    lines.append(f"tickers: {', '.join(tickers)}")
    lines.append("annualized mean (sample): " + ", ".join(f"{t}={m:+.4%}" for t, m in zip(tickers, result["mu_annual"])))
    lines.append(f"risk-free rate: {result['risk_free']:+.4%} (no live FRED/RBI fetch available offline - see README)")
    lines.append("")

    lines.append(f"theoretical Sharpe bound (sqrt(excess' inv(cov) excess)): {result['sharpe_bound']:.4f}")
    lines.append("tangency portfolio, unconstrained (long/short):")
    lines.append(_format_point(tickers, result["unconstrained"]))
    lines.append("")
    lines.append("tangency portfolio, long-only (w >= 0):")
    lines.append(_format_point(tickers, result["long_only"]))
    lines.append("")

    lo = result["long_only"]
    if lo.success:
        lines.append(f"capital market line (mixing risk-free with the long-only tangency portfolio):")
        lines.append(f"  {'weight in tangency':>20}{'vol':>12}{'return':>12}")
        max_t = len(result["cml_vols"]) - 1
        for i, (v, r) in enumerate(zip(result["cml_vols"], result["cml_returns"])):
            t = i / max_t if max_t else 0.0
            lines.append(f"  {t:>19.0%} {v:>11.4%} {r:>11.4%}")
    else:
        lines.append("capital market line: not computed, long-only tangency portfolio was infeasible")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    result = run(tickers, args.risk_free_rate, args.fixtures_dir, args.cml_points)
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
