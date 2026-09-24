"""Ledoit-Wolf shrinkage covariance vs the plain sample covariance (Day 4).

    python -m optimizer.shrinkage --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --risk-free-rate 0.0

This is the centrepiece day, not a footnote (see the hub's ``knowaboutit.md``).
Days 1-3 built everything on the plain sample covariance matrix
(:func:`optimizer.covariance.sample_mean_cov`), which with few assets and a
short window is noisy and, in higher dimensions than this fixture universe
has, can be near-singular. Ledoit & Wolf (2004), "Honey, I Shrunk the Sample
Covariance Matrix", shrink it toward a well-conditioned target - here the
standard scaled-identity target ``mu * I`` with ``mu = trace(S) / n`` - using
a shrinkage intensity chosen to minimize expected quadratic loss against the
(unknown) true covariance. The point of this module is not the shrunk matrix
itself but the comparison: how far do the optimal weights move once the
covariance input changes, with everything else (the mean vector, the
optimizer) held fixed?

Two weight problems are compared, deliberately excluding the mean vector
where it can be avoided:

- The global minimum-variance portfolio (:func:`global_min_variance_weights`)
  depends on the covariance matrix alone, not on ``mu`` at all - the cleanest
  place to isolate what shrinkage changes, since Day 1 already established
  that ``mu`` is this pipeline's noisiest, most consequential input and Day
  4 would otherwise be confounding two different sources of error.
- The long-only maximum-Sharpe tangency portfolio
  (:func:`optimizer.tangency.max_sharpe_weights_long_only`) still needs
  ``mu``, but it is the portfolio the rest of this project actually cares
  about, so it is reported too, with that caveat attached.

The shrinkage formula here was validated during development against
``sklearn.covariance.LedoitWolf`` on this repo's own fixtures (shrinkage
intensity and shrunk covariance both matched to float precision); ``sklearn``
is not added as a runtime dependency, so that check is not a committed test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from optimizer.covariance import condition_number, sample_mean_cov
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix
from optimizer.tangency import TangencyPoint, max_sharpe_weights_long_only

_SLSQP_OPTIONS = {"maxiter": 500, "ftol": 1e-12}
_FEASIBILITY_TOL = 1e-6


def ledoit_wolf_shrinkage(returns_matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Ledoit-Wolf shrinkage intensity and shrunk daily covariance.

    ``returns_matrix`` has shape ``(n_days, n_assets)``, same convention as
    :func:`optimizer.covariance.sample_mean_cov`. Shrinks toward the scaled-
    identity target ``F = mu * I``, ``mu = trace(S) / n``, using the
    closed-form asymptotically-optimal intensity from Ledoit & Wolf (2004):
    with centered observations ``x_t`` and population (divide-by-T, not
    Day 1's ddof=1) sample covariance ``S = (1/T) sum_t x_t x_t'``,

        pi_hat    = (1/T) * sum_t || x_t x_t' - S ||_F^2
        delta_hat = || S - F ||_F^2
        shrinkage = clip(pi_hat / delta_hat / T, 0, 1)

    (the ``1/n`` that normally appears in both ``pi_hat`` and ``delta_hat``
    cancels in the ratio, so it is omitted here.) The population-covariance
    convention matches the paper the formula is drawn from; at this
    project's sample sizes the gap to the ddof=1 estimator used elsewhere in
    this repo is a factor of ``T / (T - 1)``, well under 1%, and immaterial
    to the comparison this day is making.

    Returns ``(shrinkage, cov_daily_shrunk)`` where ``shrinkage`` is in
    ``[0, 1]`` (0 = no shrinkage, pure sample covariance; 1 = the target
    itself) and ``cov_daily_shrunk = (1 - shrinkage) * S + shrinkage * F``.
    """
    n_obs, n_assets = returns_matrix.shape
    if n_assets < 2:
        raise ValueError("need at least 2 assets to shrink a covariance matrix")

    x = returns_matrix - returns_matrix.mean(axis=0)
    sample_cov = x.T @ x / n_obs
    mu = float(np.trace(sample_cov) / n_assets)
    target = mu * np.eye(n_assets)

    x_sq = x**2
    pi_hat = float(((x_sq.T @ x_sq) / n_obs - sample_cov**2).sum())
    delta_hat = float(((sample_cov - target) ** 2).sum())

    if delta_hat == 0.0:
        shrinkage = 0.0
    else:
        shrinkage = float(np.clip((pi_hat / delta_hat) / n_obs, 0.0, 1.0))

    shrunk_cov = (1.0 - shrinkage) * sample_cov + shrinkage * target
    return shrinkage, shrunk_cov


@dataclass
class GMVPoint:
    """One solved global minimum-variance portfolio (no target return, no mu)."""

    weights: np.ndarray
    variance: float
    long_only: bool
    success: bool

    @property
    def vol(self) -> float:
        return self.variance**0.5


def global_min_variance_weights(cov: np.ndarray, long_only: bool) -> GMVPoint:
    """Minimum-variance weights subject only to ``sum(w) == 1`` (no target return).

    Unconstrained, this has the closed form ``w = inv(cov) @ 1 / (1' @
    inv(cov) @ 1)`` - unlike the tangency portfolio, the denominator here is
    a quadratic form ``1' @ inv(cov) @ 1``, always positive for a valid
    (positive-definite) covariance matrix, so there is no degenerate-sign
    case to guard against the way :func:`optimizer.tangency.max_sharpe_weights_unconstrained`
    has to. With ``w >= 0`` added, no closed form exists in general and this
    falls back to the same SLSQP pattern used throughout this package,
    checking constraint residuals rather than trusting ``scipy``'s own
    ``success`` flag alone.
    """
    n = cov.shape[0]
    if n < 2:
        raise ValueError("need at least 2 assets for a minimum-variance portfolio")

    if not long_only:
        ones = np.ones(n)
        inv_cov_ones = np.linalg.solve(cov, ones)
        weights = inv_cov_ones / float(ones @ inv_cov_ones)
        return GMVPoint(weights=weights, variance=float(weights @ cov @ weights), long_only=False, success=True)

    def variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    x0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, None) for _ in range(n)]

    result = minimize(variance, x0, method="SLSQP", bounds=bounds, constraints=constraints, options=_SLSQP_OPTIONS)

    weights = result.x
    budget_ok = abs(np.sum(weights) - 1.0) < _FEASIBILITY_TOL
    bounds_ok = bool(np.all(weights >= -_FEASIBILITY_TOL))
    success = bool(result.success) and budget_ok and bounds_ok

    return GMVPoint(weights=weights, variance=float(weights @ cov @ weights), long_only=True, success=success)


def weight_shift(weights_a: np.ndarray, weights_b: np.ndarray) -> dict:
    """How far one weight vector moved from another: per-asset, L1, and max-abs."""
    diff = weights_b - weights_a
    return {
        "diff": diff,
        "l1": float(np.abs(diff).sum()),
        "max_abs": float(np.abs(diff).max()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ledoit-Wolf shrinkage covariance vs the plain sample covariance, and how far optimal weights move (Day 4)."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="annualized risk-free rate for the tangency-portfolio comparison (default: 0.0, see optimizer.tangency)",
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    return parser


def run(tickers: list[str], risk_free: float, fixtures_dir: str) -> dict:
    _, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    n_obs = returns_matrix.shape[0]

    mu_daily, cov_sample_daily = sample_mean_cov(returns_matrix)
    shrinkage, cov_shrunk_daily = ledoit_wolf_shrinkage(returns_matrix)

    mu = annualize_mean(mu_daily)
    cov_sample = annualize_cov(cov_sample_daily)
    cov_shrunk = annualize_cov(cov_shrunk_daily)

    gmv_sample_lo = global_min_variance_weights(cov_sample, long_only=True)
    gmv_shrunk_lo = global_min_variance_weights(cov_shrunk, long_only=True)
    gmv_sample_uc = global_min_variance_weights(cov_sample, long_only=False)
    gmv_shrunk_uc = global_min_variance_weights(cov_shrunk, long_only=False)

    tan_sample = max_sharpe_weights_long_only(mu, cov_sample, risk_free)
    tan_shrunk = max_sharpe_weights_long_only(mu, cov_shrunk, risk_free)

    return {
        "tickers": tickers,
        "n_obs": n_obs,
        "shrinkage": shrinkage,
        "cond_sample": condition_number(cov_sample),
        "cond_shrunk": condition_number(cov_shrunk),
        "gmv_long_only": (gmv_sample_lo, gmv_shrunk_lo, weight_shift(gmv_sample_lo.weights, gmv_shrunk_lo.weights)),
        "gmv_unconstrained": (
            gmv_sample_uc,
            gmv_shrunk_uc,
            weight_shift(gmv_sample_uc.weights, gmv_shrunk_uc.weights),
        ),
        "tangency_long_only": (
            tan_sample,
            tan_shrunk,
            weight_shift(tan_sample.weights, tan_shrunk.weights) if tan_sample.success and tan_shrunk.success else None,
        ),
    }


def _format_gmv_pair(tickers: list[str], label: str, sample: GMVPoint, shrunk: GMVPoint, shift: dict) -> list[str]:
    lines = [f"{label}:"]
    if not (sample.success and shrunk.success):
        lines.append("  not computed - one side failed to converge")
        return lines
    for t, ws, wl in zip(tickers, sample.weights, shrunk.weights):
        lines.append(f"  {t:<14}sample={ws:>+9.4%}   shrunk={wl:>+9.4%}   delta={wl - ws:>+9.4%}")
    lines.append(f"  vol: sample={sample.vol:.4%}   shrunk={shrunk.vol:.4%}")
    lines.append(f"  weight movement: L1={shift['l1']:.4%}   max|delta|={shift['max_abs']:.4%}")
    return lines


def _format_tangency_pair(tickers: list[str], sample: TangencyPoint, shrunk: TangencyPoint, shift: dict | None) -> list[str]:
    lines = ["tangency portfolio, long-only (uses mu - shown for completeness, not the isolated comparison):"]
    if shift is None:
        lines.append("  not computed - one side failed to converge")
        return lines
    for t, ws, wl in zip(tickers, sample.weights, shrunk.weights):
        lines.append(f"  {t:<14}sample={ws:>+9.4%}   shrunk={wl:>+9.4%}   delta={wl - ws:>+9.4%}")
    lines.append(f"  Sharpe: sample={sample.sharpe:+.4f}   shrunk={shrunk.sharpe:+.4f}")
    lines.append(f"  weight movement: L1={shift['l1']:.4%}   max|delta|={shift['max_abs']:.4%}")
    return lines


def _format_report(result: dict) -> str:
    tickers = result["tickers"]
    lines = []
    lines.append(f"tickers: {', '.join(tickers)}   n_obs={result['n_obs']} days")
    lines.append(f"Ledoit-Wolf shrinkage intensity: {result['shrinkage']:.4f} (0=pure sample, 1=pure scaled-identity target)")
    lines.append(
        f"covariance condition number: sample={result['cond_sample']:.2f}   shrunk={result['cond_shrunk']:.2f}"
    )
    lines.append("")
    lines.append(
        "global minimum-variance portfolio (covariance only, no mu - the isolated comparison this day is about):"
    )
    lines.extend(_format_gmv_pair(tickers, "  long-only (w >= 0)", *result["gmv_long_only"]))
    lines.append("")
    lines.extend(_format_gmv_pair(tickers, "  unconstrained (long/short)", *result["gmv_unconstrained"]))
    lines.append("")
    lines.extend(_format_tangency_pair(tickers, *result["tangency_long_only"]))
    lines.append("")
    lines.append("estimation-error note:")
    if result["shrinkage"] < 0.15:
        lines.append(
            f"  shrinkage intensity is small ({result['shrinkage']:.1%}) and the sample covariance was"
        )
        lines.append(
            f"  already reasonably well-conditioned (cond={result['cond_sample']:.1f}) - with only "
            f"{len(tickers)} assets"
        )
        lines.append(
            f"  and {result['n_obs']} days of history, T >> n, which is exactly the regime where shrinkage"
        )
        lines.append(
            "  has the least to fix. The weight swings above are correspondingly modest. This is the"
        )
        lines.append(
            "  honest result for this fixture universe, not evidence shrinkage doesn't matter in general -"
        )
        lines.append(
            "  it earns its keep most in higher-dimensional, shorter-window settings than this one."
        )
    else:
        lines.append(
            f"  shrinkage intensity is material ({result['shrinkage']:.1%}) - the sample covariance carried"
        )
        lines.append(
            "  enough estimation noise that the optimizer is meaningfully sensitive to which covariance"
        )
        lines.append("  estimate it is handed. Treat either weight vector above as a point estimate, not a fact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    result = run(tickers, args.risk_free_rate, args.fixtures_dir)
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
