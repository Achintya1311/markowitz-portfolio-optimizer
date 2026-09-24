"""Sector caps, position limits, and a turnover penalty on the frontier (Day 5).

    python -m optimizer.constraints --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS \\
        --target-return -0.15 --max-position 0.5 --turnover-lambda 5

Day 2's frontier weights are not a portfolio anyone can actually hold: a
single name can be the whole book, a mandate typically caps exposure to any
one sector, and rebalancing into a fresh optimum every period costs money
the frontier ignores. This module solves the same minimum-variance-for-a-
target-return problem as :func:`optimizer.frontier.min_variance_weights`,
with three more layers on top:

- **Position limits** - ``w <= max_position`` per asset (and, long/short,
  ``w >= -max_position``), a tighter bound than the plain long-only ``w >=
  0``.
- **Sector caps** - ``sum(w[i] for i in sector) <= cap``, one inequality
  constraint per capped sector.
- **Turnover penalty** - a quadratic penalty ``turnover_lambda * sum((w -
  w_prev) ** 2)`` added to the objective, discouraging large moves away
  from a reference portfolio. The more standard turnover cost is L1
  (``sum(|w - w_prev|)``), but that is non-differentiable at zero, which
  SLSQP (used throughout this package) is not built for; the quadratic form
  is smooth, still penalizes every unit of turnover, and the resulting
  weights' actual L1 distance from the reference is reported alongside it
  so the penalty's effect is still visible in the units that matter.

Sector classification is static public information, not price data, and
needs no fetch: this pipeline's 3-ticker fixture universe's NSE sector
membership (Reliance Industries: Energy; Tata Chemicals: Chemicals;
Crompton Greaves Consumer Electricals: Consumer Durables) is hardcoded
below as :data:`DEFAULT_SECTOR_MAP`, the same classification any index
provider publishes. Because all three names sit in *different* sectors, a
sector cap applied to this real universe is mathematically identical to a
plain per-name position limit - the module is genuinely exercised against
a synthetic multi-sector universe in the test suite instead, and this
degeneracy is reported plainly rather than dressed up as a demonstration
this fixture set cannot give.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from optimizer.covariance import sample_mean_cov
from optimizer.frontier import min_variance_weights
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix

_SLSQP_OPTIONS = {"maxiter": 1000, "ftol": 1e-12}
_FEASIBILITY_TOL = 1e-6
_BINDING_TOL = 1e-4

DEFAULT_SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TATACHEM.NS": "Chemicals",
    "CROMPTON.NS": "Consumer Durables",
}


@dataclass
class ConstrainedPoint:
    """One solved minimum-variance portfolio under position/sector/turnover constraints."""

    target_return: float
    achieved_return: float
    variance: float
    weights: np.ndarray
    prev_weights: np.ndarray
    turnover_l1: float
    turnover_lambda: float
    long_only: bool
    max_position: float | None
    sector_caps: dict[str, float] = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)
    binding_position_limits: list[str] = field(default_factory=list)
    binding_sector_caps: list[str] = field(default_factory=list)
    success: bool = False

    @property
    def vol(self) -> float:
        return self.variance**0.5


def _sector_groups(tickers: list[str], sector_map: dict[str, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, ticker in enumerate(tickers):
        if ticker not in sector_map:
            raise ValueError(f"no sector assigned to {ticker!r} - pass --sector-map or extend DEFAULT_SECTOR_MAP")
        groups.setdefault(sector_map[ticker], []).append(i)
    return groups


def constrained_min_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    tickers: list[str],
    long_only: bool = True,
    sector_map: dict[str, str] | None = None,
    sector_caps: dict[str, float] | None = None,
    max_position: float | None = None,
    turnover_lambda: float = 0.0,
    prev_weights: np.ndarray | None = None,
) -> ConstrainedPoint:
    """Minimum-variance weights for a target return, under sector/position/turnover constraints.

    Same feasibility-by-residual pattern as :mod:`optimizer.frontier` and
    :mod:`optimizer.tangency`: ``success`` checks the budget, target-return,
    bound, and sector-cap constraints directly against the solved weights
    rather than trusting SLSQP's own exit flag alone.
    """
    n = len(mu)
    if n < 2:
        raise ValueError("need at least 2 assets for a constrained frontier")
    if max_position is not None and max_position <= 0:
        raise ValueError("max_position must be positive")

    if prev_weights is None:
        prev_weights = np.full(n, 1.0 / n)
    else:
        prev_weights = np.asarray(prev_weights, dtype=np.float64)
        if len(prev_weights) != n:
            raise ValueError("prev_weights must have one entry per ticker")

    lower = 0.0 if long_only else (-max_position if max_position is not None else None)
    upper = max_position
    bounds = [(lower, upper) for _ in range(n)]

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]

    sector_caps = sector_caps or {}
    if sector_caps and sector_map is None:
        raise ValueError("sector_caps given but sector_map is None")
    groups = _sector_groups(tickers, sector_map) if sector_caps else {}
    for sector, cap in sector_caps.items():
        if sector not in groups:
            raise ValueError(f"sector cap given for {sector!r}, which no ticker is assigned to")
        idx = groups[sector]
        constraints.append({"type": "ineq", "fun": (lambda w, idx=idx, cap=cap: cap - np.sum(w[idx]))})

    def objective(w: np.ndarray) -> float:
        variance = float(w @ cov @ w)
        penalty = turnover_lambda * float(np.sum((w - prev_weights) ** 2))
        return variance + penalty

    x0 = np.full(n, 1.0 / n)
    if lower is not None:
        x0 = np.maximum(x0, lower)
    if upper is not None:
        x0 = np.minimum(x0, upper)

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options=_SLSQP_OPTIONS
    )

    weights = result.x
    budget_ok = abs(np.sum(weights) - 1.0) < _FEASIBILITY_TOL
    return_ok = abs(float(weights @ mu) - target_return) < _FEASIBILITY_TOL
    bounds_ok = True
    if lower is not None:
        bounds_ok = bounds_ok and bool(np.all(weights >= lower - _FEASIBILITY_TOL))
    if upper is not None:
        bounds_ok = bounds_ok and bool(np.all(weights <= upper + _FEASIBILITY_TOL))
    caps_ok = all(np.sum(weights[groups[sector]]) <= cap + _FEASIBILITY_TOL for sector, cap in sector_caps.items())
    success = bool(result.success) and budget_ok and return_ok and bounds_ok and caps_ok

    binding_position = []
    if max_position is not None:
        for ticker, w in zip(tickers, weights):
            hit_upper = abs(w - upper) < _BINDING_TOL
            # The long-only lower bound is always 0 regardless of max_position, so a
            # weight of 0 only reflects the position *limit* (not just long-only) when
            # short positions are allowed down to -max_position.
            hit_lower = (not long_only) and abs(w - lower) < _BINDING_TOL
            if hit_upper or hit_lower:
                binding_position.append(ticker)

    sector_weights = {sector: float(np.sum(weights[idx])) for sector, idx in groups.items()}
    binding_sectors = [
        sector for sector, cap in sector_caps.items() if abs(sector_weights[sector] - cap) < _BINDING_TOL
    ]

    return ConstrainedPoint(
        target_return=target_return,
        achieved_return=float(weights @ mu),
        variance=float(weights @ cov @ weights),
        weights=weights,
        prev_weights=prev_weights,
        turnover_l1=float(np.sum(np.abs(weights - prev_weights))),
        turnover_lambda=turnover_lambda,
        long_only=long_only,
        max_position=max_position,
        sector_caps=sector_caps,
        sector_weights=sector_weights,
        binding_position_limits=binding_position,
        binding_sector_caps=binding_sectors,
        success=success,
    )


def _parse_kv_floats(spec: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        result[key.strip()] = float(value)
    return result


def _parse_sector_map(spec: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        ticker, sep, sector = pair.partition("=")
        if not sep:
            raise ValueError(f"expected TICKER=SECTOR, got {pair!r}")
        result[ticker.strip()] = sector.strip()
    return result


def _parse_prev_weights(spec: str, n: int) -> np.ndarray:
    if spec == "equal":
        return np.full(n, 1.0 / n)
    values = [float(v.strip()) for v in spec.split(",") if v.strip()]
    if len(values) != n:
        raise ValueError(f"--prev-weights has {len(values)} values, expected {n} (one per ticker)")
    weights = np.array(values, dtype=np.float64)
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(f"--prev-weights must sum to 1.0, got {weights.sum():.6f}")
    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimum-variance portfolio for a target return under sector caps, position limits, "
        "and a turnover penalty (Day 5)."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--target-return", type=float, required=True, help="annualized target portfolio return, e.g. -0.15"
    )
    parser.add_argument("--long-short", action="store_true", help="allow short positions (default: long-only)")
    parser.add_argument(
        "--max-position", type=float, default=None, help="per-asset weight cap, e.g. 0.5 for 50%% (default: none)"
    )
    parser.add_argument(
        "--sector-map",
        default=None,
        help="TICKER=SECTOR,... (default: DEFAULT_SECTOR_MAP for the shared fixture universe)",
    )
    parser.add_argument(
        "--sector-cap", default=None, help="SECTOR=CAP,... e.g. Energy=0.5,Chemicals=0.5 (default: no caps)"
    )
    parser.add_argument(
        "--turnover-lambda", type=float, default=0.0, help="quadratic turnover penalty coefficient (default: 0.0)"
    )
    parser.add_argument(
        "--prev-weights",
        default="equal",
        help="reference portfolio for the turnover penalty: 'equal' or comma-separated weights summing to 1 "
        "(default: equal)",
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    return parser


def run(
    tickers: list[str],
    target_return: float,
    fixtures_dir: str,
    long_only: bool = True,
    sector_map: dict[str, str] | None = None,
    sector_caps: dict[str, float] | None = None,
    max_position: float | None = None,
    turnover_lambda: float = 0.0,
    prev_weights: np.ndarray | None = None,
) -> dict:
    _, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    mu = annualize_mean(mu_daily)
    cov = annualize_cov(cov_daily)

    if sector_map is None:
        sector_map = {t: DEFAULT_SECTOR_MAP[t] for t in tickers if t in DEFAULT_SECTOR_MAP}
        if sector_caps and len(sector_map) != len(tickers):
            unmapped = [t for t in tickers if t not in sector_map]
            raise ValueError(f"no default sector for {unmapped} - pass --sector-map")

    baseline = min_variance_weights(mu, cov, target_return, long_only)
    constrained = constrained_min_variance(
        mu,
        cov,
        target_return,
        tickers,
        long_only=long_only,
        sector_map=sector_map,
        sector_caps=sector_caps,
        max_position=max_position,
        turnover_lambda=turnover_lambda,
        prev_weights=prev_weights,
    )

    is_real_fixture_universe = set(tickers) == set(DEFAULT_SECTOR_MAP)
    sectors_all_distinct = sector_map is not None and len(set(sector_map.values())) == len(sector_map)

    return {
        "tickers": tickers,
        "mu_annual": mu,
        "sector_map": sector_map,
        "baseline": baseline,
        "constrained": constrained,
        "degenerate_sector_caps": sectors_all_distinct and bool(sector_caps),
        "is_real_fixture_universe": is_real_fixture_universe,
    }


def _format_point_line(tickers: list[str], weights: np.ndarray) -> str:
    return ", ".join(f"{t}={w:+.4f}" for t, w in zip(tickers, weights))


def _format_report(result: dict) -> str:
    tickers = result["tickers"]
    baseline = result["baseline"]
    constrained = result["constrained"]
    lines = []
    lines.append(f"tickers: {', '.join(tickers)}")
    lines.append(f"target return: {constrained.target_return:+.4%}")
    lines.append("sector map: " + ", ".join(f"{t}={s}" for t, s in result["sector_map"].items()))
    lines.append("")

    lines.append("baseline (unconstrained-beyond-budget-and-target, Day 2's frontier):")
    if baseline.success:
        lines.append(f"  weights: {_format_point_line(tickers, baseline.weights)}")
        lines.append(f"  vol: {baseline.variance ** 0.5:.4%}")
    else:
        lines.append("  infeasible - target return is not reachable")
    lines.append("")

    lines.append("constrained (sector caps + position limit + turnover penalty):")
    if not constrained.success:
        lines.append("  infeasible under these constraints")
    else:
        lines.append(f"  weights: {_format_point_line(tickers, constrained.weights)}")
        lines.append(f"  vol: {constrained.variance ** 0.5:.4%}")
        if constrained.max_position is not None:
            binding = ", ".join(constrained.binding_position_limits) or "none"
            lines.append(f"  position limit: {constrained.max_position:.2%}   binding: {binding}")
        if constrained.sector_caps:
            for sector, cap in constrained.sector_caps.items():
                actual = constrained.sector_weights[sector]
                binding = " (binding)" if sector in constrained.binding_sector_caps else ""
                lines.append(f"  sector cap {sector}: {actual:.2%} / {cap:.2%}{binding}")
        lines.append(f"  turnover penalty: lambda={constrained.turnover_lambda:g}")
    lines.append("")

    lines.append(f"reference portfolio (turnover penalty target): {_format_point_line(tickers, constrained.prev_weights)}")
    lines.append(f"L1 turnover of constrained solution vs reference: {constrained.turnover_l1:.4%}")
    if baseline.success:
        baseline_turnover = float(np.sum(np.abs(baseline.weights - constrained.prev_weights)))
        lines.append(f"L1 turnover of baseline (unpenalized) vs reference: {baseline_turnover:.4%}")
    lines.append("")

    if result["degenerate_sector_caps"] and result["is_real_fixture_universe"]:
        lines.append(
            "note: this 3-ticker fixture universe has each ticker in its own sector, so a sector cap here "
            "is mathematically identical to a per-name position limit - it does not exercise sector caps as "
            "a distinct constraint. See tests/test_constraints.py for a synthetic multi-sector universe "
            "where sector caps genuinely bind differently from position limits, and the README's "
            "limitations section."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    sector_map = _parse_sector_map(args.sector_map) if args.sector_map else None
    sector_caps = _parse_kv_floats(args.sector_cap) if args.sector_cap else None
    prev_weights = _parse_prev_weights(args.prev_weights, len(tickers))

    result = run(
        tickers,
        args.target_return,
        args.fixtures_dir,
        long_only=not args.long_short,
        sector_map=sector_map,
        sector_caps=sector_caps,
        max_position=args.max_position,
        turnover_lambda=args.turnover_lambda,
        prev_weights=prev_weights,
    )
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
