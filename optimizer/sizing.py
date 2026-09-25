"""Position sizing for the spine (v0.4 contract, integration day).

    python -m optimizer.sizing --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS \\
        --risk-free-rate 0.0 --sector-cap 0.5 \\
        --contract outputs/sizing_contract.json --contract-ticker RELIANCE.NS

Days 3, 4, 6 and 7 all landed on the same finding: this 3-asset universe's
long-only max-Sharpe tangency portfolio -- sample or Ledoit-Wolf shrunk
covariance alike -- is a 100% single-name corner solution (RELIANCE.NS,
negative Sharpe), a direct consequence of having only three assets to pick
from. That number is not a position size any real book would take, so it is
not what this module publishes as "how much of this name to hold". Instead
it combines Day 4's Ledoit-Wolf shrunk covariance (the method named in
NEXT_STEPS.md's committed contract, ``max_sharpe_ledoit_wolf``) with Day 5's
sector-cap machinery: maximize Sharpe subject to the same budget/long-only
bounds Day 3 used, plus a per-sector weight cap. Nothing here is new
optimization theory -- the objective is Day 3's, the cap constraints are Day
5's -- just the two already-tested pieces combined for the one question a
sizing gate needs answered.

On the real 3-ticker fixture universe every ticker sits in its own NSE
sector (Day 5's own finding, restated here rather than hidden): a sector cap
is mathematically identical to a plain position limit on that one name. The
contract's committed shape still names the binding constraint
``sector_cap``, since that is what actually bound the weight, with the
degeneracy stated plainly in the CLI output and the README.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from optimizer.constraints import DEFAULT_SECTOR_MAP
from optimizer.covariance import sample_mean_cov
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix
from optimizer.shrinkage import ledoit_wolf_shrinkage
from optimizer.tangency import TangencyPoint, max_sharpe_weights_long_only

_SLSQP_OPTIONS = {"maxiter": 500, "ftol": 1e-12}
_FEASIBILITY_TOL = 1e-6
_BINDING_TOL = 1e-4

DEFAULT_SECTOR_CAP = 0.5
METHOD = "max_sharpe_ledoit_wolf"


@dataclass
class SizingPoint:
    """A sector-capped, long-only max-Sharpe tangency portfolio on a shrunk covariance."""

    weights: np.ndarray
    achieved_return: float
    variance: float
    sharpe: float
    sector_map: dict[str, str]
    sector_caps: dict[str, float]
    sector_weights: dict[str, float] = field(default_factory=dict)
    binding_sector_caps: list[str] = field(default_factory=list)
    success: bool = False

    @property
    def vol(self) -> float:
        return self.variance ** 0.5


def _sector_groups(tickers: list[str], sector_map: dict[str, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, ticker in enumerate(tickers):
        if ticker not in sector_map:
            raise ValueError(f"no sector assigned to {ticker!r} - pass --sector-map or extend DEFAULT_SECTOR_MAP")
        groups.setdefault(sector_map[ticker], []).append(i)
    return groups


def max_sharpe_weights_sector_capped(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float,
    tickers: list[str],
    sector_map: dict[str, str],
    sector_caps: dict[str, float],
) -> SizingPoint:
    """Long-only max-Sharpe tangency subject to per-sector weight caps.

    Same SLSQP negative-Sharpe objective as
    :func:`optimizer.tangency.max_sharpe_weights_long_only`, with Day 5's
    sector-cap inequality constraints added on top, checked against the
    solved weights directly rather than trusting SLSQP's own exit flag
    alone -- the same feasibility-by-residual pattern used throughout this
    package.
    """
    n = len(mu)
    if n < 2:
        raise ValueError("need at least 2 assets for a sizing solve")

    groups = _sector_groups(tickers, sector_map)
    for sector, cap in sector_caps.items():
        if sector not in groups:
            raise ValueError(f"sector cap given for {sector!r}, which no ticker is assigned to")
        if not 0 < cap <= 1:
            raise ValueError(f"sector cap for {sector!r} must be in (0, 1], got {cap}")

    def negative_sharpe(w: np.ndarray) -> float:
        port_return = w @ mu
        port_vol = float(w @ cov @ w) ** 0.5
        if port_vol == 0.0:
            return 0.0
        return -(port_return - risk_free) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for sector, cap in sector_caps.items():
        idx = groups[sector]
        constraints.append({"type": "ineq", "fun": (lambda w, idx=idx, cap=cap: cap - np.sum(w[idx]))})

    bounds = [(0.0, None) for _ in range(n)]
    x0 = np.full(n, 1.0 / n)

    result = minimize(
        negative_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints, options=_SLSQP_OPTIONS
    )

    weights = result.x
    budget_ok = abs(np.sum(weights) - 1.0) < _FEASIBILITY_TOL
    bounds_ok = bool(np.all(weights >= -_FEASIBILITY_TOL))
    sector_weights = {sector: float(np.sum(weights[idx])) for sector, idx in groups.items()}
    caps_ok = all(sector_weights[s] <= cap + _FEASIBILITY_TOL for s, cap in sector_caps.items())
    success = bool(result.success) and budget_ok and bounds_ok and caps_ok

    binding = [s for s, cap in sector_caps.items() if abs(sector_weights[s] - cap) < _BINDING_TOL]

    achieved_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    sharpe = (achieved_return - risk_free) / (variance ** 0.5) if variance > 0 else float("nan")

    return SizingPoint(
        weights=weights,
        achieved_return=achieved_return,
        variance=variance,
        sharpe=sharpe,
        sector_map=sector_map,
        sector_caps=sector_caps,
        sector_weights=sector_weights,
        binding_sector_caps=binding,
        success=success,
    )


def run(
    tickers: list[str],
    risk_free: float,
    fixtures_dir: str,
    sector_cap: float,
    sector_map: dict[str, str] | None = None,
) -> dict:
    _, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    mu_daily, _ = sample_mean_cov(returns_matrix)
    shrinkage, cov_shrunk_daily = ledoit_wolf_shrinkage(returns_matrix)
    mu = annualize_mean(mu_daily)
    cov_shrunk = annualize_cov(cov_shrunk_daily)

    if sector_map is None:
        sector_map = {t: DEFAULT_SECTOR_MAP[t] for t in tickers if t in DEFAULT_SECTOR_MAP}
        missing = [t for t in tickers if t not in sector_map]
        if missing:
            raise ValueError(f"no default sector for {missing} - pass --sector-map")

    # dict.fromkeys, not set(...), to keep sector order deterministic (a
    # plain set's iteration order is randomized per-process for strings,
    # which would otherwise make the CLI's own output non-reproducible).
    sector_caps = {sector: sector_cap for sector in dict.fromkeys(sector_map[t] for t in tickers)}

    unconstrained = max_sharpe_weights_long_only(mu, cov_shrunk, risk_free)
    capped = max_sharpe_weights_sector_capped(mu, cov_shrunk, risk_free, tickers, sector_map, sector_caps)

    is_real_fixture_universe = set(tickers) == set(DEFAULT_SECTOR_MAP)
    sectors_all_distinct = len(set(sector_map.values())) == len(sector_map)

    return {
        "tickers": tickers,
        "mu_annual": mu,
        "risk_free": risk_free,
        "shrinkage": shrinkage,
        "sector_map": sector_map,
        "sector_cap": sector_cap,
        "unconstrained": unconstrained,
        "capped": capped,
        "degenerate_sector_caps": sectors_all_distinct and is_real_fixture_universe,
        "is_real_fixture_universe": is_real_fixture_universe,
    }


def to_contract(result: dict, ticker: str) -> dict:
    """The ``sizing`` block this repo publishes to the spine (v0.4), matching
    the shape ``NEXT_STEPS.md`` committed to before this module existed:
    ``{"sizing": {"weight", "method", "constraint_binding"}}``.

    ``weight`` is the sector-capped solve's weight for ``ticker``, never the
    raw unconstrained corner solution Days 3/4/6/7 already found and
    reported as degenerate. ``constraint_binding`` names the sector whose
    cap actually bound that ticker's weight, or ``"none"`` if it did not
    bind (e.g. a ticker the optimizer wanted less of than its cap allows).
    """
    tickers = result["tickers"]
    if ticker not in tickers:
        raise ValueError(f"{ticker!r} not in {tickers}")
    capped = result["capped"]
    if not capped.success:
        raise ValueError("sector-capped sizing solve did not converge - refusing to publish a contract from it")

    idx = tickers.index(ticker)
    sector = result["sector_map"][ticker]
    constraint_binding = sector if sector in capped.binding_sector_caps else "none"

    return {
        "sizing": {
            "weight": round(float(capped.weights[idx]), 6),
            "method": METHOD,
            "constraint_binding": constraint_binding,
        }
    }


def _format_point_line(tickers: list[str], weights: np.ndarray) -> str:
    return ", ".join(f"{t}={w:+.4f}" for t, w in zip(tickers, weights))


def _format_report(result: dict) -> str:
    tickers = result["tickers"]
    lines = []
    lines.append(f"tickers: {', '.join(tickers)}")
    lines.append(f"risk-free rate: {result['risk_free']:+.4%}")
    lines.append(f"Ledoit-Wolf shrinkage intensity: {result['shrinkage']:.4f} (Day 4's centrepiece estimator)")
    lines.append("sector map: " + ", ".join(f"{t}={s}" for t, s in result["sector_map"].items()))
    lines.append(f"sector cap: {result['sector_cap']:.2%} per sector")
    lines.append("")

    unconstrained = result["unconstrained"]
    lines.append("unconstrained max-Sharpe tangency (Ledoit-Wolf shrunk covariance, Days 3+4's method):")
    if unconstrained.success:
        lines.append(f"  weights: {_format_point_line(tickers, unconstrained.weights)}")
        lines.append(f"  return: {unconstrained.achieved_return:+.4%}   vol: {unconstrained.vol:.4%}   Sharpe: {unconstrained.sharpe:+.4f}")
    else:
        lines.append("  no feasible solution")
    lines.append("")

    capped = result["capped"]
    lines.append("sector-capped max-Sharpe tangency (what this repo actually publishes, v0.4):")
    if not capped.success:
        lines.append("  infeasible under these sector caps")
    else:
        lines.append(f"  weights: {_format_point_line(tickers, capped.weights)}")
        lines.append(f"  return: {capped.achieved_return:+.4%}   vol: {capped.vol:.4%}   Sharpe: {capped.sharpe:+.4f}")
        for sector, cap in capped.sector_caps.items():
            actual = capped.sector_weights[sector]
            binding = " (binding)" if sector in capped.binding_sector_caps else ""
            lines.append(f"  sector {sector}: {actual:.2%} / {cap:.2%}{binding}")
    lines.append("")

    if result["degenerate_sector_caps"]:
        lines.append(
            "note: this 3-ticker fixture universe has each ticker in its own sector, so a sector cap here "
            "is mathematically identical to a per-name position limit - it does not exercise sector caps as "
            "a distinct constraint (same degeneracy Day 5's optimizer.constraints already found on this "
            "universe). The contract still names the binding sector, since that is what actually capped the "
            "weight; see the README's limitations section."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sector-capped, Ledoit-Wolf-shrunk max-Sharpe position sizing "
        "- the v0.4 integration day, published to STOCKSTALKER as a file contract."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.0,
        help="annualized risk-free rate (default: 0.0, see optimizer.tangency for why)",
    )
    parser.add_argument(
        "--sector-cap", type=float, default=DEFAULT_SECTOR_CAP,
        help=f"uniform per-sector weight cap applied to every sector (default: {DEFAULT_SECTOR_CAP:.0%})",
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    parser.add_argument(
        "--contract",
        metavar="PATH",
        default=None,
        help="write the v0.4 sizing contract block (see to_contract()) as JSON to this path, "
        "for the spine to read as a file -- never as a Python import",
    )
    parser.add_argument(
        "--contract-ticker", default=None, help="the ticker --contract is solved for (required with --contract)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if bool(args.contract) != bool(args.contract_ticker):
        parser.error("--contract and --contract-ticker must be given together")
    if args.contract_ticker and args.contract_ticker not in tickers:
        parser.error(f"--contract-ticker {args.contract_ticker!r} must be one of --tickers")

    result = run(tickers, args.risk_free_rate, args.fixtures_dir, args.sector_cap)
    print(_format_report(result))

    if args.contract:
        contract_path = Path(args.contract)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(to_contract(result, args.contract_ticker), indent=2) + "\n")
        print(f"wrote sizing contract for {args.contract_ticker} to {contract_path}")

    return 0 if result["capped"].success else 1


if __name__ == "__main__":
    raise SystemExit(main())
