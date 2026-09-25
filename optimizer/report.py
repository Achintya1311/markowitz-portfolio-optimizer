"""Frontier chart and a weights terminal table for the long-only portfolio (Day 7).

    python -m optimizer.report --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS --risk-free-rate 0.0

Every earlier day printed weights inline (``TICKER=+0.1234, ...``) as part of
a longer report. This is the one-page summary Days 2-4 built toward: the
long-only efficient frontier in mean-vol space (:mod:`optimizer.frontier`),
the maximum-Sharpe tangency portfolio and capital market line
(:mod:`optimizer.tangency`), plotted together and reported as an aligned
table rather than free text. It computes nothing new - every number here is
:mod:`optimizer.frontier` or :mod:`optimizer.tangency`'s own output, laid
out differently.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optimizer.covariance import sample_mean_cov
from optimizer.frontier import FrontierPoint, build_frontier
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix
from optimizer.tangency import (
    DEFAULT_RISK_FREE_RATE,
    TangencyPoint,
    capital_market_line,
    max_sharpe_weights_long_only,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_N_POINTS = 25


def plot_frontier(
    tickers: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    frontier: list[FrontierPoint],
    tangency: TangencyPoint,
    cml_vols: np.ndarray,
    cml_returns: np.ndarray,
    out_path: str | Path,
) -> None:
    """Write the mean-vol frontier, tangency point, CML and asset points to ``out_path``.

    Matplotlib with the ``Agg`` backend, same as every other chart in this
    portfolio (see e.g. ``mcsim.commodity.plot_curve_regime``) - no display
    is available or needed, only a file in ``outputs/``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solved = [p for p in frontier if p.success]
    vols = [p.variance**0.5 for p in solved]
    returns = [p.achieved_return for p in solved]

    asset_vols = np.sqrt(np.diag(cov))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(vols, returns, color="#4c72b0", linewidth=1.6, label="long-only min-variance frontier")
    ax.scatter(asset_vols, mu, color="#888888", zorder=3, label="individual assets")
    for t, v, r in zip(tickers, asset_vols, mu):
        ax.annotate(t, (v, r), textcoords="offset points", xytext=(5, 3), fontsize=8, color="#555555")

    if tangency.success:
        ax.plot(cml_vols, cml_returns, color="#dd8452", linestyle="--", linewidth=1.2, label="capital market line")
        ax.scatter([tangency.vol], [tangency.achieved_return], color="#c44e52", marker="*", s=160, zorder=4, label="max-Sharpe tangency (long-only)")

    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized return")
    ax.set_title("Efficient frontier — long-only, sample covariance")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    tickers: list[str],
    risk_free: float,
    fixtures_dir: str,
    n_points: int = DEFAULT_N_POINTS,
) -> dict:
    _, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    mu = annualize_mean(mu_daily)
    cov = annualize_cov(cov_daily)

    frontier = build_frontier(mu, cov, long_only=True, n_points=n_points)
    solved = [p for p in frontier if p.success]
    min_var_point = min(solved, key=lambda p: p.variance) if solved else None

    tangency = max_sharpe_weights_long_only(mu, cov, risk_free)
    cml_vols, cml_returns = capital_market_line(risk_free, tangency.achieved_return, tangency.vol) if tangency.success else (np.array([]), np.array([]))

    return {
        "tickers": tickers,
        "mu_annual": mu,
        "cov_annual": cov,
        "risk_free": risk_free,
        "frontier": frontier,
        "min_var_point": min_var_point,
        "tangency": tangency,
        "cml_vols": cml_vols,
        "cml_returns": cml_returns,
    }


def _weight_row(label: str, tickers: list[str], weights: np.ndarray | None, ret: float | None, vol: float | None) -> str:
    if weights is None:
        return f"  {label:<22}  no feasible solution"
    weight_cells = "  ".join(f"{w:>+9.4f}" for w in weights)
    return f"  {label:<22}  {weight_cells}  {ret:>+9.4%}  {vol:>9.4%}"


def format_weights_table(result: dict) -> str:
    tickers = result["tickers"]
    header_cells = "  ".join(f"{t:>9}" for t in tickers)
    lines = [
        f"  {'portfolio':<22}  {header_cells}  {'return':>9}  {'vol':>9}",
    ]

    mv = result["min_var_point"]
    if mv is not None:
        lines.append(_weight_row("min-variance", tickers, mv.weights, mv.achieved_return, mv.variance**0.5))
    else:
        lines.append(_weight_row("min-variance", tickers, None, None, None))

    tan = result["tangency"]
    if tan.success:
        lines.append(_weight_row("max-Sharpe tangency", tickers, tan.weights, tan.achieved_return, tan.vol))
    else:
        lines.append(_weight_row("max-Sharpe tangency", tickers, None, None, None))

    return "\n".join(lines)


def _format_report(result: dict, chart_path: str | None) -> str:
    tickers = result["tickers"]
    lines = []
    lines.append(f"tickers: {', '.join(tickers)}")
    lines.append("annualized mean (sample): " + ", ".join(f"{t}={m:+.4%}" for t, m in zip(tickers, result["mu_annual"])))
    lines.append(f"risk-free rate: {result['risk_free']:+.4%}")
    lines.append("")
    lines.append("weights (long-only):")
    lines.append(format_weights_table(result))
    lines.append("")

    n_solved = sum(1 for p in result["frontier"] if p.success)
    lines.append(f"frontier: {n_solved}/{len(result['frontier'])} target returns feasible on the long-only sweep")

    if chart_path:
        lines.append(f"wrote frontier chart to {chart_path}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frontier chart to outputs/ and a weights terminal table (Day 7)."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE,
        help="annualized risk-free rate for the tangency portfolio (default: 0.0, see optimizer.tangency)",
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    parser.add_argument(
        "--n-points", type=int, default=DEFAULT_N_POINTS, help=f"target returns to sweep for the frontier (default: {DEFAULT_N_POINTS})"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="directory the frontier chart is written to (default: outputs/)")
    parser.add_argument("--no-chart", action="store_true", help="skip writing the frontier chart PNG")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    result = run(tickers, args.risk_free_rate, args.fixtures_dir, args.n_points)

    chart_path = None
    if not args.no_chart:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = "-".join(t.replace(".", "_") for t in tickers)
        chart_path = str(out_dir / f"frontier_{slug}.png")
        plot_frontier(
            tickers, result["mu_annual"], result["cov_annual"], result["frontier"],
            result["tangency"], result["cml_vols"], result["cml_returns"], chart_path,
        )

    print(_format_report(result, chart_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
