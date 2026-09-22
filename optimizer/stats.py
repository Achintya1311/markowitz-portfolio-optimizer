"""CLI: estimate annualized mean/covariance for a ticker basket and report
how much estimation error sits inside those numbers (Day 1).

    python -m optimizer.stats --tickers RELIANCE.NS,TATACHEM.NS,CROMPTON.NS

Runs offline against ``fixtures/ohlcv`` by default. This is deliberately not
the frontier or the optimizer itself - it is the input diagnostic those later
days build on, and the frontier is only as trustworthy as the numbers
reported here.
"""
from __future__ import annotations

import argparse

from optimizer.covariance import (
    condition_number,
    correlation_matrix,
    mean_t_stats,
    sample_mean_cov,
    split_half_stability,
    standard_error_of_annualized_mean,
)
from optimizer.returns import FIXTURES_DIR, annualize_cov, annualize_mean, load_returns_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate annualized mean/covariance from OHLCV fixtures, with an estimation-error report."
    )
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"
    )
    parser.add_argument(
        "--fixtures-dir", default=str(FIXTURES_DIR), help="directory of <TICKER>.csv fixtures (default: fixtures/ohlcv)"
    )
    return parser


def run(tickers: list[str], fixtures_dir: str) -> dict:
    dates, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    n_obs = returns_matrix.shape[0]

    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    mu_annual = annualize_mean(mu_daily)
    cov_annual = annualize_cov(cov_daily)
    se_annual = standard_error_of_annualized_mean(cov_daily, n_obs)
    t_stats = mean_t_stats(mu_annual, se_annual)
    corr = correlation_matrix(cov_annual)
    cond = condition_number(cov_annual)
    split = split_half_stability(tickers, returns_matrix)

    return {
        "tickers": tickers,
        "n_obs": n_obs,
        "date_range": (dates[0], dates[-1]),
        "mu_annual": mu_annual,
        "cov_annual": cov_annual,
        "se_annual": se_annual,
        "t_stats": t_stats,
        "corr": corr,
        "condition_number": cond,
        "split_half": split,
    }


def _format_report(result: dict) -> str:
    lines = []
    tickers = result["tickers"]
    lines.append(
        f"n_obs={result['n_obs']} days   range={result['date_range'][0]} .. {result['date_range'][1]}"
    )
    lines.append("")
    lines.append(f"{'ticker':<14}{'mean (ann.)':>14}{'vol (ann.)':>14}{'SE(mean)':>14}{'t-stat':>10}")
    for i, ticker in enumerate(tickers):
        vol_annual = result["cov_annual"][i, i] ** 0.5
        lines.append(
            f"{ticker:<14}{result['mu_annual'][i]:>14.4%}{vol_annual:>14.4%}"
            f"{result['se_annual'][i]:>14.4%}{result['t_stats'][i]:>10.2f}"
        )
    lines.append("")
    lines.append("correlation matrix (annualized):")
    header = "".join(f"{t:>14}" for t in tickers)
    lines.append(" " * 14 + header)
    for i, ticker in enumerate(tickers):
        row = "".join(f"{result['corr'][i, j]:>14.3f}" for j in range(len(tickers)))
        lines.append(f"{ticker:<14}{row}")
    lines.append("")
    lines.append(f"covariance condition number: {result['condition_number']:.1f}")
    lines.append("")
    lines.append("estimation-error note:")
    max_abs_t = max(abs(t) for t in result["t_stats"])
    if max_abs_t < 2.0:
        lines.append(
            "  every mean-return estimate above has |t| < 2 - none is statistically"
        )
        lines.append(
            "  distinguishable from a zero expected return at this sample size. Any"
        )
        lines.append(
            "  frontier or tangency portfolio built from these means in later days is"
        )
        lines.append("  ranking assets on noise, not signal, until stated otherwise.")
    else:
        lines.append(
            f"  at least one mean-return estimate has |t| >= 2 (max |t|={max_abs_t:.2f}),"
        )
        lines.append(
            "  but treat that as weak evidence, not proof - see the split-half check below."
        )
    split = result["split_half"]
    lines.append(
        f"  covariance matrix moved {split.cov_frobenius_relative_change:.1%} (Frobenius norm,"
    )
    lines.append("  relative) between the first and second half of the window; per-asset")
    lines.append("  annualized mean shifted by:")
    for ticker, shift in zip(tickers, split.mean_shift):
        lines.append(f"    {ticker:<14}{shift:>+9.4%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    result = run(tickers, args.fixtures_dir)
    print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
