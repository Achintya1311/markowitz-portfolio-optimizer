"""Walk-forward comparison of portfolio rules against equal weight, with drift-aware costs.

    python -m optimizer.methods --universe largecap --train-window 252

Criteria and candidates were fixed before any result was computed: ``docs/PREREG_methods_vs_equal_weight.md``.

Every rule is evaluated on the same folds as ``optimizer.oos`` (rolling training window, non-overlapping hold)
and costed the same way, with one correction: turnover is measured from the weights a portfolio has *drifted*
to by the end of the previous hold, back to its new target. The older benchmark paid no turnover because it
reset to 1/N with no drift; that favoured equal weight, so here equal weight pays for rebalancing too.

P129: the Nifty 50 (``fixtures/index/NSEI.csv``, fetched in Actions) is shown over the same holds as a
passive reference, with no costs. It is a price index with no dividends, while the stock fixtures are
dividend-adjusted, so the comparison is biased against the index by roughly the Nifty dividend yield.
It is kept out of ``METHODS`` and the pre-registered verdict, which were fixed before it existed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from optimizer.covariance import sample_mean_cov
from optimizer.oos import TURNOVER_COST_RATE, total_simple_returns, walk_forward_folds
from optimizer.returns import annualize_cov, annualize_mean, fixture_path, load_dates, load_returns_matrix
from optimizer.shrinkage import global_min_variance_weights, ledoit_wolf_shrinkage
from optimizer.tangency import max_sharpe_weights_long_only

ROOT = Path(__file__).resolve().parent.parent
LARGECAP_DIR = ROOT / "fixtures" / "ohlcv_largecap"
INDEX_FIXTURE = ROOT / "fixtures" / "index" / "NSEI.csv"
INDEX_CAVEAT = "Nifty 50 price index, no dividends (stocks are dividend-adjusted), no costs"
FOLDS_PER_YEAR = 252 / 21


def equal_weight(train: np.ndarray) -> np.ndarray:
    n = train.shape[1]
    return np.full(n, 1.0 / n)


def min_variance(train: np.ndarray) -> np.ndarray:
    _, cov = ledoit_wolf_shrinkage(train)
    p = global_min_variance_weights(cov, long_only=True)
    return p.weights if p.success else equal_weight(train)


def inverse_vol(train: np.ndarray) -> np.ndarray:
    inv = 1.0 / train.std(axis=0, ddof=1)
    return inv / inv.sum()


def blend(train: np.ndarray) -> np.ndarray:
    return 0.5 * min_variance(train) + 0.5 * equal_weight(train)


def max_sharpe(train: np.ndarray) -> np.ndarray:
    """The repo's previous method, unchanged: long-only max-Sharpe on Ledoit-Wolf covariance, rf = 0."""
    mu_daily, _ = sample_mean_cov(train)
    _, cov = ledoit_wolf_shrinkage(train)
    sol = max_sharpe_weights_long_only(annualize_mean(mu_daily), annualize_cov(cov), 0.0)
    return sol.weights if sol.success else equal_weight(train)


METHODS = {"equal_weight": equal_weight, "min_variance": min_variance, "inverse_vol": inverse_vol,
           "blend_50_50": blend, "max_sharpe": max_sharpe}


def drift(weights: np.ndarray, test_returns: np.ndarray) -> np.ndarray:
    """Weights after a hold in which each asset earned its own total return and nothing was traded."""
    grown = weights * (1.0 + test_returns)
    return grown / grown.sum()


def walk_forward(returns: np.ndarray, rule, train_window: int, test_window: int = 21, cost_rate: float = TURNOVER_COST_RATE) -> dict:
    gross, net, turnover = [], [], []
    held = None
    for tr, te in walk_forward_folds(returns.shape[0], train_window, test_window):
        target = rule(returns[tr])
        prev = held if held is not None else target          # the very first allocation is not charged, for every rule alike
        t = float(np.abs(target - prev).sum())
        r = total_simple_returns(returns[te])
        g = float(target @ r)
        gross.append(g); net.append(g - cost_rate * t); turnover.append(t)
        held = drift(target, r)
    return {"gross": np.array(gross), "net": np.array(net), "turnover": np.array(turnover)}


def load_index_returns(price_dates: list[str], path: Path = INDEX_FIXTURE):
    """Daily log returns of the index on the stock price calendar, or None if the fixture is missing.

    The index has its own calendar. A stock day with no index close carries the previous close
    (a 0 return); a stock day before the first index close takes that first close. Returns
    ``(returns, n_filled)`` so the caller can say how many days were filled."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        closes = {row["date"]: float(row["close"]) for row in csv.DictReader(f)}
    first = min(closes)
    aligned, last, filled = [], closes[first], 0
    for d in price_dates:
        if d in closes:
            last = closes[d]
        else:
            filled += 1
        aligned.append(last)
    return np.diff(np.log(aligned)), filled


def index_walk_forward(index_returns: np.ndarray, train_window: int, test_window: int = 21) -> np.ndarray:
    """Price return of the index over each hold of the same folds every rule uses; no cost, no turnover."""
    return np.array([float(total_simple_returns(index_returns[te]))
                     for _, te in walk_forward_folds(len(index_returns), train_window, test_window)])


def summarize(net: np.ndarray, turnover: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    sd = float(net.std(ddof=1))
    return {"compounded_net": float(eq[-1] - 1.0), "sharpe_net": float(net.mean() / sd * np.sqrt(FOLDS_PER_YEAR)) if sd else float("nan"),
            "vol_annual": sd * float(np.sqrt(FOLDS_PER_YEAR)), "max_drawdown": float((eq / peak - 1.0).min()),
            "avg_turnover": float(turnover.mean()), "n_folds": int(len(net))}


def sharpe_diff_ci(a: np.ndarray, b: np.ndarray, reps: int = 5000, seed: int = 0) -> dict:
    """95% interval for Sharpe(a) - Sharpe(b) from a stationary bootstrap over paired fold returns."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    x = np.column_stack([a, b])
    blk = int(min(max(round(float(optimal_block_length(x)["stationary"].mean())), 1), 10))
    ann = float(np.sqrt(FOLDS_PER_YEAR))
    sh = lambda m: m[:, 0].mean() / m[:, 0].std(ddof=1) * ann - m[:, 1].mean() / m[:, 1].std(ddof=1) * ann
    point = sh(x)
    draws = np.array([sh(d[0][0]) for d in StationaryBootstrap(blk, x, seed=seed).bootstrap(reps)])
    return {"diff": float(point), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)), "block": blk}


def run(returns: np.ndarray, train_window: int, index_returns: np.ndarray | None = None) -> dict:
    res = {name: walk_forward(returns, rule, train_window) for name, rule in METHODS.items()}
    out = {"train_window": train_window, "methods": {n: summarize(v["net"], v["turnover"]) for n, v in res.items()}}
    ew = res["equal_weight"]["net"]
    for n in METHODS:
        if n != "equal_weight":
            out["methods"][n]["sharpe_diff_vs_equal_weight"] = sharpe_diff_ci(res[n]["net"], ew)
    if index_returns is not None:
        net = index_walk_forward(index_returns, train_window)
        out["index"] = summarize(net, np.zeros(len(net)))
        out["index"]["sharpe_diff_vs_equal_weight"] = sharpe_diff_ci(net, ew)
    return out


def verdict(w252: dict, w126: dict, name: str = "min_variance") -> dict:
    def core(r):
        m, e = r["methods"][name], r["methods"]["equal_weight"]
        return {"sharpe_ge_equal_weight": m["sharpe_net"] >= e["sharpe_net"],
                "ci_not_entirely_below_zero": m["sharpe_diff_vs_equal_weight"]["hi"] >= 0.0,
                "vol_lower": m["vol_annual"] < e["vol_annual"],
                "drawdown_no_worse": m["max_drawdown"] >= e["max_drawdown"]}
    c252, c126 = core(w252), core(w126)
    out = {**c252, "turnover_le_20pct": w252["methods"][name]["avg_turnover"] <= 0.20, "holds_at_126d": all(c126.values())}
    out["all_pass"] = all(out.values())
    return out


def build_report(tickers: list[str], fixtures_dir: str, dates, w252: dict, verdict_mv: dict) -> dict:
    """D38: the large-cap headline as plain JSON, small enough to diff, so the
    weekly data refresh can regenerate it next to the data and a test can check
    the two agree. Covers README lines 7-8 and the site's Portfolio card: the
    original max-Sharpe walk-forward (``optimizer.oos``, 252/21, rf 0) and this
    module's 252-day methods table and pre-registered verdict."""
    from optimizer import oos

    wf = oos.run(tickers, fixtures_dir, train_window=252, test_window=21, risk_free=0.0)
    r6 = lambda x: None if x is None else round(float(x), 6)  # noqa: E731
    keep = ("n_folds", "compounded_net", "sharpe_net", "vol_annual", "max_drawdown", "avg_turnover")
    report = {
        "fixture_last_date": str(dates[-1])[:10],
        "n_assets": len(tickers),
        "walk_forward_252": {
            "n_folds": len(wf.folds),
            "optimizer_before_costs": r6(wf.optimizer_compounded_return),
            "optimizer_after_costs": r6(wf.optimizer_compounded_return_net),
            "equal_weight_before_costs": r6(wf.equal_weight_compounded_return),
            "t_stat_excess_after_costs": r6(wf.t_stat_excess_net),
        },
        "methods_252": {n: {k: (m[k] if k == "n_folds" else r6(m[k])) for k in keep}
                        for n, m in w252["methods"].items()},
        "verdict_min_variance": verdict_mv,
    }
    if "index" in w252:
        report["index_252"] = {k: (w252["index"][k] if k == "n_folds" else r6(w252["index"][k])) for k in keep}
    return report


README_TABLE_ROWS = {
    "Equal weight": "equal_weight",
    "Minimum variance": "min_variance",
    "Inverse volatility": "inverse_vol",
    "50/50 blend": "blend_50_50",
    "Max-Sharpe (previous method)": "max_sharpe",
    "Nifty 50 price index (no dividends, no costs)": "index",
}


def update_readme_table(text: str, report: dict) -> str:
    """D58: rewrite the README's pre-registered table from the report, so the
    weekly refresh keeps the two in step. Only the five report columns change;
    the Sharpe-difference interval column (not in the report) is kept."""
    methods = report["methods_252"]
    lines = text.split("\n")
    for i, line in enumerate(lines):
        cells = line.split("|")
        if len(cells) < 9 or cells[1].strip() not in README_TABLE_ROWS:
            continue
        key = README_TABLE_ROWS[cells[1].strip()]
        if key == "index":
            if "index_252" not in report:
                continue
            m = report["index_252"]
        else:
            m = methods[key]
        values = (f"{m['compounded_net']:+.1%}", f"{m['sharpe_net']:.2f}", f"{m['vol_annual']:.1%}",
                  f"{m['max_drawdown']:.1%}", f"{m['avg_turnover']:.1%}")
        for j, v in enumerate(values, start=2):
            cells[j] = f" {v} "
        lines[i] = "|".join(cells)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures-dir", default=str(LARGECAP_DIR))
    ap.add_argument("--json", default=None)
    ap.add_argument("--index-fixture", default=str(INDEX_FIXTURE),
                    help="date,close index fixture for the passive leg (P129); skipped with a message if missing")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="also write the compact headline report (reports/largecap_latest.json is this, committed)")
    ap.add_argument("--update-readme", default=None, metavar="PATH",
                    help="with --report: rewrite the pre-registered table in this README from the report (D58)")
    a = ap.parse_args(argv)
    tickers = sorted(p.stem.replace("_", ".") for p in Path(a.fixtures_dir).glob("*.csv"))
    dates, returns = load_returns_matrix(tickers, a.fixtures_dir)
    loaded = load_index_returns(load_dates(fixture_path(tickers[0], a.fixtures_dir)), a.index_fixture)
    index_returns = loaded[0] if loaded else None
    w252, w126 = run(returns, 252, index_returns), run(returns, 126, index_returns)
    out = {"n_assets": len(tickers), "w252": w252, "w126": w126, "verdict_min_variance": verdict(w252, w126)}
    for label, r in (("252-day training window", w252), ("126-day training window", w126)):
        print(f"\n{label}, {r['methods']['equal_weight']['n_folds']} folds, {len(tickers)} assets, costs {TURNOVER_COST_RATE:.4%} per unit turnover (drift-aware)")
        print(f"  {'method':<14}{'net compounded':>15}{'net Sharpe':>11}{'vol':>8}{'max DD':>9}{'turnover':>10}   Sharpe diff vs EW (95% interval)")
        for n, m in r["methods"].items():
            d = m.get("sharpe_diff_vs_equal_weight")
            ds = f"{d['diff']:+.2f} ({d['lo']:+.2f} to {d['hi']:+.2f})" if d else ""
            print(f"  {n:<14}{m['compounded_net']:>+15.1%}{m['sharpe_net']:>11.2f}{m['vol_annual']:>8.1%}{m['max_drawdown']:>9.1%}{m['avg_turnover']:>10.1%}   {ds}")
        if "index" in r:
            m, d = r["index"], r["index"]["sharpe_diff_vs_equal_weight"]
            print(f"  {'nifty50_price':<14}{m['compounded_net']:>+15.1%}{m['sharpe_net']:>11.2f}{m['vol_annual']:>8.1%}{m['max_drawdown']:>9.1%}{'n/a':>10}   "
                  f"{d['diff']:+.2f} ({d['lo']:+.2f} to {d['hi']:+.2f})")
    if loaded:
        print(f"\nnifty50_price: {INDEX_CAVEAT}; same holds as every rule; "
              f"{loaded[1]} stock-calendar days with no index close carry the previous close")
    else:
        print(f"\nindex leg skipped: no index fixture at {a.index_fixture} (fetched weekly in Actions, P129)")
    print("\npre-registered verdict for min_variance:", json.dumps(out["verdict_min_variance"]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    if a.report:
        report = build_report(tickers, a.fixtures_dir, dates, w252, out["verdict_min_variance"])
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
        print(f"report written to {a.report}")
        if a.update_readme:
            readme = Path(a.update_readme)
            readme.write_text(update_readme_table(readme.read_text(), report))
            print(f"table updated in {a.update_readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
