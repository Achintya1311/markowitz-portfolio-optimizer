"""Fetch daily OHLCV fixtures for a 25-30-name NSE large-cap universe.

D4 (hub CRITIQUES.md): the committed fixture is only 3 tickers
(RELIANCE.NS, TATACHEM.NS, CROMPTON.NS) over about two years, which is why
the unconstrained long-only tangency solve is a 100% single-stock corner
in 9 of 11 walk-forward folds (``optimizer.oos``) - 3 names leave almost no
room for diversification to matter. This widens the universe with the
same free source the existing fixture already used (``yfinance``,
``auto_adjust=True``; see README "Data sources"), not a new one, and
writes the exact shape ``optimizer/returns.py`` already reads: one
``date,close,high,low,open,volume`` CSV per ticker under
``fixtures/ohlcv/``, ascending by date, every ticker sharing the identical
trading-day calendar (``load_returns_matrix`` raises if they don't).

Meant to run from ``.github/workflows/fetch-largecaps.yml``, triggered by
pushing this file or the workflow file, or ``workflow_dispatch`` - not by
hand from a cloud sandbox, since the free yfinance endpoint intermittently
429s or drops a NaN last bar from a cloud IP (hub CRITIQUES.md D8), and
Actions runners draw from a larger, less-flagged IP pool.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ohlcv"

START_DATE = "2015-01-01"

# 30 Nifty 50 constituents, picked for a continuous NSE listing well before
# 2015 (so no ticker's own IPO or demerger date silently shrinks the shared
# calendar below) and spread across more than the existing fixture's three
# sectors.
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "HCLTECH.NS", "DRREDDY.NS", "GRASIM.NS", "HEROMOTOCO.NS",
]

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5


def fixture_path(ticker: str, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    """Map a ticker like ``RELIANCE.NS`` to its fixture file - same rule as
    ``optimizer.returns.fixture_path``, duplicated here rather than
    imported so this script has no dependency on the package it feeds."""
    return fixtures_dir / f"{ticker.replace('.', '_')}.csv"


def download_one(ticker: str, start: str, end: str | None, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Download one ticker's OHLCV history, retrying on a rate limit or an
    empty/all-NaN response (D8: yfinance returns 429s and NaN last bars)."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        except Exception as exc:  # yfinance raises a mix of requests/JSON errors on 429
            last_error = exc
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            df = df.dropna(subset=["Close"])
            if not df.empty:
                return df
        time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    detail = f" (last error: {last_error})" if last_error else ""
    raise RuntimeError(f"{ticker}: no usable data after {max_retries} attempts{detail}")


def common_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersect every ticker's trading days into one shared calendar.

    ``optimizer/returns.py``'s ``load_returns_matrix`` requires every
    fixture to carry the exact same date column - a corporate action, a
    trading halt, or a late IPO in just one name must not silently
    misalign the rest.
    """
    calendars = [pd.DatetimeIndex(df.index) for df in frames.values()]
    shared = calendars[0]
    for cal in calendars[1:]:
        shared = shared.intersection(cal)
    return shared.sort_values()


def write_fixture(
    ticker: str, df: pd.DataFrame, calendar: pd.DatetimeIndex, fixtures_dir: Path = FIXTURES_DIR
) -> Path:
    aligned = df.loc[calendar]
    path = fixture_path(ticker, fixtures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "close", "high", "low", "open", "volume"])
        for ts, row in aligned.iterrows():
            writer.writerow(
                [
                    ts.strftime("%Y-%m-%d"),
                    round(float(row["Close"]), 4),
                    round(float(row["High"]), 4),
                    round(float(row["Low"]), 4),
                    round(float(row["Open"]), 4),
                    int(row["Volume"]),
                ]
            )
    return path


def fetch_and_write(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
) -> list[Path]:
    """Best-effort fetch: a ticker that still has no usable data after
    ``download_one``'s own retries (D4/Day 9 - e.g. TATAMOTORS.NS 404ing
    on yfinance as "Quote not found") is skipped and logged, not allowed to
    abort every other ticker's fetch."""
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            frames[t] = download_one(t, start, end)
        except RuntimeError as exc:
            print(f"skipping {t}: {exc}", file=sys.stderr)
    if len(frames) < 2:
        raise RuntimeError(
            f"fewer than 2 tickers had usable data ({len(frames)}/{len(tickers)} succeeded)"
        )
    calendar = common_calendar(frames)
    if len(calendar) < 2:
        raise RuntimeError("no shared trading days across the requested tickers - check the ticker list")
    return [write_fixture(t, df, calendar, fixtures_dir) for t, df in frames.items()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="comma-separated NSE tickers, e.g. RELIANCE.NS,TCS.NS",
    )
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=None, help="default: today (yfinance's own default)")
    parser.add_argument("--fixtures-dir", default=str(FIXTURES_DIR))
    args = parser.parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        print("need at least 2 tickers", file=sys.stderr)
        return 1

    paths = fetch_and_write(tickers, start=args.start, end=args.end, fixtures_dir=Path(args.fixtures_dir))
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
