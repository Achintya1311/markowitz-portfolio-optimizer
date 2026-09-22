"""Loaders for daily log returns from committed OHLCV fixtures.

Everything downstream (covariance estimation, the frontier, the tangency
portfolio) is built on daily log returns read straight off ``fixtures/ohlcv``,
never a live fetch - see the repo README for why the fixture path is the
default.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TRADING_DAYS_PER_YEAR = 252

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "ohlcv"


def fixture_path(ticker: str, fixtures_dir: str | Path = FIXTURES_DIR) -> Path:
    """Map a ticker like ``RELIANCE.NS`` to its committed fixture file."""
    return Path(fixtures_dir) / f"{ticker.replace('.', '_')}.csv"


def load_closes(csv_path: str | Path) -> np.ndarray:
    """Read a ``date,close,...`` fixture and return the close column.

    Rows are assumed to already be in ascending date order, which is how
    every fixture in this pipeline is written.
    """
    closes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            closes.append(float(row["close"]))

    if len(closes) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 rows of close prices, cannot compute returns")

    return np.asarray(closes, dtype=np.float64)


def load_dates(csv_path: str | Path) -> list[str]:
    """Read a ``date,close,...`` fixture and return just the date column, in file order."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["date"] for row in reader]


def load_daily_log_returns(csv_path: str | Path) -> np.ndarray:
    """Read a ``date,close,...`` fixture and return daily log returns."""
    return np.diff(np.log(load_closes(csv_path)))


def load_returns_matrix(
    tickers: list[str], fixtures_dir: str | Path = FIXTURES_DIR
) -> tuple[list[str], np.ndarray]:
    """Load daily log returns for several tickers, aligned on a shared calendar.

    Returns ``(dates, matrix)`` where ``matrix`` has shape ``(n_days, n_tickers)``
    and ``dates`` are the ``n_days`` return dates (one shorter than the price
    calendar, since a log return needs two prices). Raises ``ValueError`` if
    the tickers' fixtures don't share the exact same trading-day calendar -
    covariance estimated from misaligned rows is silently wrong, not just
    noisy, so this is checked rather than assumed.
    """
    if len(tickers) < 2:
        raise ValueError("need at least 2 tickers to estimate a covariance matrix")

    all_dates = [load_dates(fixture_path(t, fixtures_dir)) for t in tickers]
    reference = all_dates[0]
    for ticker, dates in zip(tickers[1:], all_dates[1:]):
        if dates != reference:
            raise ValueError(
                f"{ticker}'s trading-day calendar does not match {tickers[0]}'s - "
                "cannot align returns into one covariance matrix"
            )

    columns = [load_daily_log_returns(fixture_path(t, fixtures_dir)) for t in tickers]
    return reference[1:], np.column_stack(columns)


def annualize_mean(mu_daily: np.ndarray | float) -> np.ndarray | float:
    """Scale a daily log-return mean (per asset) to an annualized figure (iid assumption)."""
    return mu_daily * TRADING_DAYS_PER_YEAR


def annualize_cov(cov_daily: np.ndarray) -> np.ndarray:
    """Scale a daily log-return covariance matrix to annualized figures (iid assumption)."""
    return cov_daily * TRADING_DAYS_PER_YEAR
