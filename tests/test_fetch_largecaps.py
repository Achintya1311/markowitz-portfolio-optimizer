"""Tests for scripts/fetch_largecaps.py. All offline: no network, no keys."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import fetch_largecaps  # noqa: E402

from optimizer.returns import load_closes, load_dates, load_returns_matrix  # noqa: E402


def _frame(dates: list[str], close: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    n = len(idx)
    return pd.DataFrame(
        {
            "Open": [close] * n,
            "High": [close + 1] * n,
            "Low": [close - 1] * n,
            "Close": [close + i * 0.1 for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


def test_common_calendar_is_the_intersection_not_the_union():
    frames = {
        "A.NS": _frame(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "B.NS": _frame(["2024-01-01", "2024-01-03", "2024-01-04"]),
    }
    calendar = fetch_largecaps.common_calendar(frames)
    assert list(calendar.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-03"]


def test_common_calendar_is_sorted_regardless_of_input_order():
    frames = {
        "A.NS": _frame(["2024-01-03", "2024-01-01", "2024-01-02"]),
        "B.NS": _frame(["2024-01-02", "2024-01-03", "2024-01-01"]),
    }
    calendar = fetch_largecaps.common_calendar(frames)
    assert list(calendar.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_write_fixture_round_trips_through_the_loader_the_optimizer_uses(tmp_path):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
    df = _frame(dates)
    calendar = pd.DatetimeIndex(pd.to_datetime(dates))

    path = fetch_largecaps.write_fixture("TEST.NS", df, calendar, fixtures_dir=tmp_path)

    assert load_dates(path) == dates
    closes = load_closes(path)
    assert closes[0] == pytest.approx(100.0)
    assert closes[-1] == pytest.approx(100.2)


def test_write_fixture_uses_the_same_filename_rule_as_optimizer_returns(tmp_path):
    df = _frame(["2024-01-01", "2024-01-02"])
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    path = fetch_largecaps.write_fixture("RELIANCE.NS", df, calendar, fixtures_dir=tmp_path)
    assert path.name == "RELIANCE_NS.csv"


def test_fetch_and_write_aligns_every_ticker_onto_one_shared_calendar(tmp_path, monkeypatch):
    raw = {
        "A.NS": _frame(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "B.NS": _frame(["2024-01-01", "2024-01-03"], close=50.0),
    }

    def fake_download_one(ticker, start, end, max_retries=4):
        return raw[ticker]

    monkeypatch.setattr(fetch_largecaps, "download_one", fake_download_one)

    paths = fetch_largecaps.fetch_and_write(["A.NS", "B.NS"], fixtures_dir=tmp_path)

    assert len(paths) == 2
    assert load_dates(paths[0]) == load_dates(paths[1]) == ["2024-01-01", "2024-01-03"]

    # the two fixtures now share the exact calendar optimizer.returns requires
    dates, matrix = load_returns_matrix(["A.NS", "B.NS"], fixtures_dir=tmp_path)
    assert len(dates) == 1
    assert matrix.shape == (1, 2)


def test_fetch_and_write_skips_a_permanently_failing_ticker_instead_of_aborting_the_whole_batch(
    tmp_path, monkeypatch
):
    # D4/Day 9 (hub CRITIQUES.md): the real DEFAULT_TICKERS run hit exactly
    # this -- TATAMOTORS.NS 404s on yfinance ("Quote not found") after 4
    # retries, and the old fetch_and_write let that one RuntimeError from
    # download_one abort the dict comprehension for the whole 30-ticker
    # batch, so the Actions workflow's "best effort" fetch step quietly
    # wrote zero fixtures every single run.
    raw = {
        "A.NS": _frame(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "C.NS": _frame(["2024-01-01", "2024-01-02", "2024-01-03"], close=50.0),
    }

    def fake_download_one(ticker, start, end, max_retries=4):
        if ticker == "BAD.NS":
            raise RuntimeError("BAD.NS: no usable data after 4 attempts")
        return raw[ticker]

    monkeypatch.setattr(fetch_largecaps, "download_one", fake_download_one)

    paths = fetch_largecaps.fetch_and_write(["A.NS", "BAD.NS", "C.NS"], fixtures_dir=tmp_path)

    assert len(paths) == 2
    assert {p.name for p in paths} == {"A_NS.csv", "C_NS.csv"}
    assert not (tmp_path / "BAD_NS.csv").exists()


def test_fetch_and_write_raises_if_too_few_tickers_survive_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_largecaps,
        "download_one",
        lambda ticker, start, end, max_retries=4: (_ for _ in ()).throw(
            RuntimeError(f"{ticker}: no usable data after 4 attempts")
        ),
    )

    with pytest.raises(RuntimeError, match="fewer than 2 tickers had usable data"):
        fetch_largecaps.fetch_and_write(["A.NS", "B.NS"], fixtures_dir=tmp_path)


def test_fetch_and_write_raises_if_tickers_share_no_trading_day(tmp_path, monkeypatch):
    raw = {
        "A.NS": _frame(["2024-01-01"]),
        "B.NS": _frame(["2024-01-02"]),
    }
    monkeypatch.setattr(fetch_largecaps, "download_one", lambda ticker, start, end, max_retries=4: raw[ticker])

    with pytest.raises(RuntimeError, match="no shared trading days"):
        fetch_largecaps.fetch_and_write(["A.NS", "B.NS"], fixtures_dir=tmp_path)


def test_download_one_retries_then_raises_when_every_attempt_is_empty(monkeypatch):
    calls = []

    def always_empty(ticker, start=None, end=None, auto_adjust=True, progress=False):
        calls.append(ticker)
        return pd.DataFrame()

    monkeypatch.setattr(fetch_largecaps.yf, "download", always_empty)
    monkeypatch.setattr(fetch_largecaps.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="no usable data after 2 attempts"):
        fetch_largecaps.download_one("X.NS", "2015-01-01", None, max_retries=2)

    assert len(calls) == 2


def test_download_one_drops_an_incomplete_nan_last_bar(monkeypatch):
    good_rows = _frame(["2024-01-01", "2024-01-02"])
    with_nan_bar = pd.concat([good_rows, _frame(["2024-01-03"])])
    with_nan_bar.loc[with_nan_bar.index[-1], "Close"] = float("nan")

    monkeypatch.setattr(
        fetch_largecaps.yf,
        "download",
        lambda ticker, start=None, end=None, auto_adjust=True, progress=False: with_nan_bar,
    )

    df = fetch_largecaps.download_one("X.NS", "2024-01-01", None)

    assert len(df) == 2
    assert not df["Close"].isna().any()


def test_default_tickers_has_25_to_30_names_and_no_duplicates():
    assert 25 <= len(fetch_largecaps.DEFAULT_TICKERS) <= 30
    assert len(set(fetch_largecaps.DEFAULT_TICKERS)) == len(fetch_largecaps.DEFAULT_TICKERS)
    assert all(t.endswith(".NS") for t in fetch_largecaps.DEFAULT_TICKERS)


def test_main_rejects_fewer_than_two_tickers(capsys):
    rc = fetch_largecaps.main(["--tickers", "RELIANCE.NS"])
    assert rc == 1
    assert "at least 2 tickers" in capsys.readouterr().err
