"""P129 part 1: scripts/fetch_largecaps.py also writes a Nifty 50 index
fixture. All offline: download_one is monkeypatched, no network."""

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import fetch_largecaps  # noqa: E402


def _frame(dates, closes):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame({"Close": closes, "Volume": [0] * len(closes)}, index=idx)


def test_index_fixture_path_lives_outside_the_stock_universe():
    path = fetch_largecaps.index_fixture_path("^NSEI")
    assert path.name == "NSEI.csv"
    assert path.parent.name == "index"
    assert path.parent != fetch_largecaps.FIXTURES_DIR


def test_write_index_fixture_writes_date_close_ascending(tmp_path):
    df = _frame(["2024-01-03", "2024-01-01", "2024-01-02"], [102.0, 100.0, 101.123456])
    path = fetch_largecaps.write_index_fixture("^NSEI", df, tmp_path)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["date", "close"]
    assert rows[1:] == [["2024-01-01", "100.0"], ["2024-01-02", "101.1235"], ["2024-01-03", "102.0"]]


def test_fetch_index_uses_its_own_calendar_not_the_stocks(tmp_path, monkeypatch):
    # The index keeps its own trading days; Day 194 aligns it to the
    # stock calendar when it is read, so a missing stock day drops nothing here.
    monkeypatch.setattr(
        fetch_largecaps, "download_one",
        lambda t, s, e: _frame(["2024-01-01", "2024-01-02"], [10.0, 11.0]),
    )
    path = fetch_largecaps.fetch_index("^NSEI", "2024-01-01", None, tmp_path)
    assert path == tmp_path / "NSEI.csv"
    assert path.read_text().splitlines() == ["date,close", "2024-01-01,10.0", "2024-01-02,11.0"]


def test_fetch_index_failure_is_logged_not_raised(tmp_path, monkeypatch, capsys):
    def boom(t, s, e):
        raise RuntimeError(f"{t}: no usable data")

    monkeypatch.setattr(fetch_largecaps, "download_one", boom)
    assert fetch_largecaps.fetch_index("^NSEI", "2024-01-01", None, tmp_path) is None
    assert "skipping index ^NSEI" in capsys.readouterr().err
    assert not (tmp_path / "NSEI.csv").exists()


def test_main_writes_index_after_the_universe(tmp_path, monkeypatch):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

    def fake(t, s, e):
        df = _frame(dates, [100.0, 101.0, 102.0])
        for col in ("Open", "High", "Low"):
            df[col] = df["Close"]
        return df

    monkeypatch.setattr(fetch_largecaps, "download_one", fake)
    rc = fetch_largecaps.main(
        ["--tickers", "A.NS,B.NS", "--fixtures-dir", str(tmp_path / "u"),
         "--index-dir", str(tmp_path / "i")]
    )
    assert rc == 0
    assert (tmp_path / "i" / "NSEI.csv").exists()
    assert (tmp_path / "u" / "A_NS.csv").exists()


def test_main_index_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_largecaps, "download_one", lambda t, s, e: (_ for _ in ()).throw(AssertionError(t)))
    monkeypatch.setattr(fetch_largecaps, "fetch_and_write", lambda *a, **k: [])
    assert fetch_largecaps.main(["--tickers", "A.NS,B.NS", "--index", "", "--index-dir", str(tmp_path)]) == 0
    assert not any(tmp_path.iterdir())
