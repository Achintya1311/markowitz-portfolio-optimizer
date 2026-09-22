import numpy as np
import pytest

from optimizer.returns import (
    TRADING_DAYS_PER_YEAR,
    annualize_cov,
    annualize_mean,
    fixture_path,
    load_closes,
    load_daily_log_returns,
    load_dates,
    load_returns_matrix,
)

FIXTURE = "fixtures/ohlcv/RELIANCE_NS.csv"
TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def test_fixture_path_maps_dot_to_underscore():
    assert fixture_path("RELIANCE.NS").name == "RELIANCE_NS.csv"


def test_load_closes_matches_row_count():
    closes = load_closes(FIXTURE)
    with open(FIXTURE) as f:
        n_rows = sum(1 for _ in f) - 1  # minus header
    assert len(closes) == n_rows
    assert (closes > 0).all()


def test_load_daily_log_returns_is_one_shorter_than_closes():
    closes = load_closes(FIXTURE)
    returns = load_daily_log_returns(FIXTURE)
    assert len(returns) == len(closes) - 1


def test_load_daily_log_returns_matches_manual_diff_log():
    closes = load_closes(FIXTURE)
    expected = np.diff(np.log(closes))
    returns = load_daily_log_returns(FIXTURE)
    np.testing.assert_allclose(returns, expected)


def test_load_closes_rejects_too_short_file(tmp_path):
    p = tmp_path / "one_row.csv"
    p.write_text("date,close\n2024-01-01,100\n")
    with pytest.raises(ValueError, match="fewer than 2 rows"):
        load_closes(str(p))


def test_annualize_mean_scales_by_trading_days():
    assert annualize_mean(0.001) == pytest.approx(0.001 * TRADING_DAYS_PER_YEAR)


def test_annualize_cov_scales_by_trading_days():
    cov_daily = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    cov_annual = annualize_cov(cov_daily)
    np.testing.assert_allclose(cov_annual, cov_daily * TRADING_DAYS_PER_YEAR)


def test_load_returns_matrix_shape_and_dates():
    dates, matrix = load_returns_matrix(TICKERS)
    closes = load_closes(FIXTURE)
    assert matrix.shape == (len(closes) - 1, len(TICKERS))
    assert len(dates) == matrix.shape[0]
    assert dates == load_dates(FIXTURE)[1:]


def test_load_returns_matrix_columns_match_individual_loads():
    dates, matrix = load_returns_matrix(TICKERS)
    for i, ticker in enumerate(TICKERS):
        expected = load_daily_log_returns(fixture_path(ticker))
        np.testing.assert_allclose(matrix[:, i], expected)


def test_load_returns_matrix_rejects_single_ticker():
    with pytest.raises(ValueError, match="at least 2 tickers"):
        load_returns_matrix(["RELIANCE.NS"])


def test_load_returns_matrix_rejects_mismatched_calendar(tmp_path):
    aligned_dir = tmp_path
    (aligned_dir / "A.csv").write_text(
        "date,close\n2024-01-01,100\n2024-01-02,101\n2024-01-03,102\n"
    )
    (aligned_dir / "B.csv").write_text(
        "date,close\n2024-01-01,50\n2024-01-02,51\n"
    )
    with pytest.raises(ValueError, match="calendar"):
        load_returns_matrix(["A", "B"], fixtures_dir=aligned_dir)
