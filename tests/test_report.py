from optimizer.report import format_weights_table, run

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def test_run_produces_a_solved_min_variance_point():
    result = run(TICKERS, risk_free=0.0, fixtures_dir="fixtures/ohlcv")
    assert result["min_var_point"] is not None
    assert result["min_var_point"].success


def test_run_produces_a_successful_tangency_portfolio():
    result = run(TICKERS, risk_free=0.0, fixtures_dir="fixtures/ohlcv")
    assert result["tangency"].success
    assert len(result["tangency"].weights) == len(TICKERS)


def test_weights_table_lists_every_ticker_and_both_portfolios():
    result = run(TICKERS, risk_free=0.0, fixtures_dir="fixtures/ohlcv")
    table = format_weights_table(result)
    for ticker in TICKERS:
        assert ticker in table
    assert "min-variance" in table
    assert "max-Sharpe" in table


def test_frontier_weights_sum_to_one():
    result = run(TICKERS, risk_free=0.0, fixtures_dir="fixtures/ohlcv")
    mv = result["min_var_point"]
    assert abs(sum(mv.weights) - 1.0) < 1e-6
    tan = result["tangency"]
    assert abs(sum(tan.weights) - 1.0) < 1e-6
