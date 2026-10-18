import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.oos", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_and_reports_fold_count():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "folds=11" in result.stdout
    assert "walk-forward:" in result.stdout


def test_cli_reports_compounded_returns_and_win_rate():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "compounded return over all" in result.stdout
    assert "optimizer:" in result.stdout
    assert "equal-weight:" in result.stdout
    assert "win rate" in result.stdout
    assert "t-stat of mean excess:" in result.stdout


def test_cli_reports_index_fixture_limitation():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "no NSE index/benchmark fixture is committed" in result.stdout


def test_cli_accepts_window_overrides():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--train-window", "200", "--test-window", "50",
    )
    assert result.returncode == 0, result.stderr
    assert "train_window=200d" in result.stdout
    assert "test_window=50d" in result.stdout


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert a.stdout == b.stdout


def test_cli_rejects_windows_too_long_for_the_data():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--train-window", "400", "--test-window", "200",
    )
    assert result.returncode != 0
    assert "not enough" in result.stderr


def test_cli_rejects_single_ticker():
    result = run_cli("--tickers", "RELIANCE.NS")
    assert result.returncode != 0
    assert "at least 2 tickers" in result.stderr


def test_cli_reports_transaction_costs_and_after_cost_figures():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "before transaction costs" in result.stdout
    assert "transaction costs (R9, hub CRITIQUES.md)" in result.stdout
    assert "average per-fold L1 turnover" in result.stdout
    assert "sum of per-fold cost rates" in result.stdout
    assert "after transaction costs" in result.stdout
    assert "t-stat of mean excess, after costs:" in result.stdout


def test_cli_turnover_lambda_reports_penalized_solve_usage():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--turnover-lambda", "5")
    assert result.returncode == 0, result.stderr
    assert "--turnover-lambda 5" in result.stdout
    assert "turnover-penalized solve" in result.stdout


def test_cli_turnover_lambda_defaults_to_zero_and_omits_its_own_note():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "turnover-penalized solve" not in result.stdout


def test_cli_is_deterministic_with_turnover_lambda():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--turnover-lambda", "5")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--turnover-lambda", "5")
    assert a.returncode == 0, a.stderr
    assert a.stdout == b.stdout
