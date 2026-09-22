import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.stats", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_and_reports_all_tickers():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "RELIANCE.NS" in result.stdout
    assert "TATACHEM.NS" in result.stdout
    assert "CROMPTON.NS" in result.stdout
    assert "correlation matrix" in result.stdout
    assert "condition number" in result.stdout
    assert "estimation-error note" in result.stdout


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert a.stdout == b.stdout


def test_cli_rejects_single_ticker():
    result = run_cli("--tickers", "RELIANCE.NS")
    assert result.returncode != 0
    assert "at least 2 tickers" in result.stderr
