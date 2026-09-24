import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.shrinkage", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_and_reports_shrinkage_intensity():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "Ledoit-Wolf shrinkage intensity: 0.0706" in result.stdout
    assert "covariance condition number: sample=3.23   shrunk=2.94" in result.stdout


def test_cli_reports_gmv_and_tangency_sections():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "global minimum-variance portfolio" in result.stdout
    assert "long-only (w >= 0)" in result.stdout
    assert "unconstrained (long/short)" in result.stdout
    assert "tangency portfolio, long-only" in result.stdout
    assert "weight movement: L1=" in result.stdout
    assert "estimation-error note:" in result.stdout


def test_cli_accepts_risk_free_rate_override():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--risk-free-rate", "0.065"
    )
    assert result.returncode == 0, result.stderr


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert a.stdout == b.stdout


def test_cli_rejects_single_ticker():
    result = run_cli("--tickers", "RELIANCE.NS")
    assert result.returncode != 0
    assert "at least 2 tickers" in result.stderr
