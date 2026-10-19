import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.tangency", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_with_default_risk_free_rate():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "risk-free rate: +0.0000%" in result.stdout
    assert "theoretical Sharpe bound" in result.stdout
    assert "tangency portfolio, long-only (w >= 0):" in result.stdout
    assert "capital market line" in result.stdout


def test_cli_reports_degenerate_unconstrained_tangency_on_real_universe():
    # Every asset in this fixture universe has a negative Day 1 sample mean,
    # so at rf=0 the unconstrained tangency portfolio has no finite solution.
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "no finite maximum-Sharpe portfolio exists" in result.stdout


def test_cli_accepts_risk_free_rate_override():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--risk-free-rate", "0.065"
    )
    assert result.returncode == 0, result.stderr
    assert "risk-free rate: +6.5000%" in result.stdout


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert a.stdout == b.stdout


def test_cli_rejects_single_ticker():
    result = run_cli("--tickers", "RELIANCE.NS")
    assert result.returncode != 0
    assert "at least 2 tickers" in result.stderr


def test_cli_risk_free_rate_sweep_reports_each_rate_and_a_verdict():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--risk-free-rate-sweep", "0.0,0.04,0.07",
    )
    assert result.returncode == 0, result.stderr
    assert "risk-free-rate sweep (R10, hub CRITIQUES.md):" in result.stdout
    assert "+0.0000%" in result.stdout
    assert "+4.0000%" in result.stdout
    assert "+7.0000%" in result.stdout
    assert "verdict:" in result.stdout
    assert "downward-sloping sign is robust across every rate swept" in result.stdout


def test_cli_risk_free_rate_sweep_overrides_single_rate_flag():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--risk-free-rate", "0.065",
        "--risk-free-rate-sweep", "0.0,0.04",
    )
    assert result.returncode == 0, result.stderr
    assert "risk-free-rate sweep" in result.stdout
    assert "+6.5000%" not in result.stdout


def test_cli_risk_free_rate_sweep_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--risk-free-rate-sweep", "0.0,0.04,0.07")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--risk-free-rate-sweep", "0.0,0.04,0.07")
    assert a.returncode == 0, a.stderr
    assert a.stdout == b.stdout
