import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.frontier", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_with_feasible_long_only_target():
    # Between the shared fixtures' three (all-negative) Day 1 sample means.
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15"
    )
    assert result.returncode == 0, result.stderr
    assert "long-only (w >= 0)" in result.stdout
    assert "long/short (unconstrained)" in result.stdout
    assert "variance is convex in target return: True" in result.stdout


def test_cli_reports_infeasible_long_only_target():
    # 10% is above every one of this universe's (negative) asset means.
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "0.10"
    )
    assert result.returncode == 0, result.stderr
    assert "long-only" in result.stdout
    assert "infeasible" in result.stdout
    assert "long/short" in result.stdout


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15")
    assert a.stdout == b.stdout


def test_cli_rejects_single_ticker():
    result = run_cli("--tickers", "RELIANCE.NS", "--target-return", "0.0")
    assert result.returncode != 0
    assert "at least 2 tickers" in result.stderr
