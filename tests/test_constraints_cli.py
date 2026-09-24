import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.constraints", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_with_no_extra_constraints():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15")
    assert result.returncode == 0, result.stderr
    assert "baseline (unconstrained" in result.stdout
    assert "constrained (sector caps" in result.stdout


def test_cli_reports_sector_map_and_degeneracy_note():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15",
                      "--sector-cap", "Energy=0.6,Chemicals=0.6,Consumer Durables=0.6")
    assert result.returncode == 0, result.stderr
    assert "sector map: RELIANCE.NS=Energy" in result.stdout
    assert "sector cap Energy: " in result.stdout
    assert "mathematically identical to a per-name position limit" in result.stdout


def test_cli_max_position_binds_and_is_reported():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15", "--max-position", "0.6"
    )
    assert result.returncode == 0, result.stderr
    assert "position limit: 60.00%" in result.stdout
    assert "binding: RELIANCE.NS" in result.stdout


def test_cli_turnover_penalty_changes_weights():
    baseline = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15")
    penalized = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15",
        "--turnover-lambda", "50", "--prev-weights", "0.2,0.3,0.5",
    )
    assert baseline.returncode == 0 and penalized.returncode == 0
    assert baseline.stdout != penalized.stdout


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15")
    assert a.stdout == b.stdout


def test_cli_rejects_bad_prev_weights_length():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "-0.15",
        "--prev-weights", "0.5,0.5",
    )
    assert result.returncode != 0
    assert "expected 3" in result.stderr


def test_cli_rejects_infeasible_target():
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--target-return", "0.50", "--long-short",
        "--max-position", "0.6",
    )
    assert result.returncode == 0, result.stderr
    assert "infeasible" in result.stdout
