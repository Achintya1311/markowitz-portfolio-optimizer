import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.sizing", *args],
        capture_output=True,
        text=True,
    )


def test_cli_runs_with_defaults():
    result = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert result.returncode == 0, result.stderr
    assert "Ledoit-Wolf shrinkage intensity" in result.stdout
    assert "unconstrained max-Sharpe tangency" in result.stdout
    assert "sector-capped max-Sharpe tangency" in result.stdout
    assert "each ticker in its own sector" in result.stdout


def test_cli_writes_contract_file(tmp_path):
    out = tmp_path / "sizing_contract.json"
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--sector-cap", "0.5",
        "--contract", str(out),
        "--contract-ticker", "RELIANCE.NS",
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    contract = json.loads(out.read_text())
    assert contract == {
        "sizing": {
            "weight": 0.5,
            "method": "max_sharpe_ledoit_wolf",
            "constraint_binding": "Energy",
        }
    }
    assert f"wrote sizing contract for RELIANCE.NS to {out}" in result.stdout


def test_cli_rejects_contract_without_ticker(tmp_path):
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--contract", str(tmp_path / "c.json"),
    )
    assert result.returncode != 0
    assert "--contract and --contract-ticker must be given together" in result.stderr


def test_cli_rejects_contract_ticker_outside_universe(tmp_path):
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--contract", str(tmp_path / "c.json"),
        "--contract-ticker", "NOTATICKER.NS",
    )
    assert result.returncode != 0
    assert "must be one of --tickers" in result.stderr


def test_cli_is_deterministic():
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS")
    assert a.stdout == b.stdout


def test_cli_rejects_single_ticker():
    result = run_cli("--tickers", "RELIANCE.NS")
    assert result.returncode != 0
    assert "at least 2" in result.stderr
