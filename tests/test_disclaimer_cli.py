"""S16: every named-stock CLI prints PLAN.md rule 1's disclaimer in its own output."""
import subprocess
import sys

import pytest

from optimizer import DISCLAIMER

TICKERS = "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS"

CLI_ARGS = {
    "optimizer.stats": ["--tickers", TICKERS],
    "optimizer.frontier": ["--tickers", TICKERS, "--target-return", "-0.15"],
    "optimizer.tangency": ["--tickers", TICKERS],
    "optimizer.shrinkage": ["--tickers", TICKERS],
    "optimizer.constraints": ["--tickers", TICKERS, "--target-return", "-0.15"],
    "optimizer.oos": ["--tickers", TICKERS],
    "optimizer.report": ["--tickers", TICKERS, "--no-chart"],
    "optimizer.sizing": ["--tickers", TICKERS],
}


def test_disclaimer_is_plan_rule_1_line():
    assert DISCLAIMER == (
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    )


@pytest.mark.parametrize("module", sorted(CLI_ARGS))
def test_cli_prints_disclaimer(module):
    result = subprocess.run(
        [sys.executable, "-m", module, *CLI_ARGS[module]],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert DISCLAIMER in result.stdout


@pytest.mark.parametrize("module", ["optimizer.tangency", "optimizer.oos"])
def test_sweep_mode_prints_disclaimer(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--tickers", TICKERS, "--risk-free-rate-sweep", "0.0,0.04"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert DISCLAIMER in result.stdout
