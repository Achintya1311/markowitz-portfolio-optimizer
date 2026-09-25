import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.report", *args],
        capture_output=True,
        text=True,
    )


def test_cli_writes_chart_and_prints_weights_table(tmp_path):
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--output-dir", str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "min-variance" in result.stdout
    assert "max-Sharpe" in result.stdout
    assert "RELIANCE.NS" in result.stdout

    chart_path = tmp_path / "frontier_RELIANCE_NS-TATACHEM_NS-CROMPTON_NS.png"
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0
    assert str(chart_path) in result.stdout


def test_cli_no_chart_skips_writing_the_png(tmp_path):
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--output-dir", str(tmp_path), "--no-chart",
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("*.png")) == []
    assert "wrote frontier chart" not in result.stdout


def test_cli_is_deterministic(tmp_path):
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--output-dir", str(tmp_path), "--no-chart")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--output-dir", str(tmp_path), "--no-chart")
    assert a.stdout == b.stdout
