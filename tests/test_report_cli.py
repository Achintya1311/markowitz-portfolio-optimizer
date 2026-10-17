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


def test_cli_writes_chart_on_the_large_cap_universe_without_name_too_long(tmp_path):
    # P19: a plain dash-joined ticker slug is 253 chars on the real 29-name
    # large-cap fixture (D4/D9), 2 short of a typical 255-byte NAME_MAX, and
    # used to raise OSError before writing any chart.
    import os

    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures", "ohlcv_largecap")
    tickers = sorted(f[:-4].replace("_", ".", 1) for f in os.listdir(fixtures_dir) if f.endswith(".csv"))
    assert len(tickers) >= 20  # sanity: this is the large-cap universe, not the 3-ticker one

    result = run_cli("--tickers", ",".join(tickers), "--fixtures-dir", fixtures_dir, "--output-dir", str(tmp_path))

    assert result.returncode == 0, result.stderr
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
    assert len(pngs[0].name) <= 255
    assert pngs[0].stat().st_size > 0
    assert str(pngs[0]) in result.stdout
