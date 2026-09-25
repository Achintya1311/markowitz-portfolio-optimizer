import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "optimizer.audit", *args],
        capture_output=True,
        text=True,
    )


def test_cli_clean_on_committed_fixtures(tmp_path):
    result = run_cli(
        "--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS",
        "--output-dir", str(tmp_path), "--date", "2026-09-26",
    )
    assert result.returncode == 0, result.stderr
    assert "fold boundaries" in result.stdout
    assert "causality" in result.stdout
    assert "no look-ahead detected" in result.stdout

    report_path = tmp_path / "audit_2026-09-26.md"
    assert report_path.exists()
    text = report_path.read_text()
    assert "clean" in text


def test_cli_is_deterministic(tmp_path):
    a = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--output-dir", str(tmp_path), "--date", "2026-09-26")
    b = run_cli("--tickers", "RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", "--output-dir", str(tmp_path), "--date", "2026-09-26")
    assert a.stdout == b.stdout
