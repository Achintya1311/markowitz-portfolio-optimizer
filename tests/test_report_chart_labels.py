"""S34: the frontier PNG is shared on its own, so it carries the rule 1 line and its data window."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizer import DISCLAIMER
from optimizer.report import main
from optimizer.returns import FIXTURES_DIR, load_returns_matrix

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def test_frontier_chart_carries_disclaimer_and_fixture_window(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    assert main(["--tickers", ",".join(TICKERS), "--output-dir", str(tmp_path)]) == 0
    (fig,) = figures
    texts = [t.get_text() for t in fig.texts] + [ax.get_title() for ax in fig.axes]
    dates, _ = load_returns_matrix(TICKERS, FIXTURES_DIR)
    assert DISCLAIMER in texts
    assert any(f"historical daily returns, {dates[0]} to {dates[-1]}" in t for t in texts), texts
