"""D58: the README's pre-registered table is the committed report, at the printed precision."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT / "reports" / "largecap_latest.json").read_text())["methods_252"]
ROWS = {
    "Equal weight": "equal_weight",
    "Minimum variance": "min_variance",
    "Inverse volatility": "inverse_vol",
    "50/50 blend": "blend_50_50",
    "Max-Sharpe (previous method)": "max_sharpe",
}


def _table() -> dict[str, list[str]]:
    out = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in ROWS:
            out[cells[0]] = cells
    return out


@pytest.mark.parametrize("label", sorted(ROWS))
def test_methods_table_row_matches_the_committed_report(label):
    cells = _table()[label]
    m = REPORT[ROWS[label]]
    assert cells[1] == f"{m['compounded_net']:+.1%}"
    assert cells[2] == f"{m['sharpe_net']:.2f}"
    assert cells[3] == f"{m['vol_annual']:.1%}"
    assert cells[4] == f"{m['max_drawdown']:.1%}"
    assert cells[5] == f"{m['avg_turnover']:.1%}"
