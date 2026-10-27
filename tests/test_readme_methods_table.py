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


def test_update_readme_table_rewrites_drifted_cells_and_keeps_the_interval_column():
    from optimizer.methods import update_readme_table

    text = (ROOT / "README.md").read_text()
    stale = text.replace("| Minimum variance | +254.5% |", "| Minimum variance | +999.9% |")
    fixed = update_readme_table(stale, {"methods_252": REPORT})
    assert fixed == text
    assert "-0.19 (-0.51 to +0.07)" in fixed


def test_update_readme_table_is_a_no_op_when_the_table_matches():
    from optimizer.methods import update_readme_table

    text = (ROOT / "README.md").read_text()
    assert update_readme_table(text, {"methods_252": REPORT}) == text


def test_refresh_workflow_updates_and_commits_the_readme():
    wf = (ROOT / ".github" / "workflows" / "fetch-largecaps.yml").read_text()
    assert "--update-readme README.md" in wf
    assert "git add fixtures/ohlcv_largecap reports/largecap_latest.json README.md" in wf
