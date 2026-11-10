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


def test_update_readme_table_rewrites_drifted_cells():
    from optimizer.methods import update_readme_table

    text = (ROOT / "README.md").read_text()
    stale = text.replace(f"| Minimum variance | {REPORT['min_variance']['compounded_net']:+.1%} |", "| Minimum variance | +999.9% |")
    assert stale != text
    fixed = update_readme_table(stale, {"methods_252": REPORT})
    assert fixed == text


def test_update_readme_table_is_a_no_op_when_the_table_matches():
    from optimizer.methods import update_readme_table

    text = (ROOT / "README.md").read_text()
    assert update_readme_table(text, {"methods_252": REPORT}) == text


def test_refresh_workflow_updates_and_commits_the_readme():
    wf = (ROOT / ".github" / "workflows" / "fetch-largecaps.yml").read_text()
    assert "--update-readme README.md" in wf
    assert "git add fixtures/ohlcv_largecap fixtures/index reports/largecap_latest.json README.md" in wf


# D60: the interval column and the section's fold count come from the report too.
def _report_with(diff_lo_hi, n_folds):
    import copy

    rep = copy.deepcopy(json.loads((ROOT / "reports" / "largecap_latest.json").read_text()))
    for m in rep["methods_252"].values():
        m["n_folds"] = n_folds
    d, lo, hi = diff_lo_hi
    rep["methods_252"]["max_sharpe"]["sharpe_diff_vs_equal_weight"] = {"diff": d, "lo": lo, "hi": hi}
    return rep


def test_update_readme_table_rewrites_a_stale_interval_from_the_report():
    from optimizer.methods import update_readme_table

    text = (ROOT / "README.md").read_text()
    fixed = update_readme_table(text, _report_with((-0.123, -0.456, 0.789), 127))
    row = next(line for line in fixed.splitlines() if line.startswith("| Max-Sharpe (previous method) |"))
    assert row.rstrip().endswith("| -0.12 (-0.46 to +0.79) |")


def test_update_readme_table_rewrites_the_methods_section_fold_count():
    from optimizer.methods import update_readme_table

    text = (ROOT / "README.md").read_text()
    fixed = update_readme_table(text, _report_with((-0.42, -0.96, 0.02), 131))
    section = fixed.split("## Which rule, if any, beats equal weight?")[1].split("\n## ")[0]
    assert "21-day hold, 131 non-overlapping folds" in section
    assert "with 131 folds none of the gaps" in section
    assert "126 folds" not in section and "126 holds" not in section
    # outside the section only the headline lines follow the report (D63); the dated 26 Oct figures stay
    def outside(t):
        head = t.split("## Which rule, if any, beats equal weight?")[0]
        return [l for l in head.splitlines() if not l.startswith(("**Finding", "**Update", "**Data refresh"))]
    assert outside(fixed) == outside(text)
    assert "On the fixtures to 26 Oct 2026 (126 folds)" in fixed


def test_committed_report_stores_every_rule_interval_and_the_readme_quotes_it():
    rep = json.loads((ROOT / "reports" / "largecap_latest.json").read_text())
    cells = _table()
    for label, key in ROWS.items():
        if key == "equal_weight":
            continue
        d = rep["methods_252"][key]["sharpe_diff_vs_equal_weight"]
        assert cells[label][6] == f"{d['diff']:+.2f} ({d['lo']:+.2f} to {d['hi']:+.2f})"
    section = (ROOT / "README.md").read_text().split("## Which rule, if any, beats equal weight?")[1].split("\n## ")[0]
    assert f"21-day hold, {rep['methods_252']['equal_weight']['n_folds']} non-overlapping folds" in section
