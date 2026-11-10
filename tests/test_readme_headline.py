"""D63, S35: the README's Finding, Update and Data-refresh lines quote the committed
report (the weekly refresh rewrites them with the table), and say the returns are before tax."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
REPORT = json.loads((ROOT / "reports" / "largecap_latest.json").read_text())


def _head() -> str:
    return "\n".join(l for l in README.splitlines()[:12] if l.startswith(("**Finding", "**Update", "**Data refresh")))


def test_headline_quotes_the_committed_report():
    from optimizer.methods import headline_figures

    head = _head()
    for fig in headline_figures(REPORT):
        assert fig in head, fig


def test_update_readme_table_rewrites_a_stale_headline():
    from optimizer.methods import update_readme_table

    stale = README
    for a, b in (("127 folds; Day 9", "126 folds; Day 9"), ("13.8% vs 15.0%", "13.9% vs 15.0%"),
                 ("(Sharpe difference -0.01,", "(Sharpe difference -0.00,"), ("+225.0% for max-Sharpe", "+252.2% for max-Sharpe")):
        assert a in stale, a
        stale = stale.replace(a, b, 1)
    stale = re.sub(r"fixtures ending \d+ \w+ \d{4} \(\d+ folds\)", "fixtures ending 26 Oct 2026 (126 folds)", stale)
    assert update_readme_table(stale, REPORT) == README


def test_update_readme_table_leaves_a_current_readme_alone():
    from optimizer.methods import update_readme_table

    assert update_readme_table(README, REPORT) == README


def test_returns_are_labelled_before_tax():
    head = _head()
    assert "before tax" in head
    assert "| Rule (net of costs, before tax) |" in README
    limitations = README.split("## Limitations and what would make me wrong")[1]
    assert "No income tax is modelled" in limitations
