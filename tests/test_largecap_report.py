"""D38: the committed large-cap report must match what the committed fixtures
produce. The weekly refresh regenerates both together, so a refresh that moves
the walk-forward headline (or adds a fold) shows up in the report, never silently."""

import json
from pathlib import Path

import pytest

from optimizer import methods

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "largecap_latest.json"


def _assert_same(fresh, committed, path=""):
    assert type(fresh) is type(committed) or isinstance(committed, float), path
    if isinstance(committed, dict):
        assert set(fresh) == set(committed), path
        for k in committed:
            _assert_same(fresh[k], committed[k], f"{path}.{k}")
    elif isinstance(committed, float):
        assert fresh == pytest.approx(committed, abs=1e-6), path
    else:
        assert fresh == committed, path


def test_committed_report_matches_the_committed_fixtures(tmp_path):
    out = tmp_path / "largecap.json"
    assert methods.main(["--report", str(out)]) == 0
    _assert_same(json.loads(out.read_text()), json.loads(REPORT.read_text()))


def test_report_carries_the_headline_fields():
    r = json.loads(REPORT.read_text())
    assert {"fixture_last_date", "n_assets", "walk_forward_252", "methods_252", "verdict_min_variance"} <= set(r)
    wf = r["walk_forward_252"]
    assert {"n_folds", "optimizer_before_costs", "optimizer_after_costs", "equal_weight_before_costs",
            "t_stat_excess_after_costs"} <= set(wf)
    assert {"equal_weight", "min_variance", "inverse_vol", "blend_50_50", "max_sharpe"} == set(r["methods_252"])
