"""P129 part 2: the price-only Nifty 50 leg beside equal weight and the rules."""

import json
from pathlib import Path

import numpy as np
import pytest

from optimizer import methods

ROOT = Path(__file__).resolve().parent.parent


def _write_index(path, rows):
    path.write_text("date,close\n" + "".join(f"{d},{c}\n" for d, c in rows))
    return path


def test_load_index_returns_aligns_to_the_stock_calendar(tmp_path):
    p = _write_index(tmp_path / "i.csv", [("2024-01-02", 100.0), ("2024-01-03", 110.0), ("2024-01-05", 121.0)])
    calendar = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    r, filled = methods.load_index_returns(calendar, p)
    # 01-01 takes the first close, 01-04 carries 01-03's close: both 0 return.
    assert filled == 2
    np.testing.assert_allclose(r, np.log([1.0, 1.1, 1.0, 1.1]))


def test_load_index_returns_missing_fixture_is_none(tmp_path):
    assert methods.load_index_returns(["2024-01-01", "2024-01-02"], tmp_path / "missing.csv") is None


def test_index_walk_forward_compounds_each_hold_with_no_cost():
    r = np.log(np.full(10, 1.01))
    net = methods.index_walk_forward(r, train_window=4, test_window=3)
    # folds hold rows 4-6 and 7-9, same slicing as every rule
    np.testing.assert_allclose(net, [1.01 ** 3 - 1, 1.01 ** 3 - 1])


def test_run_adds_index_leg_outside_methods_and_keeps_verdict_inputs():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0004, 0.01, size=(400, 3))
    idx = rets.mean(axis=1)
    out = methods.run(rets, 200, index_returns=idx)
    assert set(out["methods"]) == set(methods.METHODS)
    leg = out["index"]
    assert leg["n_folds"] == out["methods"]["equal_weight"]["n_folds"]
    assert leg["avg_turnover"] == 0.0
    assert "sharpe_diff_vs_equal_weight" in leg
    assert "index" not in methods.run(rets, 200)


def test_cli_prints_the_index_leg_and_its_caveat(capsys):
    assert methods.main([]) == 0
    out = capsys.readouterr().out
    assert "nifty50_price" in out
    assert "price index, no dividends" in out


def test_cli_skips_the_leg_with_a_message_when_the_fixture_is_missing(tmp_path, capsys):
    assert methods.main(["--index-fixture", str(tmp_path / "none.csv")]) == 0
    assert "index leg skipped" in capsys.readouterr().out


def test_report_carries_index_252():
    r = json.loads((ROOT / "reports" / "largecap_latest.json").read_text())
    assert r["index_252"]["n_folds"] == r["methods_252"]["equal_weight"]["n_folds"]
