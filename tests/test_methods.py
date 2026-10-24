import numpy as np
import pytest

from optimizer.methods import (blend, drift, equal_weight, inverse_vol, min_variance, sharpe_diff_ci, verdict, walk_forward)


def _panel(n=1200, k=6, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.linspace(0.008, 0.025, k)
    common = rng.normal(0, 0.006, (n, 1))
    return 0.0003 + common + rng.normal(0, 1, (n, k)) * vol


def test_every_rule_returns_a_long_only_fully_invested_portfolio():
    r = _panel()
    for rule in (equal_weight, min_variance, inverse_vol, blend):
        w = rule(r[:252])
        assert w.sum() == pytest.approx(1.0, abs=1e-6) and (w >= -1e-9).all()


def test_min_variance_and_inverse_vol_tilt_toward_the_calm_assets():
    w_mv, w_iv = min_variance(_panel()[:500]), inverse_vol(_panel()[:500])
    assert w_mv[0] > w_mv[-1] and w_iv[0] > w_iv[-1]


def test_drift_moves_weight_to_the_winner_and_keeps_the_sum():
    w = drift(np.array([0.5, 0.5]), np.array([0.10, -0.10]))
    assert w.sum() == pytest.approx(1.0) and w[0] > 0.5 > w[1]


def test_equal_weight_now_pays_for_rebalancing_back_from_drift():
    r = _panel()
    out = walk_forward(r, equal_weight, 252)
    assert out["turnover"][0] == 0.0                              # first allocation is free for every rule alike
    assert (out["turnover"][1:] > 0).all()                        # later folds pay to undo drift (the old benchmark paid nothing)
    assert (out["net"][1:] < out["gross"][1:]).all()


def test_rule_only_ever_sees_a_full_training_window():
    seen = []

    def spy(train):
        seen.append(train.shape[0])
        return equal_weight(train)

    walk_forward(_panel(), spy, 252)
    assert set(seen) == {252}


def test_sharpe_diff_interval_brackets_the_point_estimate_and_flags_a_real_gap():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.02, 0.03, 120), rng.normal(0.0, 0.03, 120)
    d = sharpe_diff_ci(a, b, reps=800)
    assert d["lo"] < d["diff"] < d["hi"] and d["lo"] > 0


def test_verdict_requires_every_pre_registered_criterion():
    eq = {"sharpe_net": 1.0, "vol_annual": 0.2, "max_drawdown": -0.3, "avg_turnover": 0.03, "compounded_net": 1.0}
    mv = {"sharpe_net": 1.1, "vol_annual": 0.15, "max_drawdown": -0.2, "avg_turnover": 0.1, "compounded_net": 0.9,
          "sharpe_diff_vs_equal_weight": {"diff": 0.1, "lo": -0.2, "hi": 0.4}}
    good = {"methods": {"equal_weight": eq, "min_variance": mv}}
    assert verdict(good, good)["all_pass"]
    bad = {"methods": {"equal_weight": eq, "min_variance": {**mv, "avg_turnover": 0.3}}}
    assert not verdict(bad, good)["all_pass"]
