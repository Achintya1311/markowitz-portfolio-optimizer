import numpy as np
import pytest

from optimizer.constraints import (
    DEFAULT_SECTOR_MAP,
    constrained_min_variance,
    _parse_kv_floats,
    _parse_prev_weights,
    _parse_sector_map,
)
from optimizer.covariance import sample_mean_cov
from optimizer.frontier import min_variance_weights
from optimizer.returns import annualize_cov, annualize_mean, load_returns_matrix

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def _real_mu_cov():
    _, returns_matrix = load_returns_matrix(TICKERS)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    return annualize_mean(mu_daily), annualize_cov(cov_daily)


# --- no constraints: should match Day 2's frontier exactly (same convex QP) ---


def test_no_constraints_matches_day2_frontier():
    mu, cov = _real_mu_cov()
    target = float(np.mean(mu))
    baseline = min_variance_weights(mu, cov, target, long_only=True)
    constrained = constrained_min_variance(mu, cov, target, TICKERS, long_only=True)

    assert baseline.success and constrained.success
    np.testing.assert_allclose(constrained.weights, baseline.weights, atol=1e-5)
    assert constrained.variance == pytest.approx(baseline.variance, abs=1e-8)


def test_budget_and_return_constraints_hold():
    mu, cov = _real_mu_cov()
    target = float(mu[0])
    point = constrained_min_variance(mu, cov, target, TICKERS, long_only=False, max_position=0.9)
    assert point.success
    assert np.sum(point.weights) == pytest.approx(1.0, abs=1e-6)
    assert point.achieved_return == pytest.approx(target, abs=1e-6)


# --- position limits ---


def test_position_limit_binds_on_synthetic_universe():
    # Asset A has by far the highest mean; at this target the unconstrained
    # solution wants more of it than a 30% position limit allows, and there is
    # comfortable headroom in the other four (spread-mean) assets to make up
    # the target return once A is capped.
    tickers = ["A", "B", "C", "D", "E"]
    mu = np.array([0.20, 0.08, 0.05, 0.05, 0.02])
    cov = np.diag([0.01, 0.02, 0.02, 0.02, 0.02])
    target = 0.10

    unconstrained = constrained_min_variance(mu, cov, target, tickers, long_only=True)
    assert unconstrained.success
    assert unconstrained.weights[0] > 0.3

    capped = constrained_min_variance(mu, cov, target, tickers, long_only=True, max_position=0.3)
    assert capped.success
    assert capped.weights[0] == pytest.approx(0.3, abs=1e-3)
    assert capped.binding_position_limits == ["A"]
    assert np.sum(capped.weights) == pytest.approx(1.0, abs=1e-6)
    assert capped.achieved_return == pytest.approx(target, abs=1e-6)


def test_position_limit_does_not_flag_ordinary_long_only_zero_weights():
    # 2-asset analytic corner solution: weight is exactly 0 on A because of
    # the plain long-only bound (w >= 0), not because a 150%-of-book position
    # limit did anything - that zero must not be reported as a binding
    # position limit.
    mu = np.array([0.10, 0.02])
    cov = np.diag([0.02, 0.02])
    point = constrained_min_variance(mu, cov, 0.02, ["A", "B"], long_only=True, max_position=1.5)
    assert point.success
    np.testing.assert_allclose(point.weights, [0.0, 1.0], atol=1e-6)
    assert point.binding_position_limits == []


def test_position_limit_too_tight_is_infeasible():
    mu = np.array([0.10, 0.02])
    cov = np.array([[0.02, 0.0], [0.0, 0.02]])
    # 2 assets, each capped at 0.3: budget constraint sum(w)==1 cannot be met.
    point = constrained_min_variance(mu, cov, 0.05, ["A", "B"], long_only=True, max_position=0.3)
    assert not point.success


def test_max_position_must_be_positive():
    mu, cov = _real_mu_cov()
    with pytest.raises(ValueError):
        constrained_min_variance(mu, cov, float(mu[0]), TICKERS, max_position=0.0)


# --- sector caps ---


def test_sector_cap_binds_on_synthetic_multi_sector_universe():
    # 4 assets, 2 per sector. Sector 1 (assets 0,1) has the better mean, so
    # the unconstrained optimum concentrates there beyond a 40% sector cap;
    # sector 2's spread means (0.03, 0.01) give the capped solution somewhere
    # to redistribute into and still hit the target.
    tickers = ["A", "B", "C", "D"]
    sector_map = {"A": "Sector1", "B": "Sector1", "C": "Sector2", "D": "Sector2"}
    mu = np.array([0.08, 0.06, 0.03, 0.01])
    cov = np.diag([0.02, 0.02, 0.03, 0.03])
    target = 0.04

    unconstrained = constrained_min_variance(mu, cov, target, tickers, long_only=True, sector_map=sector_map)
    assert unconstrained.success
    sector1_weight = unconstrained.weights[0] + unconstrained.weights[1]
    assert sector1_weight > 0.4

    capped = constrained_min_variance(
        mu, cov, target, tickers, long_only=True, sector_map=sector_map, sector_caps={"Sector1": 0.4}
    )
    assert capped.success
    capped_sector1 = capped.weights[0] + capped.weights[1]
    assert capped_sector1 == pytest.approx(0.4, abs=1e-3)
    assert "Sector1" in capped.binding_sector_caps
    assert capped.sector_weights["Sector1"] == pytest.approx(capped_sector1, abs=1e-6)


def test_sector_cap_unknown_sector_raises():
    mu, cov = _real_mu_cov()
    with pytest.raises(ValueError):
        constrained_min_variance(
            mu, cov, float(mu[0]), TICKERS, sector_map=DEFAULT_SECTOR_MAP, sector_caps={"Nonexistent": 0.5}
        )


def test_sector_caps_require_sector_map():
    mu, cov = _real_mu_cov()
    with pytest.raises(ValueError):
        constrained_min_variance(mu, cov, float(mu[0]), TICKERS, sector_caps={"Energy": 0.5})


def test_sector_cap_on_real_universe_is_degenerate_with_position_limit():
    # Documented finding: with each of the 3 real tickers in its own sector,
    # capping every sector at X is the same constraint as capping every
    # position at X.
    mu, cov = _real_mu_cov()
    target = float(np.mean(mu))
    cap = 0.5

    via_sector_cap = constrained_min_variance(
        mu,
        cov,
        target,
        TICKERS,
        long_only=True,
        sector_map=DEFAULT_SECTOR_MAP,
        sector_caps={sector: cap for sector in DEFAULT_SECTOR_MAP.values()},
    )
    via_position_limit = constrained_min_variance(
        mu, cov, target, TICKERS, long_only=True, max_position=cap
    )
    assert via_sector_cap.success and via_position_limit.success
    np.testing.assert_allclose(via_sector_cap.weights, via_position_limit.weights, atol=1e-4)


# --- turnover penalty ---


def test_turnover_penalty_pulls_weights_toward_reference():
    mu = np.array([0.10, 0.02])
    cov = np.array([[0.03, 0.0], [0.0, 0.03]])
    prev_weights = np.array([0.5, 0.5])
    # target reachable exactly by prev_weights, so a huge lambda should collapse onto it
    target = float(prev_weights @ mu)

    unpenalized = constrained_min_variance(mu, cov, target, ["A", "B"], long_only=True, prev_weights=prev_weights)
    penalized = constrained_min_variance(
        mu, cov, target, ["A", "B"], long_only=True, prev_weights=prev_weights, turnover_lambda=1e6
    )

    assert unpenalized.success and penalized.success
    np.testing.assert_allclose(penalized.weights, prev_weights, atol=1e-3)
    assert penalized.turnover_l1 < unpenalized.turnover_l1 + 1e-9


def test_turnover_penalty_tradeoff_is_monotonic_on_real_fixtures():
    # Two guaranteed properties of this convex-objective-mix, for any lambda1
    # < lambda2 (standard parametric-QP argument, adding the two optimality
    # inequalities against each other): the *squared* L2 distance to the
    # reference portfolio is non-increasing in lambda, and variance is
    # non-decreasing in lambda. (L1 turnover, unlike squared L2, is not
    # guaranteed to move monotonically - see the CLI's real-fixture report
    # for a case where it doesn't, recorded as a limitation.)
    mu, cov = _real_mu_cov()
    target = float(np.mean(mu))
    prev_weights = np.array([0.2, 0.3, 0.5])

    points = [
        constrained_min_variance(
            mu, cov, target, TICKERS, long_only=True, prev_weights=prev_weights, turnover_lambda=lam
        )
        for lam in (0.0, 1.0, 5.0, 20.0, 50.0)
    ]
    assert all(p.success for p in points)
    l2_sq = [float(np.sum((p.weights - prev_weights) ** 2)) for p in points]
    variances = [p.variance for p in points]

    assert all(a >= b - 1e-9 for a, b in zip(l2_sq, l2_sq[1:]))
    assert all(a <= b + 1e-9 for a, b in zip(variances, variances[1:]))


def test_prev_weights_length_must_match_tickers():
    mu, cov = _real_mu_cov()
    with pytest.raises(ValueError):
        constrained_min_variance(mu, cov, float(mu[0]), TICKERS, prev_weights=np.array([0.5, 0.5]))


# --- input validation ---


def test_rejects_single_asset():
    with pytest.raises(ValueError):
        constrained_min_variance(np.array([0.1]), np.array([[0.04]]), 0.1, ["A"])


# --- CLI arg parsing helpers ---


def test_parse_kv_floats():
    assert _parse_kv_floats("Energy=0.5,Chemicals=0.3") == {"Energy": 0.5, "Chemicals": 0.3}


def test_parse_kv_floats_rejects_malformed():
    with pytest.raises(ValueError):
        _parse_kv_floats("Energy0.5")


def test_parse_sector_map():
    assert _parse_sector_map("RELIANCE.NS=Energy,TATACHEM.NS=Chemicals") == {
        "RELIANCE.NS": "Energy",
        "TATACHEM.NS": "Chemicals",
    }


def test_parse_prev_weights_equal():
    np.testing.assert_allclose(_parse_prev_weights("equal", 4), np.full(4, 0.25))


def test_parse_prev_weights_explicit():
    np.testing.assert_allclose(_parse_prev_weights("0.2,0.3,0.5", 3), [0.2, 0.3, 0.5])


def test_parse_prev_weights_rejects_wrong_length():
    with pytest.raises(ValueError):
        _parse_prev_weights("0.5,0.5", 3)


def test_parse_prev_weights_rejects_bad_sum():
    with pytest.raises(ValueError):
        _parse_prev_weights("0.5,0.6", 2)
