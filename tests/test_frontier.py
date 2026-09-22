import numpy as np
import pytest

from optimizer.covariance import sample_mean_cov
from optimizer.frontier import (
    analytic_two_asset_weights,
    build_frontier,
    feasible_target_range,
    is_convex,
    min_variance_weights,
)
from optimizer.returns import annualize_cov, annualize_mean, load_returns_matrix

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def _real_mu_cov():
    _, returns_matrix = load_returns_matrix(TICKERS)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    return annualize_mean(mu_daily), annualize_cov(cov_daily)


@pytest.mark.parametrize("long_only", [True, False])
def test_min_variance_matches_analytic_two_asset(long_only):
    mu = np.array([0.12, -0.04])
    cov = np.array([[0.05, 0.01], [0.01, 0.03]])
    # Strictly between the two means, so long-only weights land in (0, 1) anyway.
    target = 0.03

    point = min_variance_weights(mu, cov, target, long_only=long_only)
    analytic = analytic_two_asset_weights(mu, target)

    assert point.success
    np.testing.assert_allclose(point.weights, analytic, atol=1e-6)
    assert point.achieved_return == pytest.approx(target, abs=1e-6)


def test_min_variance_matches_analytic_two_asset_on_real_fixture_pair():
    # RELIANCE.NS / TATACHEM.NS from the shared Stock Stalker fixtures.
    _, returns_matrix = load_returns_matrix(["RELIANCE.NS", "TATACHEM.NS"])
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    mu = annualize_mean(mu_daily)
    cov = annualize_cov(cov_daily)

    target = float(np.mean(mu))  # strictly between the two real means
    point = min_variance_weights(mu, cov, target, long_only=False)
    analytic = analytic_two_asset_weights(mu, target)

    assert point.success
    np.testing.assert_allclose(point.weights, analytic, atol=1e-6)


def test_budget_constraint_holds():
    mu, cov = _real_mu_cov()
    point = min_variance_weights(mu, cov, target_return=float(mu[0]), long_only=False)
    assert point.success
    assert np.sum(point.weights) == pytest.approx(1.0, abs=1e-6)


def test_long_only_weights_are_non_negative():
    mu, cov = _real_mu_cov()
    lo, hi = feasible_target_range(mu, long_only=True)
    midpoint = (lo + hi) / 2
    point = min_variance_weights(mu, cov, midpoint, long_only=True)
    assert point.success
    assert np.all(point.weights >= -1e-8)


def test_long_only_infeasible_outside_convex_hull_of_means():
    mu, cov = _real_mu_cov()
    _, hi = feasible_target_range(mu, long_only=True)
    unreachable = hi + 1.0  # 100pp above the best single-asset mean
    point = min_variance_weights(mu, cov, unreachable, long_only=True)
    assert not point.success


def test_long_short_can_reach_targets_long_only_cannot():
    mu, cov = _real_mu_cov()
    _, hi = feasible_target_range(mu, long_only=True)
    beyond_long_only = hi + 0.05
    long_only = min_variance_weights(mu, cov, beyond_long_only, long_only=True)
    long_short = min_variance_weights(mu, cov, beyond_long_only, long_only=False)
    assert not long_only.success
    assert long_short.success
    assert np.any(long_short.weights < 0)  # needed leverage/shorting to get there


def test_feasible_target_range_long_only_is_min_max_of_means():
    mu = np.array([0.10, -0.05, 0.02])
    lo, hi = feasible_target_range(mu, long_only=True)
    assert lo == pytest.approx(-0.05)
    assert hi == pytest.approx(0.10)


def test_frontier_variance_is_convex_in_target_return():
    mu, cov = _real_mu_cov()
    frontier = build_frontier(mu, cov, long_only=False, n_points=15)
    assert all(p.success for p in frontier)
    variances = np.array([p.variance for p in frontier])
    assert is_convex(variances)


def test_frontier_minimum_sits_below_both_endpoints():
    mu, cov = _real_mu_cov()
    frontier = build_frontier(mu, cov, long_only=False, n_points=15)
    variances = np.array([p.variance for p in frontier])
    assert variances.min() <= variances[0]
    assert variances.min() <= variances[-1]


def test_is_convex_detects_a_concave_sequence():
    concave = np.array([1.0, 3.0, 4.0, 3.0, 1.0])  # inverted parabola, not convex
    assert not is_convex(concave)


def test_analytic_two_asset_rejects_wrong_asset_count():
    with pytest.raises(ValueError):
        analytic_two_asset_weights(np.array([0.1, 0.2, 0.3]), 0.1)


def test_analytic_two_asset_rejects_equal_means():
    with pytest.raises(ValueError):
        analytic_two_asset_weights(np.array([0.1, 0.1]), 0.1)


def test_min_variance_rejects_single_asset():
    with pytest.raises(ValueError):
        min_variance_weights(np.array([0.1]), np.array([[0.04]]), 0.1, long_only=False)
