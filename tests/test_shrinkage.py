import numpy as np
import pytest

from optimizer.covariance import sample_mean_cov
from optimizer.returns import annualize_cov, load_returns_matrix
from optimizer.shrinkage import (
    global_min_variance_weights,
    ledoit_wolf_shrinkage,
    weight_shift,
)

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def _real_returns_matrix():
    _, returns_matrix = load_returns_matrix(TICKERS)
    return returns_matrix


# --- ledoit_wolf_shrinkage -------------------------------------------------


def test_shrinkage_matches_sklearn_reference_value_on_real_fixtures():
    # Cross-checked during development against sklearn.covariance.LedoitWolf
    # fit on this exact fixture universe: shrinkage_=0.07058252076287826,
    # matching to float precision. sklearn is not a runtime dependency of
    # this repo, so that check is pinned here as a plain regression value
    # instead of an importable test dependency.
    shrinkage, _ = ledoit_wolf_shrinkage(_real_returns_matrix())
    assert shrinkage == pytest.approx(0.07058252076287826, abs=1e-9)


def test_shrunk_cov_matches_sklearn_reference_values_on_real_fixtures():
    _, cov_shrunk_daily = ledoit_wolf_shrinkage(_real_returns_matrix())
    cov_shrunk_annual = annualize_cov(cov_shrunk_daily)
    expected = np.array(
        [
            [0.04477621, 0.01546577, 0.01557464],
            [0.01546577, 0.08130382, 0.02190955],
            [0.01557464, 0.02190955, 0.08009635],
        ]
    )
    np.testing.assert_allclose(cov_shrunk_annual, expected, atol=1e-6)


def test_shrinkage_intensity_in_unit_interval():
    rng = np.random.default_rng(0)
    data = rng.normal(scale=0.01, size=(120, 5))
    shrinkage, _ = ledoit_wolf_shrinkage(data)
    assert 0.0 <= shrinkage <= 1.0


def test_shrinkage_rejects_single_asset():
    with pytest.raises(ValueError, match="at least 2 assets"):
        ledoit_wolf_shrinkage(np.zeros((10, 1)))


def test_shrunk_cov_is_the_stated_convex_combination():
    rng = np.random.default_rng(1)
    data = rng.normal(scale=0.02, size=(80, 4))
    shrinkage, shrunk = ledoit_wolf_shrinkage(data)

    x = data - data.mean(axis=0)
    sample_cov = x.T @ x / data.shape[0]
    mu = np.trace(sample_cov) / data.shape[1]
    target = mu * np.eye(data.shape[1])
    expected = (1 - shrinkage) * sample_cov + shrinkage * target

    np.testing.assert_allclose(shrunk, expected, atol=1e-12)


def test_shrinkage_pulls_correlation_toward_zero_on_correlated_data():
    rng = np.random.default_rng(2)
    n_obs, n_assets = 60, 6
    common_factor = rng.normal(scale=0.02, size=(n_obs, 1))
    idiosyncratic = rng.normal(scale=0.005, size=(n_obs, n_assets))
    data = common_factor + idiosyncratic  # strongly correlated across assets

    shrinkage, shrunk = ledoit_wolf_shrinkage(data)
    _, sample_cov = sample_mean_cov(data)

    assert shrinkage > 0.0
    sample_offdiag = np.abs(sample_cov[np.triu_indices(n_assets, k=1)]).sum()
    shrunk_offdiag = np.abs(shrunk[np.triu_indices(n_assets, k=1)]).sum()
    assert shrunk_offdiag < sample_offdiag


def test_shrinkage_saturates_toward_one_when_true_covariance_is_the_target():
    # If the true covariance genuinely is a scaled identity (the shrinkage
    # target), every off-diagonal entry the sample covariance shows is pure
    # sampling noise, and with enough observations the estimator should
    # detect that and shrink nearly all the way to the target - not partial
    # credit for noise that isn't there.
    rng = np.random.default_rng(7)
    cov_true = np.eye(5) * 0.0003
    data = rng.multivariate_normal(mean=np.zeros(5), cov=cov_true, size=3000)
    shrinkage, _ = ledoit_wolf_shrinkage(data)
    assert shrinkage > 0.99


def test_shrinkage_falls_toward_zero_as_observations_grow_for_real_correlation():
    # Mirror image of the above: when assets genuinely are correlated (the
    # true covariance has real off-diagonal structure, unlike the target),
    # more data means the sample covariance is a more reliable estimate of
    # that real structure, and the optimal shrinkage intensity should fall
    # toward zero rather than erasing a real signal.
    rng = np.random.default_rng(5)
    n_assets = 6
    loadings = rng.normal(size=(n_assets, n_assets))
    cov_true = loadings @ loadings.T * 0.0001 + np.eye(n_assets) * 0.0002
    data = rng.multivariate_normal(mean=np.zeros(n_assets), cov=cov_true, size=5000)

    shrinkage_small_t, _ = ledoit_wolf_shrinkage(data[:100])
    shrinkage_large_t, _ = ledoit_wolf_shrinkage(data[:5000])
    assert shrinkage_large_t < shrinkage_small_t


# --- global_min_variance_weights -------------------------------------------


def test_gmv_unconstrained_matches_closed_form_two_asset():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    point = global_min_variance_weights(cov, long_only=False)
    ones = np.ones(2)
    expected = np.linalg.solve(cov, ones)
    expected = expected / expected.sum()
    np.testing.assert_allclose(point.weights, expected, atol=1e-10)
    assert point.success


def test_gmv_weights_sum_to_one():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T + np.eye(5) * 0.01  # positive definite
    for long_only in (False, True):
        point = global_min_variance_weights(cov, long_only=long_only)
        assert point.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_gmv_long_only_matches_unconstrained_when_already_nonnegative():
    # A near-diagonal, uncorrelated covariance has a positive unconstrained
    # GMV solution, so the long-only bound should not bind and both methods
    # should land on (close to) the same point.
    cov = np.diag([0.01, 0.02, 0.03])
    unconstrained = global_min_variance_weights(cov, long_only=False)
    long_only = global_min_variance_weights(cov, long_only=True)
    assert np.all(unconstrained.weights >= -1e-9)
    np.testing.assert_allclose(long_only.weights, unconstrained.weights, atol=1e-4)


def test_gmv_rejects_single_asset():
    with pytest.raises(ValueError, match="at least 2 assets"):
        global_min_variance_weights(np.array([[0.04]]), long_only=False)


def test_gmv_on_real_fixtures_is_feasible_and_long_only_nonnegative():
    _, cov_daily = sample_mean_cov(_real_returns_matrix())
    cov_annual = annualize_cov(cov_daily)
    point = global_min_variance_weights(cov_annual, long_only=True)
    assert point.success
    assert point.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(point.weights >= -1e-6)


def test_gmv_variance_is_no_larger_than_equal_weight():
    # By construction the GMV portfolio minimizes variance over the budget
    # simplex, so it cannot do worse than the naive 1/N benchmark Day 6-7
    # compares against.
    _, cov_daily = sample_mean_cov(_real_returns_matrix())
    cov_annual = annualize_cov(cov_daily)
    n = cov_annual.shape[0]
    equal_weight = np.full(n, 1.0 / n)
    equal_weight_variance = float(equal_weight @ cov_annual @ equal_weight)

    point = global_min_variance_weights(cov_annual, long_only=True)
    assert point.variance <= equal_weight_variance + 1e-9


# --- weight_shift ------------------------------------------------------------


def test_weight_shift_zero_for_identical_vectors():
    w = np.array([0.5, 0.3, 0.2])
    shift = weight_shift(w, w)
    assert shift["l1"] == pytest.approx(0.0)
    assert shift["max_abs"] == pytest.approx(0.0)


def test_weight_shift_l1_and_max_abs():
    a = np.array([0.6, 0.4])
    b = np.array([0.5, 0.5])
    shift = weight_shift(a, b)
    np.testing.assert_allclose(shift["diff"], [-0.1, 0.1])
    assert shift["l1"] == pytest.approx(0.2)
    assert shift["max_abs"] == pytest.approx(0.1)
