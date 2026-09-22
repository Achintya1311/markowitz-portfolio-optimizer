import numpy as np
import pytest

from optimizer.covariance import (
    condition_number,
    correlation_matrix,
    mean_t_stats,
    sample_mean_cov,
    split_half_stability,
    standard_error_of_annualized_mean,
)
from optimizer.returns import TRADING_DAYS_PER_YEAR, load_returns_matrix

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def test_sample_mean_cov_matches_numpy_directly():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 3))
    mu, cov = sample_mean_cov(data)
    np.testing.assert_allclose(mu, data.mean(axis=0))
    np.testing.assert_allclose(cov, np.cov(data, rowvar=False, ddof=1))


def test_sample_mean_cov_shapes():
    _, returns_matrix = load_returns_matrix(TICKERS)
    mu, cov = sample_mean_cov(returns_matrix)
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)


def test_standard_error_shrinks_with_more_observations():
    rng = np.random.default_rng(1)
    daily = rng.normal(scale=0.02, size=(500, 2))
    _, cov_daily = sample_mean_cov(daily)
    se_100 = standard_error_of_annualized_mean(cov_daily, n_obs=100)
    se_400 = standard_error_of_annualized_mean(cov_daily, n_obs=400)
    # SE scales as 1/sqrt(n): quadrupling n halves it.
    np.testing.assert_allclose(se_400, se_100 / 2, rtol=1e-8)


def test_standard_error_known_closed_form():
    sigma_daily = 0.02
    cov_daily = np.array([[sigma_daily**2]])
    se = standard_error_of_annualized_mean(cov_daily, n_obs=252)
    expected = TRADING_DAYS_PER_YEAR * sigma_daily / np.sqrt(252)
    assert se[0] == pytest.approx(expected)


def test_mean_t_stats_ratio():
    mu = np.array([0.10, -0.05])
    se = np.array([0.20, 0.10])
    t = mean_t_stats(mu, se)
    np.testing.assert_allclose(t, [0.5, -0.5])


def test_condition_number_identity_is_one():
    assert condition_number(np.eye(3)) == pytest.approx(1.0)


def test_condition_number_rises_with_correlation():
    low_corr = np.array([[1.0, 0.1], [0.1, 1.0]])
    high_corr = np.array([[1.0, 0.95], [0.95, 1.0]])
    assert condition_number(high_corr) > condition_number(low_corr)


def test_correlation_matrix_diagonal_is_one():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    corr = correlation_matrix(cov)
    np.testing.assert_allclose(np.diag(corr), [1.0, 1.0])
    assert corr[0, 1] == pytest.approx(0.01 / (0.2 * 0.3))


def test_split_half_stability_on_real_fixtures():
    _, returns_matrix = load_returns_matrix(TICKERS)
    split = split_half_stability(TICKERS, returns_matrix)
    assert split.mean_shift.shape == (3,)
    assert split.cov_frobenius_relative_change >= 0.0
    # Two independently-estimated covariance matrices from real market data
    # essentially never come out bit-identical.
    assert not np.allclose(split.cov_annual_first_half, split.cov_annual_second_half)


def test_split_half_stability_zero_for_identical_halves():
    rng = np.random.default_rng(2)
    half = rng.normal(scale=0.01, size=(100, 2))
    both_halves = np.vstack([half, half])
    split = split_half_stability(["A", "B"], both_halves)
    np.testing.assert_allclose(split.mean_shift, 0.0, atol=1e-12)
    assert split.cov_frobenius_relative_change == pytest.approx(0.0, abs=1e-10)
