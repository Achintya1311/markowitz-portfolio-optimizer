import numpy as np
import pytest
from scipy.optimize import minimize

from optimizer.covariance import sample_mean_cov
from optimizer.returns import annualize_cov, annualize_mean, load_returns_matrix
from optimizer.tangency import (
    capital_market_line,
    max_sharpe_bound,
    max_sharpe_weights_long_only,
    max_sharpe_weights_unconstrained,
    sharpe_ratio,
)

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]

# Synthetic universe with positive excess returns, so ones @ inv(cov) @ excess > 0
# and the closed-form tangency portfolio is well-defined (not the degenerate case
# the real fixture universe hits).
_POS_MU = np.array([0.15, 0.10, 0.08])
_POS_COV = np.array(
    [
        [0.04, 0.01, 0.005],
        [0.01, 0.03, 0.004],
        [0.005, 0.004, 0.02],
    ]
)


def _real_mu_cov():
    _, returns_matrix = load_returns_matrix(TICKERS)
    mu_daily, cov_daily = sample_mean_cov(returns_matrix)
    return annualize_mean(mu_daily), annualize_cov(cov_daily)


def _numeric_max_sharpe(mu, cov, risk_free, long_only):
    n = len(mu)

    def negative_sharpe(w):
        port_return = w @ mu
        port_vol = float(w @ cov @ w) ** 0.5
        return -(port_return - risk_free) / port_vol

    bounds = [(0.0, None) for _ in range(n)] if long_only else None
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        negative_sharpe,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    assert result.success
    return result.x, -result.fun


def test_unconstrained_matches_theoretical_sharpe_bound_when_well_defined():
    rf = 0.02
    point = max_sharpe_weights_unconstrained(_POS_MU, _POS_COV, rf)
    bound = max_sharpe_bound(_POS_MU, _POS_COV, rf)

    assert point.success
    assert point.sharpe == pytest.approx(bound, abs=1e-8)
    assert np.sum(point.weights) == pytest.approx(1.0, abs=1e-8)


def test_unconstrained_matches_numeric_optimum_when_well_defined():
    rf = 0.02
    point = max_sharpe_weights_unconstrained(_POS_MU, _POS_COV, rf)
    numeric_weights, numeric_sharpe = _numeric_max_sharpe(_POS_MU, _POS_COV, rf, long_only=False)

    np.testing.assert_allclose(point.weights, numeric_weights, atol=1e-5)
    assert point.sharpe == pytest.approx(numeric_sharpe, abs=1e-6)


def test_unconstrained_is_degenerate_when_denominator_is_negative():
    # Real fixture universe: every asset's annualized sample mean is negative
    # (Day 1's finding), so at rf=0 every excess return is negative and
    # ones @ inv(cov) @ excess < 0 - the classic "no finite tangency
    # portfolio" case, not a bug to paper over.
    mu, cov = _real_mu_cov()
    point = max_sharpe_weights_unconstrained(mu, cov, risk_free=0.0)
    assert not point.success


def test_unconstrained_never_silently_returns_the_minimum_sharpe_portfolio():
    # Regression check for the exact bug this module used to have: dividing
    # by a negative ones @ inv(cov) @ excess lands on the Sharpe-*minimizing*
    # direction, not the maximizing one. If max_sharpe_weights_unconstrained
    # ever reports success in this branch, its Sharpe must not be the
    # negative of the theoretical bound (the minimum), which is what the
    # naive (unfixed) formula silently produced.
    mu, cov = _real_mu_cov()
    bound = max_sharpe_bound(mu, cov, risk_free=0.0)
    point = max_sharpe_weights_unconstrained(mu, cov, risk_free=0.0)
    if point.success:
        assert point.sharpe != pytest.approx(-bound, abs=1e-3)


def test_long_only_matches_numeric_optimum_on_real_fixtures():
    mu, cov = _real_mu_cov()
    point = max_sharpe_weights_long_only(mu, cov, risk_free=0.0)
    numeric_weights, numeric_sharpe = _numeric_max_sharpe(mu, cov, risk_free=0.0, long_only=True)

    assert point.success
    np.testing.assert_allclose(point.weights, numeric_weights, atol=1e-4)
    assert point.sharpe == pytest.approx(numeric_sharpe, abs=1e-5)


def test_long_only_weights_are_non_negative_and_sum_to_one():
    mu, cov = _real_mu_cov()
    point = max_sharpe_weights_long_only(mu, cov, risk_free=0.0)
    assert point.success
    assert np.all(point.weights >= -1e-8)
    assert np.sum(point.weights) == pytest.approx(1.0, abs=1e-6)


def test_long_only_sharpe_never_exceeds_theoretical_bound():
    # The long-only feasible set is a subset of the full-investment
    # (any sign) set, so its best achievable Sharpe can never beat the
    # unconstrained bound, on any input.
    mu, cov = _real_mu_cov()
    bound = max_sharpe_bound(mu, cov, risk_free=0.0)
    point = max_sharpe_weights_long_only(mu, cov, risk_free=0.0)
    assert point.success
    assert point.sharpe <= bound + 1e-8


def test_long_only_finds_positive_sharpe_when_one_asset_has_positive_excess_return():
    mu = np.array([0.12, -0.05, -0.10])
    cov = np.array([[0.05, 0.01, 0.01], [0.01, 0.04, 0.01], [0.01, 0.01, 0.03]])
    point = max_sharpe_weights_long_only(mu, cov, risk_free=0.0)
    assert point.success
    assert point.sharpe > 0


def test_sharpe_ratio_matches_dataclass_field():
    mu, cov = _real_mu_cov()
    point = max_sharpe_weights_long_only(mu, cov, risk_free=0.0)
    assert sharpe_ratio(point.weights, mu, cov, risk_free=0.0) == pytest.approx(point.sharpe, abs=1e-10)


def test_sharpe_ratio_rejects_zero_variance_portfolio():
    mu = np.array([0.1, 0.1])
    cov = np.zeros((2, 2))
    with pytest.raises(ValueError):
        sharpe_ratio(np.array([0.5, 0.5]), mu, cov, risk_free=0.0)


def test_max_sharpe_weights_unconstrained_rejects_single_asset():
    with pytest.raises(ValueError):
        max_sharpe_weights_unconstrained(np.array([0.1]), np.array([[0.04]]), risk_free=0.0)


def test_max_sharpe_weights_long_only_rejects_single_asset():
    with pytest.raises(ValueError):
        max_sharpe_weights_long_only(np.array([0.1]), np.array([[0.04]]), risk_free=0.0)


def test_capital_market_line_passes_through_risk_free_point():
    vols, returns = capital_market_line(risk_free=0.05, tangency_return=0.15, tangency_vol=0.20, n_points=5)
    assert vols[0] == pytest.approx(0.0)
    assert returns[0] == pytest.approx(0.05)


def test_capital_market_line_passes_through_tangency_point():
    vols, returns = capital_market_line(risk_free=0.05, tangency_return=0.15, tangency_vol=0.20, n_points=5, max_leverage=1.0)
    assert vols[-1] == pytest.approx(0.20)
    assert returns[-1] == pytest.approx(0.15)


def test_capital_market_line_is_linear_in_vol():
    vols, returns = capital_market_line(risk_free=0.05, tangency_return=0.15, tangency_vol=0.20, n_points=11)
    expected_slope = (0.15 - 0.05) / 0.20
    # return = rf + slope * vol for every point (skip vol=0 to avoid 0/0)
    for v, r in zip(vols[1:], returns[1:]):
        assert r == pytest.approx(0.05 + expected_slope * v, abs=1e-10)


def test_capital_market_line_rejects_zero_tangency_vol():
    with pytest.raises(ValueError):
        capital_market_line(risk_free=0.05, tangency_return=0.15, tangency_vol=0.0)
