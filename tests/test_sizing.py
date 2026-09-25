import numpy as np
import pytest

from optimizer.constraints import DEFAULT_SECTOR_MAP
from optimizer.covariance import sample_mean_cov
from optimizer.returns import annualize_cov, annualize_mean, load_returns_matrix
from optimizer.shrinkage import ledoit_wolf_shrinkage
from optimizer.sizing import (
    max_sharpe_weights_sector_capped,
    run,
    to_contract,
)
from optimizer.tangency import max_sharpe_weights_long_only

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


def _real_mu_shrunk_cov():
    _, returns_matrix = load_returns_matrix(TICKERS)
    mu_daily, _ = sample_mean_cov(returns_matrix)
    _, cov_shrunk_daily = ledoit_wolf_shrinkage(returns_matrix)
    return annualize_mean(mu_daily), annualize_cov(cov_shrunk_daily)


# --- max_sharpe_weights_sector_capped ---


def test_uncapped_sector_cap_matches_plain_tangency():
    # A cap of 1.0 on every sector cannot change the *optimum* (the budget
    # constraint already caps any single sector at 100%), so the capped
    # solve should reproduce Day 3's plain long-only tangency weights
    # exactly - though since the real optimum already sits at exactly 100%
    # in one sector, a 1.0 cap is trivially "binding" at that boundary too.
    mu, cov = _real_mu_shrunk_cov()
    plain = max_sharpe_weights_long_only(mu, cov, 0.0)
    capped = max_sharpe_weights_sector_capped(
        mu, cov, 0.0, TICKERS, DEFAULT_SECTOR_MAP, {s: 1.0 for s in DEFAULT_SECTOR_MAP.values()}
    )
    assert plain.success and capped.success
    np.testing.assert_allclose(capped.weights, plain.weights, atol=1e-4)


def test_sector_cap_binds_on_real_universe():
    # Day 3/4's finding: the unconstrained tangency portfolio is 100%
    # RELIANCE.NS. A 50% cap on its (single-member) sector must bind.
    mu, cov = _real_mu_shrunk_cov()
    capped = max_sharpe_weights_sector_capped(
        mu, cov, 0.0, TICKERS, DEFAULT_SECTOR_MAP, {s: 0.5 for s in DEFAULT_SECTOR_MAP.values()}
    )
    assert capped.success
    reliance_idx = TICKERS.index("RELIANCE.NS")
    assert capped.weights[reliance_idx] == pytest.approx(0.5, abs=1e-3)
    assert "Energy" in capped.binding_sector_caps
    assert np.sum(capped.weights) == pytest.approx(1.0, abs=1e-6)
    assert np.all(capped.weights >= -1e-6)


def test_sector_cap_binds_differently_from_position_limit_on_synthetic_universe():
    # Same shape as Day 5's own synthetic multi-sector test: two names share
    # a sector, so a sector cap redistributes weight differently from a
    # per-name position limit would.
    tickers = ["A", "B", "C"]
    sector_map = {"A": "Sector1", "B": "Sector1", "C": "Sector2"}
    mu = np.array([0.10, 0.08, 0.02])
    cov = np.diag([0.02, 0.02, 0.02])

    capped = max_sharpe_weights_sector_capped(mu, cov, 0.0, tickers, sector_map, {"Sector1": 0.5, "Sector2": 1.0})
    assert capped.success
    sector1_weight = capped.weights[0] + capped.weights[1]
    assert sector1_weight == pytest.approx(0.5, abs=1e-3)
    assert "Sector1" in capped.binding_sector_caps
    # Unlike a position limit, the cap is on the sector total, not on A alone -
    # both A and B can still be individually above what a 50% position limit
    # on each name would have allowed.
    assert capped.weights[0] > 0.0 and capped.weights[1] > 0.0


def test_unknown_sector_raises():
    mu, cov = _real_mu_shrunk_cov()
    with pytest.raises(ValueError):
        max_sharpe_weights_sector_capped(mu, cov, 0.0, TICKERS, DEFAULT_SECTOR_MAP, {"Nonexistent": 0.5})


def test_sector_cap_out_of_range_raises():
    mu, cov = _real_mu_shrunk_cov()
    with pytest.raises(ValueError):
        max_sharpe_weights_sector_capped(mu, cov, 0.0, TICKERS, DEFAULT_SECTOR_MAP, {"Energy": 1.5})


# --- run() / to_contract() ---


def test_run_flags_real_universe_as_degenerate():
    result = run(TICKERS, 0.0, "fixtures/ohlcv", 0.5)
    assert result["degenerate_sector_caps"] is True
    assert result["is_real_fixture_universe"] is True
    assert result["capped"].success


def test_to_contract_shape_and_values():
    result = run(TICKERS, 0.0, "fixtures/ohlcv", 0.5)
    contract = to_contract(result, "RELIANCE.NS")
    assert set(contract.keys()) == {"sizing"}
    sizing = contract["sizing"]
    assert set(sizing.keys()) == {"weight", "method", "constraint_binding"}
    assert sizing["method"] == "max_sharpe_ledoit_wolf"
    assert sizing["weight"] == pytest.approx(0.5, abs=1e-3)
    assert sizing["constraint_binding"] == "Energy"


def test_to_contract_reports_none_when_ticker_not_binding():
    result = run(TICKERS, 0.0, "fixtures/ohlcv", 0.5)
    contract = to_contract(result, "CROMPTON.NS")
    # CROMPTON.NS gets 0% in the capped solve on the real universe, well
    # under its own 50% sector cap, so that cap does not bind for it.
    assert contract["sizing"]["constraint_binding"] == "none"


def test_to_contract_rejects_unknown_ticker():
    result = run(TICKERS, 0.0, "fixtures/ohlcv", 0.5)
    with pytest.raises(ValueError):
        to_contract(result, "NOTATICKER.NS")


def test_to_contract_refuses_when_solve_infeasible():
    result = run(TICKERS, 0.0, "fixtures/ohlcv", 0.5)
    result["capped"].success = False
    with pytest.raises(ValueError):
        to_contract(result, "RELIANCE.NS")
