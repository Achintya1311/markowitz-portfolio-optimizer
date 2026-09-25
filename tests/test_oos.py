import numpy as np
import pytest

from optimizer.oos import (
    OOSResult,
    FoldResult,
    run,
    run_fold,
    total_simple_returns,
    walk_forward_folds,
)
from optimizer.returns import FIXTURES_DIR, load_returns_matrix

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


# --- walk_forward_folds ---


def test_folds_are_non_overlapping_and_contiguous():
    folds = walk_forward_folds(n_days=100, train_window=50, test_window=10)
    for train_slice, test_slice in folds:
        assert train_slice.stop == test_slice.start
        assert test_slice.stop - test_slice.start == 10
        assert train_slice.stop - train_slice.start == 50


def test_folds_step_forward_by_test_window():
    folds = walk_forward_folds(n_days=100, train_window=50, test_window=10)
    starts = [train_slice.start for train_slice, _ in folds]
    assert starts == sorted(starts)
    assert all(b - a == 10 for a, b in zip(starts, starts[1:]))


def test_folds_drop_incomplete_trailing_window():
    # 100 days, train=50, test=21: fold 0 uses [0:50]/[50:71], fold 1 would need
    # [21:71]/[71:92], fold 2 would need [42:92]/[92:113] which overruns 100 days.
    folds = walk_forward_folds(n_days=100, train_window=50, test_window=21)
    for _, test_slice in folds:
        assert test_slice.stop <= 100


def test_no_folds_when_data_too_short():
    assert walk_forward_folds(n_days=10, train_window=50, test_window=10) == []


def test_train_window_must_be_at_least_two():
    with pytest.raises(ValueError, match="train_window"):
        walk_forward_folds(n_days=100, train_window=1, test_window=10)


def test_test_window_must_be_positive():
    with pytest.raises(ValueError, match="test_window"):
        walk_forward_folds(n_days=100, train_window=50, test_window=0)


# --- total_simple_returns ---


def test_total_simple_return_matches_price_ratio():
    # log returns of [10 -> 11 -> 9.9] should compound to the same total
    # simple return as the direct price ratio, for two "assets" side by side.
    prices_a = np.array([10.0, 11.0, 9.9])
    prices_b = np.array([5.0, 5.5, 6.05])
    log_returns = np.column_stack([np.diff(np.log(prices_a)), np.diff(np.log(prices_b))])

    total = total_simple_returns(log_returns)

    expected_a = prices_a[-1] / prices_a[0] - 1.0
    expected_b = prices_b[-1] / prices_b[0] - 1.0
    np.testing.assert_allclose(total, [expected_a, expected_b])


def test_total_simple_return_zero_for_flat_prices():
    log_returns = np.zeros((5, 3))
    np.testing.assert_allclose(total_simple_returns(log_returns), np.zeros(3))


# --- run_fold ---


def test_run_fold_weights_sum_to_one_when_successful():
    _, returns_matrix = load_returns_matrix(TICKERS)
    dates, _ = load_returns_matrix(TICKERS)
    fold = run_fold(
        returns_matrix, dates, slice(0, 200), slice(200, 221), fold_index=0, risk_free=0.0
    )
    if fold.optimizer_success:
        assert np.sum(fold.optimizer_weights) == pytest.approx(1.0, abs=1e-6)
        assert np.all(fold.optimizer_weights >= -1e-9)


def test_run_fold_equal_weight_return_matches_manual_calc():
    _, returns_matrix = load_returns_matrix(TICKERS)
    dates, _ = load_returns_matrix(TICKERS)
    train_slice, test_slice = slice(0, 200), slice(200, 221)
    fold = run_fold(returns_matrix, dates, train_slice, test_slice, fold_index=0, risk_free=0.0)

    manual = total_simple_returns(returns_matrix[test_slice])
    expected_equal_weight = float(np.mean(manual))
    assert fold.equal_weight_return == pytest.approx(expected_equal_weight, abs=1e-9)


def test_run_fold_records_test_and_train_date_boundaries():
    _, returns_matrix = load_returns_matrix(TICKERS)
    dates, _ = load_returns_matrix(TICKERS)
    train_slice, test_slice = slice(0, 200), slice(200, 221)
    fold = run_fold(returns_matrix, dates, train_slice, test_slice, fold_index=3, risk_free=0.0)

    assert fold.fold == 3
    assert fold.train_start_date == dates[0]
    assert fold.train_end_date == dates[199]
    assert fold.test_start_date == dates[200]
    assert fold.test_end_date == dates[220]


def test_run_fold_falls_back_to_equal_weight_when_infeasible(monkeypatch):
    # max_sharpe_weights_long_only essentially never fails to converge in
    # practice (see optimizer.tangency's own tests), so exercise the
    # fallback branch directly rather than hunting for a numeric case that
    # trips it.
    import optimizer.oos as oos_module
    from optimizer.tangency import TangencyPoint

    def fake_infeasible(mu, cov, risk_free):
        return TangencyPoint(
            weights=np.array([1.0, 0.0]), achieved_return=0.0, variance=0.0, sharpe=float("nan"),
            long_only=True, success=False,
        )

    monkeypatch.setattr(oos_module, "max_sharpe_weights_long_only", fake_infeasible)

    _, returns_matrix = load_returns_matrix(TICKERS[:2])
    dates, _ = load_returns_matrix(TICKERS[:2])
    fold = run_fold(returns_matrix, dates, slice(0, 40), slice(40, 60), fold_index=0, risk_free=0.0)

    assert fold.used_fallback
    assert not fold.optimizer_success
    np.testing.assert_allclose(fold.optimizer_weights, [0.5, 0.5])
    assert fold.optimizer_return == pytest.approx(fold.equal_weight_return)


# --- run / OOSResult aggregation ---


def test_run_on_real_fixtures_produces_expected_fold_count():
    result = run(TICKERS, str(FIXTURES_DIR), train_window=252, test_window=21, risk_free=0.0)
    assert len(result.folds) == 11
    assert result.unused_days == 500 - (252 + 11 * 21)


def test_run_rejects_too_short_a_history():
    with pytest.raises(ValueError, match="not enough"):
        run(TICKERS, str(FIXTURES_DIR), train_window=400, test_window=200, risk_free=0.0)


def test_compounded_return_matches_product_of_fold_returns():
    result = run(TICKERS, str(FIXTURES_DIR), train_window=252, test_window=21, risk_free=0.0)
    manual_opt = np.prod([1.0 + f.optimizer_return for f in result.folds]) - 1.0
    manual_eq = np.prod([1.0 + f.equal_weight_return for f in result.folds]) - 1.0
    assert result.optimizer_compounded_return == pytest.approx(manual_opt)
    assert result.equal_weight_compounded_return == pytest.approx(manual_eq)


def test_win_rate_is_fraction_of_folds_optimizer_beats_equal_weight():
    result = run(TICKERS, str(FIXTURES_DIR), train_window=252, test_window=21, risk_free=0.0)
    manual_wins = sum(1 for f in result.folds if f.excess_return > 0)
    assert result.win_rate == pytest.approx(manual_wins / len(result.folds))


def test_run_is_deterministic():
    a = run(TICKERS, str(FIXTURES_DIR), train_window=252, test_window=21, risk_free=0.0)
    b = run(TICKERS, str(FIXTURES_DIR), train_window=252, test_window=21, risk_free=0.0)
    assert a.optimizer_compounded_return == b.optimizer_compounded_return
    assert a.equal_weight_compounded_return == b.equal_weight_compounded_return


def test_oos_result_aggregation_from_synthetic_folds():
    folds = [
        FoldResult(0, "d0", "d1", "d2", "d3", 0.1, np.array([1.0, 0.0]), True, False, 0.05, 0.02),
        FoldResult(1, "d3", "d4", "d5", "d6", 0.1, np.array([0.0, 1.0]), True, False, -0.01, 0.01),
    ]
    result = OOSResult(
        tickers=["A", "B"], train_window=50, test_window=10, risk_free=0.0, folds=folds, unused_days=0
    )
    assert result.optimizer_compounded_return == pytest.approx(1.05 * 0.99 - 1.0)
    assert result.equal_weight_compounded_return == pytest.approx(1.02 * 1.01 - 1.0)
    assert result.win_rate == pytest.approx(0.5)
    assert result.mean_excess == pytest.approx((0.03 + (-0.02)) / 2)
    assert result.n_fallback == 0
