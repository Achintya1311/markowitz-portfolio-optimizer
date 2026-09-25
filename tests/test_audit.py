import numpy as np

from optimizer.audit import (
    fold_boundary_violations,
    fold_causality_violations,
    run_audit,
)
from optimizer.oos import walk_forward_folds
from optimizer.returns import FIXTURES_DIR, load_returns_matrix

TICKERS = ["RELIANCE.NS", "TATACHEM.NS", "CROMPTON.NS"]


# --- fold_boundary_violations ---


def test_boundary_clean_on_generator_output():
    folds = walk_forward_folds(n_days=500, train_window=252, test_window=21)
    assert fold_boundary_violations(folds) == []


def test_boundary_catches_train_test_gap():
    folds = [(slice(0, 50), slice(51, 60))]
    violations = fold_boundary_violations(folds)
    assert len(violations) == 1
    assert "gap or overlap" in violations[0]


def test_boundary_catches_train_test_overlap():
    folds = [(slice(0, 50), slice(40, 60))]
    violations = fold_boundary_violations(folds)
    assert len(violations) == 1
    assert "gap or overlap" in violations[0]


def test_boundary_catches_overlapping_test_windows():
    folds = [(slice(0, 50), slice(50, 70)), (slice(10, 60), slice(60, 65))]
    violations = fold_boundary_violations(folds)
    assert any("scored twice" in v for v in violations)


def test_boundary_catches_empty_test_window():
    folds = [(slice(0, 50), slice(50, 50))]
    violations = fold_boundary_violations(folds)
    assert any("empty test window" in v for v in violations)


# --- fold_causality_violations ---


def test_causality_clean_on_committed_fixtures():
    dates, returns_matrix = load_returns_matrix(TICKERS, FIXTURES_DIR)
    folds = walk_forward_folds(returns_matrix.shape[0], train_window=252, test_window=21)
    assert fold_causality_violations(returns_matrix, dates, folds, risk_free=0.0) == []


def test_causality_flags_a_fold_that_reads_past_its_own_test_window():
    dates, returns_matrix = load_returns_matrix(TICKERS, FIXTURES_DIR)
    folds = walk_forward_folds(returns_matrix.shape[0], train_window=252, test_window=21)

    # A leaking training slice for fold 0 that reads past its own test
    # window into fold 1's data - simulates the exact bug the audit exists
    # to catch (fold 1 must exist for there to be anything to leak from).
    assert len(folds) >= 2
    train_slice, test_slice = folds[0]
    leaking_train_slice = slice(train_slice.start, test_slice.stop + 21)
    leaking_folds = [(leaking_train_slice, test_slice)] + folds[1:]

    violations = fold_causality_violations(returns_matrix, dates, leaking_folds, risk_free=0.0)
    assert len(violations) >= 1
    assert "fold 0" in violations[0]
    assert "would not have had yet" in violations[0]


# --- run_audit end to end ---


def test_run_audit_is_clean_on_committed_fixtures():
    report = run_audit(TICKERS, str(FIXTURES_DIR), train_window=252, test_window=21, risk_free=0.0)
    assert report["clean"]
    assert report["boundary_clean"]
    assert report["causality_clean"]
    assert report["n_folds"] > 0
