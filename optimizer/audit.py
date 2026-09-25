"""Day 7: pipeline audit for the walk-forward out-of-sample module.

Two checks, both mechanical rather than by-inspection, in the spirit of
Day 2's convexity check and Day 6's own honesty about what the walk-forward
test does and does not prove:

1. **Fold boundaries.** :func:`optimizer.oos.walk_forward_folds` is the one
   function standing between this project and the "optimizing and reporting
   on the same window is not a result" trap NEXT_STEPS.md's own "traps"
   section warns about. It must produce, for every fold, a training window
   that ends exactly where the test window begins (no gap that wastes data,
   no overlap that leaks the holdout into training) and test windows that
   never overlap each other (so no trading day is ever scored twice). This
   checks both invariants directly against the fold generator's own output.

2. **Decision causality.** ``run_fold`` is supposed to pick a fold's weights
   using only that fold's training window - never the test window it is
   about to be scored against, and never any later fold's data. That is
   true by construction today (``train_returns = returns_matrix[train_slice]``),
   but "by construction" is exactly the kind of claim a later refactor (an
   expanding window, a shared cache keyed by ticker instead of by window)
   can quietly break without any existing test catching it, since every
   existing test runs on the one committed fixture set. This checks it
   directly: a fold's chosen weights on the full return series must be
   identical to its weights when every row after that fold's own test
   window is simply not there yet. If truncating the future ever changes a
   past fold's decision, the walk-forward test is not out-of-sample and the
   audit fails.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from optimizer.oos import (
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    run_fold,
    walk_forward_folds,
)
from optimizer.returns import FIXTURES_DIR, load_returns_matrix
from optimizer.tangency import DEFAULT_RISK_FREE_RATE

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

_WEIGHT_TOL = 1e-9


def fold_boundary_violations(folds: list[tuple[slice, slice]]) -> list[str]:
    """Gap/overlap checks on the (train, test) slices a walk-forward run uses.

    Training windows are allowed - expected - to overlap each other fold to
    fold (it is a rolling, not expanding, window). What must never happen:
    a fold's own train/test boundary having a gap or overlap, or two folds'
    test windows overlapping (the same day scored as "out-of-sample" twice).
    """
    violations = []
    for i, (train_slice, test_slice) in enumerate(folds):
        if train_slice.stop != test_slice.start:
            violations.append(
                f"fold {i}: training window ends at row {train_slice.stop} but test window "
                f"starts at row {test_slice.start} - gap or overlap between training and holdout"
            )
        if test_slice.stop <= test_slice.start:
            violations.append(f"fold {i}: empty test window ({test_slice.start}:{test_slice.stop})")
    for i in range(1, len(folds)):
        prev_test, this_test = folds[i - 1][1], folds[i][1]
        if this_test.start < prev_test.stop:
            violations.append(
                f"fold {i}: test window starts at row {this_test.start}, before fold {i - 1}'s "
                f"test window ends at row {prev_test.stop} - the same day would be scored twice"
            )
    return violations


def fold_causality_violations(
    returns_matrix: np.ndarray,
    dates: list[str],
    folds: list[tuple[slice, slice]],
    risk_free: float,
) -> list[str]:
    """Folds whose chosen weights change once data past their own test window is removed.

    For each fold, recomputes ``run_fold`` on a copy of the data truncated
    to end exactly at that fold's own test window - i.e. as if no trading
    day after it had happened yet - and compares the resulting weights
    against the same fold run on the full series. A fold's weight decision
    depending on rows past its own test window is impossible today given
    how ``run_fold`` slices its input, but that is exactly the kind of
    invariant a refactor could break silently; this checks the observable
    behaviour instead of trusting the slicing.
    """
    violations = []
    for i, (train_slice, test_slice) in enumerate(folds):
        full_fold = run_fold(returns_matrix, dates, train_slice, test_slice, i, risk_free)

        cutoff = test_slice.stop
        truncated_matrix = returns_matrix[:cutoff]
        truncated_dates = dates[:cutoff]
        truncated_fold = run_fold(truncated_matrix, truncated_dates, train_slice, test_slice, i, risk_free)

        if not np.allclose(full_fold.optimizer_weights, truncated_fold.optimizer_weights, atol=_WEIGHT_TOL):
            violations.append(
                f"fold {i} ({full_fold.test_start_date}..{full_fold.test_end_date}): weights on the full "
                f"series {full_fold.optimizer_weights!r} differ from weights with all data after this "
                f"fold's test window removed {truncated_fold.optimizer_weights!r} - the decision depends "
                "on data the strategy would not have had yet"
            )
    return violations


def run_audit(
    tickers: list[str], fixtures_dir: str, train_window: int, test_window: int, risk_free: float
) -> dict[str, Any]:
    dates, returns_matrix = load_returns_matrix(tickers, fixtures_dir)
    n_days = returns_matrix.shape[0]
    folds = walk_forward_folds(n_days, train_window, test_window)

    boundary_findings = fold_boundary_violations(folds)
    causality_findings = fold_causality_violations(returns_matrix, dates, folds, risk_free) if not boundary_findings else []

    return {
        "tickers": tickers,
        "train_window": train_window,
        "test_window": test_window,
        "n_folds": len(folds),
        "boundary_findings": boundary_findings,
        "boundary_clean": not boundary_findings,
        "causality_findings": causality_findings,
        "causality_clean": not causality_findings,
        "clean": not boundary_findings and not causality_findings,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Pipeline audit - optimizer.oos walk-forward, {', '.join(report['tickers'])}, {report['n_folds']} folds")

    if report["boundary_clean"]:
        print("  fold boundaries: no gap, overlap, or double-scored day detected")
    else:
        print(f"  fold boundaries: {len(report['boundary_findings'])} violation(s)")
        for v in report["boundary_findings"]:
            print(f"    - {v}")

    if not report["boundary_findings"]:
        if report["causality_clean"]:
            print("  causality: no look-ahead detected - every fold's weights are unchanged with later data removed")
        else:
            print(f"  causality: {len(report['causality_findings'])} violation(s)")
            for v in report["causality_findings"]:
                print(f"    - {v}")
    else:
        print("  causality: skipped - fold boundaries must hold first")


def write_markdown(report: dict[str, Any], path: Path, generated: str) -> None:
    lines = [
        f"# Pipeline audit — {generated}",
        "",
        "Checks two things about `optimizer.oos`'s walk-forward test: that "
        "every fold's training/test boundary has no gap or overlap and no "
        "two folds' test windows overlap, and that each fold's chosen "
        "weights are unchanged when every trading day after that fold's own "
        "test window is removed - i.e. the decision never depends on data "
        "the strategy would not have had yet.",
        "",
        f"**Fold boundaries:** {'clean' if report['boundary_clean'] else 'violations found'}. "
        f"**Causality:** {'clean' if report['causality_clean'] else 'violations found'}.",
        "",
    ]
    if report["boundary_findings"]:
        lines.append("## Fold boundary violations")
        lines.append("")
        lines += [f"- {v}" for v in report["boundary_findings"]]
        lines.append("")
    if report["causality_findings"]:
        lines.append("## Causality violations")
        lines.append("")
        lines += [f"- {v}" for v in report["causality_findings"]]
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tickers", default="RELIANCE.NS,TATACHEM.NS,CROMPTON.NS", help="comma-separated tickers"
    )
    parser.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--test-window", type=int, default=DEFAULT_TEST_WINDOW)
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    parser.add_argument("--fixtures-dir", default=str(FIXTURES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--date", default=None, help="override the generated date (mainly for tests)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    generated = args.date or date.today().isoformat()

    report = run_audit(tickers, args.fixtures_dir, args.train_window, args.test_window, args.risk_free_rate)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(report, out_dir / f"audit_{generated}.md", generated)

    print_report(report)
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
