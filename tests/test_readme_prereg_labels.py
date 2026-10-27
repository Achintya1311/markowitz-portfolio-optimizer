"""P107, P108: the pre-registration claim matches git, and the Sharpe basis is stated."""

from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()


def test_preregistration_sentence_matches_the_git_record():
    for token in ("f5a59cf", "46927f7", "16:33:20", "16:35:54", "order"):
        assert token in README


def test_preregistered_sharpe_is_labelled_rf_zero():
    assert "| Rule (net of costs) | Compounded | Sharpe (rf = 0) |" in README
    lead = next(line for line in README.splitlines() if line.startswith("**Update (24 Oct 2026"))
    assert "Sharpe at rf = 0 (0.94 vs 1.13)" in lead
