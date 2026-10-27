"""D57: README Data sources matches what the repo actually uses.

The three-name fixtures are byte copies of STOCKSTALKER's bars, pinned by
hash and compared with the sibling checkout when one exists.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "STOCKSTALKER" / "fixtures" / "ohlcv"
PINNED = {
    "RELIANCE_NS.csv": "81d022cb3dff50dc316d94dc1a58b64947a991850c6595480165a2723b623369",
    "TATACHEM_NS.csv": "986700d1bd15887486c77d240cbcd7f668664efc9d4a207b47de898e47a585ed",
    "CROMPTON_NS.csv": "db66db9aba5594544631f8887e7540101257e953cdec5f65ef76318b5a54327a",
}
SOURCE_COMMIT = "d9fe2d8"


def _data_sources() -> str:
    text = (ROOT / "README.md").read_text()
    return text.split("## Data sources", 1)[1].split("\n## ", 1)[0]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_three_name_copy_matches_the_pinned_hash(name):
    assert hashlib.sha256((ROOT / "fixtures" / "ohlcv" / name).read_bytes()).hexdigest() == PINNED[name]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_three_name_copy_matches_stockstalker_when_checked_out(name):
    sibling = SIBLING / name
    if not sibling.is_file():
        pytest.skip("STOCKSTALKER is not checked out next to this repo")
    assert (ROOT / "fixtures" / "ohlcv" / name).read_bytes() == sibling.read_bytes()


def test_data_sources_say_what_the_repo_uses():
    sources = _data_sources()
    assert "FRED / RBI" not in sources
    assert "assumed parameter" in sources
    for token in ("fixtures/ohlcv/", "fixtures/ohlcv_largecap/", "auto_adjust=True",
                  "back-adjusted", "fetch-largecaps.yml", SOURCE_COMMIT, *PINNED.values()):
        assert token in sources
