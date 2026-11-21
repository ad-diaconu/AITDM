"""Shared fixtures: the synthetic WISDM data, windowed and split into folds once per session."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data_splitting", "experiments", "fl_app", "tests"):
    sys.path.insert(0, str(ROOT / sub))


@pytest.fixture(scope="session")
def windows(tmp_path_factory):
    """(x, y, users, is_eval, folds) of the synthetic data, as prepare_wisdm.py builds them."""
    import prepare_wisdm
    from synthetic_wisdm import make

    raw = tmp_path_factory.mktemp("raw") / "raw.txt"
    make(raw)
    df, _ = prepare_wisdm.load_and_clean(raw)
    x, y, users, is_eval = prepare_wisdm.make_windows(df, 200, 100, 200, 1000)
    folds = prepare_wisdm.make_folds(y, users, is_eval, 5, 42)
    return x, y, users, is_eval, folds
