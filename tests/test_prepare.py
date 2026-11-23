"""data_splitting/prepare_wisdm.py: cleaning, windows and subject-wise folds."""

import numpy as np
import prepare_wisdm


def test_windows_shape_and_labels(windows):
    x, y, users, is_eval, _ = windows
    assert x.shape[1:] == (200, 3) and x.dtype == np.float32 and np.isfinite(x).all()
    assert len(x) == len(y) == len(users) == len(is_eval)
    assert set(np.unique(y)) == set(range(len(prepare_wisdm.CLASSES)))


def test_folds_have_no_user_leak_and_every_class(windows):
    _, y, users, is_eval, folds = windows
    test_users = []
    for tr, va, te in folds:
        sets = [set(users[i]) for i in (tr, va, te)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert is_eval[va].all() and is_eval[te].all()  # evaluation windows do not overlap
        for idx in (tr, va, te):
            assert set(y[idx]) == set(range(len(prepare_wisdm.CLASSES)))
        test_users += sorted(sets[2])
    assert len(test_users) == len(set(test_users))  # every user is a test user at most once


def test_cleaning_drops_duplicates_and_corrupt_rows(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("1,Walking,1300000000000,1.0,2.0,3.0;\n"
                   "1,Walking,1300000000000,1.0,2.0,3.0;\n"      # duplicate
                   "1,Walking,1300000000050,99.0,2.0,3.0;\n"     # |value| > 40
                   "1,Walking,,1.0,2.0,3.0;\n"                   # empty timestamp
                   "2,Sitting,1300000000100,0.1,9.8,0.2;\n")
    df, log = prepare_wisdm.load_and_clean(raw)
    assert len(df) == 2
    assert log["C3_duplicate_rows"] == 1 and log["C4_corrupt_rows"] == 1
