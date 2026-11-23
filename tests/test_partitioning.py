"""data_splitting/partitioning.py: the federated splits S0-S3 and the opt-in corruptions."""

import numpy as np
import pytest
from partitioning import CLASSES, apply_label_noise, apply_sensor_skew, assign, jsd, valid


def split(windows, name, k, attempt=0):
    x, y, users, _, folds = windows
    tr, va, te = folds[0]
    return (tr, va, te), assign(name, k, tr, va, te, y, users, seed=42, attempt=attempt)


@pytest.mark.parametrize("name", ["s0", "s1", "s2", "s3"])
def test_every_window_on_exactly_one_client(windows, name):
    (tr, va, te), clients = split(windows, name, 3)
    for idx, c in zip((tr, va, te), clients):
        assert len(c) == len(idx) and set(np.unique(c)) <= {0, 1, 2}


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_user_level_splits_keep_users_whole(windows, name):
    _, _, users, _, _ = windows
    (tr, va, te), clients = split(windows, name, 3)
    for idx, c in zip((tr, va, te), clients):
        for u in np.unique(users[idx]):
            assert len(set(c[users[idx] == u])) == 1


def test_val_and_test_users_never_split(windows):
    """Even S0 / S3 (window-level training splits) keep evaluation users on one client."""
    _, _, users, _, _ = windows
    for name in ("s0", "s3"):
        (_, va, te), (_, c_va, c_te) = split(windows, name, 3)
        for idx, c in ((va, c_va), (te, c_te)):
            assert all(len(set(c[users[idx] == u])) == 1 for u in np.unique(users[idx]))


def test_same_seed_same_split(windows):
    _, a = split(windows, "s2", 3)
    _, b = split(windows, "s2", 3)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    _, c = split(windows, "s2", 3, attempt=1)
    assert not all(np.array_equal(p, q) for p, q in zip(a, c))


def test_valid_rejects_empty_client(windows):
    _, _, users, _, _ = windows
    (tr, va, te), (c_tr, c_va, c_te) = split(windows, "s1", 3)
    assert valid(c_tr, c_va, c_te, tr, users, 3, 0.05)
    assert not valid(np.zeros_like(c_tr), c_va, c_te, tr, users, 3, 0.05)


def test_jsd_bounds():
    p = np.array([1.0, 0, 0])
    assert jsd(p, p) == 0
    assert jsd(p, np.array([0, 1.0, 0])) == pytest.approx(1.0)


def test_label_noise_rate():
    labels = np.random.default_rng(0).integers(0, len(CLASSES), size=5000)
    noisy = apply_label_noise(labels, 0.2, seed=0)
    assert 0.17 < np.mean(noisy != labels) < 0.23
    assert np.array_equal(apply_label_noise(labels, 0.0), labels)


def test_sensor_skew_leaves_client_0_unchanged():
    x = np.random.default_rng(0).normal(size=(20, 200, 3)).astype(np.float32)
    assert np.allclose(apply_sensor_skew(x, 0, 3), x)
    skewed = apply_sensor_skew(x, 2, 3)
    assert skewed.shape == x.shape and skewed.dtype == x.dtype and not np.allclose(skewed, x)
