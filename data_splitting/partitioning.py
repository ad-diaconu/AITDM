"""Partitioning algorithms for the WISDM federation (docs/experiment_setup_federated.md, section 4).

The splits assign the windows of one subject-wise fold to K clients:

    s0  IID: training windows dealt at random (window level); val / test users packed by user
    s1  natural: every user (and all their windows) on one client, balanced by window count
    s2  user-level Dirichlet(alpha) on each user's dominant class: label skew, users stay whole
    s3  window-level Dirichlet(alpha): every class's training windows split by q[c] ~ Dir(alpha)

plus "real world" corruptions applied after the split (opt-in, split_data.py --label-noise /
--sensor-skew). Everything is pure NumPy and reproducible from the seed.
"""

from __future__ import annotations

import numpy as np

CLASSES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown"]
SPLITS = {
    "s0": "IID (window level)",
    "s1": "natural (user level, balanced)",
    "s2": "user-level Dirichlet alpha=0.3 on dominant class",
    "s3": "window-level Dirichlet alpha=0.3",
}
ALPHA = 0.3


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) of two label distributions."""
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pack_users(users: np.ndarray, sizes: dict, k: int, rng) -> dict:
    """Balanced packing (as the tutorial's partitioning.group_split): shuffle, then largest
    first onto the client with the fewest windows. Returns {user: client}."""
    users = users[rng.permutation(len(users))]
    users = users[np.argsort(-np.array([sizes[u] for u in users]), kind="stable")]
    load, owner = np.zeros(k), {}
    for u in users:
        c = int(np.argmin(load))
        owner[int(u)] = c
        load[c] += sizes[u]
    return owner


def dominant_class(idx, y, users) -> dict:
    return {int(u): int(np.bincount(y[idx][users[idx] == u], minlength=len(CLASSES)).argmax())
            for u in np.unique(users[idx])}


def assign(split: str, k: int, tr, va, te, y, users, seed: int, attempt: int):
    """Return per-split arrays of client ids (aligned with tr, va, te)."""
    rng = np.random.default_rng(seed + attempt)
    count = lambda idx: {int(u): int((users[idx] == u).sum()) for u in np.unique(users[idx])}  # noqa: E731

    def by_owner(idx, owner):
        return np.array([owner[int(u)] for u in users[idx]])

    if split in ("s0", "s1"):
        if split == "s0":
            c_tr = np.empty(len(tr), int)
            c_tr[rng.permutation(len(tr))] = np.arange(len(tr)) % k
        else:
            c_tr = by_owner(tr, pack_users(np.unique(users[tr]), count(tr), k, rng))
        c_va = by_owner(va, pack_users(np.unique(users[va]), count(va), k, rng))
        c_te = by_owner(te, pack_users(np.unique(users[te]), count(te), k, rng))
        return c_tr, c_va, c_te

    q = rng.dirichlet(ALPHA * np.ones(k), size=len(CLASSES))  # q[c] = client shares of class c

    def users_by_q(idx):
        dom = dominant_class(idx, y, users)
        return by_owner(idx, {u: int(rng.choice(k, p=q[c])) for u, c in dom.items()})

    if split == "s2":
        c_tr = users_by_q(tr)
    else:  # s3: split every class's training windows by q[c]
        c_tr = np.empty(len(tr), int)
        for c in range(len(CLASSES)):
            pos = np.where(y[tr] == c)[0]
            pos = pos[rng.permutation(len(pos))]
            cuts = (np.cumsum(q[c]) * len(pos)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(pos, cuts)):
                c_tr[chunk] = client
    return c_tr, users_by_q(va), users_by_q(te)


def valid(c_tr, c_va, c_te, tr, users, k: int, min_frac: float) -> bool:
    """Every client has >= min_frac of the training windows from >= 3 users, and some
    validation and test users."""
    for c in range(k):
        n = (c_tr == c).sum()
        if n < min_frac * len(tr) or len(np.unique(users[tr][c_tr == c])) < 3:
            return False
        if not (c_va == c).any() or not (c_te == c).any():
            return False
    return True


# ---------------------------------------------------------------------------
# Extra "real world" corruptions applied after partitioning (adapted from the tutorial)
# ---------------------------------------------------------------------------
def apply_sensor_skew(x: np.ndarray, client_id: int, num_clients: int, seed: int = 42) -> np.ndarray:
    """Feature skew for (N, T, C) accelerometer windows: every client has a differently
    calibrated phone. Each axis gets a client-specific gain (0.9x ... 1.1x) and offset, plus
    measurement noise that grows from client 0 (clean) to the last client."""
    rng = np.random.default_rng(seed + client_id)
    t = client_id / max(num_clients - 1, 1)  # 0 ... 1
    c = x.shape[-1]
    scale = x.reshape(-1, c).std(axis=0) + 1e-8 if len(x) else np.ones(c)
    gain = 1.0 + rng.uniform(-0.1, 0.1, size=c) * t
    offset = rng.normal(0.0, 0.2, size=c) * scale * t
    noise = rng.normal(0.0, 0.1 * t, size=x.shape) * scale
    return (x * gain + offset + noise).astype(x.dtype)


def apply_label_noise(labels: np.ndarray, noise_rate: float, num_classes: int = len(CLASSES),
                      seed: int = 42) -> np.ndarray:
    """Flip a fraction `noise_rate` of labels to a different, random class: a client with an
    unreliable annotator (or a data poisoner)."""
    if noise_rate <= 0:
        return labels
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    flip = rng.random(len(labels)) < noise_rate
    offsets = rng.integers(1, num_classes, size=flip.sum())
    noisy[flip] = (labels[flip] + offsets) % num_classes
    return noisy
