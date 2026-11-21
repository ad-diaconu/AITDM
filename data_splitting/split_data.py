"""Federated partitions of the WISDM windows (docs/experiment_setup_federated.md, section 4).

For every fold, split strategy and number of clients K, assigns the fold's train / val / test
users (or windows, for the window-level splits) to K clients and writes

    <out>/<split>_K<K>/fold<f>/client_<i>.npz   x_/y_/u_ train, val, test (raw m/s^2, float32)
    <out>/<split>_K<K>/fold<f>/test.npz          union of the clients' test windows
    <out>/<split>_K<K>/fold<f>/meta.json         normalisation, sizes, users, JSD, label shares
    <out>/<split>_K<K>/label_distribution.png    per-client label shares of every fold

Inputs stay un-normalised (they compress 3x better); the clients z-score them with the fold's
pooled training mean/std from meta.json, i.e. the values one federated aggregation of
(n, sum x, sum x^2) per channel would give (identical to the centralised baseline).

    python wisdm_baseline/make_fl_partitions.py --splits s0 s1 s2 --clients 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pack_users(users: np.ndarray, sizes: dict, k: int, rng) -> dict:
    """Balanced packing (as partitioning.group_split): shuffle, then largest first onto the
    client with the fewest windows."""
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
    for c in range(k):
        n = (c_tr == c).sum()
        if n < min_frac * len(tr) or len(np.unique(users[tr][c_tr == c])) < 3:
            return False
        if not (c_va == c).any() or not (c_te == c).any():
            return False
    return True


def plot(path: Path, shares: list, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folds = len(shares)
    fig, axes = plt.subplots(1, folds, figsize=(3.2 * folds, 3), sharey=True)
    colors = plt.cm.tab10(np.arange(len(CLASSES)))
    for f, ax in enumerate(np.atleast_1d(axes)):
        bottom = np.zeros(len(shares[f]))
        for c in range(len(CLASSES)):
            vals = np.array([s[c] for s in shares[f]])
            ax.bar(range(len(vals)), vals, bottom=bottom, color=colors[c], label=CLASSES[c])
            bottom += vals
        ax.set_title(f"fold {f}")
        ax.set_xlabel("client")
        ax.set_xticks(range(len(shares[f])))
    np.atleast_1d(axes)[0].set_ylabel("share of training windows")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.6, 1))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--windows", type=Path, default=Path("data/wisdm/windows.npz"))
    p.add_argument("--out", type=Path, default=Path("data/partitions/wisdm"))
    p.add_argument("--splits", nargs="+", default=["s0", "s1", "s2"], choices=list(SPLITS))
    p.add_argument("--clients", type=int, nargs="+", default=[5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-frac", type=float, default=0.05)
    args = p.parse_args()

    data = dict(np.load(args.windows))
    x, y, users = data["x"], data["y"], data["user"]
    n_folds = sum(1 for key in data if key.startswith("train"))
    for split in args.splits:
        for k in args.clients:
            root = args.out / f"{split}_K{k}"
            all_shares = []
            for f in range(n_folds):
                tr, va, te = data[f"train{f}"], data[f"val{f}"], data[f"test{f}"]
                for attempt in range(500):
                    c_tr, c_va, c_te = assign(split, k, tr, va, te, y, users, args.seed, attempt)
                    if valid(c_tr, c_va, c_te, tr, users, k, args.min_frac):
                        break
                else:
                    raise SystemExit(f"{split} K={k} fold {f}: no valid assignment in 500 draws")
                # pooled training statistics, exactly as wisdm_baseline/train.py (torch .std(): ddof=1)
                flat = x[tr].reshape(-1, 3).astype(np.float64)
                mean, std = flat.mean(0), flat.std(0, ddof=1)
                fold_dir = root / f"fold{f}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                g = np.bincount(y[tr], minlength=len(CLASSES)) / len(tr)
                clients, shares = [], []
                for c in range(k):
                    parts = {"train": tr[c_tr == c], "val": va[c_va == c], "test": te[c_te == c]}
                    arrays = {}
                    for s, idx in parts.items():
                        arrays.update({f"x_{s}": x[idx], f"y_{s}": y[idx], f"u_{s}": users[idx]})
                    np.savez_compressed(fold_dir / f"client_{c}.npz", **arrays)
                    share = np.bincount(y[parts["train"]], minlength=len(CLASSES)) / len(parts["train"])
                    shares.append(share.tolist())
                    clients.append({
                        "client": c, "jsd_to_pooled": jsd(share, g), "train_label_share": share.tolist(),
                        **{f"{s}_windows": int(len(idx)) for s, idx in parts.items()},
                        **{f"{s}_users": sorted(int(u) for u in np.unique(users[idx])) for s, idx in parts.items()},
                        **{f"{s}_per_class": np.bincount(y[idx], minlength=len(CLASSES)).tolist()
                           for s, idx in parts.items()}})
                all_shares.append(shares)
                np.savez_compressed(fold_dir / "test.npz", x_test=x[te], y_test=y[te], u_test=users[te])
                # sanity: every window of the fold is on exactly one client, users disjoint for user splits
                for s, idx, cl in (("train", tr, c_tr), ("val", va, c_va), ("test", te, c_te)):
                    assert sum(c[f"{s}_windows"] for c in clients) == len(idx)
                    if split != "s0" and not (split == "s3" and s == "train"):
                        owners = {int(u): set(cl[users[idx] == u]) for u in np.unique(users[idx])}
                        assert all(len(o) == 1 for o in owners.values()), f"user on two clients ({s})"
                meta = {"split": split, "description": SPLITS[split], "num_clients": k, "fold": f,
                        "num_classes": len(CLASSES), "classes": CLASSES, "kind": "timeseries",
                        "sample_shape": list(x.shape[1:]), "seed": args.seed, "attempt": attempt,
                        "alpha": ALPHA if split in ("s2", "s3") else None,
                        "normalisation": {"mean": mean.tolist(), "std": std.tolist(),
                                          "source": "pooled training windows of the fold (federated sums)"},
                        "jsd_mean": float(np.mean([c["jsd_to_pooled"] for c in clients])),
                        "jsd_max": float(np.max([c["jsd_to_pooled"] for c in clients])),
                        "clients": clients}
                (fold_dir / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"{split}_K{k} fold{f}: attempt {attempt}, train windows per client "
                      f"{[c['train_windows'] for c in clients]}, users "
                      f"{[len(c['train_users']) for c in clients]}, JSD mean {meta['jsd_mean']:.3f}")
            plot(root / "label_distribution.png", all_shares, f"{split.upper()} ({SPLITS[split]}), K={k}")


if __name__ == "__main__":
    main()
