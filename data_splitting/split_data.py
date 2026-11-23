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

    python data_splitting/split_data.py --splits s0 s1 s2 --clients 5

Stage 2 (opt-in, written to <split>_K<K>-<tag>/ so the plain splits stay untouched):

    # client 4 has an unreliable annotator: 50 % of its training labels are wrong
    python data_splitting/split_data.py --splits s1 --clients 5 --label-noise 0,0,0,0,0.5
    # every client a differently calibrated phone (gain / offset / noise per axis)
    python data_splitting/split_data.py --splits s1 --clients 5 --sensor-skew
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partitioning import (  # noqa: E402
    ALPHA,
    CLASSES,
    SPLITS,
    apply_label_noise,
    apply_sensor_skew,
    assign,
    jsd,
    valid,
)


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


def parse_label_noise(text: str, k: int) -> list[float]:
    if not text:
        return [0.0] * k
    rates = [float(r) for r in text.split(",")]
    if len(rates) != k or not all(0 <= r <= 1 for r in rates):
        raise SystemExit(f"--label-noise needs {k} comma-separated rates in [0, 1], got {text!r}")
    return rates


def variant_tag(args, k: int) -> str:
    parts = []
    if any(parse_label_noise(args.label_noise, k)):
        parts.append("noise" + "-".join(f"{r:g}" for r in parse_label_noise(args.label_noise, k)))
    if args.sensor_skew:
        parts.append("skew")
    return args.tag or "_".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--windows", type=Path, default=Path("data/wisdm/windows.npz"))
    p.add_argument("--out", type=Path, default=Path("data/partitions/wisdm"))
    p.add_argument("--splits", nargs="+", default=["s0", "s1", "s2"], choices=list(SPLITS))
    p.add_argument("--clients", type=int, nargs="+", default=[5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-frac", type=float, default=0.05)
    bias = p.add_argument_group("extra real-world bias (opt-in; output <split>_K<K>-<tag>)")
    bias.add_argument("--label-noise", default="",
                      help="per-client fraction of wrong TRAINING labels, e.g. 0,0,0,0,0.5")
    bias.add_argument("--sensor-skew", action="store_true",
                      help="per-client gain / offset / noise on the x, y, z axes (all of its windows)")
    bias.add_argument("--tag", default="", help="name of the variant (default: from the options)")
    args = p.parse_args()

    data = dict(np.load(args.windows))
    x, y, users = data["x"], data["y"], data["user"]
    n_folds = sum(1 for key in data if key.startswith("train"))
    for split in args.splits:
        for k in args.clients:
            noise = parse_label_noise(args.label_noise, k)
            tag = variant_tag(args, k)
            root = args.out / (f"{split}_K{k}" + (f"-{tag}" if tag else ""))
            all_shares = []
            for f in range(n_folds):
                tr, va, te = data[f"train{f}"], data[f"val{f}"], data[f"test{f}"]
                for attempt in range(500):
                    c_tr, c_va, c_te = assign(split, k, tr, va, te, y, users, args.seed, attempt)
                    if valid(c_tr, c_va, c_te, tr, users, k, args.min_frac):
                        break
                else:
                    raise SystemExit(f"{split} K={k} fold {f}: no valid assignment in 500 draws")
                fold_dir = root / f"fold{f}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                parts = [{"train": tr[c_tr == c], "val": va[c_va == c], "test": te[c_te == c]}
                         for c in range(k)]
                # per-client arrays (+ optional corruptions; clean labels stay in y for the stats)
                client_arrays = []
                for c in range(k):
                    arrays = {}
                    for s, idx in parts[c].items():
                        xs, ys = x[idx], y[idx]
                        if args.sensor_skew:
                            xs = apply_sensor_skew(xs, c, k, seed=args.seed + 1000 * f)
                        if s == "train" and noise[c] > 0:
                            ys = apply_label_noise(ys, noise[c], seed=args.seed + 1000 * f + c)
                        arrays.update({f"x_{s}": xs, f"y_{s}": ys, f"u_{s}": users[idx]})
                    client_arrays.append(arrays)
                # pooled training statistics, exactly as experiments/centralised.py (torch .std(): ddof=1)
                if args.sensor_skew:
                    flat = np.concatenate([a["x_train"] for a in client_arrays]).reshape(-1, 3).astype(np.float64)
                else:
                    flat = x[tr].reshape(-1, 3).astype(np.float64)
                mean, std = flat.mean(0), flat.std(0, ddof=1)
                g = np.bincount(y[tr], minlength=len(CLASSES)) / len(tr)
                clients, shares = [], []
                for c in range(k):
                    np.savez_compressed(fold_dir / f"client_{c}.npz", **client_arrays[c])
                    share = np.bincount(y[parts[c]["train"]], minlength=len(CLASSES)) / len(parts[c]["train"])
                    shares.append(share.tolist())
                    clients.append({
                        "client": c, "jsd_to_pooled": jsd(share, g), "train_label_share": share.tolist(),
                        **{f"{s}_windows": int(len(idx)) for s, idx in parts[c].items()},
                        **{f"{s}_users": sorted(int(u) for u in np.unique(users[idx])) for s, idx in parts[c].items()},
                        **{f"{s}_per_class": np.bincount(y[idx], minlength=len(CLASSES)).tolist()
                           for s, idx in parts[c].items()}})
                    if noise[c] > 0:
                        clients[-1]["train_label_noise"] = noise[c]
                        clients[-1]["train_labels_flipped"] = int(
                            (client_arrays[c]["y_train"] != y[parts[c]["train"]]).sum())
                all_shares.append(shares)
                if args.sensor_skew:  # the test windows as the clients' phones recorded them
                    x_test = np.empty_like(x[te])
                    for c in range(k):
                        x_test[c_te == c] = client_arrays[c]["x_test"]
                else:
                    x_test = x[te]
                np.savez_compressed(fold_dir / "test.npz", x_test=x_test, y_test=y[te], u_test=users[te])
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
                if tag:
                    meta["variant"] = {"tag": tag, "label_noise": noise, "sensor_skew": args.sensor_skew}
                (fold_dir / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"{root.name} fold{f}: attempt {attempt}, train windows per client "
                      f"{[c['train_windows'] for c in clients]}, users "
                      f"{[len(c['train_users']) for c in clients]}, JSD mean {meta['jsd_mean']:.3f}")
            plot(root / "label_distribution.png", all_shares,
                 f"{split.upper()} ({SPLITS[split]}), K={k}" + (f", {tag}" if tag else ""))


if __name__ == "__main__":
    main()
