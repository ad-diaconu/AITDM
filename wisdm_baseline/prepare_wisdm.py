"""WISDM Actitracker v2.0 -> cleaned, resampled 10 s windows + 5 subject-wise folds.

Implements sections 3-5 of docs/experiment_setup_centralised_baseline.md.
Runs locally (needs pandas + scikit-learn); the cluster only needs the output .npz.

    python wisdm_baseline/prepare_wisdm.py \
        --input data/raw/wisdm/WISDM_at_v2.0_raw.txt --out data/wisdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown"]
RATE_MS = 50  # 20 Hz


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    log = {}
    df = pd.read_csv(
        path, header=None, names=["user", "activity", "ts", "x", "y", "z"], dtype=str
    )
    log["rows_read"] = len(df)
    df["z"] = df["z"].str.rstrip(";")
    df = df.dropna()
    log["C2_empty_rows"] = log["rows_read"] - len(df)
    for col in ["ts", "x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n = len(df)
    df = df.dropna()
    log["unparseable_rows"] = n - len(df)
    df["user"] = df["user"].astype(int)
    n = len(df)
    df = df.drop_duplicates()
    log["C3_duplicate_rows"] = n - len(df)
    corrupt = (
        (df[["x", "y", "z"]].abs().max(axis=1) > 40) | (df.ts < 1e12) | (df.ts > 1.5e12)
    )
    log["C4_corrupt_rows"] = int(corrupt.sum())
    df = df[~corrupt].sort_values(["user", "ts"], kind="stable")
    assert not df.duplicated(
        ["user", "ts"]
    ).any(), "duplicate (user, timestamp) after cleaning"
    unknown = set(df.activity) - set(CLASSES)
    assert not unknown, f"unexpected labels {unknown}"
    log["rows_clean"] = len(df)
    log["users_clean"] = int(df.user.nunique())
    return df, log


def make_windows(
    df: pd.DataFrame, window: int, train_stride: int, eval_stride: int, max_gap_ms: int
):
    """Segment (user / activity change or gap > max_gap_ms), resample to 20 Hz, window.
    Windows are cut every `train_stride`; `is_eval` marks those on the `eval_stride` grid.
    """
    assert eval_stride % train_stride == 0
    dt = df.groupby("user").ts.diff()
    new_seg = (
        (df.user != df.user.shift())
        | (df.activity != df.activity.shift())
        | (dt > max_gap_ms)
        | dt.isna()
    )
    xs, ys, users, is_eval = [], [], [], []
    for _, seg in df.groupby(new_seg.cumsum()):
        t = seg.ts.to_numpy()
        if t[-1] - t[0] < (window - 1) * RATE_MS:
            continue
        grid = np.arange(t[0], t[-1] + 1, RATE_MS)
        sig = np.stack(
            [np.interp(grid, t, seg[c].to_numpy()) for c in "xyz"], 1
        ).astype(np.float32)
        label, user = CLASSES.index(seg.activity.iloc[0]), int(seg.user.iloc[0])
        for start in range(0, len(sig) - window + 1, train_stride):
            xs.append(sig[start : start + window])
            ys.append(label)
            users.append(user)
            is_eval.append(start % eval_stride == 0)
    return (
        np.stack(xs),
        np.array(ys, np.int64),
        np.array(users, np.int64),
        np.array(is_eval),
    )


def make_folds(y, users, is_eval, n_folds: int, seed: int):
    folds = []
    outer = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    for pool, test in outer.split(y, y, users):
        inner = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        tr_i, va_i = next(inner.split(pool, y[pool], users[pool]))
        train, val = pool[tr_i], pool[va_i]
        val, test = val[is_eval[val]], test[is_eval[test]]
        sets = [set(users[i]) for i in (train, val, test)]
        assert not (
            sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
        ), "user leak"
        for name, idx in (("train", train), ("val", val), ("test", test)):
            missing = set(range(len(CLASSES))) - set(y[idx])
            assert not missing, f"fold {len(folds)} {name} misses classes {missing}"
        folds.append((train, val, test))
    test_users = np.concatenate([np.unique(users[f[2]]) for f in folds])
    assert len(test_users) == len(set(test_users)), "a user is a test user twice"
    return folds


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/wisdm"))
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--train-stride", type=int, default=100)
    p.add_argument("--eval-stride", type=int, default=200)
    p.add_argument("--max-gap-ms", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df, log = load_and_clean(args.input)
    print("cleaning:", log)
    x, y, users, is_eval = make_windows(
        df, args.window, args.train_stride, args.eval_stride, args.max_gap_ms
    )
    assert x.dtype == np.float32 and np.isfinite(x).all()
    print(
        f"windows {x.shape} from {len(np.unique(users))} users, per class {np.bincount(y)}"
    )
    folds = make_folds(y, users, is_eval, args.folds, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    arrays = {"x": x, "y": y, "user": users, "is_eval": is_eval}
    meta = {
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "kind": "timeseries",
        "layout": "(N, T, C) = (N, window, xyz), m/s^2, not normalised",
        "params": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "cleaning": log,
        "n_windows": int(len(y)),
        "folds": [],
    }
    for k, (tr, va, te) in enumerate(folds):
        arrays.update({f"train{k}": tr, f"val{k}": va, f"test{k}": te})
        meta["folds"].append(
            {
                split: {
                    "windows": int(len(idx)),
                    "users": sorted(int(u) for u in np.unique(users[idx])),
                    "per_class": np.bincount(y[idx], minlength=len(CLASSES)).tolist(),
                }
                for split, idx in (("train", tr), ("val", va), ("test", te))
            }
        )
        print(
            f"fold {k}: "
            + ", ".join(
                f"{s} {meta['folds'][k][s]['windows']} windows / "
                f"{len(meta['folds'][k][s]['users'])} users"
                for s in ("train", "val", "test")
            )
        )
    np.savez_compressed(args.out / "windows.npz", **arrays)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out / 'windows.npz'} and meta.json")


if __name__ == "__main__":
    main()
