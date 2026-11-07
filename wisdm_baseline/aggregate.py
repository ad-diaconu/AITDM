"""Aggregate outputs/wisdm/<model>/fold<k>/seed<s>/results.json into
outputs/wisdm/summary.json and a self-contained outputs/wisdm/report.html
(the page the `wisdm-results` canvas shows; it also opens in any browser).

    python wisdm_baseline/aggregate.py --results outputs/wisdm --meta data/wisdm/meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

CLASSES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown"]
CURVES = ["train_loss", "val_loss", "test_loss", "train_acc", "val_f1", "test_f1"]


def mean_std(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def summarise_model(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda r: (r["fold"], r["seed"]))
    worst = [min(r["test_per_user_accuracy"].values()) for r in runs]
    median = [
        float(np.median(list(r["test_per_user_accuracy"].values()))) for r in runs
    ]
    below_half = [
        float(np.mean(np.array(list(r["test_per_user_accuracy"].values())) < 0.5))
        for r in runs
    ]
    out = {
        "params": runs[0]["params"],
        "metrics": {
            "val_f1": mean_std([r["val"]["f1_macro"] for r in runs]),
            "test_f1": mean_std([r["test"]["f1_macro"] for r in runs]),
            "test_bacc": mean_std([r["test"]["balanced_accuracy"] for r in runs]),
            "test_acc": mean_std([r["test"]["accuracy"] for r in runs]),
            "test_worst_user_acc": mean_std(worst),
            "test_median_user_acc": mean_std(median),
            "test_users_below_50pct": mean_std(below_half),
            "best_epoch": mean_std([r["best_epoch"] for r in runs]),
            "epochs_run": mean_std([r["epochs_run"] for r in runs]),
            "train_seconds": mean_std([r["train_seconds"] for r in runs]),
        },
        "per_class": {
            split: {
                c: {
                    m: mean_std([r[split]["per_class"][c][m] for r in runs])
                    for m in ("precision", "recall", "f1")
                }
                for c in CLASSES
            }
            for split in ("val", "test")
        },
        "confusion": {
            split: np.sum([r[split]["confusion_matrix"] for r in runs], axis=0).tolist()
            for split in ("val", "test")
        },
        "runs": [
            {
                "fold": r["fold"],
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "epochs_run": r["epochs_run"],
                "val_f1": r["val"]["f1_macro"],
                "test_f1": r["test"]["f1_macro"],
                "test_acc": r["test"]["accuracy"],
                "test_bacc": r["test"]["balanced_accuracy"],
            }
            for r in runs
        ],
        "user_accuracy": [
            a for r in runs for a in r["test_per_user_accuracy"].values()
        ],
    }
    max_ep = max(len(r["history"]) for r in runs)
    curves = {}
    for key in CURVES:
        mean, std, n = [], [], []
        for e in range(max_ep):
            vals = [r["history"][e][key] for r in runs if len(r["history"]) > e]
            mean.append(float(np.mean(vals)))
            std.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            n.append(len(vals))
        curves[key] = {"mean": mean, "std": std, "n": n}
    out["curves"] = curves
    out["best_epochs"] = [r["best_epoch"] for r in runs]
    return out


def majority_baseline(meta: dict) -> dict:
    accs, f1s = [], []
    for fold in meta["folds"]:
        counts = np.array(fold["test"]["per_class"], dtype=float)
        majority = int(np.argmax(np.array(fold["train"]["per_class"])))
        p = counts[majority] / counts.sum()
        accs.append(p)
        f1s.append((2 * p / (1 + p)) / len(counts))
    return {
        "class": CLASSES[majority],
        "test_acc": mean_std(accs),
        "test_f1": mean_std(f1s),
        "test_bacc": mean_std([1 / len(CLASSES)] * len(accs)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("outputs/wisdm"))
    p.add_argument("--meta", type=Path, default=Path("data/wisdm/meta.json"))
    p.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("report_template.html"),
    )
    args = p.parse_args()

    runs: dict[str, list] = {}
    for path in sorted(args.results.glob("*/fold*/seed*/results.json")):
        r = json.loads(path.read_text())
        runs.setdefault(r["model"], []).append(r)
    if not runs:
        raise SystemExit(f"no results.json under {args.results}")
    meta = json.loads(args.meta.read_text())

    summary = {
        "classes": CLASSES,
        "models": {m: summarise_model(rs) for m, rs in runs.items()},
        "majority": majority_baseline(meta),
        "data": {
            "cleaning": meta["cleaning"],
            "params": meta["params"],
            "n_windows": meta["n_windows"],
            "folds": [
                {
                    s: {
                        "windows": f[s]["windows"],
                        "users": len(f[s]["users"]),
                        "per_class": f[s]["per_class"],
                    }
                    for s in ("train", "val", "test")
                }
                for f in meta["folds"]
            ],
        },
        "config": next(iter(runs.values()))[0]["config"],
    }
    if {"fcn", "lstm"} <= runs.keys():
        a = {(r["fold"], r["seed"]): r["test"]["f1_macro"] for r in runs["fcn"]}
        b = {(r["fold"], r["seed"]): r["test"]["f1_macro"] for r in runs["lstm"]}
        keys = sorted(a.keys() & b.keys())
        diff = np.array([a[k] - b[k] for k in keys])
        summary["comparison"] = {
            "metric": "test macro-F1",
            "pairs": len(keys),
            "mean_diff_fcn_minus_lstm": float(diff.mean()),
            "fcn_better": int((diff > 0).sum()),
            "wilcoxon_p": float(wilcoxon(diff).pvalue),
        }

    (args.results / "summary.json").write_text(json.dumps(summary, indent=1))
    html = args.template.read_text().replace("/*__DATA__*/null", json.dumps(summary))
    (args.results / "report.html").write_text(html)
    for m, s in summary["models"].items():
        mt = s["metrics"]
        print(
            f"{m:5s} ({mt['test_f1']['n']} runs) val F1 {mt['val_f1']['mean']:.3f}±{mt['val_f1']['std']:.3f} | "
            f"test F1 {mt['test_f1']['mean']:.3f}±{mt['test_f1']['std']:.3f}, "
            f"bal.acc {mt['test_bacc']['mean']:.3f}±{mt['test_bacc']['std']:.3f}, "
            f"acc {mt['test_acc']['mean']:.3f}±{mt['test_acc']['std']:.3f}, "
            f"worst user {mt['test_worst_user_acc']['mean']:.3f}"
        )
    if "comparison" in summary:
        c = summary["comparison"]
        print(
            f"FCN - LSTM test F1: {c['mean_diff_fcn_minus_lstm']:+.3f}, FCN better in "
            f"{c['fcn_better']}/{c['pairs']}, Wilcoxon p = {c['wilcoxon_p']:.3f}"
        )
    print(f"wrote {args.results / 'summary.json'} and {args.results / 'report.html'}")


if __name__ == "__main__":
    main()
