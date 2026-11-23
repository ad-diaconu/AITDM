"""Where does a trained WISDM model fail? Per-class metrics, confusion matrix, per-client and
per-user accuracy and the hardest windows (highest loss), adapted from the tutorial's
explainability/error_analysis.py to accelerometer windows (docs/05_explainability.md).

    # a federated run: model and partitions are found from its results.json
    python explainability/error_analysis.py --run outputs/wisdm_fl/s2_K5/fcn-fedavg/fold0/seed1

    # a centralised run (needs a partition folder of the same fold for the per-client view)
    python explainability/error_analysis.py --run outputs/wisdm/fcn/fold0/seed1 \
        --partitions data/partitions/wisdm/s1_K5/fold0

    # the hardest TRAINING windows of client 1 (candidates for wrong labels / artefacts)
    python explainability/error_analysis.py --run <run> --data client_1 --split train

Diagnose on --data val (default: all clients' validation users). --data test only for the final
report. Output in the run folder (or --out-dir): errors_<data>.png (confusion matrix + recall),
hardest_<data>.png (x/y/z traces of the hardest windows), errors_<data>.json (all numbers).
Note: for FedBN runs final_model.pt holds the averaged BatchNorm layers (the "global BN" model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fl_app"))
from aitdm_fl.task import CLASSES, build_model, confusion, load_split, load_state, predict, scores  # noqa: E402


def load_run(run: Path, partitions: Path | None) -> tuple[torch.nn.Module, Path, dict]:
    """(model with the run's weights, partition folder, results.json) of a run folder."""
    result = json.loads((run / "results.json").read_text())
    weights = run / "final_model.pt" if (run / "final_model.pt").exists() else run / "model.pt"
    if not weights.exists():
        raise SystemExit(f"no final_model.pt / model.pt in {run} (model weights are not in git: "
                         "run this where the run was trained, e.g. on fep)")
    if partitions is None:
        if "split" not in result:
            raise SystemExit("centralised run: pass --partitions data/partitions/wisdm/<split>_K<K>/"
                             f"fold{result['fold']}")
        partitions = REPO / "data/partitions/wisdm" / f"{result['split']}_K{result['num_clients']}" / f"fold{result['fold']}"
    model = build_model(result["model"])
    load_state(model, torch.load(weights, map_location="cpu"))
    model.eval()
    return model, partitions, result


def accuracy_by(groups: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    return {int(g): {"windows": int((groups == g).sum()), "accuracy": float(np.mean(pred[groups == g] == y[groups == g])),
                     "f1_macro": scores(confusion(y[groups == g], pred[groups == g]))["f1_macro"]}
            for g in np.unique(groups)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run folder with results.json + weights")
    p.add_argument("--partitions", type=Path, help="fold folder (default: the run's own)")
    p.add_argument("--data", default="val", help="'val' (all clients' validation users), 'client_<i>' or 'test'")
    p.add_argument("--split", choices=["train", "val", "test"], default="val", help="for client_<i>")
    p.add_argument("--num-windows", type=int, default=12, help="hardest windows to plot")
    p.add_argument("--out-dir", type=Path, help="default: the run folder")
    args = p.parse_args()
    out_dir = args.out_dir or args.run

    model, pdir, _ = load_run(args.run, args.partitions)
    x, y, users, owner, tag = load_split(str(pdir), args.data, args.split)
    logits, pred = predict(model, x, torch.device("cpu"))
    losses = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y), reduction="none").numpy()
    probs = torch.from_numpy(logits).softmax(1).numpy()
    s = scores(confusion(y, pred))
    cm = np.array(s["confusion_matrix"])

    print(f"\n{tag}: {len(y)} windows, {len(np.unique(users))} users, accuracy {s['accuracy']:.3f}, "
          f"balanced accuracy {s['balanced_accuracy']:.3f}, macro-F1 {s['f1_macro']:.3f}\n")
    print(f"{'class':<10} support  recall  precision     f1  most confused with")
    for c, name in enumerate(CLASSES):
        pc = s["per_class"][name]
        row = cm[c].copy()
        row[c] = 0
        confused = f"{CLASSES[row.argmax()]} ({row.max()})" if row.max() else "-"
        print(f"{name:<10} {pc['support']:>7}  {pc['recall']:>6.2f}  {pc['precision']:>9.2f}  {pc['f1']:>5.2f}  {confused}")

    per_client = accuracy_by(owner, y, pred) if owner is not None else {}
    if per_client:
        print(f"\n{'client':<8} windows  accuracy  macro-F1")
        for cid, r in per_client.items():
            print(f"{cid:<8} {r['windows']:>7}  {r['accuracy']:>8.3f}  {r['f1_macro']:>8.3f}")
    per_user = accuracy_by(users, y, pred)
    worst = sorted(per_user, key=lambda u: per_user[u]["accuracy"])[:5]
    print("\nworst users: " + ", ".join(f"{u} ({per_user[u]['accuracy']:.2f}, {per_user[u]['windows']} windows)"
                                        for u in worst))

    hardest = np.argsort(-losses)[:args.num_windows]
    print(f"\nhardest {len(hardest)} windows (index: true -> predicted, loss):")
    for i in hardest[:10]:
        print(f"  #{i:<6} user {users[i]:<4} {CLASSES[y[i]]:>9} -> {CLASSES[pred[i]]:<9} loss {losses[i]:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    axes[0].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                             color="white" if norm[i, j] > 0.5 else "black")
    axes[0].set_xticks(range(len(CLASSES)), CLASSES, rotation=30, fontsize=8)
    axes[0].set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    axes[0].set_xlabel("predicted")
    axes[0].set_title("Confusion matrix (colour = fraction of the true class)")
    recall = [s["per_class"][c]["recall"] for c in CLASSES]
    bars = axes[1].barh(CLASSES, recall, color=["tab:red" if r < 0.3 else "tab:blue" for r in recall])
    axes[1].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Recall per class (red < 0.3)")
    fig.suptitle(f"{args.run} | {tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"errors_{tag}.png", dpi=120)
    plt.close(fig)

    cols = 3
    rows = int(np.ceil(len(hardest) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.2 * rows), squeeze=False, sharey=True)
    t = np.arange(x.shape[1]) / 20  # 20 Hz -> seconds
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, hardest):
        ax.axis("on")
        for ch, (name, col) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
            ax.plot(t, x[i, :, ch], color=col, lw=0.8, label=name)
        ax.set_title(f"#{i} user {users[i]}: {CLASSES[y[i]]} -> {CLASSES[pred[i]]} ({probs[i, pred[i]]:.0%})",
                     fontsize=8, color="red")
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Hardest windows (highest loss), z-scored x / y / z over {t[-1] + 0.05:.0f} s | {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"hardest_{tag}.png", dpi=110)
    plt.close(fig)

    report = {"data": tag, "partitions": str(pdir), **{k: s[k] for k in ("accuracy", "balanced_accuracy", "f1_macro")},
              "per_class": s["per_class"], "confusion_matrix": s["confusion_matrix"], "per_client": per_client,
              "per_user": per_user,
              "hardest": [{"index": int(i), "user": int(users[i]), "true": CLASSES[y[i]], "pred": CLASSES[pred[i]],
                           "loss": float(losses[i])} for i in hardest]}
    (out_dir / f"errors_{tag}.json").write_text(json.dumps(report, indent=1))
    print(f"\nSaved errors_{tag}.png, hardest_{tag}.png, errors_{tag}.json in {out_dir}")


if __name__ == "__main__":
    main()
