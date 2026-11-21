"""Permutation importance for a trained WISDM model, adapted from the tutorial's
explainability/feature_importance.py to (N, T, 3) accelerometer windows (docs/05_explainability.md).

Idea: shuffle one part of the input between windows, which breaks its link with the label,
and measure how much macro-F1 (and accuracy) drops. Big drop = the model relies on it.

* --group-by axis      the x, y or z axis of the whole window (which axis matters?)
* --group-by segment   one time segment of all axes, e.g. second 3-4 of 10 (where in the window?)
* --group-by axis-segment  both: a (segment x axis) heat map

    python explainability/feature_importance.py --run outputs/wisdm_fl/s2_K5/fcn-fedavg/fold0/seed1
    python explainability/feature_importance.py --run <run> --group-by segment --segments 10
    python explainability/feature_importance.py --run <run> --untrained    # sanity check: ~0 everywhere

Output: importance_<data>_<group-by>.png and .json in the run folder (or --out).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_analysis import load_run  # noqa: E402

from aitdm_fl.task import build_model, confusion, load_split, predict, scores  # noqa: E402

AXES = "xyz"


def groups_of(group_by: str, t: int, segments: int) -> dict[str, tuple[slice, list[int]]]:
    """name -> (time slice, axes) of the input that is shuffled together."""
    edges = np.linspace(0, t, segments + 1).astype(int)
    segs = {f"{edges[i] / 20:.0f}-{edges[i + 1] / 20:.0f}s": slice(edges[i], edges[i + 1]) for i in range(segments)}
    if group_by == "axis":
        return {a: (slice(0, t), [i]) for i, a in enumerate(AXES)}
    if group_by == "segment":
        return {name: (s, [0, 1, 2]) for name, s in segs.items()}
    return {f"{name} {a}": (s, [i]) for name, s in segs.items() for i, a in enumerate(AXES)}


def evaluate(model, x, y) -> tuple[float, float]:
    _, pred = predict(model, x, torch.device("cpu"))
    s = scores(confusion(y, pred))
    return s["f1_macro"], s["accuracy"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--partitions", type=Path, help="fold folder (default: the run's own)")
    p.add_argument("--data", default="val", help="'val', 'client_<i>' or 'test' (final report only)")
    p.add_argument("--group-by", choices=["axis", "segment", "axis-segment"], default="axis")
    p.add_argument("--segments", type=int, default=10, help="time segments per window")
    p.add_argument("--repeats", type=int, default=5, help="shuffles per group (averaged)")
    p.add_argument("--untrained", action="store_true", help="sanity check: a randomly initialised model")
    p.add_argument("--max-windows", type=int, default=5000, help="subsample for speed (0 = all)")
    p.add_argument("--out", type=Path, help="output PNG (default: in the run folder)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, pdir, result = load_run(args.run, args.partitions)
    if args.untrained:
        torch.manual_seed(args.seed)
        model = build_model(result["model"]).eval()
    x, y, _, _, tag = load_split(str(pdir), args.data)
    rng = np.random.default_rng(args.seed)
    if args.max_windows and len(y) > args.max_windows:
        keep = np.sort(rng.choice(len(y), args.max_windows, replace=False))
        x, y = x[keep], y[keep]
    base_f1, base_acc = evaluate(model, x, y)
    print(f"{tag}: {len(y)} windows, macro-F1 {base_f1:.3f}, accuracy {base_acc:.3f}")

    groups = groups_of(args.group_by, x.shape[1], args.segments)
    imp, spread, imp_acc = {}, {}, {}
    for name, (tsl, axes) in groups.items():
        drops, drops_acc = [], []
        for _ in range(args.repeats):
            xs = x.copy()
            perm = rng.permutation(len(x))
            for a in axes:
                xs[:, tsl, a] = x[perm][:, tsl, a]
            f1, acc = evaluate(model, xs, y)
            drops.append(base_f1 - f1)
            drops_acc.append(base_acc - acc)
        imp[name], spread[name], imp_acc[name] = float(np.mean(drops)), float(np.std(drops)), float(np.mean(drops_acc))
    ranking = sorted(imp, key=imp.get, reverse=True)
    print(f"\nmost important ({args.group_by}): macro-F1 drop when shuffled (± std over {args.repeats})")
    for g in ranking[:10]:
        print(f"  {g:<14} {imp[g]:+.3f} ± {spread[g]:.3f}   (accuracy {imp_acc[g]:+.3f})")

    suffix = "_untrained" if args.untrained else ""
    out = args.out or args.run / f"importance_{tag}_{args.group_by}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"Permutation importance per {args.group_by} | {tag} | macro-F1 {base_f1:.3f}"
    if args.untrained:
        title = "UNTRAINED model (sanity check) | " + title
    if args.group_by == "axis-segment":
        names = list(dict.fromkeys(n.split(" ")[0] for n in groups))
        grid = np.array([[imp[f"{s} {a}"] for s in names] for a in AXES])
        fig, ax = plt.subplots(figsize=(10, 2.8))
        im = ax.imshow(grid, cmap="Reds", aspect="auto", vmin=min(0.0, grid.min()))
        ax.set_yticks(range(3), list(AXES))
        ax.set_xticks(range(len(names)), names, rotation=45, fontsize=8)
        plt.colorbar(im, ax=ax, label="macro-F1 drop")
    else:
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(groups) + 1)))
        shown = ranking[::-1] if args.group_by == "axis" else list(groups)[::-1]
        ax.barh(shown, [imp[g] for g in shown], xerr=[spread[g] for g in shown], color="tab:red")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("macro-F1 drop when shuffled (higher = more important)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    out.with_suffix(".json").write_text(json.dumps(
        {"data": tag, "group_by": args.group_by, "repeats": args.repeats, "windows": int(len(y)),
         "baseline_f1_macro": base_f1, "baseline_accuracy": base_acc, "untrained": args.untrained,
         "importance_f1": {g: imp[g] for g in ranking}, "importance_f1_std": {g: spread[g] for g in ranking},
         "importance_accuracy": {g: imp_acc[g] for g in ranking}}, indent=1))
    print(f"\nSaved {out} (+ .json)")


if __name__ == "__main__":
    main()
