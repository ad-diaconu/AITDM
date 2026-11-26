"""Tutorial experiment (E0): print what the few-minute demo run produced.

    python experiments/tutorial.py                 # after: bash Fep/submit.sh tutorial
    python experiments/tutorial.py --out outputs/tutorial

Reads outputs/tutorial/central/fcn/fold0/seed1/results.json           (centralised.py)
      outputs/tutorial/fl/s2_K5/fcn-{fedavg,fedbn}/fold0/seed1/results.json (run_seeds.py)
written by Fep/jobs/tutorial.sbatch (walk-through: docs/tutorial.md).

One fold, one seed, a few epochs / rounds: it shows the pipeline and where every number comes
from. It is NOT a result: real comparisons are 5 folds x 3 seeds, paired, with Holm
(.claude/experiment-review.md). The Stage 1 numbers are printed next to it for orientation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Stage 1, test macro-F1 mean over 15 runs (docs/experiment_setup_federated.md, S2 K = 5, FCN)
STAGE1 = {"centralised": 0.539, "FedAvg": 0.533, "FedBN (personalised)": 0.538, "FedBN (global BN)": 0.528}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def row(name: str, r: dict, part: dict, best: int, clients: dict | None) -> list[str]:
    pc = part["test"]["per_class"]
    worst = min(c["test"]["f1_macro"] for c in clients.values()) if clients else None
    return [name, str(best), f"{part['val']['f1_macro']:.3f}", f"{part['test']['f1_macro']:.3f}",
            f"{part['test']['balanced_accuracy']:.3f}", "-" if worst is None else f"{worst:.3f}",
            f"{pc['LyingDown']['f1']:.3f}", f"{pc['Stairs']['f1']:.3f}", f"{r['train_seconds']:.0f}",
            f"{STAGE1.get(name, float('nan')):.3f}"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "outputs/tutorial")
    args = p.parse_args()

    central = load(args.out / "central/fcn/fold0/seed1/results.json")
    fl = {s: load(args.out / f"fl/s2_K5/fcn-{s}/fold0/seed1/results.json") for s in ("fedavg", "fedbn")}
    if central is None and not any(fl.values()):
        raise SystemExit(f"nothing in {args.out}: run `bash Fep/submit.sh tutorial` first (docs/tutorial.md)")

    rows = []
    if central:
        rows.append(row("centralised", central, central, central["best_epoch"], None))
    if fl["fedavg"]:
        r = fl["fedavg"]
        rows.append(row("FedAvg", r, r, r["best_round"], r["per_client"]))
    if fl["fedbn"]:
        r = fl["fedbn"]
        rows.append(row("FedBN (personalised)", r, r, r["best_round"], r["per_client"]))
        g = r["global_bn"]
        rows.append(row("FedBN (global BN)", r, g, g["best_round"], g["per_client"]))

    head = ["method", "best ep/rnd", "val F1", "test F1", "test bAcc", "worst client", "LyingDown",
            "Stairs", "seconds", "Stage 1 (15 runs)"]
    widths = [max(len(h), *(len(x[i]) for x in rows)) for i, h in enumerate(head)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    print("\nTutorial E0: FCN on WISDM, fold 0, seed 1; federated = S2 (user-Dirichlet 0.3), K = 5 clients")
    print("macro-F1 unless noted; 'best ep/rnd' was chosen on the VALIDATION users, test is only reported\n")
    print(line(head))
    print(line(["-" * w for w in widths]))
    for x in rows:
        print(line(x))

    for s, r in fl.items():
        if r:
            f1 = {k: c["test"]["f1_macro"] for k, c in sorted(r["per_client"].items())}
            users = {k: c["test_users"] for k, c in sorted(r["per_client"].items())}
            print(f"\n{s} per client (test macro-F1, test users): "
                  + "  ".join(f"c{k} {v:.2f} ({users[k]})" for k, v in f1.items()))
    print()
    for s, r in fl.items():
        if r:
            curve = [h["val_f1"] for h in r["history"]]
            print(f"{s} validation macro-F1 per round: " + " ".join(f"{v:.2f}" for v in curve)
                  + f"   (best round {r['best_round']})")

    print("\nOne fold and one seed with a short budget: differences here are noise, not findings.")
    print("Stage 1 column = mean of 5 folds x 3 seeds with the full budget (30 epochs / 40 rounds).")
    print(f"Files: {args.out} (git-ignored; delete it when done)")


if __name__ == "__main__":
    main()
