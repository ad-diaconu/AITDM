"""Aggregate the federated runs and compare them with the centralised baseline.

    python wisdm_baseline/aggregate_fl.py

Reads  outputs/wisdm_fl/<split>_K<K>/<method>/fold<f>/seed<s>/results.json  (run_fl.py)
       outputs/wisdm/<model>/fold<f>/seed<s>/{results.json, model.pt}       (Stage 1)
       data/partitions/wisdm/<split>_K<K>/fold<f>/{meta.json, test.npz}
Writes outputs/wisdm_fl/summary_fl.json and the self-contained outputs/wisdm_fl/report.html.

The centralised models are re-evaluated on every client's test users, so per-client fairness
numbers exist for the centralised baseline too (same windows, same normalisation).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path[:0] = [str(REPO), str(REPO / "fl_app")]
from aitdm_fl.task import build_model, load_state, scores  # noqa: E402
from wisdm_baseline.train import CLASSES  # noqa: E402

SPLIT_NAMES = {"s0": "S0 IID", "s1": "S1 natural", "s2": "S2 user-Dirichlet α=0.3",
               "s3": "S3 window-Dirichlet α=0.3"}


def mean_std(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def paired(a: dict, b: dict) -> dict | None:
    """a, b: {(fold, seed): value}. Mean difference a - b, sign count, Wilcoxon p."""
    keys = sorted(a.keys() & b.keys())
    if len(keys) < 2:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    return {"pairs": len(keys), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
            "wins": int((d > 0).sum()), "p": p}


def holm(tests: list[dict]) -> None:
    """Holm-Bonferroni within one family, in place (adds p_holm)."""
    valid = [t for t in tests if t.get("p") is not None]
    order = sorted(range(len(valid)), key=lambda i: valid[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(valid) - rank) * valid[i]["p"]))
        valid[i]["p_holm"] = running


def per_client_from_users(cm_by_user: dict, owner: dict, k: int) -> list[float]:
    """Per-client macro-F1 from per-user confusion matrices and user -> client."""
    cms = [np.zeros((6, 6), int) for _ in range(k)]
    for u, cm in cm_by_user.items():
        cms[owner[u]] += cm
    return [scores(cm)["f1_macro"] for cm in cms]


class Central:
    """Stage 1 models, evaluated per test user of each fold (cached)."""

    def __init__(self, root: Path, partitions: Path):
        self.root, self.partitions = root, partitions
        self.cache = {}
        self.results = {}
        for path in sorted(root.glob("*/fold*/seed*/results.json")):
            r = json.loads(path.read_text())
            self.results[(r["model"], r["fold"], r["seed"])] = r

    def user_cms(self, model: str, fold: int, seed: int, pdir: Path) -> dict:
        key = (model, fold, seed)
        if key not in self.cache:
            meta = json.loads((pdir / "meta.json").read_text())["normalisation"]
            mean, std = np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)
            d = np.load(pdir / "test.npz")
            x = ((d["x_test"] - mean) / std).astype(np.float32)
            net = build_model(model)
            load_state(net, torch.load(self.root / model / f"fold{fold}" / f"seed{seed}" / "model.pt",
                                       map_location="cpu"))
            net.eval()
            with torch.no_grad():
                pred = torch.cat([net(torch.from_numpy(x[i:i + 1024])).argmax(1)
                                  for i in range(0, len(x), 1024)]).numpy()
            y, users = d["y_test"], d["u_test"]
            cms = {}
            for u in np.unique(users):
                m = users == u
                cm = np.zeros((6, 6), int)
                np.add.at(cm, (y[m], pred[m]), 1)
                cms[int(u)] = cm
            total = sum(cms.values())
            stage1 = np.array(self.results[key]["test"]["confusion_matrix"])
            self.cache[key] = {"cms": cms, "windows_differing": int(np.abs(total - stage1).sum() // 2),
                               "f1_diff": abs(scores(total)["f1_macro"] - scores(stage1)["f1_macro"])}
        return self.cache[key]


def summarise(runs: list[dict], central_by_key: dict, central: Central, pdirs: dict, k: int,
              test_key: str = "test", val_key: str = "val", sub: str | None = None) -> dict:
    """Summary of one method (runs over folds x seeds). `sub` = 'global_bn' for FedBN global-BN."""
    def part(r):
        return r[sub] if sub else r
    test_f1 = {(r["fold"], r["seed"]): part(r)["test"]["f1_macro"] for r in runs}
    worst, spread, below, median = [], [], [], []
    for r in runs:
        f1s = [c["test"]["f1_macro"] for c in part(r)["per_client"].values()]
        worst.append(min(f1s))
        spread.append(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0)
        acc = np.array(list(part(r)["test_per_user_accuracy"].values()))
        below.append(float((acc < 0.5).mean()))
        median.append(float(np.median(acc)))
    best = [part(r)["best_round"] if sub else r["best_round"] for r in runs]
    comm = [r["comm_mb_per_round_fedbn_net" if r["strategy"] == "fedbn" and not sub else "comm_mb_per_round"]
            for r in runs]
    out = {
        "runs": len(runs),
        "metrics": {
            "val_f1": mean_std([part(r)["val"]["f1_macro"] for r in runs]),
            "test_f1": mean_std(list(test_f1.values())),
            "test_bacc": mean_std([part(r)["test"]["balanced_accuracy"] for r in runs]),
            "test_acc": mean_std([part(r)["test"]["accuracy"] for r in runs]),
            "worst_client_f1": mean_std(worst), "client_f1_std": mean_std(spread),
            "median_user_acc": mean_std(median), "users_below_50": mean_std(below),
            "best_round": mean_std(best), "mb_to_best": mean_std([b * c for b, c in zip(best, comm)]),
            "train_seconds": mean_std([r["train_seconds"] for r in runs]),
        },
        "per_class": {c: mean_std([part(r)["test"]["per_class"][c]["f1"] for r in runs]) for c in CLASSES},
        "confusion": np.sum([part(r)["test"]["confusion_matrix"] for r in runs], axis=0).tolist(),
        "confusion_val": np.sum([part(r)["val"]["confusion_matrix"] for r in runs], axis=0).tolist(),
        "client_f1": [c["test"]["f1_macro"] for r in runs for c in part(r)["per_client"].values()],
        "client_users": [c["test_users"] for r in runs for c in part(r)["per_client"].values()],
        "by_run": {f"{f}-{s}": v for (f, s), v in test_f1.items()},
        "by_run_worst": {f"{r['fold']}-{r['seed']}": w for r, w in zip(runs, worst)},
        "hit_round_limit": float(np.mean([(part(r)["stop_round"] if sub else r["stop_round"]) == r["rounds_run"]
                                          and (part(r)["best_round"] if sub else r["best_round"]) > r["rounds_run"] - int(r["config"]["patience"])
                                          for r in runs])),
    }
    if not sub:
        hist = defaultdict(list)
        for r in runs:
            for h in r["history"]:
                for key, v in h.items():
                    if key != "round":
                        hist[key].append((h["round"], v))
        rounds = max(len(r["history"]) for r in runs)
        out["curves"] = {key: {"mean": [float(np.mean([v for rr, v in vals if rr == i])) for i in range(1, rounds + 1)],
                               "std": [float(np.std([v for rr, v in vals if rr == i], ddof=1)) for i in range(1, rounds + 1)]}
                         for key, vals in hist.items()}
        out["stop_round"] = mean_std([r["stop_round"] for r in runs])
    model = runs[0]["model"]
    if central_by_key is not None:
        out["vs_central"] = paired(test_f1, central_by_key[model])
    return out


def central_summary(central: Central, model: str, pdirs: dict, k: int) -> dict:
    """Centralised baseline on the same test users, with per-client numbers for this split."""
    runs = [r for (m, f, s), r in central.results.items() if m == model]
    worst, spread, client_f1, client_users = [], [], [], []
    for r in runs:
        pdir = pdirs[r["fold"]]
        meta = json.loads((pdir / "meta.json").read_text())
        owner = {u: c["client"] for c in meta["clients"] for u in c["test_users"]}
        cms = central.user_cms(model, r["fold"], r["seed"], pdir)["cms"]
        f1s = per_client_from_users(cms, owner, k)
        worst.append(min(f1s))
        spread.append(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0)
        client_f1 += f1s
        client_users += [len(c["test_users"]) for c in meta["clients"]]
    return {"worst_client_f1": mean_std(worst), "client_f1_std": mean_std(spread),
            "client_f1": client_f1, "client_users": client_users}


def partition_summary(root: Path) -> dict:
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("fold*/meta.json"))]
    return {"jsd_mean": mean_std([m["jsd_mean"] for m in metas]), "jsd_max": float(max(m["jsd_max"] for m in metas)),
            "fold0": [{"train_windows": c["train_windows"], "train_users": len(c["train_users"]),
                       "val_users": len(c["val_users"]), "test_users": len(c["test_users"]),
                       "share": c["train_label_share"]} for c in metas[0]["clients"]],
            "size_cv": mean_std([np.std(w := [c["train_windows"] for c in m["clients"]]) / np.mean(w) for m in metas]),
            "missing_classes": mean_std([sum(s == 0 for s in c["train_label_share"]) for m in metas for c in m["clients"]])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fl", type=Path, default=REPO / "outputs/wisdm_fl")
    p.add_argument("--central", type=Path, default=REPO / "outputs/wisdm")
    p.add_argument("--partitions", type=Path, default=REPO / "data/partitions/wisdm")
    p.add_argument("--template", type=Path, default=Path(__file__).with_name("report_fl_template.html"))
    args = p.parse_args()

    central = Central(args.central, args.partitions)
    central_f1 = {m: {(f, s): r["test"]["f1_macro"] for (mm, f, s), r in central.results.items() if mm == m}
                  for m in ("fcn", "lstm")}
    central_stage1 = {}
    for m in ("fcn", "lstm"):
        rs = [r for (mm, _, _), r in central.results.items() if mm == m]
        central_stage1[m] = {
            "val_f1": mean_std([r["val"]["f1_macro"] for r in rs]), "test_f1": mean_std([r["test"]["f1_macro"] for r in rs]),
            "test_bacc": mean_std([r["test"]["balanced_accuracy"] for r in rs]),
            "test_acc": mean_std([r["test"]["accuracy"] for r in rs]),
            "best_epoch": mean_std([r["best_epoch"] for r in rs]),
            "per_class": {c: mean_std([r["test"]["per_class"][c]["f1"] for r in rs]) for c in CLASSES},
            "confusion": np.sum([r["test"]["confusion_matrix"] for r in rs], axis=0).tolist(),
            "confusion_val": np.sum([r["val"]["confusion_matrix"] for r in rs], axis=0).tolist(),
            "by_run": {f"{r['fold']}-{r['seed']}": r["test"]["f1_macro"] for r in rs},
            "median_user_acc": mean_std([np.median(list(r["test_per_user_accuracy"].values())) for r in rs]),
            "users_below_50": mean_std([np.mean(np.array(list(r["test_per_user_accuracy"].values())) < 0.5) for r in rs]),
            "curves": None}

    groups = defaultdict(list)
    for path in sorted(args.fl.glob("*_K*/*/fold*/seed*/results.json")):
        exp, method = path.parts[-5], path.parts[-4]
        groups[(exp, method)].append(json.loads(path.read_text()))

    experiments = {}
    for (exp, method), runs in sorted(groups.items()):
        split, k = exp.split("_K")[0], int(exp.split("_K")[1])
        pdirs = {f: args.partitions / exp / f"fold{f}" for f in range(5)}
        e = experiments.setdefault(exp, {"split": split, "split_name": SPLIT_NAMES.get(split, split), "K": k,
                                         "methods": {}, "partition": partition_summary(args.partitions / exp)})
        e["methods"][method] = summarise(runs, central_f1, central, pdirs, k)
        e["methods"][method]["model"], e["methods"][method]["strategy"] = runs[0]["model"], runs[0]["strategy"]
        if runs[0]["strategy"] == "fedbn":
            g = summarise(runs, central_f1, central, pdirs, k, sub="global_bn")
            g["model"], g["strategy"] = runs[0]["model"], "fedbn-globalbn"
            e["methods"][f"{method}-globalbn"] = g
            e["methods"][method]["fedbn_distinct_bn"] = float(np.mean([r["fedbn_check"]["distinct"] for r in runs]))
        if k > 1:
            for m in ("fcn", "lstm"):
                e.setdefault("central", {})[m] = central_summary(central, m, pdirs, k)

    # paired tests: one family per experiment (split, K)
    for exp, e in experiments.items():
        tests = []
        for name, s in e["methods"].items():
            if s.get("vs_central"):
                s["vs_central"]["label"] = f"{name} vs centralised {s['model'].upper()}"
                tests.append(s["vs_central"])
        for base in ("fcn-fedbn", "fcn-fedbn-e3"):
            avg = base.replace("fedbn", "fedavg")
            if base in e["methods"] and avg in e["methods"]:
                for variant in (base, f"{base}-globalbn"):
                    a = {tuple(map(int, kk.split("-"))): v for kk, v in e["methods"][variant]["by_run"].items()}
                    b = {tuple(map(int, kk.split("-"))): v for kk, v in e["methods"][avg]["by_run"].items()}
                    t = paired(a, b)
                    if t:
                        t["label"] = f"{variant} vs {avg}"
                        e["methods"][variant]["vs_fedavg"] = t
                        tests.append(t)
                    # fairness: worst-client macro-F1 of the same run (separate family)
                    aw = {tuple(map(int, kk.split("-"))): v for kk, v in e["methods"][variant]["by_run_worst"].items()}
                    bw = {tuple(map(int, kk.split("-"))): v for kk, v in e["methods"][avg]["by_run_worst"].items()}
                    tw = paired(aw, bw)
                    if tw:
                        tw["label"] = f"worst client: {variant} vs {avg}"
                        e["methods"][variant]["vs_fedavg_worst"] = tw
                        e.setdefault("_fair", []).append(tw)
        holm(tests)
        holm(e.pop("_fair", []))

    checks = {"central_models_reevaluated": len(central.cache),
              "central_reeval_windows_differing": sum(v["windows_differing"] for v in central.cache.values()),
              "central_reeval_max_f1_diff": max((v["f1_diff"] for v in central.cache.values()), default=0.0)}
    summary = {"classes": CLASSES, "central": central_stage1, "experiments": experiments, "checks": checks,
               "split_names": SPLIT_NAMES}
    (args.fl / "summary_fl.json").write_text(json.dumps(summary))
    html = args.template.read_text().replace("/*__DATA__*/null", json.dumps(summary))
    (args.fl / "report.html").write_text(html)

    print(f"centralised re-evaluation vs Stage 1: {checks['central_reeval_windows_differing']} windows differ "
          f"over {checks['central_models_reevaluated']} models, max |dF1| {checks['central_reeval_max_f1_diff']:.4f}")
    for m in ("fcn", "lstm"):
        c = central_stage1[m]["test_f1"]
        print(f"central {m:4s}: test F1 {c['mean']:.3f}±{c['std']:.3f}")
    for exp, e in experiments.items():
        print(f"--- {exp} ({e['split_name']}, K={e['K']}), JSD {e['partition']['jsd_mean']['mean']:.3f}")
        for name, s in e["methods"].items():
            mt, vc = s["metrics"], s.get("vs_central") or {}
            vf = s.get("vs_fedavg") or {}
            print(f"  {name:22s} n={s['runs']:2d} test F1 {mt['test_f1']['mean']:.3f}±{mt['test_f1']['std']:.3f} "
                  f"worst client {mt['worst_client_f1']['mean']:.3f} best rnd {mt['best_round']['mean']:4.1f} | "
                  f"vs central {vc.get('mean_diff', float('nan')):+.3f} ({vc.get('wins', '-')}/{vc.get('pairs', '-')}, "
                  f"p_holm {vc.get('p_holm', float('nan')):.3f})"
                  + (f" | vs FedAvg {vf['mean_diff']:+.3f} ({vf['wins']}/{vf['pairs']}, p_holm {vf.get('p_holm', float('nan')):.3f})" if vf else "")
                  + (f" | worst-client vs FedAvg {s['vs_fedavg_worst']['mean_diff']:+.3f} ({s['vs_fedavg_worst']['wins']}/{s['vs_fedavg_worst']['pairs']}, p_holm {s['vs_fedavg_worst'].get('p_holm', float('nan')):.3f})" if s.get('vs_fedavg_worst') else "")
                  + (f" | hit round limit {s['hit_round_limit']*100:.0f}%" if s.get('hit_round_limit') else ""))
    print(f"wrote {args.fl / 'summary_fl.json'} and {args.fl / 'report.html'}")


if __name__ == "__main__":
    main()
