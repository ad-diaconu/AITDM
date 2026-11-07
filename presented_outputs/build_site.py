"""Collect the metrics of every run under outputs/ and build the mini-site in presented_outputs/site/.

Walks the outputs folder recursively and reads every *.json file:
  * run results (anything with test/val metrics, a fold and a seed), centralised or federated;
    FedBN runs additionally yield a "<method>-globalbn" variant from their `global_bn` block;
  * summary files with paired statistical tests (`vs_central`, `vs_fedavg`, ...), if present.
Everything is written to site/data.js; open site/index.html in a browser (no server needed).

    python presented_outputs/build_site.py
    python presented_outputs/build_site.py --outputs outputs --site presented_outputs/site
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_f1", "val_acc", "test_loss", "test_f1", "test_acc")
SPLIT_NAMES = {"s0": "S0 IID", "s1": "S1 natural (users)", "s2": "S2 user-Dirichlet α=0.3",
               "s3": "S3 window-Dirichlet α=0.3"}


def r4(x):
    return None if x is None else round(float(x), 4)


def is_metrics(block) -> bool:
    return isinstance(block, dict) and "f1_macro" in block


def is_run(d) -> bool:
    return isinstance(d, dict) and is_metrics(d.get("test")) and "fold" in d and "seed" in d


def metric_block(m: dict) -> dict:
    return {"f1": r4(m.get("f1_macro")), "acc": r4(m.get("accuracy")), "bacc": r4(m.get("balanced_accuracy"))}


def per_class(m: dict) -> dict:
    return {c: {k: r4(v.get(k)) for k in ("precision", "recall", "f1")} | {"support": v.get("support")}
            for c, v in (m.get("per_class") or {}).items()}


def user_stats(acc: dict) -> dict:
    vals = [float(v) for v in (acc or {}).values()]
    if not vals:
        return {}
    return {"n": len(vals), "median": r4(statistics.median(vals)), "min": r4(min(vals)),
            "below50": r4(sum(v < 0.5 for v in vals) / len(vals)), "values": [r4(v) for v in vals]}


def history(hist: list, prefix_map: dict | None = None) -> dict:
    """Column-oriented learning curves; prefix_map renames e.g. gval_f1 -> val_f1 (FedBN global BN)."""
    if not hist:
        return {}
    out = {"x": [h.get("round", h.get("epoch")) for h in hist]}
    for key in HISTORY_KEYS:
        src = (prefix_map or {}).get(key, key)
        if any(src in h for h in hist):
            out[key] = [r4(h.get(src)) for h in hist]
    return out


def clients(per_client: dict) -> list:
    return [{"id": int(cid), "f1": r4(c["test"]["f1_macro"]), "acc": r4(c["test"]["accuracy"]),
             "val_f1": r4(c.get("val", {}).get("f1_macro")), "users": c.get("test_users")}
            for cid, c in sorted((per_client or {}).items(), key=lambda kv: int(kv[0]))]


def make_run(d: dict, rel: Path, sub: str | None = None) -> dict:
    """One flat record; `sub="global_bn"` builds the FedBN global-BN variant of a FedBN run."""
    part = d[sub] if sub else d
    federated = "num_clients" in d or "strategy" in d
    parts = rel.parts
    if federated:
        experiment = next((p for p in parts if "_K" in p), f"{d.get('split', '?')}_K{d.get('num_clients', '?')}")
        method = parts[parts.index(experiment) + 1] if experiment in parts else f"{d['model']}-{d['strategy']}"
    else:
        experiment, method = "central", f"{d['model']}-central"
    if sub:
        method += "-globalbn"
    pc = clients(part.get("per_client"))
    client_f1 = [c["f1"] for c in pc]
    best = part.get("best_round") if sub else d.get("best_round", d.get("best_epoch"))
    hist_map = {"val_loss": "gval_loss", "val_f1": "gval_f1", "val_acc": "gval_acc",
                "test_loss": "gtest_loss", "test_f1": "gtest_f1", "test_acc": "gtest_acc"} if sub else None
    comm = d.get("comm_mb_per_round_fedbn_net" if d.get("strategy") == "fedbn" and not sub else "comm_mb_per_round")
    return {
        "id": str(rel.parent) + (" [global BN]" if sub else ""),
        "path": str(rel),
        "kind": "federated" if federated else "central",
        "experiment": experiment,
        "split": d.get("split"),
        "K": d.get("num_clients"),
        "method": method,
        "model": d.get("model"),
        "strategy": ("fedbn-globalbn" if sub else d.get("strategy")) if federated else "central",
        "fold": d["fold"],
        "seed": d["seed"],
        "val": metric_block(part["val"]) if is_metrics(part.get("val")) else {},
        "test": metric_block(part["test"]),
        "best": best,
        "length": d.get("rounds_run", d.get("epochs_run")),
        "stop": part.get("stop_round") if sub else d.get("stop_round"),
        "seconds": r4(d.get("train_seconds")),
        "params": d.get("params"),
        "comm_mb_round": r4(comm),
        "comm_mb_to_best": r4(comm * best) if comm and best else None,
        "per_class": per_class(part["test"]),
        "cm": part["test"].get("confusion_matrix"),
        "cm_val": (part.get("val") or {}).get("confusion_matrix"),
        "clients": pc,
        "worst_client": min(client_f1) if len(client_f1) > 1 else None,
        "client_std": r4(statistics.stdev(client_f1)) if len(client_f1) > 1 else None,
        "users": user_stats(part.get("test_per_user_accuracy")),
        "history": history(d.get("history"), hist_map),
        "partition": d.get("partition"),
        "config": d.get("config"),
        "fedbn_distinct": (d.get("fedbn_check") or {}).get("distinct") if not sub else None,
    }


def collect_tests(summary: dict, rel: Path) -> list:
    """Paired tests (mean diff, wins, p, Holm p) from summary files such as summary_fl.json."""
    tests = []
    for exp, e in (summary.get("experiments") or {}).items():
        for method, m in (e.get("methods") or {}).items():
            for kind in ("vs_central", "vs_fedavg", "vs_fedavg_worst"):
                t = m.get(kind)
                if isinstance(t, dict) and "mean_diff" in t:
                    tests.append({"experiment": exp, "method": method, "kind": kind, "label": t.get("label"),
                                  "pairs": t.get("pairs"), "mean_diff": r4(t["mean_diff"]), "wins": t.get("wins"),
                                  "p": t.get("p"), "p_holm": t.get("p_holm"), "source": str(rel)})
    comp = summary.get("comparison")
    if isinstance(comp, dict) and "wilcoxon_p" in comp:
        tests.append({"experiment": "central", "method": "fcn-central", "kind": "fcn_vs_lstm",
                      "label": "FCN vs LSTM (centralised)", "pairs": comp.get("pairs"),
                      "mean_diff": r4(comp.get("mean_diff_fcn_minus_lstm")), "wins": comp.get("fcn_better"),
                      "p": comp.get("wilcoxon_p"), "p_holm": None, "source": str(rel)})
    return tests


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outputs", type=Path, default=REPO / "outputs")
    p.add_argument("--site", type=Path, default=HERE / "site")
    args = p.parse_args()

    runs, tests, scanned, skipped, classes = [], [], 0, [], None
    for path in sorted(args.outputs.rglob("*.json")):
        scanned += 1
        rel = path.relative_to(args.outputs)
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{rel}: {exc}")
            continue
        if is_run(d):
            runs.append(make_run(d, rel))
            if isinstance(d.get("global_bn"), dict) and is_metrics(d["global_bn"].get("test")):
                runs.append(make_run(d, rel, sub="global_bn"))
            classes = classes or list((d["test"].get("per_class") or {}).keys())
        elif isinstance(d, dict):
            found = collect_tests(d, rel)
            tests += found
            if not found:
                skipped.append(f"{rel}: no run metrics")
            classes = classes or d.get("classes")
    if not runs:
        raise SystemExit(f"no run results found under {args.outputs}")

    data = {"generated": datetime.now().isoformat(timespec="seconds"), "source": str(args.outputs),
            "scanned_files": scanned, "skipped": skipped, "classes": classes or [],
            "split_names": SPLIT_NAMES, "runs": runs, "tests": tests}
    args.site.mkdir(parents=True, exist_ok=True)
    out = args.site / "data.js"
    out.write_text("window.DATA = " + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    groups = {(r["experiment"], r["method"]) for r in runs}
    print(f"scanned {scanned} JSON files: {len(runs)} runs ({len(groups)} experiment/method groups), "
          f"{len(tests)} statistical tests, {len(skipped)} other files")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB); open {args.site / 'index.html'}")


if __name__ == "__main__":
    main()
