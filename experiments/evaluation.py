"""Shared evaluation and statistics for experiments/summarize.py and summarize_fl.py, so that
centralised and federated runs are summarised and compared in exactly the same way.

* mean_std:   mean +- sample std (ddof = 1) over runs (folds x seeds)
* paired:     paired comparison of two methods over the same (fold, seed) runs (Wilcoxon)
* holm:       Holm-Bonferroni correction within one family of tests
* Central:    re-evaluates the centralised Stage 1 models on every client's test users, so that
              per-client (fairness) numbers exist for the centralised baseline too
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fl_app"))  # works without `pip install -e fl_app` too
from aitdm_fl.task import CLASSES, build_model, load_state, scores  # noqa: E402

K = len(CLASSES)


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
    cms = [np.zeros((K, K), int) for _ in range(k)]
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
                cm = np.zeros((K, K), int)
                np.add.at(cm, (y[m], pred[m]), 1)
                cms[int(u)] = cm
            total = sum(cms.values())
            stage1 = np.array(self.results[key]["test"]["confusion_matrix"])
            self.cache[key] = {"cms": cms, "windows_differing": int(np.abs(total - stage1).sum() // 2),
                               "f1_diff": abs(scores(total)["f1_macro"] - scores(stage1)["f1_macro"])}
        return self.cache[key]
