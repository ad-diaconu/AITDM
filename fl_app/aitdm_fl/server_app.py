"""Flower ServerApp: FedAvg aggregation for both FedAvg and FedBN (FedBN clients ignore the
averaged BN values), exact pooled metrics from summed client confusion counts, and selection
of the best round on federated validation macro-F1 (early stopping with `patience`,
applied to the per-round curve). Writes results.json in the format of the centralised runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from aitdm_fl.task import bn_state, build_model, meta, run_config, scores, set_seed, shared_state

app = ServerApp()
PREFIXES = ("val", "test", "gval", "gtest")


def select_round(f1_by_round: list[float], patience: int) -> tuple[int, int]:
    """(best round, round at which early stopping would stop) - rounds are 1-based."""
    best, best_f1 = 1, -1.0
    for r, f1 in enumerate(f1_by_round, start=1):
        if f1 > best_f1:
            best, best_f1 = r, f1
        elif r - best >= patience:
            return best, r
    return best, len(f1_by_round)


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = run_config(context)
    pdir = str(cfg["partitions-dir"])
    info = meta(pdir)
    k = int(info["num_clients"])
    set_seed(int(cfg["seed"]))
    model = build_model(cfg["model"])
    initial = ArrayRecord(shared_state(model))
    n_params = sum(p.numel() for p in model.parameters())
    shared_bytes = sum(v.numel() * v.element_size() for v in shared_state(model).values())
    bn_bytes = sum(v.numel() * v.element_size() for v in bn_state(model).values())

    train_log, eval_log, stash = [], [], {}

    def train_aggr(contents, _key) -> MetricRecord:
        recs = [c["metrics"] for c in contents]
        n = np.array([r["num-examples"] for r in recs], float)
        loss = float(np.dot(n, [r["train_loss"] for r in recs]) / n.sum())
        train_log.append({"train_loss": loss,
                          "per_client": {int(r["client"]): float(r["train_loss"]) for r in recs}})
        return MetricRecord({"train_loss": loss})

    def eval_aggr(contents, _key) -> MetricRecord:
        recs = sorted((c["metrics"] for c in contents), key=lambda r: int(r["client"]))
        entry, summary = {"clients": {}}, {}
        for p in PREFIXES:
            if f"{p}_cm" not in recs[0]:
                continue
            cm = np.sum([np.array(r[f"{p}_cm"]) for r in recs], axis=0)
            n = sum(int(r[f"{p}_n"]) for r in recs)
            entry[p] = {"cm": cm.tolist(), "loss": sum(float(r[f"{p}_loss_sum"]) for r in recs) / max(n, 1), "n": n}
            summary[f"{p}_f1"] = scores(cm)["f1_macro"]
        for r in recs:
            entry["clients"][int(r["client"])] = {
                "bn_signature": float(r["bn_signature"]),
                **{p: {"cm": list(r[f"{p}_cm"]), "users": list(r[f"{p}_users"]),
                       "user_correct": list(r[f"{p}_user_correct"]), "user_total": list(r[f"{p}_user_total"])}
                   for p in PREFIXES if f"{p}_cm" in r}}
        eval_log.append(entry)
        return MetricRecord(summary)

    def keep_arrays(server_round: int, arrays: ArrayRecord):
        stash[server_round] = {k_: v.clone() for k_, v in arrays.to_torch_state_dict().items()}
        return None

    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0, min_train_nodes=k,
                      min_evaluate_nodes=k, min_available_nodes=k,
                      train_metrics_aggr_fn=train_aggr, evaluate_metrics_aggr_fn=eval_aggr)
    t0 = time.time()
    rounds = int(cfg["num-server-rounds"])
    strategy.start(grid=grid, initial_arrays=initial, num_rounds=rounds,
                   train_config=ConfigRecord({"lr": float(cfg["learning-rate"])}),
                   evaluate_config=ConfigRecord({}), evaluate_fn=keep_arrays)
    seconds = time.time() - t0
    assert len(eval_log) == len(train_log) == rounds, (len(eval_log), len(train_log))

    history = []
    for r, (tr, ev) in enumerate(zip(train_log, eval_log), start=1):
        h = {"round": r, "train_loss": tr["train_loss"]}
        for p in PREFIXES:
            if p in ev:
                s = scores(ev[p]["cm"])
                h.update({f"{p}_loss": ev[p]["loss"], f"{p}_f1": s["f1_macro"], f"{p}_acc": s["accuracy"]})
        history.append(h)
    patience = int(cfg["patience"])
    best, stop = select_round([h["val_f1"] for h in history], patience)

    def evaluation_at(rnd: int, prefix: str, val_prefix: str) -> dict:
        ev = eval_log[rnd - 1]
        per_client, user_acc = {}, {}
        for cid, c in ev["clients"].items():
            per_client[cid] = {"val": scores(c[val_prefix]["cm"]), "test": scores(c[prefix]["cm"]),
                               "test_users": len(c[prefix]["users"]),
                               "val_users": len(c[val_prefix]["users"]),
                               "bn_signature": c["bn_signature"]}
            for u, ok, tot in zip(c[prefix]["users"], c[prefix]["user_correct"], c[prefix]["user_total"]):
                user_acc[int(u)] = ok / tot
        return {"round": rnd, "val": scores(ev[val_prefix]["cm"]), "test": scores(ev[prefix]["cm"]),
                "per_client": per_client, "test_per_user_accuracy": user_acc}

    selected = evaluation_at(best, "test", "val")
    result = {
        "model": cfg["model"], "strategy": cfg["strategy"], "split": info["split"], "num_clients": k,
        "fold": info["fold"], "seed": int(cfg["seed"]), "best_round": best, "stop_round": stop,
        "rounds_run": rounds, "train_seconds": seconds, "params": n_params,
        "comm_mb_per_round": 2 * k * shared_bytes / 1e6,
        "comm_mb_per_round_fedbn_net": 2 * k * (shared_bytes - bn_bytes) / 1e6,
        "config": {key: cfg[key] for key in ("num-server-rounds", "local-epochs", "learning-rate",
                                             "batch-size", "clip", "patience")},
        "partition": {"jsd_mean": info["jsd_mean"], "jsd_max": info["jsd_max"],
                      "train_windows": [c["train_windows"] for c in info["clients"]],
                      "train_users": [len(c["train_users"]) for c in info["clients"]]},
        "val": selected["val"], "test": selected["test"], "per_client": selected["per_client"],
        "test_per_user_accuracy": selected["test_per_user_accuracy"], "history": history,
    }
    if cfg["strategy"] == "fedbn":
        g_best, g_stop = select_round([h["gval_f1"] for h in history], patience)
        g = evaluation_at(g_best, "gtest", "gval")
        result["global_bn"] = {"best_round": g_best, "stop_round": g_stop, "val": g["val"], "test": g["test"],
                               "per_client": g["per_client"], "test_per_user_accuracy": g["test_per_user_accuracy"]}
        sigs = [c["bn_signature"] for c in selected["per_client"].values()]
        result["fedbn_check"] = {"bn_signatures": sigs, "distinct": len(set(np.round(sigs, 6))) == len(sigs)}

    save_dir = Path(str(cfg["save-dir"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stash[best], save_dir / "final_model.pt")
    tmp = save_dir / "results.json.tmp"
    tmp.write_text(json.dumps(result))
    tmp.replace(save_dir / "results.json")
    print(f"[{cfg['model']}-{cfg['strategy']} {info['split']}_K{k} fold{info['fold']} seed{cfg['seed']}] "
          f"best round {best} (stop {stop}): val F1 {result['val']['f1_macro']:.3f}, "
          f"test F1 {result['test']['f1_macro']:.3f}, acc {result['test']['accuracy']:.3f}, "
          f"{seconds:.0f}s", flush=True)
