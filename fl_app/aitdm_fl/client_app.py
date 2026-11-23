"""Flower ClientApp: one site with its own users (docs/experiment_setup_federated.md).

FedAvg: train the received global model, send it back.
FedBN:  the same, but the BatchNorm layers never leave the client. They are kept in
        context.state between rounds and replace the received (averaged) BN values.
        strategy='fedbn' (= FedAvg aggregation) or local-bn=true with any other strategy.
Evaluation returns confusion counts, loss sums and per-user counts on the client's
validation and test users, which the server sums into exact pooled metrics.

Stage 2 (docs/04_aggregation_strategies.md): FedProx's proximal term, class-balanced loss,
and simulated attackers (malicious-clients / attack / attack-inflate-examples).
"""

from __future__ import annotations

from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from aitdm_fl.task import (balanced_class_weights, bn_state, build_model, derive_seed, evaluate,
                           get_device, load_client_file, load_state, run_config, set_seed,
                           shared_state, train, uses_local_bn)

app = ClientApp()


def data_path(context: Context, cfg: dict) -> Path:
    """Which client file does this node own?

    * Simulation: Flower sets partition-id (0 ... num-supernodes-1) -> partitions-dir/client_<id>.npz
    * Deployment: each SuperNode points to its own file with
      --node-config "data-path='/abs/path/fold0/client_2.npz'" (scripts/start_supernode.sh)
    """
    if "data-path" in context.node_config:
        return Path(str(context.node_config["data-path"]))
    return Path(str(cfg["partitions-dir"])) / f"client_{int(context.node_config['partition-id'])}.npz"


def partition_id(context: Context, cfg: dict) -> int:
    if "partition-id" in context.node_config:
        return int(context.node_config["partition-id"])
    return int(data_path(context, cfg).stem.split("_")[1])


def is_malicious(context: Context, cfg: dict) -> bool:
    """Simulation: run config malicious-clients='1,3' (partition ids).
    Deployment: --node-config "data-path='...' malicious=true"."""
    if str(context.node_config.get("malicious", "false")).lower() in ("true", "1"):
        return True
    ids = str(cfg["malicious-clients"]).replace(" ", "")
    return ids != "" and str(partition_id(context, cfg)) in ids.split(",")


def poison(global_state: dict, local_state: dict, attack: str) -> dict:
    """Model poisoning: replace the honest update with a harmful one.

    * "sign-flip": push the model 5x in the OPPOSITE direction of learning
    * "noise":     send random Gaussian weights
    * "none":      send the honestly trained weights (e.g. a data poisoner with wrong labels,
                   data_splitting/split_data.py --label-noise, combined with attack-inflate-examples)
    """
    if attack == "none":
        return local_state
    poisoned = {}
    for name, w_local in local_state.items():
        w_global = global_state[name]
        if not torch.is_floating_point(w_local):
            poisoned[name] = w_local
        elif attack == "sign-flip":
            poisoned[name] = w_global - 5.0 * (w_local - w_global)
        elif attack == "noise":
            poisoned[name] = torch.randn_like(w_local)
        else:
            raise ValueError(f"Unknown attack '{attack}'. Use 'sign-flip', 'noise' or 'none'.")
    return poisoned


def receive_model(msg: Message, cfg: dict, context: Context, local_bn: bool):
    model = build_model(cfg["model"])
    load_state(model, msg.content["arrays"].to_torch_state_dict())
    if local_bn:
        if "bn" in context.state:
            load_state(model, context.state["bn"].to_torch_state_dict(), partial=True)
        elif int(msg.content["config"]["server-round"]) > 1:
            raise RuntimeError("FedBN: the client's BatchNorm state was lost between rounds")
    return model


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    cfg = run_config(context)
    pid = partition_id(context, cfg)
    rnd = int(msg.content["config"]["server-round"])
    seed = derive_seed(int(cfg["seed"]), pid, rnd)
    set_seed(seed)
    x, y, _ = load_client_file(str(data_path(context, cfg)))["train"]
    local_bn = uses_local_bn(cfg)
    model = receive_model(msg, cfg, context, local_bn=local_bn)
    malicious = is_malicious(context, cfg)
    global_state = {k: v.clone() for k, v in shared_state(model).items()} if malicious else None
    weights = balanced_class_weights(y) if str(cfg["loss-weighting"]) == "balanced" else None
    loss = train(model, x, y, epochs=int(cfg["local-epochs"]), lr=float(cfg["learning-rate"]),
                 batch_size=int(cfg["batch-size"]), clip=float(cfg["clip"]), seed=seed,
                 device=get_device(),
                 proximal_mu=float(msg.content["config"].get("proximal-mu", 0.0)),  # set by FedProx
                 class_weights=weights)
    if local_bn:
        context.state["bn"] = ArrayRecord(bn_state(model))
    state = shared_state(model)
    num_examples = int(len(y))  # FedAvg weights the aggregation by it
    if malicious:
        state = poison(global_state, state, str(cfg["attack"]))
        # nothing stops a client from lying about its size: FedAvg trusts it
        num_examples *= int(cfg["attack-inflate-examples"])
    metrics = MetricRecord({"train_loss": float(loss), "num-examples": num_examples, "client": pid,
                            "unit-weight": 1})  # equal weights, used when differential privacy is on
    return Message(content=RecordDict({"arrays": ArrayRecord(state), "metrics": metrics}),
                   reply_to=msg)


def _flatten(prefix: str, res: dict) -> dict:
    return {f"{prefix}_cm": res["cm"], f"{prefix}_loss_sum": res["loss_sum"], f"{prefix}_n": res["n"],
            f"{prefix}_users": res["users"], f"{prefix}_user_correct": res["user_correct"],
            f"{prefix}_user_total": res["user_total"]}


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    cfg = run_config(context)
    pid = partition_id(context, cfg)
    data = load_client_file(str(data_path(context, cfg)))
    device = get_device()
    local_bn = uses_local_bn(cfg)
    out = {"client": pid, "num-examples": int(len(data["val"][1])), "unit-weight": 1}
    if local_bn:
        # the received model carries the client-size-weighted average of all BN layers:
        # what a NEW site without its own BN statistics would use
        global_bn = receive_model(msg, cfg, context, local_bn=False)
        out.update(_flatten("gval", evaluate(global_bn, *data["val"], device)))
        out.update(_flatten("gtest", evaluate(global_bn, *data["test"], device)))
    model = receive_model(msg, cfg, context, local_bn=local_bn)
    out.update(_flatten("val", evaluate(model, *data["val"], device)))
    out.update(_flatten("test", evaluate(model, *data["test"], device)))
    bn = bn_state(model)
    out["bn_signature"] = float(sum(v.double().sum() for k, v in bn.items() if k.endswith("running_mean"))) if bn else 0.0
    return Message(content=RecordDict({"metrics": MetricRecord(out)}), reply_to=msg)
