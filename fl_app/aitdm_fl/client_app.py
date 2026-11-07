"""Flower ClientApp: one site with its own users (docs/experiment_setup_federated.md).

FedAvg: train the received global model, send it back.
FedBN:  the same, but the BatchNorm layers never leave the client. They are kept in
        context.state between rounds and replace the received (averaged) BN values.
Evaluation returns confusion counts, loss sums and per-user counts on the client's
validation and test users, which the server sums into exact pooled metrics.
"""

from __future__ import annotations

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from aitdm_fl.task import (bn_state, build_model, derive_seed, evaluate, get_device, load_client,
                           load_state, run_config, set_seed, shared_state, train)

app = ClientApp()


def partition_id(context: Context) -> int:
    return int(context.node_config["partition-id"])


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
    pid = partition_id(context)
    rnd = int(msg.content["config"]["server-round"])
    seed = derive_seed(int(cfg["seed"]), pid, rnd)
    set_seed(seed)
    x, y, _ = load_client(str(cfg["partitions-dir"]), pid)["train"]
    fedbn = cfg["strategy"] == "fedbn"
    model = receive_model(msg, cfg, context, local_bn=fedbn)
    loss = train(model, x, y, epochs=int(cfg["local-epochs"]), lr=float(cfg["learning-rate"]),
                 batch_size=int(cfg["batch-size"]), clip=float(cfg["clip"]), seed=seed,
                 device=get_device())
    if fedbn:
        context.state["bn"] = ArrayRecord(bn_state(model))
    metrics = MetricRecord({"train_loss": float(loss), "num-examples": int(len(y)), "client": pid})
    return Message(content=RecordDict({"arrays": ArrayRecord(shared_state(model)), "metrics": metrics}),
                   reply_to=msg)


def _flatten(prefix: str, res: dict) -> dict:
    return {f"{prefix}_cm": res["cm"], f"{prefix}_loss_sum": res["loss_sum"], f"{prefix}_n": res["n"],
            f"{prefix}_users": res["users"], f"{prefix}_user_correct": res["user_correct"],
            f"{prefix}_user_total": res["user_total"]}


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    cfg = run_config(context)
    pid = partition_id(context)
    data = load_client(str(cfg["partitions-dir"]), pid)
    device = get_device()
    fedbn = cfg["strategy"] == "fedbn"
    out = {"client": pid, "num-examples": int(len(data["val"][1]))}
    if fedbn:
        # the received model carries the client-size-weighted average of all BN layers:
        # what a NEW site without its own BN statistics would use
        global_bn = receive_model(msg, cfg, context, local_bn=False)
        out.update(_flatten("gval", evaluate(global_bn, *data["val"], device)))
        out.update(_flatten("gtest", evaluate(global_bn, *data["test"], device)))
    model = receive_model(msg, cfg, context, local_bn=fedbn)
    out.update(_flatten("val", evaluate(model, *data["val"], device)))
    out.update(_flatten("test", evaluate(model, *data["test"], device)))
    bn = bn_state(model)
    out["bn_signature"] = float(sum(v.double().sum() for k, v in bn.items() if k.endswith("running_mean"))) if bn else 0.0
    return Message(content=RecordDict({"metrics": MetricRecord(out)}), reply_to=msg)
