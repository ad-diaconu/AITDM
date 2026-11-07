"""Data, models, local training and evaluation for the WISDM federation (no Flower here).

The models and the metric function are imported from wisdm_baseline/train.py, so the
federated runs use exactly the networks of the centralised baseline.
"""

from __future__ import annotations

import json
import os
import random
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from wisdm_baseline.train import CLASSES, FCN, LSTMNet  # noqa: E402

MODELS = {"fcn": FCN, "lstm": LSTMNet}
DEFAULTS = {
    "partitions-dir": "", "save-dir": "", "model": "fcn", "strategy": "fedavg", "seed": 1,
    "num-server-rounds": 40, "local-epochs": 1, "learning-rate": 1e-3, "batch-size": 64,
    "clip": 1.0, "patience": 10, "min-nodes": 2,
}


def run_config(context) -> dict:
    """Defaults < pyproject run config (flwr run) < JSON file named in $AITDM_FL_CONFIG_FILE
    (run_simulation; re-read on every call because Ray workers can outlive one run)."""
    cfg = dict(DEFAULTS)
    cfg.update(dict(context.run_config))
    path = os.environ.get("AITDM_FL_CONFIG_FILE")
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


def build_model(name: str) -> nn.Module:
    return MODELS[name](in_ch=3, n_classes=len(CLASSES))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_seed(seed: int, *parts) -> int:
    return zlib.crc32(":".join(str(p) for p in (seed, *parts)).encode())


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ state dict helpers
def is_bn_key(model: nn.Module, key: str) -> bool:
    module_name = key.rsplit(".", 1)[0]
    module = dict(model.named_modules()).get(module_name)
    return isinstance(module, nn.modules.batchnorm._BatchNorm)


def shared_state(model: nn.Module) -> dict:
    """What travels to the server: everything except the integer BN step counters."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.endswith("num_batches_tracked")}


def bn_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if is_bn_key(model, k) and not k.endswith("num_batches_tracked")}


def load_state(model: nn.Module, state: dict, partial: bool = False) -> None:
    """Load a full shared state (all but num_batches_tracked) or, with partial=True, a subset
    such as the client's own BatchNorm layers."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert partial or all(k.endswith("num_batches_tracked") for k in missing), missing


# ------------------------------------------------------------------ data
@lru_cache(maxsize=8)
def _normalisation(partitions_dir: str) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((Path(partitions_dir) / "meta.json").read_text())["normalisation"]
    return np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)


@lru_cache(maxsize=8)
def load_client(partitions_dir: str, client: int) -> dict:
    """{split: (x normalised float32 (N, T, C), y int64, user int64)} of one client."""
    mean, std = _normalisation(partitions_dir)
    data = np.load(Path(partitions_dir) / f"client_{client}.npz")
    return {s: (((data[f"x_{s}"] - mean) / std).astype(np.float32), data[f"y_{s}"].astype(np.int64),
                data[f"u_{s}"].astype(np.int64)) for s in ("train", "val", "test")}


def meta(partitions_dir: str) -> dict:
    return json.loads((Path(partitions_dir) / "meta.json").read_text())


# ------------------------------------------------------------------ training / evaluation
def train(model, x, y, epochs: int, lr: float, batch_size: int, clip: float, seed: int, device) -> float:
    """Local training (fresh Adam every round, as in the tutorial app). Returns mean loss."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    gen = torch.Generator().manual_seed(seed)
    loss_sum, n = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen).to(device)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            loss_sum += loss.item() * len(b)
            n += len(b)
    return loss_sum / max(n, 1)


@torch.no_grad()
def evaluate(model, x, y, users, device) -> dict:
    """Confusion counts, loss sum and per-user counts: everything sums across clients."""
    k = len(CLASSES)
    if len(y) == 0:
        return {"cm": [0] * k * k, "loss_sum": 0.0, "n": 0, "users": [], "user_correct": [], "user_total": []}
    model.to(device).eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, preds = 0.0, []
    for i in range(0, len(y), 2048):
        xb = torch.from_numpy(x[i:i + 2048]).to(device)
        out = model(xb)
        loss_sum += crit(out, torch.from_numpy(y[i:i + 2048]).to(device)).item()
        preds.append(out.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (y, pred), 1)
    uniq = np.unique(users)
    correct = [int((pred[users == u] == y[users == u]).sum()) for u in uniq]
    total = [int((users == u).sum()) for u in uniq]
    return {"cm": cm.reshape(-1).tolist(), "loss_sum": float(loss_sum), "n": int(len(y)),
            "users": uniq.tolist(), "user_correct": correct, "user_total": total}


def scores(cm) -> dict:
    """Pooled metrics from a (summed) confusion matrix, as wisdm_baseline/train.metrics()."""
    k = len(CLASSES)
    cm = np.asarray(cm, dtype=np.int64).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k),
                   where=(precision + recall) > 0)
    present = support > 0
    total = cm.sum()
    return {"accuracy": float(tp.sum() / total) if total else 0.0,
            "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
            "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
            "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])} for i, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist()}
