"""Models, data loading, training and evaluation for WISDM: everything that is *not*
Flower-specific lives here (no Flower import), so experiments/centralised.py and the
explainability scripts use exactly the networks and metrics of the federated runs.
"""

from __future__ import annotations

import json
import os
import random
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

CLASSES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown"]

# Every run-config key the app understands, with its default. fl_app/pyproject.toml holds the
# same keys for `flwr run`; experiments/run_seeds.py overrides them per run.
DEFAULTS = {
    "partitions-dir": "", "save-dir": "", "model": "fcn", "strategy": "fedavg", "local-bn": False,
    "seed": 1, "num-server-rounds": 40, "local-epochs": 1, "learning-rate": 1e-3, "batch-size": 64,
    "clip": 1.0, "patience": 10, "min-nodes": 2, "loss-weighting": "none",
    # aggregation strategies (docs/04_aggregation_strategies.md)
    "proximal-mu": 0.1, "server-momentum": 0.9, "server-lr": 0.01, "trim-beta": 0.2,
    "num-malicious": 1, "qfedavg-q": 0.1, "qfedavg-client-lr": 1.0,
    # central differential privacy (0 = off)
    "dp-noise-multiplier": 0.0, "dp-clipping-norm": 1.0, "dp-delta": 1e-5,
    # simulated attackers
    "malicious-clients": "", "attack": "sign-flip", "attack-inflate-examples": 1,
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


def uses_local_bn(cfg: dict) -> bool:
    """FedBN: BatchNorm layers stay on the client. 'fedbn' = FedAvg aggregation + local BN."""
    return cfg["strategy"] == "fedbn" or str(cfg["local-bn"]).lower() in ("true", "1")


# ------------------------------------------------------------------ models
class FCN(nn.Module):
    """Wang, Yan & Oates (2017): 3 conv blocks + global average pooling."""

    def __init__(self, in_ch: int = 3, n_classes: int = 6):
        super().__init__()

        def block(i, o, k):
            return [nn.Conv1d(i, o, k, padding="same"), nn.BatchNorm1d(o), nn.ReLU()]

        self.features = nn.Sequential(
            *block(in_ch, 128, 8), *block(128, 256, 5), *block(256, 128, 3)
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, T, C)
        return self.head(self.features(x.transpose(1, 2)).mean(-1))


class LSTMNet(nn.Module):
    """2-layer many-to-one LSTM, last time step -> dropout -> linear."""

    def __init__(self, in_ch: int = 3, n_classes: int = 6, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


MODELS = {"fcn": FCN, "lstm": LSTMNet}


def build_model(name: str) -> nn.Module:
    return MODELS[name](in_ch=3, n_classes=len(CLASSES))


# ------------------------------------------------------------------ reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_seed(seed: int, *parts) -> int:
    """A different but reproducible seed for every (client, round, ...) combination."""
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


def normalise(x: np.ndarray, partitions_dir: str) -> np.ndarray:
    """z-score raw m/s^2 windows with the fold's pooled training mean / std (meta.json)."""
    mean, std = _normalisation(str(partitions_dir))
    return ((x - mean) / std).astype(np.float32)


@lru_cache(maxsize=8)
def load_client_file(path: str) -> dict:
    """{split: (x normalised float32 (N, T, C), y int64, user int64)} of one client file.
    The normalisation comes from meta.json next to it (written by data_splitting/split_data.py)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run data_splitting/split_data.py first and pass the "
                                "ABSOLUTE fold folder as partitions-dir")
    data = np.load(p)
    return {s: (normalise(data[f"x_{s}"], str(p.parent)), data[f"y_{s}"].astype(np.int64),
                data[f"u_{s}"].astype(np.int64)) for s in ("train", "val", "test")}


def load_client(partitions_dir: str, client: int) -> dict:
    return load_client_file(str(Path(partitions_dir) / f"client_{client}.npz"))


def meta(partitions_dir: str) -> dict:
    return json.loads((Path(partitions_dir) / "meta.json").read_text())


def load_split(partitions_dir: str, which: str = "val", split: str = "val"):
    """Data for the analysis scripts in explainability/. Returns (x, y, users, owner, tag).

    * "val"        all clients' validation users together (default; owner = client id)
    * "client_<i>" one client's data (its train / val / test split, see `split`)
    * "test"       the fold's test users (test.npz): only for the final report
    """
    pdir = Path(partitions_dir)
    if which == "val":
        files = sorted(pdir.glob("client_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
        if not files:
            raise FileNotFoundError(f"No client_<i>.npz files in {pdir}")
        parts = [load_client_file(str(f))["val"] for f in files]
        owner = np.concatenate([np.full(len(p[1]), int(f.stem.split("_")[1])) for f, p in zip(files, parts)])
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]), owner, "val")
    if which.startswith("client_"):
        x, y, u = load_client_file(str(pdir / f"{which}.npz"))[split]
        return x, y, u, None, f"{which}_{split}"
    if which == "test":
        print("NOTE: you are looking at the TEST users. Use them once, for the final numbers of a "
              "model you have already chosen. Diagnose and compare on --data val instead.")
        d = np.load(pdir / "test.npz")
        return (normalise(d["x_test"], str(pdir)), d["y_test"].astype(np.int64),
                d["u_test"].astype(np.int64), None, "test")
    raise ValueError("--data must be 'val', 'client_<i>' or 'test'")


# ------------------------------------------------------------------ training / evaluation
def balanced_class_weights(labels: np.ndarray, num_classes: int = len(CLASSES)) -> torch.Tensor:
    """Loss weights inversely proportional to class frequency, computed by every client from
    its OWN labels. Classes the client does not have get weight 1 (never in its loss anyway)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.ones(num_classes)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def train(model, x, y, epochs: int, lr: float, batch_size: int, clip: float, seed: int, device,
          proximal_mu: float = 0.0, class_weights: torch.Tensor | None = None) -> float:
    """Local training (fresh Adam every round, as in the tutorial app). Returns mean loss.

    proximal_mu > 0 is FedProx: a penalty keeps the local model close to the received global
    model. class_weights re-weights the loss per class. Both are no-ops at their defaults."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    global_params = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    gen = torch.Generator().manual_seed(seed)
    loss_sum, n = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen).to(device)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xt[b]), yt[b])
            if global_params is not None:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            loss_sum += loss.item() * len(b)
            n += len(b)
    return loss_sum / max(n, 1)


@torch.no_grad()
def predict(model, x: np.ndarray, device=None, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(logits as float32 numpy, predicted class) for normalised windows x."""
    device = device or next(model.parameters()).device
    model.to(device).eval()
    outs = [model(torch.from_numpy(x[i:i + batch]).to(device)).float().cpu() for i in range(0, len(x), batch)]
    logits = torch.cat(outs).numpy() if outs else np.zeros((0, len(CLASSES)), np.float32)
    return logits, logits.argmax(1)


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
    cm = confusion(y, pred)
    uniq = np.unique(users)
    correct = [int((pred[users == u] == y[users == u]).sum()) for u in uniq]
    total = [int((users == u).sum()) for u in uniq]
    return {"cm": cm.reshape(-1).tolist(), "loss_sum": float(loss_sum), "n": int(len(y)),
            "users": uniq.tolist(), "user_correct": correct, "user_total": total}


def confusion(y: np.ndarray, pred: np.ndarray, k: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((k, k), np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    return cm


def scores(cm) -> dict:
    """Pooled metrics from a (summed) confusion matrix: accuracy, balanced accuracy and
    macro-F1 over the classes present, per-class precision / recall / F1."""
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


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """scores() of the confusion matrix of labels y and predictions pred."""
    return scores(confusion(y, pred))
