"""Centralised training of FCN (Conv1D) or LSTM on one subject-wise WISDM fold.

Needs only torch + numpy (runs in the cluster container). Model selection uses the
validation users only; test loss/metrics are logged per epoch for plotting, never
used for any decision.

    python wisdm_baseline/train.py --data data/wisdm/windows.npz --model fcn --fold 0 --seeds 1 2 3

Writes <out>/<model>/fold<k>/seed<s>/{results.json, model.pt}.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

CLASSES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown"]


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


def metrics(y: np.ndarray, pred: np.ndarray, k: int = len(CLASSES)) -> dict:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(k),
        where=(precision + recall) > 0,
    )
    present = support > 0
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "f1_macro": float(f1[present].mean()),
        "per_class": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, x, y, criterion, batch: int = 2048):
    model.eval()
    losses, preds = [], []
    for i in range(0, len(y), batch):
        out = model(x[i : i + batch])
        losses.append(criterion(out, y[i : i + batch]).item() * len(out))
        preds.append(out.argmax(1))
    pred = torch.cat(preds).cpu().numpy()
    return sum(losses) / len(y), pred


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args, data, seed: int, device) -> tuple[dict, dict]:
    k = args.fold
    tr, va, te = data[f"train{k}"], data[f"val{k}"], data[f"test{k}"]
    x_all = torch.from_numpy(data["x"])
    mean = x_all[tr].reshape(-1, 3).mean(0)
    std = x_all[tr].reshape(-1, 3).std(0)
    norm = lambda idx: ((x_all[idx] - mean) / std).to(device)  # noqa: E731
    y_all = torch.from_numpy(data["y"])
    x_tr, x_va, x_te = norm(tr), norm(va), norm(te)
    y_tr, y_va, y_te = (y_all[i].to(device) for i in (tr, va, te))

    set_seed(seed)
    model = MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    history, best = [], {"f1": -1.0, "epoch": -1, "state": None}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr), generator=gen).to(device)
        loss_sum, n_seen, correct = 0.0, 0, 0
        for i in range(0, len(perm), args.batch_size):
            b = perm[i : i + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            out = model(x_tr[b])
            loss = criterion(out, y_tr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            loss_sum += loss.item() * len(b)
            n_seen += len(b)
            correct += (out.argmax(1) == y_tr[b]).sum().item()
        val_loss, val_pred = evaluate(model, x_va, y_va, criterion)
        test_loss, test_pred = evaluate(model, x_te, y_te, criterion)
        vm, tm = metrics(data["y"][va], val_pred), metrics(data["y"][te], test_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / n_seen,
                "train_acc": correct / n_seen,
                "val_loss": val_loss,
                "val_acc": vm["accuracy"],
                "val_f1": vm["f1_macro"],
                "test_loss": test_loss,
                "test_acc": tm["accuracy"],
                "test_f1": tm["f1_macro"],
                "seconds": time.time() - t0,
            }
        )
        h = history[-1]
        print(
            f"  [{args.model} fold{k} seed{seed}] ep {epoch:2d} train {h['train_loss']:.3f} "
            f"val {val_loss:.3f}/{h['val_f1']:.3f} test {test_loss:.3f}/{h['test_f1']:.3f} "
            f"({h['seconds']:.1f}s)",
            flush=True,
        )
        if vm["f1_macro"] > best["f1"]:
            best = {
                "f1": vm["f1_macro"],
                "epoch": epoch,
                "state": {
                    n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                },
            }
        elif epoch - best["epoch"] >= args.patience:
            break

    model.load_state_dict(best["state"])
    _, val_pred = evaluate(model, x_va, y_va, criterion)
    _, test_pred = evaluate(model, x_te, y_te, criterion)
    test_users = data["user"][te]
    per_user = {
        int(u): float(
            (test_pred[test_users == u] == data["y"][te][test_users == u]).mean()
        )
        for u in np.unique(test_users)
    }
    return {
        "model": args.model,
        "fold": k,
        "seed": seed,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "train_seconds": time.time() - t_start,
        "params": sum(p.numel() for p in model.parameters()),
        "config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "clip": args.clip,
            "optimizer": "adam",
            "selection": "best val macro-F1",
            "device": str(device),
            "torch": torch.__version__,
        },
        "normalisation": {"mean": mean.tolist(), "std": std.tolist()},
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "val": metrics(data["y"][va], val_pred),
        "test": metrics(data["y"][te], test_pred),
        "test_per_user_accuracy": per_user,
        "history": history,
    }, best["state"]


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/wisdm/windows.npz"))
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument(
        "--limit", type=int, default=0, help="smoke test: use only N training windows"
    )
    p.add_argument("--out", type=Path, default=Path("outputs/wisdm"))
    p.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}, "
        f"torch {torch.__version__}",
        flush=True,
    )
    data = dict(np.load(args.data))
    if args.limit:
        data[f"train{args.fold}"] = data[f"train{args.fold}"][: args.limit]

    for seed in args.seeds:
        run_dir = args.out / args.model / f"fold{args.fold}" / f"seed{seed}"
        if (run_dir / "results.json").exists():
            print(f"skip {run_dir} (done)")
            continue
        result, state = run(args, data, seed, device)
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.save_model:
            torch.save(state, run_dir / "model.pt")
        tmp = run_dir / "results.json.tmp"
        tmp.write_text(json.dumps(result))
        tmp.replace(run_dir / "results.json")
        print(
            f"{args.model} fold{args.fold} seed{seed}: best epoch {result['best_epoch']}, "
            f"val F1 {result['val']['f1_macro']:.3f}, test F1 {result['test']['f1_macro']:.3f}, "
            f"acc {result['test']['accuracy']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
