"""Run federated WISDM experiments: one Flower simulation per (fold, seed).

    python wisdm_baseline/run_fl.py --split s1 --clients 5 --model fcn --strategy fedbn \
        --folds 0 1 2 3 4 --seeds 1 2 3

Results: outputs/wisdm_fl/<split>_K<K>/<model>-<strategy>/fold<f>/seed<s>/{results.json, final_model.pt}.
Finished runs are skipped, so an interrupted job can simply be resubmitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FL_APP = REPO / "fl_app"
# Ray workers import the apps by module name: make both packages importable for them too
os.environ["PYTHONPATH"] = os.pathsep.join([str(FL_APP), str(REPO), os.environ.get("PYTHONPATH", "")])
sys.path[:0] = [str(FL_APP), str(REPO)]
os.environ.setdefault("RAY_DEDUP_LOGS", "0")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--partitions-root", type=Path, default=REPO / "data/partitions/wisdm")
    p.add_argument("--out", type=Path, default=REPO / "outputs/wisdm_fl")
    p.add_argument("--split", required=True)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--model", choices=["fcn", "lstm"], required=True)
    p.add_argument("--strategy", choices=["fedavg", "fedbn"], required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--tag", default="", help="suffix of the experiment name, e.g. for ablations")
    p.add_argument("--gpus", type=float, default=None, help="GPUs for the simulation (default: all visible)")
    p.add_argument("--cpus", type=int, default=None)
    args = p.parse_args()

    import torch
    from flwr.simulation import run_simulation

    from aitdm_fl.client_app import app as client_app
    from aitdm_fl.server_app import app as server_app

    gpus = args.gpus if args.gpus is not None else float(torch.cuda.device_count())
    cpus = args.cpus or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    ray_tmp = f"/tmp/ray_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    backend = {"client_resources": {"num_cpus": max(1, cpus // args.clients),
                                    "num_gpus": (gpus / args.clients) if gpus else 0.0},
               # address="local" + own temp dir: never attach to another process's Ray cluster
               # (several simulations can share a node)
               "init_args": {"num_cpus": cpus, "num_gpus": int(gpus), "include_dashboard": False,
                             "logging_level": "warning", "log_to_driver": True,
                             "address": "local", "_temp_dir": ray_tmp}}
    name = f"{args.model}-{args.strategy}" + (f"-{args.tag}" if args.tag else "")
    cfg_file = Path(tempfile.gettempdir()) / f"aitdm_fl_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}.json"
    os.environ["AITDM_FL_CONFIG_FILE"] = str(cfg_file)

    for fold in args.folds:
        pdir = (args.partitions_root / f"{args.split}_K{args.clients}" / f"fold{fold}").resolve()
        if not (pdir / "meta.json").exists():
            raise SystemExit(f"missing partitions {pdir}; run wisdm_baseline/make_fl_partitions.py")
        for seed in args.seeds:
            save = (args.out / f"{args.split}_K{args.clients}" / name / f"fold{fold}" / f"seed{seed}").resolve()
            if (save / "results.json").exists():
                print(f"skip {save} (done)", flush=True)
                continue
            cfg = {"partitions-dir": str(pdir), "save-dir": str(save), "model": args.model,
                   "strategy": args.strategy, "seed": seed, "num-server-rounds": args.rounds,
                   "local-epochs": args.local_epochs, "patience": args.patience,
                   "min-nodes": args.clients}
            cfg_file.write_text(json.dumps(cfg))
            print(f"=== {name} {args.split}_K{args.clients} fold{fold} seed{seed} "
                  f"(backend {backend['client_resources']})", flush=True)
            run_simulation(server_app=server_app, client_app=client_app, num_supernodes=args.clients,
                           backend_config=backend)
            if not (save / "results.json").exists():
                raise SystemExit(f"run failed: no results.json in {save}")
    cfg_file.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(ray_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
