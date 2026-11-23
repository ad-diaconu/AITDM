"""Run federated WISDM experiments: one Flower simulation per (fold, seed).

    python experiments/run_seeds.py --split s1 --clients 5 --model fcn --strategy fedbn \
        --folds 0 1 2 3 4 --seeds 1 2 3

    # Stage 2: any aggregation strategy, central DP, attackers (docs/04_aggregation_strategies.md)
    python experiments/run_seeds.py --split s1 --clients 5 --model fcn --strategy krum \
        --config "malicious-clients='4' attack='sign-flip'" --tag signflip4

Results: outputs/wisdm_fl/<split>_K<K>[-<partition tag>]/<model>-<strategy>[-localbn][-<tag>]/fold<f>/seed<s>/
{results.json, final_model.pt}. Finished runs are skipped, so an interrupted job can simply be resubmitted.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FL_APP = REPO / "fl_app"
# Ray workers import the apps by module name: make the package importable for them too
os.environ["PYTHONPATH"] = os.pathsep.join([str(FL_APP), str(REPO), os.environ.get("PYTHONPATH", "")])
sys.path[:0] = [str(FL_APP), str(REPO)]
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

STRATEGIES = ["fedavg", "fedbn", "fedprox", "fedavgm", "fedadam", "fedyogi",
              "fedmedian", "fedtrimmedavg", "krum", "qfedavg"]
# set by this script itself, never through --config
OWN_KEYS = {"partitions-dir", "save-dir", "model", "strategy", "local-bn", "seed", "num-server-rounds",
            "local-epochs", "patience", "min-nodes"}


def parse_overrides(text: str) -> dict:
    """flwr-style run-config overrides: "dp-noise-multiplier=1.0 attack='noise'" -> dict."""
    out = {}
    for item in shlex.split(text, posix=False):
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--config: expected key=value, got {item!r}")
        out[key] = tomllib.loads(f'"{key}" = {value}')[key]
    bad = OWN_KEYS & out.keys()
    if bad:
        raise SystemExit(f"--config: set {sorted(bad)} with this script's own options instead")
    from aitdm_fl.task import DEFAULTS
    unknown = out.keys() - DEFAULTS.keys()
    if unknown:  # a typo would otherwise be ignored silently by the app
        raise SystemExit(f"--config: unknown key(s) {sorted(unknown)}; known: {sorted(DEFAULTS.keys() - OWN_KEYS)}")
    return out


def method_name(model: str, strategy: str, local_bn: bool, tag: str) -> str:
    name = f"{model}-{strategy}" + ("-localbn" if local_bn and strategy != "fedbn" else "")
    return name + (f"-{tag}" if tag else "")


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--partitions-root", type=Path, default=REPO / "data/partitions/wisdm")
    p.add_argument("--out", type=Path, default=REPO / "outputs/wisdm_fl")
    p.add_argument("--split", required=True)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--partition-tag", default="",
                   help="use <split>_K<K>-<tag> (e.g. a split_data.py --label-noise variant)")
    p.add_argument("--model", choices=["fcn", "lstm"], required=True)
    p.add_argument("--strategy", choices=STRATEGIES, required=True,
                   help="aggregation; 'fedbn' = FedAvg with BatchNorm kept on the clients")
    p.add_argument("--local-bn", action="store_true", help="FedBN-style local BatchNorm with any strategy")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--config", default="", help="extra run-config overrides, as for flwr run, e.g. "
                   "\"dp-noise-multiplier=1.0 malicious-clients='4'\" (keys: fl_app/pyproject.toml)")
    p.add_argument("--tag", default="", help="suffix of the experiment name, e.g. for ablations")
    p.add_argument("--gpus", type=float, default=None, help="GPUs for the simulation (default: all visible)")
    p.add_argument("--cpus", type=int, default=None)
    args = p.parse_args()
    overrides = parse_overrides(args.config)

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
    name = method_name(args.model, args.strategy, args.local_bn, args.tag)
    exp = f"{args.split}_K{args.clients}" + (f"-{args.partition_tag}" if args.partition_tag else "")
    cfg_file = Path(tempfile.gettempdir()) / f"aitdm_fl_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}.json"
    os.environ["AITDM_FL_CONFIG_FILE"] = str(cfg_file)

    for fold in args.folds:
        pdir = (args.partitions_root / exp / f"fold{fold}").resolve()
        if not (pdir / "meta.json").exists():
            raise SystemExit(f"missing partitions {pdir}; run data_splitting/split_data.py")
        for seed in args.seeds:
            save = (args.out / exp / name / f"fold{fold}" / f"seed{seed}").resolve()
            if (save / "results.json").exists():
                print(f"skip {save} (done)", flush=True)
                continue
            cfg = {"partitions-dir": str(pdir), "save-dir": str(save), "model": args.model,
                   "strategy": args.strategy, "seed": seed, "num-server-rounds": args.rounds,
                   "local-epochs": args.local_epochs, "patience": args.patience,
                   "min-nodes": args.clients, **overrides}
            if args.local_bn:
                cfg["local-bn"] = True
            cfg_file.write_text(json.dumps(cfg))
            print(f"=== {name} {exp} fold{fold} seed{seed} "
                  f"(backend {backend['client_resources']})", flush=True)
            run_simulation(server_app=server_app, client_app=client_app, num_supernodes=args.clients,
                           backend_config=backend)
            if not (save / "results.json").exists():
                raise SystemExit(f"run failed: no results.json in {save}")
            if os.environ.get("AITDM_COMMIT"):  # set by Fep/jobs/_job_common.sh: the code this run used
                (save / "COMMIT").write_text(os.environ["AITDM_COMMIT"] + "\n")
    cfg_file.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(ray_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
