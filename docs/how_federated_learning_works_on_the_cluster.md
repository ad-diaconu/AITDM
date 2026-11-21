# How federated learning works on the cluster

A centralised run is simple: one job trains one model on all the data. Federated learning (FL) with Flower needs
a **server** and several **clients**: the clients train on their own data, the server combines their weights.
On fep this still runs as **one Slurm job**: the server and all clients live inside it. There are two ways to
arrange them.

## 1. Simulation: what our experiments use

Jobs: `fl_array` → [`experiments/run_seeds.py`](../experiments/run_seeds.py) → `flwr.simulation.run_simulation()`.

```text
one Slurm job (1 node, 1 GPU, 10 CPUs)
└── python experiments/run_seeds.py ...        one process
    ├── ServerApp (server_app.py)              global model, rounds, aggregation
    └── Ray (Flower's Simulation Engine)       K virtual clients
        ├── ClientApp, partition 0 → reads only client_0.npz
        ├── ClientApp, partition 1 → reads only client_1.npz
        └── ...                                each gets 1/K of the GPU, they train in parallel
```

**One round** (we run 40):

1. The server sends the current global weights to all K clients.
2. Each client trains 1 local epoch on its own users' windows and sends back only the weights and its number of
   windows.
3. The server aggregates them: FedAvg = weighted average (Stage 2: median, Krum, DP noise, ...).
4. Each client evaluates the new model on its own validation and test users and sends back confusion counts.
5. The server sums the counts into exact pooled metrics.

After the last round the server picks the best round on validation macro-F1 and writes `results.json` and
`final_model.pt`.

Good to know:

- **The data is never pooled.** A client only opens its own file; only weights and counts travel. The algorithm is
  exactly federated learning, even though everything runs on one machine. This is the standard way to do FL research.
- **Clients are not permanent processes.** Ray starts a worker whenever a client must train or evaluate. What a
  client must remember between rounds (FedBN: its own BatchNorm layers) is kept by Flower in its `context.state`.
- **One array task = 15 simulations** (5 folds × 3 seeds), one after another, or spread over 2 GPUs with
  `--gres=gpu:2`. One FCN run with K = 5 and 40 rounds takes about 50 s on an A100.
- **Why not one job per client?** The account may run only 2 jobs at once, and jobs on different nodes cannot easily
  find each other. One job holding the whole federation avoids both problems.

## 2. Deployment: closer to reality

Job: `deployment` ([`Fep/jobs/deployment.sbatch`](../Fep/jobs/deployment.sbatch)).

```text
one Slurm job, --ntasks = K + 1 (may span several nodes)
├── task 0: SuperLink            the server side, listens on a port
├── task 1: SuperNode 0          --node-config "data-path=.../client_0.npz"
├── ...
└── task K: SuperNode K-1
then `flwr run`: the ServerApp runs at the SuperLink, the ClientApps inside the SuperNodes
```

The server and every client are **separate processes**, possibly on different nodes, talking over the network by IP
address (gRPC), exactly as separate machines would. Each SuperNode only knows the path of its own file. The algorithm
and the results are the same as in simulation, but there are more moving parts and it is slower. It is useful to
demonstrate real FL or to test networking, not for the hundreds of runs of an experiment matrix.

## Summary

| | Centralised | FL simulation (experiments) | FL deployment |
|---|---|---|---|
| Slurm jobs per run | 1 | 1 (15 runs per array task) | 1 (K + 1 tasks inside) |
| Processes | 1 | 1 + Ray workers | 1 server + K clients |
| Who sees the data | the model sees all users | each client only its own users | each client only its own users |
| What moves | nothing | weights + counts, in memory | weights + counts, over the network |

Commands for both: [Fep/02_running_on_slurm.md](../Fep/02_running_on_slurm.md) (section 2: experiment matrices,
section 4: ad-hoc and deployment runs). What the federated experiments are:
[experiment_setup_federated.md](experiment_setup_federated.md).
