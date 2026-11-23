# 2. Running the experiments on fep with Slurm

> Before you start: [set up fep](01_environment_setup.md) and build the data (`bash Fep/submit.sh reproduce data`).

## 0. Slurm in five commands

| Command | What it does |
|---|---|
| `bash Fep/submit.sh JOB ...` | Submits one of our jobs (a file in [`Fep/jobs/`](jobs/)). Prints the job id and its log file |
| `squeue --me` | Your jobs: `PD` = waiting (the last column says why), `R` = running |
| `tail -f ~/AITDM_Project/slurm/aitdm-<job>-<id>.out` | Follows a log live (Ctrl+C stops *watching*, not the job) |
| `scancel <id>` | Stops a job (`scancel -u $USER` stops all of yours) |
| `sacct -X -o jobid,jobname,state,elapsed` | Your finished jobs and how long they took |

`Fep/submit.sh` is a thin wrapper around `sbatch`: it starts the job in the repository root and writes
the log to `~/AITDM_Project/slurm/`. Every job is named `aitdm-<job>` (e.g. `aitdm-fl_array`), so AITDM jobs
are easy to tell apart from your other projects' jobs: `squeue --me --name=aitdm-fl_array`, or
`sacct -X --format=JobID,JobName%28,State,Elapsed | grep aitdm-`. Any `sbatch` option **before** the job name overrides the job's
defaults: `-p ucsx --gres=gpu:1` (partition, GPU), `-t 04:00:00` (time limit), `-c 16` (CPUs),
`--array=0-8%2` (array tasks). Arguments **after** the job name go to the job.

## 1. Partitions: which queue can I use?

```bash
bash Fep/my_partitions.sh          # the partitions your account may use
sinfo -s                           # how busy they are (A/I/O/T = allocated/idle/other/total nodes)
```

| Partition | Hardware (per node) | Account `student` | We use it for |
|---|---|---|---|
| `haswell` | 32 CPUs, no GPU | yes | smoke test, data, summaries, `command` (starts quickly) |
| `ucsx` | 3 x A100 | yes | GPU runs when `dgxa100` is busy (`-p ucsx`) |
| `dgxa100` | 8 x A100 | yes, often busy | default of `centralised_array` and `fl_array` |
| `dgxh100` | 8 x H100 | yes, often busy | alternative GPU partition |
| `xl` | 2 x P100 | yes, but GPUs too old for current PyTorch | no |

Limits of the `student` account: **at most 2 jobs running at a time** (the others wait with
`(QOSMaxJobsPerUserLimit)`, hence `%2` in the array commands), **12 hours** per job, **3 GPUs** and
3 nodes per job. On haswell memory follows the CPUs (1.7 GB per CPU).

Measured on an A100: one federated run (FCN, K = 5, 40 rounds) about 50 s, LSTM about 80 s; one
centralised seed a few minutes.

## 2. The experiment matrices

Every experiment is "5 subject-wise folds x seeds 1, 2, 3". Finished `(fold, seed)` runs are skipped, so
a job that failed or hit its time limit can simply be submitted again.

**Stage 1, centralised baseline** ([setup](../docs/experiment_setup_centralised_baseline.md)): one array
task per (model, fold).

```bash
bash Fep/submit.sh centralised_array                 # tasks 0-9: fcn folds 0-4, lstm folds 0-4
bash Fep/submit.sh --array=3 centralised_array       # re-run one task
```

Results: `outputs/wisdm/<model>/fold<f>/seed<s>/{results.json, model.pt}`.

**Federated** ([setup](../docs/experiment_setup_federated.md)): one array task per line of a run list in
[`experiments/runs/`](../experiments/runs/) (a line = the arguments of
[`experiments/run_seeds.py`](../experiments/run_seeds.py)).

```bash
bash Fep/submit.sh --array=0 fl_array experiments/runs/fl_sanity.txt          # K = 1 check first
bash Fep/submit.sh --array=0-8%2 fl_array experiments/runs/fl_primary.txt     # 9 lines
bash Fep/submit.sh --array=0-10%2 fl_array experiments/runs/fl_secondary.txt  # 11 lines
# 2 GPUs per task: the 5 folds are spread over them (2 processes)
bash Fep/submit.sh --array=0-8%2 --gres=gpu:2 --cpus-per-task=20 --mem=64G fl_array experiments/runs/fl_primary.txt
```

The array range is `0-(number of non-comment lines - 1)`. Results:
`outputs/wisdm_fl/<split>_K<K>/<model>-<strategy>[-tag]/fold<f>/seed<s>/{results.json, final_model.pt}`.
Inside a task the K clients share the task's GPU (`num_gpus = 1/K` each).

**Summaries and reports** (CPU, a few minutes; the FL summary re-evaluates the centralised `model.pt`
files on every client's test users, so they must exist):

```bash
bash Fep/submit.sh reproduce summary
# = python experiments/summarize.py; python experiments/summarize_fl.py
```

They write `outputs/wisdm/{summary.json, report.html}` and `outputs/wisdm_fl/{summary_fl.json, report.html}`.

**Committing results** ([CONTRIBUTING.md](../CONTRIBUTING.md#27-rules-for-results)): commit the code first,
run, then commit the `results.json` / summaries in a separate `results(...)` commit with
`Code: <hash>` and `Config: experiments/runs/<list>.txt` trailers. Model weights stay out of git.

## 3. Stage 2: strategies, privacy, attacks

[`docs/04_aggregation_strategies.md`](../docs/04_aggregation_strategies.md) explains the options. In a run list:

```text
--split s2 --clients 5 --model fcn --strategy krum --config "malicious-clients='4' attack='sign-flip'" --tag signflip4
--split s2 --clients 5 --model fcn --strategy fedavg --config "dp-noise-multiplier=1.0" --tag dp1.0
```

[`experiments/runs/stage2_example.txt`](../experiments/runs/stage2_example.txt) is a starting point. Label-noise /
sensor-skew partitions are made once with `split_data.py` (CPU job) and selected with `--partition-tag`:

```bash
bash Fep/submit.sh command python data_splitting/split_data.py --splits s2 --clients 5 --label-noise 0,0,0,0,0.5
```

## 4. Ad-hoc runs: one fold with `flwr run`

For trying something on one fold (not for the reported numbers), the tutorial's `simulation` job runs
`flwr run` through [`scripts/run_simulation.sh`](../scripts/run_simulation.sh) (needs the `.venv`):

```bash
bash Fep/submit.sh simulation data/partitions/wisdm/s2_K5/fold0 "num-server-rounds=10 strategy='fedprox'"
bash Fep/submit.sh -p ucsx --gres=gpu:1 simulation data/partitions/wisdm/s2_K5/fold0
```

Every key of [`fl_app/pyproject.toml`](../fl_app/pyproject.toml) can be overridden this way. Results go to
`outputs/sim-<split>_K<K>-fold<f>-<timestamp>-job<id>/` (git-ignored). Inside a job Flower gets its own
ports, `FLWR_HOME` and Ray folder ([`scripts/_common.sh`](../scripts/_common.sh)), so several jobs can
share a node.

**Deployment** (every component a separate process, connected by IP address, as on real sites): a SuperLink and
one SuperNode per client, each a Slurm task of one job. `--ntasks` must be K + 1:

```bash
bash Fep/submit.sh deployment data/partitions/wisdm/s1_K5/fold0                 # 6 tasks (default)
bash Fep/submit.sh -n 4 deployment data/partitions/wisdm/s1_K3/fold0            # K = 3
bash Fep/submit.sh -N 1 -p ucsx --gres=gpu:1 deployment data/partitions/wisdm/s1_K5/fold0
```

Each SuperNode only gets the path of its own `client_<i>.npz` (`--node-config "data-path=..."`); component
logs go to `logs/<jobid>/`.

## 5. Writing your own job

Copy a file of [`Fep/jobs/`](jobs/), change the `#SBATCH` lines and the command, keep
`source Fep/jobs/_job_common.sh ...` (it checks the working directory, prints a header, activates the
environment and defines `"${PY[@]}"` for the `.venv` or the container), then
`bash Fep/submit.sh my_job ...`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Job waits with `(QOSMaxJobsPerUserLimit)` | 2 of your jobs are running; it starts when one ends |
| Job waits with `(Resources)` / `(Priority)` on dgxa100 | try `-p ucsx` or `-p dgxh100` |
| `run failed: no results.json` | read the traceback above it in the log; the finished runs are kept, resubmit |
| An array task ended with `TIMEOUT` | resubmit the same task (`--array=<i>`): finished runs are skipped |
| `ModuleNotFoundError: flwr` | the job did not find the environment: `.venv` (setup_fep.sh) or `export AITDM_IMAGE=...` |
| `Address already in use` | only for ad-hoc runs outside a job: `bash scripts/stop_flower.sh` |
| `summarize_fl.py`: `No such file ... model.pt` | the centralised models are missing: run `centralised_array` first (weights are not in git) |
