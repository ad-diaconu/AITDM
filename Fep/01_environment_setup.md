# 1. Environment setup on fep

| | |
|---|---|
| Where you type commands | the **login node** `fep.grid.pub.ro` (over SSH) |
| Where the training runs | a **compute node**, through a Slurm job (`bash Fep/submit.sh ...`) |
| Data | `~/AITDM_Project/data`, linked into the clone as `data/` |
| Results | `outputs/` in the clone (committed to git, except the model weights) |

> Never train on the login node. It is shared by everybody and long processes there are killed.
> Installing packages, copying data, editing files and short scripts are fine there.

## 0. Connect

```bash
ssh <your.username>@fep.grid.pub.ro
```

Your home folder is shared between the login node and all compute nodes, so whatever you create there is
visible to your jobs. Students have **50 GB**: a `.venv` takes about 6 GB, the WISDM data (windows + all
partitions) about 1.5 GB. Check with `du -sh ~/AITDM_Project ~/.cache`.

## 1. Clone and install (once, on the login node)

```bash
git clone https://github.com/ad-diaconu/AITDM.git ~/AITDM_Project/AITDM
cd ~/AITDM_Project/AITDM
bash Fep/setup_fep.sh
```

Already have a clone with a real `data/` folder? Follow
[Moving an existing fep clone](README.md#moving-an-existing-fep-clone-to-this-layout-do-this-before-git-pull) first.

The script

1. creates `~/AITDM_Project/` with `data/`, `logs/` and `slurm/`,
2. replaces `data/` and `logs/` of the clone with links to those folders (an existing `data/` is moved there),
3. creates the Python environment, one of two:

| | `bash Fep/setup_fep.sh` (default) | `bash Fep/setup_fep.sh --container` |
|---|---|---|
| What | `.venv/` in the clone from [`requirements.txt`](../requirements.txt) (as the tutorial) | the apptainer image `~/NNProj/convnext.sif` (PyTorch 2.9.1 + CUDA 12.6) plus the venv `~/.venvs/aitdm-fl` on top of it, with Flower 1.41 |
| When | a fresh account / clone | the environment every result in `outputs/` was produced with |
| Jobs use it | automatically | when `AITDM_IMAGE` is set: put `export AITDM_IMAGE=~/NNProj/convnext.sif` in `~/.bashrc` |
| Install time | about 15 min, 6 GB (PyTorch with CUDA) | a minute (only Flower is added) |

Both run every job; `simulation` and `deployment` (which start the `flwr` command line tools) need the
`.venv`. Both print `CUDA available: False` at the end of the install: expected, the login node has no GPU.
Other locations: `AITDM_PROJECT=/some/path`, `AITDM_IMAGE=...`, `AITDM_PYTHON=...` (set them when you
submit, too).

## 2. Check it on a compute node

```bash
bash Fep/submit.sh smoke_test                       # CPU (haswell)
bash Fep/submit.sh -p ucsx --gres=gpu:1 smoke_test  # GPU: the first line must say CUDA available: True
```

```text
Submitted job 301234
Log:    /export/home/.../AITDM_Project/slurm/aitdm-smoke_test-301234.out
Follow: tail -f /export/home/.../AITDM_Project/slurm/aitdm-smoke_test-301234.out     Cancel: scancel 301234
```

The smoke test runs the whole pipeline on a small synthetic dataset in the WISDM format: windows,
partitions (also with label noise and sensor skew), a centralised epoch, federated runs with FedAvg,
FedBN, Krum against a sign-flip attacker and differential privacy, both summaries, the explainability
scripts and one `flwr run`. The log must end with `>>> SMOKE TEST PASSED`.

## 3. Get the data

The raw archive `WISDM_at_latest.tar.gz` (WISDM Actitracker v2.0) is expected in `~/AITDM_Project/`.
Then one CPU job builds everything (the 9 partition sets are 1.3 GB of compressed `.npz`, so give it
time; files that exist already are kept):

```bash
bash Fep/submit.sh reproduce data
```

It runs, in order ([details](../docs/experiment_setup_centralised_baseline.md)):

```bash
python data_splitting/get_data.py --archive ~/AITDM_Project/WISDM_at_latest.tar.gz  # -> data/raw/wisdm/
python data_splitting/prepare_wisdm.py --input data/raw/wisdm/WISDM_at_v2.0_raw.txt --out data/wisdm
python data_splitting/split_data.py --splits s0 --clients 1 3 5     # and s1 s2 s3 with K = 3, 5
```

No archive yet? `python data_splitting/get_data.py --download` fetches it from the WISDM Lab (the job
does that by itself when the archive is missing).

## Editing the code

Edit on fep with `nano` / `vim`, or with VS Code's *Remote - SSH* extension connected to
`fep.grid.pub.ro`; or work locally, push a branch and `git pull` on fep (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
A job uses the files as they are **when it starts**: editing `task.py` while a job waits in the queue
changes what that job runs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: no .venv yet` | `bash Fep/setup_fep.sh`, or `export AITDM_IMAGE=~/NNProj/convnext.sif` to use the container |
| `Disk quota exceeded` during the install | `du -sh ~/* ~/.cache`; `rm -rf ~/.cache/pip` frees about 3 GB after the install |
| `CUDA available: False` on a GPU partition | add `--gres=gpu:1`: without it Slurm gives no GPU, even on a GPU node |
| `Invalid account or account/partition combination specified` | `bash Fep/my_partitions.sh` lists the partitions you may use ([partitions](02_running_on_slurm.md#1-partitions-which-queue-can-i-use)) |
| `data/` is an ordinary folder, not a link | run `bash Fep/setup_fep.sh` again: it moves the content to `~/AITDM_Project/data` and links it |
| `missing partitions .../fold0` | `bash Fep/submit.sh reproduce data` (or `split_data.py` for that split / K) |
