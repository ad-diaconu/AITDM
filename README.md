# AITDM – Federated Human Activity Recognition on WISDM

Project for **AI for Trustworthy Decision Making**: activity recognition (6 classes) from smartphone
accelerometer windows of the [WISDM Actitracker v2.0](https://www.cis.fordham.edu/wisdm/dataset.php) dataset,
trained centrally and federated with [Flower](https://flower.ai/) across simulated sites (clients = groups of users).

* **Stage 1:** centralised baseline (FCN and LSTM) vs. FedAvg / FedBN under IID, natural and Dirichlet
  label-skew splits with K = 1, 3, 5 clients; 5 subject-wise folds x 3 seeds, paired tests.
* **Stage 2:** other aggregation strategies (FedProx, FedAvgM, FedAdam/Yogi, FedMedian, FedTrimmedAvg,
  Krum, q-FedAvg), central differential privacy with ε reporting, simulated attackers, label noise and
  sensor skew, error analysis and permutation importance.

The structure follows the course starter kit
[AITDM_FLOWER_PROJECT_TUTORIAL](https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL),
adapted to WISDM. **The experiments run on the fep cluster: start with [Fep/README.md](Fep/README.md).**

## Quick start on fep

```bash
git clone https://github.com/ad-diaconu/AITDM.git ~/AITDM_Project/AITDM && cd ~/AITDM_Project/AITDM
bash Fep/setup_fep.sh                         # once (an existing clone: see Fep/README.md first!)
bash Fep/submit.sh smoke_test                 # the whole pipeline on synthetic data
bash Fep/submit.sh reproduce data             # WISDM archive -> windows -> partitions
bash Fep/submit.sh centralised_array          # Stage 1 centralised baseline
bash Fep/submit.sh --array=0-8%2 fl_array experiments/runs/fl_primary.txt
bash Fep/submit.sh reproduce summary          # summaries and reports
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Raw data | [`data_splitting/get_data.py`](data_splitting/get_data.py) | `data/raw/wisdm/WISDM_at_v2.0_raw.txt` |
| 2. Clean, resample to 20 Hz, 10 s windows, 5 subject-wise folds | [`data_splitting/prepare_wisdm.py`](data_splitting/prepare_wisdm.py) | `data/wisdm/{windows.npz, meta.json}` |
| 3. Federated partitions (S0 IID, S1 natural, S2 / S3 Dirichlet) | [`data_splitting/split_data.py`](data_splitting/split_data.py) | `data/partitions/wisdm/<split>_K<K>/fold<f>/` |
| 4a. Centralised baseline | [`experiments/centralised.py`](experiments/centralised.py) | `outputs/wisdm/<model>/fold<f>/seed<s>/` |
| 4b. Federated runs (one Flower simulation per fold x seed) | [`experiments/run_seeds.py`](experiments/run_seeds.py) + [`fl_app/`](fl_app/) | `outputs/wisdm_fl/<split>_K<K>/<method>/fold<f>/seed<s>/` |
| 5. Statistics and reports | [`experiments/summarize.py`](experiments/summarize.py), [`summarize_fl.py`](experiments/summarize_fl.py) | `outputs/wisdm*/summary*.json`, `report.html` |
| 6. Explanations | [`explainability/`](explainability/) | figures + JSON next to a run |

The full protocol: [centralised baseline](docs/experiment_setup_centralised_baseline.md),
[federated experiments](docs/experiment_setup_federated.md), [Stage 2 strategies](docs/04_aggregation_strategies.md),
[explainability](docs/05_explainability.md). How we commit code and results: [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository structure

```text
AITDM/
├── README.md                    <- you are here
├── CONTRIBUTING.md              <- commit / branch / results conventions
├── dirichlet_explorer.html      <- interactive Dirichlet distribution (tutorial; open in a browser)
├── requirements.txt             <- pinned dependencies (Flower 1.41, PyTorch 2.9.1, ...)
├── requirements-dev.txt         <- + pytest, ruff (tests / CI)
├── Fep/                         <- the fep cluster: setup, submit wrapper, Slurm jobs, guides
├── data_splitting/
│   ├── get_data.py              <- WISDM archive -> data/raw/wisdm/
│   ├── prepare_wisdm.py         <- cleaning, windows, subject-wise folds
│   ├── split_data.py            <- federated partitions (+ optional label noise / sensor skew)
│   └── partitioning.py          <- the split algorithms (pure NumPy)
├── experiments/
│   ├── centralised.py           <- centralised baseline, one fold, several seeds
│   ├── run_seeds.py             <- federated experiment: folds x seeds, any strategy
│   ├── summarize.py / summarize_fl.py / evaluation.py   <- statistics, paired tests, reports
│   ├── runs/                    <- run lists: one line = one fl_array task
│   ├── templates/               <- HTML report templates
│   └── reproduce.sh             <- re-creates everything (data | central | fl | summary)
├── fl_app/                      <- the Flower app (ServerApp, ClientApp, task.py: models / data / training)
├── explainability/              <- error_analysis.py, feature_importance.py
├── scripts/                     <- environment setup + Flower launch scripts used by the fep jobs
├── tests/                       <- unit tests, synthetic WISDM data, end-to-end smoke test (CI)
├── docs/                        <- experiment protocols and guides
├── outputs/                     <- committed results (results.json, summaries, reports; no *.pt)
└── data/        (git-ignored)   <- raw data, windows, partitions (on fep: a link to ~/AITDM_Project/data)
```

## Data

**No data is stored in this repository** (`data/` is git-ignored; it is regenerated by `data_splitting/`).
WISDM v2.0 is released by the [WISDM Lab](https://www.cis.fordham.edu/wisdm/) (Fordham University); cite the
papers listed in its `readme.txt` when you use it.

## Running locally (optional)

Everything also runs on a laptop with Python 3.11 – 3.13 (CPU is fine for the smoke test and single folds):

```bash
bash scripts/setup_env.sh --cpu && source .venv/bin/activate   # Windows: Git Bash, .venv/Scripts/activate
python -m pytest -q tests && bash tests/smoke_test.sh
```
