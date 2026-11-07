# Experiment setup – Federated learning on WISDM Actitracker (FedAvg vs. FedBN)

> It builds on the centralised baseline ([`experiment_setup_centralised_baseline.md`](experiment_setup_centralised_baseline.md))
> and changes **only** where the data lives and how the model is trained. Data, folds, models, optimiser, metrics and test users
> are identical. All split statistics below were measured on `data/wisdm/windows.npz` (5 subject-wise folds).

---

## 1. Goal and research questions

**Task:** the same 6-class HAR task (10 s raw accelerometer windows, 20 Hz, subject-wise 5-fold CV),
trained by **K = 5 federated clients** (K = 3 as a sensitivity check) whose data is **non-IID**, with
**FedAvg** and **FedBN**, compared with the centralised baseline.

| # | Question | Measured by |
|---|---|---|
| RQ1 | How much does federated training cost compared with pooling the data? | Δ test macro-F1 (FL − centralised), paired over (fold, seed) |
| RQ2 | Does keeping BatchNorm local (FedBN) help under the per-user feature shift of WISDM? | FCN: FedBN vs. FedAvg, same split |
| RQ3 | How does the gap grow with the degree of non-IID-ness? | Gap across splits S0 → S1 → S2 (→ S3) |
| RQ4 | Is the federated model fair across clients? | Per-client macro-F1: worst client, std across clients |

**Centralised reference** (same folds / seeds / test users, `outputs/wisdm/summary.json`):
FCN 0.539 ± 0.023, LSTM 0.532 ± 0.035 test macro-F1; majority class 0.104.

---

## 2. What stays identical to the centralised baseline

| Item | Value (unchanged) |
|---|---|
| Data, cleaning, resampling, windows | `wisdm_baseline/prepare_wisdm.py` → `data/wisdm/windows.npz` (27 961 windows, 204 users) |
| Folds | The 5 outer subject-wise folds and their inner validation split (seed 42) |
| Train / val / test **users** per fold | Identical. Only their assignment to clients is new |
| Input normalisation | Per-channel z-score with the fold's **pooled training** mean/std (computed federatedly, §5) |
| Models | FCN (267 526 params, with BatchNorm) and 2-layer LSTM (200 966 params, no normalisation layers) |
| Optimiser | Adam, lr 1e-3, batch 64, gradient clipping 1.0 |
| Seeds | 1, 2, 3 (model initialisation, shuffling); split seed 42 |
| Metrics | Macro-F1 (primary), balanced accuracy, accuracy, per-class, confusion matrix, per-user accuracy |
| Selection | Best **validation** macro-F1 (epoch → *round*); test never used for decisions |

The union of all clients' test users **is** the fold's test set, so the centralised, FedAvg and FedBN
numbers are computed on exactly the same windows.

---

## 3. Federation design

**A client is a site**: an organisation that collects data from its own group of users (a clinic, a
company wellness programme, a study centre). This is the cross-silo setting for which K = 3–5 is realistic,
and it allows FedBN's per-client models to be evaluated on new users of the *same* site (§8).

* **K = 5 clients** (primary): ~26 training users per client, still enough users per client for the per-site
  model to be meaningful. **K = 3** (≈ 44 users per client) as a sensitivity run for S1 and S2.
* **Full participation**: all K clients train in every round (`fraction-train = fraction-evaluate = 1.0`,
  `min-nodes = K`).
* A fold's **train, validation and test users are all assigned to clients** with the same rule (§4), so every
  client owns `train`, `val` and `test` users. Train/val/test remain user-disjoint as in Stage 1.

---

## 4. Non-IID data splits

WISDM is non-IID in two ways. The split strategies control how much of each the clients see.

**Feature shift (per user, from phone placement/orientation).** In static postures, the mean gravity
vector differs ~10× more between users than within a user (measured, m/s²):

| Class | Users | Between-user std of mean (x, y, z) | Within-user std (x, y, z) |
|---|---|---|---|
| Sitting | 80 | 4.84, 3.26, 5.85 | 0.58, 0.45, 0.64 |
| Standing | 65 | 2.59, 7.14, 1.63 | 0.53, 0.46, 0.50 |
| LyingDown | 32 | 5.33, 1.44, 5.87 | 0.48, 0.35, 0.64 |

Keeping each user on one client therefore gives every client its own input distribution: the setting
FedBN was designed for.

**Label and quantity skew.** 135 of 225 users have a single activity, Stairs exists for 19 users only, and
users contribute between a few and thousands of windows.

### 4.1 Split strategies

| ID | Name | Rule | Role |
|---|---|---|---|
| **S0** | IID (window level) | Training windows shuffled and dealt evenly to K clients | Reference: no skew. A user's windows end up on all clients |
| **S1** | Natural (user level) | Users shuffled (seed 42), then packed largest-first onto the client with the fewest windows (as `partitioning.group_split`) | **Realistic**: feature shift, mild label skew, equal sizes |
| **S2** | Label-skewed users, Dirichlet α = 0.3 (user level) | Each user gets a *dominant class* (most windows). For every class c draw q_c ~ Dir(α·1_K); assign each user to client k ~ Categorical(q_{dominant class}) | **Main non-IID setting**: feature + label + quantity skew; users never split |
| S3 | Window-level Dirichlet α = 0.3 (optional stress test) | Per class c, q_c ~ Dir(α·1_K), split that class's windows by q_c (the tutorial's `--strategy dirichlet`) | Strong label skew, but each user spread over ~4 clients (weak feature shift) |

**Constraints (all splits):** every client gets ≥ 5 % of the fold's training windows and ≥ 3 training users.
If a draw violates this, redraw with seed 42 + attempt (deterministic). Splits are created once per fold and
reused for every method and seed. Validation and test users follow the same rule (S1: balanced packing,
S2/S3: the same q_c, S0: random users).

### 4.2 Measured non-IID-ness (training data, all 5 folds)

JSD = Jensen–Shannon divergence (base 2) between a client's label distribution and the fold's pooled one
(0 = identical, 1 = disjoint). Size CV = std/mean of client sizes.

| K | Split | JSD mean | JSD max | Missing classes per client (mean / max) | Size CV | Users per client |
|---|---|---|---|---|---|---|
| 5 | S0 IID | 0.000 | 0.001 | 0 / 0 | 0.00 | all (~128) |
| 5 | **S1 natural** | 0.011 | 0.034 | 0.08 / 1 | 0.00 | ~26 |
| 5 | S2 user Dirichlet α = 1.0 | 0.042 | 0.154 | 0.12 / 1 | 0.36 | ~26 |
| 5 | **S2 user Dirichlet α = 0.3** | 0.058 | 0.222 | 0.28 / 3 | 0.72 | ~26 |
| 5 | S3 window Dirichlet α = 0.3 | 0.252 | 0.700 | 1.76 / 4 (< 0.5 %) | 0.89 | ~3.9 clients per user |
| 3 | S1 natural | 0.006 | 0.014 | 0 / 0 | 0.00 | ~44 |
| 3 | S2 user Dirichlet α = 0.3 | 0.074 | 0.237 | 0 / 0 | 0.43 | ~44 |
| 3 | S3 window Dirichlet α = 0.3 | 0.120 | 0.228 | 1.13 / 3 (< 0.5 %) | 0.70 | ~2.8 clients per user |

Fold 0, K = 5 examples (% of the client's training windows):

| Split | Client | Windows | Users | Walk | Jog | Stairs | Sit | Stand | Lying |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 0 | 3 289 | 25 | 43 | 5 | 1 | 25 | 18 | 7 |
| S1 | 3 | 3 288 | 25 | 54 | 17 | 2 | 16 | 9 | 2 |
| S2 | 0 | 8 450 | 69 | 62 | 14 | 2 | 11 | 7 | 4 |
| S2 | 1 | 1 381 | 6 | 23 | 13 | 2 | 21 | 9 | 32 |
| S2 | 2 | 1 039 | 11 | 61 | 19 | 2 | 13 | 6 | **0** |
| S2 | 3 | 3 232 | 28 | 22 | 6 | 3 | 42 | 21 | 6 |

Takeaway: keeping users intact limits label skew (many users per client average it out). S1 is mostly feature
shift, S2 adds real label and quantity skew. Strong label skew (S3) is only possible by splitting users across
clients, which removes the feature shift. That is why S3 is an optional stress test and not the main setting.

`make_fl_partitions.py` (§9) writes these statistics and one label-distribution plot per split to `meta.json`
and `label_distribution.png`, as `split_data.py` does in the tutorial.

---

## 5. Input normalisation in a federation

The centralised baseline z-scores with the pooled training mean/std of the fold. In FL these are computed
**without sharing data**: each client sends per channel (n_k, Σx, Σx²) once before round 1; the server sums
them and returns mean and std. The values are identical to the centralised ones, so the preprocessing is
unchanged. In the simulation the same numbers are written into the partition files; document that this is
equivalent to one aggregation round (it reveals only three sums per channel).

---

## 6. Algorithms

### FedAvg (McMahan et al., 2017)

Round r: the server sends w_r. Each client k runs E local epochs of Adam on its training windows, and the
server averages w_{r+1} = Σ_k (n_k / n) w_r^k with n_k = number of training windows of client k.

* **BatchNorm under FedAvg:** the tutorial app sends the whole `state_dict`, so BN affine parameters **and**
  running statistics (`running_mean`, `running_var`) are averaged. This is standard FedAvg; keep it, but
  exclude the integer `num_batches_tracked` buffers from averaging (or cast them back to `int64`).
  They are unused with BN momentum 0.1.

### FedBN (Li et al., ICLR 2021)

Like FedAvg, except that **all BatchNorm layers stay local**: γ, β, `running_mean`, `running_var` (and
`num_batches_tracked`) of the 3 BN layers are never averaged. Each client keeps its own BN layers across
rounds and uses them for training and evaluation. Only the conv and linear weights are averaged.
Communication drops slightly: the 1 024 BN affine parameters (of the FCN's 267 526) and 1 024 running-statistic
values stay local.

* **Applies to FCN only.** The LSTM has no normalisation layers, so FedBN *is* FedAvg for it. Report the
  LSTM with FedAvg and state this explicitly. (A FedBN-style LSTM would need an architecture change, e.g.
  BatchNorm on the input. That is a new model that would also need a new centralised baseline, so it is
  out of scope.)

### Run matrix

| Model | FedAvg | FedBN |
|---|---|---|
| FCN | ✔ | ✔ |
| LSTM | ✔ | n/a (identical to FedAvg) |

---

## 7. Training configuration

| Setting | Value |
|---|---|
| Clients | K = 5 (primary), K = 3 (sensitivity), full participation |
| Rounds | max R = 40 |
| Local epochs | E = 1 (primary). Ablation E = 3 with R = 14 (same ≈ 40 passes over the data) |
| Client optimiser | Adam, lr 1e-3, batch 64, grad-clip 1.0. Re-initialised every round, as in the tutorial app (keep this; the optimiser state is not sent) |
| Aggregation weights | n_k = training windows of client k |
| Loss | Unweighted cross-entropy, as in the centralised baseline |
| Model selection | Round with the best **federated validation macro-F1** (§8.1); early stop after 10 rounds without improvement |
| Seeds | 1, 2, 3; per-client per-round shuffling seed = `derive_seed(seed, client, round)` (tutorial) |
| Compute budget | E · R ≈ 40 passes vs. ≤ 30 centralised epochs (FL usually needs more passes; selection makes the exact number irrelevant) |

Communication per run (upper bound, no early stop): 2 directions × K × R × model size =
2 × 5 × 40 × 1.07 MB ≈ **428 MB** for FCN, ≈ 322 MB for LSTM.

---

## 8. Evaluation protocol

### 8.1 Federated validation (for model selection, every round)

Each client evaluates the round's model on its validation users and returns the **36 confusion-matrix
counts**, not averaged scores. The server sums them, which gives exactly the pooled validation macro-F1 that
the centralised baseline uses, without sharing any data. (The tutorial's weighted average of per-client
balanced accuracy is **not** equal to pooled macro-F1, so it is not used for selection.)

* FedAvg: model = global model w_r.
* FedBN: model = w_r with the client's own BN layers.

### 8.2 Test evaluation (reported numbers)

In the same round, each client also evaluates on its test users and returns test confusion counts. Only
the counts of the **round selected on validation** are reported. This mirrors the centralised script, which
logs test metrics per epoch but reports the best-validation epoch.

| Method | Test model for a test user of client k | Test windows |
|---|---|---|
| Centralised (Stage 1) | the centralised model | all test users of the fold |
| FedAvg | global model w_r* | the same |
| FedBN (personalised) | w_r* + client k's BN layers | the same, routed to their client |
| FedBN (global-BN, secondary) | w_r* + BN parameters averaged over clients | the same: performance for a **new site** with no local BN |

* Pooled metrics: sum the confusion matrices over clients, then compute macro-F1 / balanced accuracy / accuracy.
* Per-client metrics: each client's test users, for **all** methods. The Stage 1 centralised models
  (`outputs/wisdm/<model>/fold<k>/seed<s>/model.pt`) are re-evaluated per client post hoc.
* Per-user accuracy, as in Stage 1.
* Learning curves: per round, pooled train loss (weighted by n_k), validation/test loss and macro-F1.

### 8.3 Comparisons and statistics

* Unit of comparison: (fold, seed) pairs → 15 paired values per method and split.
* Primary tests (Wilcoxon signed-rank, two-sided, Holm-corrected within each family):
  FedAvg vs. centralised, FedBN vs. centralised, FedBN vs. FedAvg (FCN), per split.
* Report mean ± std (ddof = 1) and the paired mean difference with its sign count (e.g. "+0.012, 11/15").

### 8.4 Results table template (one per K)

| Split | Model | Method | Val macro-F1 | Test macro-F1 | Δ vs. central | Test bal. acc. | Worst-client F1 | Client F1 std | Rounds to best | MB sent |
|---|---|---|---|---|---|---|---|---|---|---|
| – | FCN | Centralised | 0.569 | 0.539 | – | 0.548 | | | – | 0 |
| S1 | FCN | FedAvg | | | | | | | | |
| S1 | FCN | FedBN | | | | | | | | |
| S1 | LSTM | FedAvg | | | | | | | | |
| S2 | … | … | | | | | | | | |

---

## 9. Implementation plan

Copy the tutorial's Flower app into this repository (`fl_app/`, keep its attribution) and adapt it. Do not
change the tutorial clone.

| File | Change |
|---|---|
| `wisdm_baseline/make_fl_partitions.py` (new) | From `windows.npz`: per fold × split × K, write `data/partitions/wisdm/<split>_K<K>/fold<f>/client_<i>.npz` with `x_/y_/u_` for `train`, `val`, `test` (normalised float32, `(N, 200, 3)`), `test.npz` (union of the clients' test users), `meta.json` (classes, users per client, JSD, sizes, normalisation constants), `label_distribution.png`. Roughly 20 MB per fold directory, ~0.5–1 GB for all splits, folds and K |
| `fl_app/aitdm_fl/task.py` | `load_npz_arrays()`: return float32 `(N, T, C)` unchanged (no /255). `build_model(name)`: `fcn` / `lstm` from `wisdm_baseline/train.py`. Remove the MONAI import. `evaluate()` also returns the confusion matrix |
| `fl_app/aitdm_fl/client_app.py` | (1) **FedBN:** after `load_state_dict(global)`, overwrite the BN keys from `context.state["bn"]` (if present); after training, store the BN keys back into `context.state["bn"]`. Do the same in `evaluate`. (2) Evaluate on `val` **and** `test`; return the confusion counts (`val_cm_i_j`, `test_cm_i_j`) plus loss sums as `MetricRecord`. (3) Gradient clipping 1.0 |
| `fl_app/aitdm_fl/server_app.py` | `strategy = fedavg \| fedbn` (FedBN = FedAvg aggregation; the client ignores the global BN values). Custom `evaluate_metrics_aggr_fn` that **sums** the confusion counts. Track the best round on validation macro-F1, keep that round's global arrays in memory, early stop with patience 10. Save `final_model.pt` (best round), `metrics.json` (per-round curves) and `results.json` in the Stage 1 format (`val`, `test`, `per_client`, `history`). Skip `num_batches_tracked` when averaging |
| `fl_app/pyproject.toml` | New run-config keys: `model`, `strategy`, `patience`, `num-clients`. Defaults: `num-server-rounds = 40`, `local-epochs = 1`, `batch-size = 64` |
| `wisdm_baseline/run_fl.py` (new) | Loop over folds × seeds for one (split, K, model, strategy): `flwr run` simulation with `num-supernodes = K`, skip finished runs |
| `wisdm_baseline/fep/fl_array.sbatch` (new) | Slurm array over (split, model, strategy, fold), seeds inside the task |
| `wisdm_baseline/aggregate_fl.py` (new) | Combine with `outputs/wisdm/summary.json`: paired tests, per-client fairness, curves, an FL section in `report.html` (canvas `wisdm-results`) |

**FedBN state in simulation:** in Flower simulation a node's `context.state` persists between rounds. Verify
this in the smoke test (§11) before running the matrix. If it does not persist in your Flower version, store
each client's BN tensors in `<save-dir>/client_<i>_bn.pt` instead.

Outputs: `outputs/wisdm_fl/<split>_K<K>/<model>-<strategy>/fold<f>/seed<s>/{results.json, final_model.pt, client_<i>_bn.pt}`.

---

## 10. Running on fep

* **Environment:** a venv on top of the existing image, inheriting torch 2.9.1 and CUDA 12.6 (Python 3.11):
  `apptainer exec ~/NNProj/convnext.sif python -m venv --system-site-packages ~/.venvs/aitdm-fl`,
  then `pip install "flwr[simulation]==1.41.0" scikit-learn` (from the login node, PyPI is reachable).
  Run everything as `apptainer exec --nv ~/NNProj/convnext.sif ~/.venvs/aitdm-fl/bin/python ...`.
* **Partitions:** create them locally (`make_fl_partitions.py`) and `rsync` them to `~/AITDM_Project/AITDM/data/`.
* **Slurm:** partition `dgxa100` (the student account cannot use `ml`), `--gres=gpu:1`, 8 CPUs, at most
  2 jobs at a time (QoS). One GPU is shared by the K simulated clients (`client-resources: num-gpus = 1/K`).
* **Run count (primary):** 3 splits (S0, S1, S2) × 3 model/strategy rows × 5 folds × 3 seeds = **135 runs**.
  Secondary: S3 (45), K = 3 for S1 and S2 (90), E = 3 ablation on S2 for FCN (30).
* **Time:** a centralised run took 12–25 s on an A100. Expect about 1–3 min per FL run (Ray start-up plus
  K clients sharing one GPU), i.e. a few GPU-hours for the primary matrix. Measure it in the smoke test.

---

## 11. Sanity checks (before the full matrix)

- [ ] Union of clients' train/val/test users = the fold's users; no user on two clients (S1, S2); users disjoint across splits within a fold.
- [ ] Normalisation constants in the partitions equal the Stage 1 values of the fold.
- [ ] **S0 + FedAvg with K = 1** reproduces the centralised result (within seed noise). This checks the whole pipeline.
- [ ] FedBN: after round 2, client BN parameters differ between clients and equal each client's own last state; conv weights equal the global ones.
- [ ] Summed client confusion matrices give the same macro-F1 as evaluating the pooled arrays directly.
- [ ] The same seed gives the same metrics.

---

## 12. Hypotheses (to be tested, not results)

* H1: S0 and S1 FedAvg stay close to centralised: client label distributions are nearly identical
  (JSD ≤ 0.03), and centralised training also mixes users.
* H2: FedBN ≥ FedAvg on S1/S2 for **personalised** evaluation (local BN absorbs each site's orientation
  statistics), but FedBN with global BN is ≤ FedAvg for new sites.
* H3: The gap to centralised grows S0 < S1 < S2 < S3, mostly through rare classes (Stairs, LyingDown) and
  the small S2 clients (worst-client F1).

---

## 13. Threats to validity

* **Site = random group of users**: real sites differ more (populations, phone models, instructions). S1/S2
  probably *underestimate* real feature shift.
* **Personalised FedBN uses the test user's site membership.** This is realistic (a site serves its users), but it is a
  different deployment from one global model. Always report the global-BN number next to it.
* **Small S2 clients** (e.g. 6 users) make per-client metrics noisy. Report users per client with every per-client number.
* **Fresh Adam state per round** differs from centralised training: part of the FL gap may be an optimiser effect.
* **Federated normalisation** reveals per-channel sums (negligible, but say so). No differential privacy or secure aggregation in this stage.
* Same 15 (fold, seed) pairs everywhere: the tests are paired, but folds are not independent samples (CV overlap).

---

## 14. Deliverables of this stage

- [ ] `make_fl_partitions.py` + partitions, `meta.json` and label-distribution plots per split
- [ ] Adapted Flower app (`fl_app/`) with FedAvg / FedBN, confusion-count evaluation and best-round selection
- [ ] Smoke test passing §11 (including K = 1 ≈ centralised)
- [ ] 135 primary runs (+ secondary) on fep, `outputs/wisdm_fl/`
- [ ] Aggregated report: tables of §8.4, learning curves per round, confusion matrices, per-client fairness plots, communication cost
- [ ] Discussion of RQ1–RQ4

## References

* McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 (FedAvg).
* Li, Jiang, Zhang, Kamp & Dou, *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*, ICLR 2021.
* Hsu, Qi & Brown, *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, 2019 (Dirichlet label skew).
* Kairouz et al., *Advances and Open Problems in Federated Learning*, 2021 (cross-silo vs. cross-device, non-IID taxonomy).
* Flower documentation: <https://flower.ai/docs/framework/> (simulation, `Context.state`, strategies).
* Baseline experiment: [`experiment_setup_centralised_baseline.md`](experiment_setup_centralised_baseline.md).
