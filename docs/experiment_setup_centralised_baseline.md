# Experiment setup – Centralised baseline on WISDM Actitracker (Conv1D vs. LSTM)

> Scope: **centralised training only**. Every choice below is made so that the same data files,
> splits, models and evaluation can be reused unchanged for the federated stage later.
> All dataset facts and counts in this document were measured on
> `~/AITDM_Project/WISDM_at_latest.tar.gz`; pilot numbers are from a prototype run of this protocol.

---

## 1. Goal and research question

**Task:** 6-class human activity recognition from raw 3-axis smartphone accelerometer windows.

**Question:** Under a *subject-independent* evaluation (the model is tested on people it never saw),
how do two off-the-shelf sequence models compare: a **1D-CNN (FCN)** and a **2-layer LSTM**?

**Output of this stage:** for each model, mean ± std of macro-F1, balanced accuracy and accuracy
over 5 subject-wise folds × 3 seeds, plus per-class and per-user results. These numbers are the
reference ("upper bound with pooled data") for every later federated experiment.

---

## 2. Dataset: verified facts

| Item | Value |
|---|---|
| Archive | `~/AITDM_Project/WISDM_at_latest.tar.gz` (note: in `~/AITDM_Project/`, not `~`) |
| Version | WISDM Actitracker **v2.0** (Oct 2013), Fordham WISDM Lab |
| File used | `WISDM_at_v2.0_raw.txt` (labeled, 167 MB, 3 005 411 lines) |
| Files **not** used | `*_unlabeled_raw.txt` (2.3 GB, label `NoLabel`), `*_unlabeled_transformed.arff` (labels are the *predictions* of WISDM's own model, so they would leak another model's errors into training), `*_transformed.arff` (46 hand-crafted features; optional sanity baseline only) |
| Format | `user,activity,timestamp_ms,x,y,z;` (trailing `;`, no header) |
| Sensor | phone accelerometer, m/s², includes gravity (±9.81 on the vertical axis at rest) |
| Nominal rate | 20 Hz (median Δt = 50 ms; 1 % of gaps ≥ 150 ms, some devices ~42 ms) |
| Users | 225 (labeled data) |
| Classes (6) | `Walking`, `Jogging`, `Stairs`, `Sitting`, `Standing`, `LyingDown` |
| Demographics | `WISDM_at_v2.0_demographics.txt`: height, sex, age, weight, leg injury (not used for training; useful later for fairness analysis) |
| Licence / citation | Cite Lockhart et al., SensorKDD 2011 (see `readme.txt`); include `readme.txt` if redistributed. **Never commit the data.** |

**Documentation inconsistencies to be aware of:** `raw_about.txt` lists `Upstairs`/`Downstairs`,
but the file only contains a merged `Stairs` class. Timestamps are milliseconds, not nanoseconds
(the example line in the docs is from WISDM v1.1).

### Class / user imbalance (after cleaning, 10 s windows)

| Class | Raw rows | Users with the class | Non-overlapping 10 s windows |
|---|---|---|---|
| Walking | 1 255 923 (42.1 %) | 189 | ~5 900 |
| Sitting | 663 706 (22.3 %) | 88 | ~2 860 |
| Jogging | 438 871 (14.7 %) | 48 | ~1 680 |
| Standing | 288 873 (9.7 %) | 68 | ~1 300 |
| LyingDown | 275 967 (9.3 %) | 35 | ~1 350 |
| Stairs | 57 425 (1.9 %) | **19** | **~280** |

* 135 of 225 users have only **one** activity; only 6 users have all six.
* Windows per user are very skewed (median ≈ 55, max ≈ 3 660 with 50 % overlap).
  → accuracy alone is misleading; **macro-F1 is the primary metric**.

---

## 3. Data cleaning (deterministic, in this order)

| # | Rule | Rows affected |
|---|---|---|
| C1 | Strip trailing `;`, parse as CSV with explicit column names | – |
| C2 | Drop empty rows (`579,,;` pattern) | 24 646 (4 users) |
| C3 | Drop **exact duplicate** rows | 227 690 (7.6 %) |
| C4 | Drop corrupt rows: any \|axis\| > 40 m/s², or timestamp outside [1e12, 1.5e12] ms (e.g. `5.9E20`, negative timestamps) | 563 (7 users) |
| C5 | Sort by `(user, timestamp)`; assert no duplicate `(user, timestamp)` remains | 0 |

Keep values with 20 < |a| ≤ 40 m/s² (9 664 rows): they are real peaks during jogging
on phones with a ±4 g range. Log the counts of every rule in `meta.json`.

---

## 4. Pre-processing: from rows to windows

1. **Segmentation.** A new segment starts whenever the user changes, the activity changes, or the
   time gap to the previous sample is **> 1 s**. A window never crosses a segment boundary,
   so it never mixes two activities or two recording sessions.
2. **Resampling.** Linearly interpolate each segment onto an exact 50 ms grid (20 Hz).
   This removes device-dependent jitter (42–52 ms) and short dropouts.
3. **Windowing.** Window length **T = 200 samples (10 s)**, the same length WISDM uses for its
   own transformed features (Kwapisz et al. 2010).
   * training windows: stride 100 (50 % overlap, ~2× data);
   * validation and test windows: stride 200 (no overlap, every second counted once).
   * Drop segments shorter than 10 s.
4. **Label** = the segment's activity (fixed order: `Walking, Jogging, Stairs, Sitting, Standing, LyingDown` → 0…5).
5. **Normalisation.** Per-channel z-score, mean/std computed on the **training windows of the fold
   only**, stored in `meta.json`, and applied to train/val/test. (Do not normalise per window:
   it removes the gravity direction, which is the main cue for Sitting/Standing/LyingDown.)
   For the later FL stage these constants are reused as fixed values (alternative with no
   data-dependent statistics: divide by 9.81 to get units of g).
6. **Array layout:** `x` = `float32` array of shape `(N, 200, 3)` (time, channel); `y` = `int64`;
   additionally store `user` per window (for per-user metrics and leakage checks).

Expected result: ~28 000 windows (with training overlap) from ~204 users.

---

## 5. Splitting protocol (the part that makes the baseline valid)

**Subject-wise 5-fold cross-validation** with `sklearn.model_selection.StratifiedGroupKFold`
(`groups = user`, `y = window label`, `shuffle=True`, `random_state=42`):

* Outer loop: fold *k* = **test users** (~20 %).
* Inner split on the remaining users: another `StratifiedGroupKFold(5, random_state=42)` →
  first fold = **validation users** (~16 % of all), rest = **training users** (~64 %).
* Train / val / test are **disjoint in users** (assert this in code).

Why not the tutorial defaults (`--test-fraction`, random `train_val_split`)?
They split *samples* at random, so overlapping windows of the same person land in train and test.
Pilot measurement (FCN, same hyper-parameters):

| Split | Users shared train/test | Test acc. | Test macro-F1 |
|---|---|---|---|
| Random window split | 199 of 204 | 0.864 | 0.791 |
| Subject-wise split | 0 | 0.707 | 0.534 |

The random split overestimates macro-F1 by ~0.26. Report it at most as a clearly-labelled
"leaky / subject-dependent" reference, never as the baseline.

Why 5 folds and not one 80/20 split? With one split the test set contained only **2 LyingDown users**
(and fold composition varies a lot). Measured composition of the 5 outer folds (seed 42):

| Fold | Test users | Test windows | Users per class (Walk, Jog, Stairs, Sit, Stand, Lying) |
|---|---|---|---|
| 0 | 43 | 3 697 | 38, 7, 4, 7, 13, 2 |
| 1 | 35 | 2 621 | 33, 11, 5, 13, 11, 7 |
| 2 | 43 | 2 599 | 30, 11, 2, 23, 17, 11 |
| 3 | 46 | 2 619 | 34, 10, 5, 25, 11, 6 |
| 4 | 37 | 2 632 | 34, 8, 3, 12, 13, 6 |

Every fold contains every class. Each user is a test user exactly once.

**FL compatibility.** Within each fold, group the training users and the validation users into
*N* clients (e.g. N = 5, greedy balanced packing as in `partitioning.group_split`), so each
`client_<i>.npz` has its own user-disjoint `train` and `val` part. The centralised baseline simply
pools all clients; the FL experiments later use the very same files.

---

## 6. Models ("off-the-shelf", fixed before seeing test results)

Input `(B, 200, 3)`. Both models have a similar size (~0.2–0.27 M parameters).

| | **Conv1D: FCN** (Wang, Yan & Oates, IJCNN 2017) | **LSTM** (2-layer, many-to-one) |
|---|---|---|
| Layers | Conv1d(3→128, k=8) – BN – ReLU → Conv1d(128→256, k=5) – BN – ReLU → Conv1d(256→128, k=3) – BN – ReLU → global average pooling → Linear(128→6) | LSTM(input 3, hidden 128, 2 layers, `batch_first=True`, dropout 0.2 between layers) → last time step → Dropout(0.5) → Linear(128→6) |
| Padding | `same` | – |
| Input transform | `x.transpose(1, 2)` → `(B, 3, 200)` | none |
| Parameters | 267 526 | 200 966 |
| CPU time / epoch (pilot, 16 threads, ~16 k windows) | ~9 s | ~27 s |

Note for later: FCN contains BatchNorm; under non-IID FL consider FedBN or GroupNorm. Keep BN for the
centralised baseline (it is part of the off-the-shelf architecture).

---

## 7. Training configuration (identical for both models)

| Setting | Value |
|---|---|
| Loss | Cross-entropy (main run); variant: class-weighted (`loss-weighting balanced`) |
| Optimiser | Adam, lr = 1e-3, no weight decay |
| Batch size | 64, shuffled each epoch with a seeded `torch.Generator` |
| Gradient clipping | global L2 norm 1.0 (needed for the LSTM, harmless for FCN) |
| Epochs | max 30; evaluate on **validation** after every epoch; keep the checkpoint with the best validation **macro-F1**; early stop with patience 8 |
| Seeds | 1, 2, 3 (model initialisation, dropout, shuffling); data split seed fixed at 42 |
| Runs | 2 models × 5 folds × 3 seeds = **30 runs** (+ variants) |
| Hardware | CPU is enough (~1 h FCN, ~3–4 h LSTM for all 15 runs); on fep use one GPU per job |

### 7.1 Data augmentation (ablation, not part of the main baseline)

The main baseline is trained **without** augmentation. Augmentation is a separate, clearly-labelled
variant. Rules: training batches only (never val/test), applied on raw m/s² **before** z-scoring,
seeded, identical for both models, chosen on validation macro-F1.

Pilot ablation (FCN, all 5 subject-wise folds, seed 1, 20 epochs, best-val checkpoint;
Δ = paired difference to no augmentation, test macro-F1):

| Augmentation (per window) | Val macro-F1 | Test macro-F1 | Δ test | Folds improved |
|---|---|---|---|---|
| none | 0.580 ± 0.034 | 0.546 ± 0.027 | – | – |
| axis sign flip (each axis × ±1, p = 0.5) | 0.605 ± 0.057 | **0.581 ± 0.045** | **+0.036** | 4/5 |
| scaling (per-axis gain ~ N(1, 0.1)) | 0.576 ± 0.022 | 0.562 ± 0.031 | +0.017 | 4/5 |
| jitter + scaling + rotation ±15° | 0.548 ± 0.035 | 0.557 ± 0.033 | +0.012 | 2/5 |
| rotation ±15° (random axis) | 0.562 ± 0.030 | 0.540 ± 0.040 | −0.006 | 2/5 |
| jitter (Gaussian, σ = 0.5 m/s² ≈ 0.05 g) | 0.560 ± 0.017 | 0.538 ± 0.034 | −0.007 | 1/5 |
| jitter + scaling | 0.569 ± 0.035 | 0.538 ± 0.022 | −0.008 | 2/5 |
| uniform random 3D rotation | 0.471 ± 0.040 | 0.445 ± 0.030 | **−0.101** | 0/5 |

Findings:

* **Full random rotation hurts badly** (Standing recall 0.64 → 0.19, LyingDown → 0): static postures
  differ only by the gravity direction, which an orientation-invariant model can no longer see. Do not use it.
* Jitter and small rotations: no measurable effect. Scaling: small gain, smaller than the fold-to-fold std.
* Axis sign flips are the only promising option (they mimic the phone being upside-down / reversed
  in the pocket; Sitting recall 0.67 → 0.85), but LyingDown gets worse (0.10 → 0.04). Note that
  a single-axis flip is a mirror image, not a physical rotation; flipping two axes at once
  (a 180° rotation) is the physically plausible version and worth testing.
* No augmentation fixes the two main failures (LyingDown↔Sitting, Stairs).
* With 5 folds × 1 seed no gain is statistically significant (a Wilcoxon test on 5 pairs cannot
  reach p < 0.05). Confirm with 3 seeds (15 pairs) and on the LSTM (not tested yet) before claiming an improvement.

**Hyper-parameter tuning (optional, small, fair):** only on fold 0's *validation* users, the same budget
for both models, e.g. lr ∈ {1e-3, 3e-4}, LSTM hidden ∈ {64, 128}, FCN unchanged. Freeze the choice,
then run all folds. Never look at test numbers while tuning.

---

## 8. Evaluation and reporting

**Metrics** (all on test users of each fold, using the best-validation checkpoint):

* **Primary:** macro-F1.
* Secondary: balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrix.
* Per-user accuracy (min / median over test users) → a fairness reference for the FL stage.
* Trivial references: majority class (always `Walking`: acc ≈ 0.34–0.49 depending on fold,
  macro-F1 ≈ 0.08–0.11) and, optionally, a classic model on the ARFF features.

**Aggregation:** report mean ± std (ddof = 1) over the 15 runs per model (and per fold over 3 seeds
in an appendix). Compare the models pairwise per (fold, seed) and use a Wilcoxon signed-rank
test on the 15 paired macro-F1 values.

**Results table template:**

| Model | Val macro-F1 | Test macro-F1 | Test bal. acc. | Test acc. | Worst-user acc. | Params | Train time |
|---|---|---|---|---|---|---|---|
| Majority class | – | | | | | 0 | – |
| FCN (Conv1D) | | | | | | 267 k | |
| LSTM | | | | | | 201 k | |
| FCN, random split (leaky, reference only) | | | | | | 267 k | |

### Full results (fep, `dgxa100` A100, Slurm job 294409, 2 × 5 folds × 3 seeds)

Produced by `wisdm_baseline/` (`prepare_wisdm.py` → `fep/train_array.sbatch` → `aggregate.py`);
interactive report: `outputs/wisdm/report.html` (also shown by the `wisdm-results` canvas).

| Model | Val macro-F1 | Test macro-F1 | Test bal. acc. | Test acc. | Median user acc. | Users < 50 % acc. | Params |
|---|---|---|---|---|---|---|---|
| Majority class (Walking) | – | 0.104 | 0.167 | 0.456 | – | – | 0 |
| FCN (Conv1D) | 0.569 ± 0.030 | **0.539 ± 0.023** | 0.548 ± 0.031 | 0.747 ± 0.029 | 0.963 | 19.9 % | 267 526 |
| LSTM | 0.570 ± 0.023 | 0.532 ± 0.035 | 0.539 ± 0.040 | 0.734 ± 0.025 | 0.942 | 21.0 % | 200 966 |

FCN − LSTM test macro-F1 = +0.007 (FCN better in 9/15 paired runs, Wilcoxon p = 0.45): **no significant
difference**. Per-class test F1 (FCN / LSTM): Walking 0.89 / 0.88, Jogging 0.93 / 0.90, Sitting 0.60 / 0.61,
Standing 0.69 / 0.66, LyingDown 0.08 / 0.09 (≈ 90 % predicted as Sitting), Stairs 0.03 / 0.06.
FCN's validation loss rises after the first epochs while its macro-F1 stays flat (over-confident predictions),
so selecting on macro-F1 rather than loss matters. FCN trains ~2× faster per run (12 s vs 25 s on an A100).

### Pilot results (fold 0, seed 1, 15 epochs, best validation epoch) – sanity reference only

| Model / variant | Test acc. | Test bal. acc. | Test macro-F1 |
|---|---|---|---|
| FCN | 0.707 | 0.576 | 0.534 |
| LSTM | 0.700 | 0.577 | 0.544 |
| FCN + balanced class weights | 0.693 | 0.571 | 0.531 |
| FCN, random window split (leaky) | 0.864 | 0.773 | 0.791 |

Observed in every subject-wise pilot run: **LyingDown is predicted as Sitting** (both are static,
they differ only in phone orientation, which varies between people) and **Stairs is almost never
recognised** (~350 training windows from ~12 users). Walking/Jogging F1 ≈ 0.87–0.90.
Expect your full results in this range; a much higher subject-wise score indicates a leak.

---

## 9. Required changes to the tutorial code

The tutorial pipeline (`AITDM_FLOWER_PROJECT_TUTORIAL`) does **not** support raw sequences out of the box:

| File / function | Problem | Change |
|---|---|---|
| `data_splitting/split_data.py` | Collapses each sequence into mean/std/min/max feature vectors; random sample-level test/val split | New script `data_splitting/prepare_wisdm.py` implementing §3–§5, writing `data/partitions/wisdm_fold<k>/client_<i>.npz` (`x_train, y_train, u_train, x_val, y_val, u_val`), `test.npz` (`x_test, y_test, u_test`) and `meta.json` (`num_classes`, `class_names`, `kind: "timeseries"`, normalisation mean/std, cleaning counts, user lists per split) |
| `fl_app/aitdm_fl/task.py` → `load_npz_arrays()` | A 3-D array is treated as grayscale images: divided by 255 and reshaped to `(N,1,H,W)` | Return `float32` 3-D arrays unchanged (images are `uint8`, so branch on `x.dtype == np.uint8` or on `meta.json["kind"]`) |
| `task.py` → `build_model()` / `model_from_test_set()` | Only MONAI CNN (images) or MLP (vectors) | Add `build_fcn()` and `build_lstm()`; select with a `model` argument (`"fcn"`, `"lstm"`) when the sample shape is 2-D |
| `experiments/centralised.py` | Fixed epochs, no per-epoch validation, no model choice | Add `--model {fcn,lstm}`, per-epoch validation with best-macro-F1 checkpoint + patience, gradient clipping, batch size 64; store `model` in `metrics.json` `run_config` |
| `experiments/evaluation.py` → `evaluate_model()` | Rebuilds the model without knowing which architecture | Read `run_config["model"]` from `metrics.json` and pass it to `model_from_test_set()`; add per-class report and per-user accuracy (from `u_val` / `u_test`) |
| `explainability/feature_importance.py` | Assumes 300 named features | Optional: permutation importance per axis (x/y/z) or per time segment |

`ArrayDataset` (augmentation only for 4-D images), `summarize.py` and `run_seeds.py` work unchanged.

---

## 10. Step-by-step procedure

```bash
# 0. Environment (tutorial venv: Python 3.12, torch 2.14, scikit-learn, pandas)
cd ~/AITDM_Project/AITDM_FLOWER_PROJECT_TUTORIAL   # or your project copy of it
source .venv/bin/activate

# 1. Extract ONLY the labeled data + docs (skip the 2.3 GB unlabeled file)
mkdir -p data/raw/wisdm
tar xzf ~/AITDM_Project/WISDM_at_latest.tar.gz -C data/raw/wisdm --strip-components=5 \
    --wildcards '*/WISDM_at_v2.0_raw.txt' '*/readme.txt' '*_about.txt' '*/WISDM_at_v2.0_demographics.txt'

# 2. Clean, window and split into 5 subject-wise folds (5 clients each, for later FL)
python data_splitting/prepare_wisdm.py --input data/raw/wisdm/WISDM_at_v2.0_raw.txt \
    --window 200 --train-stride 100 --eval-stride 200 --max-gap-ms 1000 \
    --folds 5 --num-clients 5 --seed 42 --out data/partitions/wisdm

# 3. Sanity checks (must pass before training, see §11)

# 4. Train both models on all folds, seeds 1 2 3
for k in 0 1 2 3 4; do
  for m in fcn lstm; do
    python experiments/centralised.py --partitions data/partitions/wisdm_fold$k \
        --model $m --epochs 30 --patience 8 --batch-size 64 --lr 1e-3 \
        --seeds 1 2 3 --name central-$m-fold$k
  done
done

# 5. Validation table (development), then the final table with test numbers once
python experiments/summarize.py outputs/experiments/central-*
python experiments/summarize.py outputs/experiments/central-* --with-test

# 6. Error analysis on validation data (per class, confusion matrix, hardest users)
python explainability/error_analysis.py --run outputs/experiments/central-fcn-fold0/seed1
```

On fep, run steps 2 and 4 as Slurm jobs (`bash Fep/submit.sh centralised ...`), never on the login node.

---

## 11. Sanity checks (automate them as asserts / unit tests)

- [ ] Cleaning counts match §3 (24 646 empty, 227 690 duplicates, 563 corrupt rows).
- [ ] `x.dtype == float32`, `x.shape[1:] == (200, 3)`, no NaN/Inf; labels in 0…5.
- [ ] Train, val and test **user sets are pairwise disjoint** in every fold; every user is a test user exactly once.
- [ ] Every class is present in train, val and test of every fold.
- [ ] Normalisation statistics come from training windows only (≈ mean [0.1, 2.9, 0.5], std [6.0, 8.0, 4.8] m/s²).
- [ ] An untrained model scores ≈ chance; a model can overfit 256 training windows to ~100 % (pipeline works).
- [ ] Majority-class baseline computed per fold.
- [ ] Re-running with the same seed gives identical metrics.

---

## 12. Threats to validity / limitations to state in the report

* **Label noise / ambiguity:** Sitting vs. LyingDown is not separable for many users with a pocket phone;
  report the confusion matrix and discuss it, do not hide it by merging classes (a 5-class
  merged variant can be an *additional* analysis).
* **Few Stairs users (19):** per-fold Stairs F1 has very high variance; report per-class std.
* **User imbalance:** a few users contribute thousands of windows; the window-level metrics are
  dominated by them → also report per-user metrics.
* **Self-reported labels** in "training mode" of the app; no video ground truth.
* **Phone placement/orientation** is uncontrolled (also the main source of non-IID-ness in the FL stage).
* Results are **not comparable** to papers that use WISDM v1.1 (36 users) or random window splits.

---

## 13. Deliverables of this stage

- [ ] `prepare_wisdm.py` + `meta.json` per fold (reproducible, seeded, cleaning log)
- [ ] `task.py` / `centralised.py` / `evaluation.py` changes from §9, with unit tests for the sanity checks
- [ ] 30 runs in `outputs/experiments/central-{fcn,lstm}-fold{0..4}/seed{1,2,3}/`
- [ ] Final table (§8), confusion matrices, per-class and per-user results, Wilcoxon test
- [ ] Short discussion: which model is better, where both fail (LyingDown, Stairs), and what this means for FL

## References

* Lockhart et al., *Design Considerations for the WISDM Smart Phone-Based Sensor Mining Architecture*, SensorKDD 2011 (dataset citation).
* Kwapisz, Weiss & Moore, *Activity Recognition using Cell Phone Accelerometers*, SensorKDD 2010 (10 s windows, transformed features).
* Weiss & Lockhart, *The Impact of Personalization on Smartphone-Based Activity Recognition*, AAAI-12 Workshop (subject-dependent vs. independent models).
* Wang, Yan & Oates, *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline*, IJCNN 2017 (FCN).
* Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997.
