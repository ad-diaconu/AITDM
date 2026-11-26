# Backlog

Single source of truth for the [planner](https://github.com/users/ad-diaconu/projects/2). Edit here, then sync
with the `gh-backlog` skill; do not edit the issues on GitHub. Keep descriptions to one or two sentences.

**Status:** `todo` · `ready` · `in progress` · `in review` · `blocked` · `done` · `rejected`
**Priority:** P0 (required) · P1 (recommended) · P2 (optional)

Stage 1 (centralised baseline, FedAvg / FedBN on S0–S3) is done: protocols and results in
`docs/experiment_setup_centralised_baseline.md` and `docs/experiment_setup_federated.md`. Stage 2 options are in
`docs/04_aggregation_strategies.md` and `docs/05_explainability.md`. Every experiment is 5 subject-wise folds × seeds
1, 2, 3, compared paired with its reference (Wilcoxon, Holm), and decided on validation users only.

## Research questions

RQ1–RQ4 are the Stage 1 questions (`docs/experiment_setup_federated.md` §1). RQ5–RQ8 are proposed for Stage 2.

| RQ | Question | Experiments | Status |
|---|---|---|---|
| RQ1 | How much does federated training cost compared with pooling the data? | E1, E2, E3, E6, E7 | answered: nothing measurable while every user stays on one site (S0–S2, K = 5: FCN within ±0.015 of centralised, LSTM within 0.017 with R = 80); losses appear only with strong skew (E4, E5) |
| RQ2 | Does FedBN help under WISDM's per-user feature shift? | E3, E5, E12 | answered for the natural shift: personalised FedBN is never significantly worse and is better where the skew is strongest (S2 K = 3: +0.058 vs FedAvg; S2 K = 5 worst client 0.135 → 0.280); the gain is entirely the local BatchNorm (global BN ≈ FedAvg). E12 tests an explicit sensor shift |
| RQ3 | How does the gap grow with non-IID-ness (S0 → S3)? | E3, E4, E5 | partly: the pooled gap is ~0 for S0–S2 and −0.046 for S3; within S2 the damage shows per client, not in the pooled metric |
| RQ4 | Is the federated model fair across clients? | E3, E9 | answered for FedAvg / FedBN: FedAvg leaves the small, skewed S2 sites behind (worst client 0.135, below the centralised model's 0.161); E9 tests q-FedAvg and a class-balanced loss |
| RQ5 | Can a server-side strategy recover the non-IID loss better than FedBN? | E8 | open |
| RQ6 | How robust is the federation to a malicious or careless site? | E10, E11 | open |
| RQ7 | What does client-level differential privacy cost in macro-F1? | E13 | open |
| RQ8 | Why do LyingDown and Stairs fail? | E14, E15, E16, E17, E18, E19 | open. Hypothesis: besides phone orientation (Weiss & Lockhart 2012, 2014), windows recorded without gravity, mostly from user 1603 (739 of fold 0's 752 LyingDown test windows), drive the LyingDown → Sitting confusion |

## E. Experiments

| ID | Title | Description | Branch | P | Status |
|---|---|---|---|---|---|
| E0 | tutorial: the workflow in a few minutes | A demo, not a result: centralised FCN (5 epochs) and FedAvg / FedBN on S2 K = 5 (10 rounds), fold 0 and seed 1 only, on one GPU (bash Fep/submit.sh tutorial). Walk it from this row through both review agents to done, as in docs/tutorial.md. | `exp-tutorial-demo` | P2 | todo |
| E1 | centralised baseline: FCN vs. LSTM | Both models on 5 subject-wise folds × 3 seeds, best validation macro-F1: the reference for every federated run. FCN 0.539 ± 0.023, LSTM 0.532 ± 0.035, no significant difference. RQ1. | `stage1-experiments` | P0 | done |
| E2 | K = 1 sanity check | S0 with one client and FedAvg must reproduce the centralised result: 0.548 ± 0.031 vs. 0.539 (Δ +0.010, n.s.). RQ1. | `stage1-experiments` | P0 | done |
| E3 | FedAvg vs. FedBN on S0–S2, K = 5 | The primary matrix: FCN FedAvg / FedBN and LSTM FedAvg on the IID, natural and user-Dirichlet splits. Federated training costs nothing measurable, and FedBN doubles the worst S2 client (0.135 → 0.280). RQ1–RQ4. | `stage1-experiments` | P0 | done |
| E4 | strong label skew (S3, K = 5) | Window-level Dirichlet split: FedAvg loses 0.046 against centralised (2/15, p_Holm 0.021), FedBN recovers most of it (−0.014). RQ3. | `stage1-experiments` | P1 | done |
| E5 | K = 3 sensitivity | S1 and S2 with three clients: on S2 FedAvg drops to 0.499 and FedBN beats it by +0.058 (14/15, p_Holm 0.004). RQ2, RQ3. | `stage1-experiments` | P1 | done |
| E6 | local epochs E = 3 | S2 K = 5 with E = 3 and R = 14 (the same ~40 passes over the data): no change for FedAvg or FedBN. RQ1. | `stage1-experiments` | P2 | done |
| E7 | LSTM round budget (R = 80) | With R = 40 the LSTM's best round was often in the last 10 rounds; with R = 80 its gap on S0 / S1 shrinks to −0.000 / −0.008. RQ1. | `stage1-experiments` | P2 | done |
| E8 | non-IID strategies where FedAvg loses | FedProx, FedAvgM, FedAdam and FedYogi (and FedProx with local BN) on S3 K = 5 and S2 K = 3, where FedAvg loses 0.046 / 0.039; server-lr and proximal-mu tuned on fold 0's validation users. Does any server-side fix match FedBN? RQ5. | `exp-strategies-s3-s2k3` | P0 | todo |
| E9 | fairness: q-FedAvg and a class-balanced loss | q-FedAvg and loss-weighting='balanced' (with FedAvg and with FedBN) on S2 K = 5, judged on the worst client (FedBN: 0.280) and on Stairs / LyingDown F1; R = 80 for q-FedAvg. RQ4. | `exp-fairness-s2-k5` | P1 | todo |
| E10 | robustness to a sign-flip attacker | Client 4 sends sign-flipped updates on S2 K = 5: FedAvg vs. FedMedian, FedTrimmedAvg and Krum, each attacked and clean, so the cost of robustness for honest non-IID sites is measured too. RQ6. | `exp-signflip-s2-k5` | P0 | todo |
| E11 | noisy labels at one site | Client 4 with 50 % wrong training labels (split_data.py --label-noise 0,0,0,0,0.5) on S2 K = 5: FedAvg vs. FedMedian and FedTrimmedAvg against the clean split. RQ6. | `exp-label-noise-s2-k5` | P1 | todo |
| E12 | differently calibrated phones | S1 K = 5 with --sensor-skew (gain, offset and noise per client, client 0 clean): does FedBN's gain over FedAvg grow under an explicit feature shift? RQ2. | `exp-sensor-skew-s1-k5` | P1 | todo |
| E13 | central differential privacy | FedAvg on S2 K = 5 with noise multiplier 0.1 / 0.5 / 1.0, R = 40 and a check at R = 80: macro-F1 against ε at δ = 1e-5 (client-level guarantee, K = 5 only). RQ7. | `exp-dp-s2-k5` | P0 | todo |
| E14 | error analysis and axis importance | error_analysis.py and feature_importance.py (by axis and by segment, with the untrained control) on the centralised FCN and on FedAvg / FedBN of S2 K = 5: which axis separates the static postures, and what the hardest windows look like. RQ8. | `exp-explainability-s2-k5` | P1 | todo |
| E15 | audit of windows without gravity | 59 % of LyingDown and 45 % of Sitting windows have a mean below 3 m/s², most from user 1603. Characterise them (users, sessions, dynamic classes, WISDM's transformed.arff) and freeze the flat-window rule before any model is re-scored. RQ8. | `data-flat-window-audit` | P0 | todo |
| E16 | LyingDown with and without flat windows | Re-score the saved Stage 1 models (centralised, FedAvg, FedBN global BN) on flat vs. gravity windows, with a placebo that drops as many random gravity LyingDown windows. Is the near-zero LyingDown F1 mostly the flat windows? Needs E15 and B7. RQ8. | `eval-flat-windows` | P0 | todo |
| E17 | what the models predict for a flat window | Feed synthetic windows (flat at zero, flat with small noise, gravity on each axis) to the saved models and record the predicted class; Sitting is expected for flat input. RQ8. | `eval-flat-window-probe` | P2 | todo |
| E18 | retrain without flat windows | Only if E16 confirms the effect: a new data version without flat windows (opt-in flag, new folder), the centralised FCN and S2 K = 5 FedAvg / FedBN re-run on the same folds and seeds and reported next to the originals; also fixes model selection in folds 1–4, where user 1603 is a validation user. RQ8. | `exp-noflat-retrain` | P1 | blocked |
| E19 | personal fine-tuning on a few labelled minutes | Weiss & Lockhart cut the LyingDown → Sitting confusion more than tenfold with personal models from ~2 minutes per activity. Fine-tune the global model on a few windows of each test user and evaluate on the rest of that user's windows. RQ8. | `exp-personal-finetune` | P2 | todo |

## B. Build

The pipeline the experiments run on: data, models and the Flower app, cluster, statistics, Stage 2 tooling and
explanations. B7 unblocks E15–E18.

| ID | Title | Description | Branch | P | Status |
|---|---|---|---|---|---|
| B1 | data pipeline | WISDM archive → cleaning → 20 Hz, 10 s windows → 5 subject-wise folds (prepare_wisdm.py), and the federated partitions S0–S3 for K = 1, 3, 5 with label noise / sensor skew options (split_data.py). | `refactor-tutorial-structure` | P0 | done |
| B2 | models and the Flower app | FCN and LSTM, FedAvg and FedBN (local BatchNorm), evaluation by summed confusion counts, best round on federated validation macro-F1. | `refactor-tutorial-structure` | P0 | done |
| B3 | fep cluster and CI | Environment setup, submit wrapper, Slurm jobs (centralised_array, fl_array, smoke test), the sync / submit / status / fetch skills, unit tests and the end-to-end smoke test in CI. | `refactor-tutorial-structure` | P0 | done |
| B4 | statistics and reports | summarize.py / summarize_fl.py: mean ± std over 15 runs, paired Wilcoxon with Holm, worst client, per-user accuracy, HTML reports. | `refactor-tutorial-structure` | P0 | done |
| B5 | Stage 2 tooling | Aggregation strategies (FedProx, FedAvgM, FedAdam / Yogi, FedMedian, FedTrimmedAvg, Krum, q-FedAvg), central DP with ε accounting, simulated attackers, class-balanced loss. | `refactor-tutorial-structure` | P0 | done |
| B6 | explainability scripts | error_analysis.py (per class, per client, hardest windows) and feature_importance.py (permutation importance by axis and time segment). | `refactor-tutorial-structure` | P1 | done |
| B7 | flat-window tooling | One shared definition of a window without gravity (stored per window by prepare_wisdm.py, no change to the existing outputs) and a script that re-scores saved models on flat vs. gravity windows with the placebo control. Needed by E15–E18. | `eval-flat-window-tooling` | P0 | todo |
