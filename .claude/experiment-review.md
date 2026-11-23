# Experiment review: shared context (AITDM)

Read by both review agents (`experiment-plan-reviewer`, before a run; `experiment-results-reviewer`, after it).
Keep it short and true: when a fact here changes (a baseline, a metric, a known pitfall), update it here.

## Where the ground truth lives

- Protocols, research questions, hypotheses, Stage 1 results: `docs/experiment_setup_centralised_baseline.md`,
  `docs/experiment_setup_federated.md` (RQ1–RQ4 §1, hypotheses §12, threats §13, results §14).
- Stage 2 options (strategies, DP, attacks, label noise, sensor skew): `docs/04_aggregation_strategies.md`;
  explainability: `docs/05_explainability.md`.
- Rules for commits and results (`results(...)` commits, `Code:` / `Config:` trailers, `!` = results-breaking):
  `CONTRIBUTING.md` §2.
- Code: `data_splitting/prepare_wisdm.py` (cleaning, windows, folds), `data_splitting/split_data.py` +
  `partitioning.py` (federated splits), `fl_app/aitdm_fl/task.py` (models FCN / LSTM, metrics, local training),
  `client_app.py` (FedBN, attackers), `server_app.py` (strategies, DP, best-round selection, `results.json`),
  `experiments/centralised.py`, `run_seeds.py`, `summarize.py`, `summarize_fl.py`, `evaluation.py`; run lists
  `experiments/runs/*.txt`; cluster jobs `Fep/jobs/*.sbatch`.
- Outputs: `outputs/wisdm/<model>/fold<f>/seed<s>/` (centralised), `outputs/wisdm_fl/<split>_K<K>[-<partition tag>]/
  <model>-<strategy>[-localbn][-<tag>]/fold<f>/seed<s>/` (federated), each with `results.json` (+ `COMMIT` for cluster
  runs); summaries `outputs/wisdm/summary.json`, `outputs/wisdm_fl/summary_fl.json`.

## Research questions (short)

Stage 1: RQ1 how much does FL cost vs. pooling the data? RQ2 does FedBN (local BatchNorm) help under WISDM's per-user
feature shift? RQ3 how does the gap grow with non-IID-ness (S0 → S1 → S2 → S3)? RQ4 is the federated model fair
across clients (worst client, spread)? Stage 2 (no numbered RQs in the docs yet: the plan must state its own
question and hypothesis): other aggregation strategies under label skew, robustness to attackers and noisy labels,
fairness (q-FedAvg, balanced loss), privacy (central DP, ε), explanations.

## Hard rules

1. **Test users decide nothing.** The epoch / round, every hyperparameter (`server-lr`, `proximal-mu`, `qfedavg-q`,
   `trim-beta`, DP clipping, rounds, local epochs) and the choice between variants are made on the **validation**
   users. Test numbers are reported, never used to pick. (Per-round test curves exist in `history` only for plots.)
2. **Subject-wise, fixed data.** No user in two of train / val / test (asserted in `prepare_wisdm.py`). User-level
   splits (S1, S2) keep every user on one client. Partitions are built with seed 42 and kept fixed: a comparison
   never regenerates them, and a new split or variant gets a new folder (`--partition-tag`).
3. **5 folds × seeds 1, 2, 3 = 15 paired runs, mean ± std (ddof 1).** Comparisons are paired over (fold, seed),
   Wilcoxon, Holm within one experiment (fairness = a separate family). A difference smaller than the run-to-run
   std, or not significant after Holm, is not a finding. Folds overlap (CV), so p-values are optimistic.
4. **One change at a time.** A comparison holds everything else fixed: same split, K, model, rounds, local epochs,
   seeds, partitions. References: the centralised model of the same architecture, and `<model>-fedavg[-tag]` of the
   same experiment (`summarize_fl.py` pairs them automatically; an attacked method needs the attacked FedAvg line
   with the same tag, and a clean line too).
5. **Committed code only.** A run's `COMMIT` must be a clean hash (no `-dirty`) for committed results. Never edit a
   `results.json` by hand; code and results go in separate commits (`Code:` trailer).
6. **Report the right numbers.** Macro-F1 (and balanced accuracy), not accuracy alone (WISDM is imbalanced: majority
   class macro-F1 0.104). Fairness: worst-client macro-F1 + spread, with users per client. Personalised FedBN always
   next to its global-BN number. DP: ε with δ and the number of rounds.
7. **Results-breaking changes** (preprocessing, windows, folds, partitions, metric definitions, app defaults in
   `task.py` DEFAULTS / `pyproject.toml`) need `!` and a re-run of everything they invalidate.

## Known numbers to compare against (test macro-F1, mean ± std over 15 runs, fep A100, Stage 1)

| What | FCN | LSTM |
|---|---|---|
| Majority class | 0.104 | 0.104 |
| Centralised (best val epoch, ≤ 30 epochs, patience 8) | 0.539 ± 0.023 (bal. acc 0.548, acc 0.747) | 0.532 ± 0.035 (bal. acc 0.539, acc 0.734) |
| FedAvg K = 1 (S0, R = 30): must ≈ centralised | 0.548 ± 0.031 (Δ +0.010, p 0.19) | – |

| Split, K = 5 (JSD) | FCN FedAvg | FCN FedBN (personalised) | FCN FedBN global BN | LSTM FedAvg R40 / R80 |
|---|---|---|---|---|
| S0 IID (0.000) | 0.536 ± 0.022 | 0.531 ± 0.021 | 0.537 ± 0.021 | 0.522 / 0.531 |
| S1 natural (0.011) | 0.545 ± 0.038 | 0.554 ± 0.032 | 0.542 ± 0.034 | 0.503 / 0.524 |
| S2 user-Dirichlet 0.3 (0.079) | 0.533 ± 0.048, worst client 0.135 | 0.538 ± 0.025, worst 0.280 | 0.528 ± 0.046, worst 0.132 | 0.527 / 0.515 |
| S3 window-Dirichlet 0.3 (0.163) | 0.492 ± 0.074 (−0.046 vs central, p_Holm 0.021) | 0.524 ± 0.060 | 0.489 ± 0.076 | 0.498 / – |

K = 3: S1 every method within ±0.015 of centralised; S2 FedAvg 0.499 ± 0.082, FedBN 0.557 ± 0.037 (+0.058 vs FedAvg,
14/15, p_Holm 0.004). S2 K = 5 with E = 3 / R = 14: FedAvg 0.532, FedBN 0.545 (n.s.). Centralised FCN worst client on
S2 K = 5 users: 0.161. A Stage 2 method is judged against these, on the same split and K.

## Known pitfalls

- **Federated runs are not bit-reproducible.** Ray returns the client replies in varying order, so the float sum of
  the aggregation differs (~1e-8 in round 1, ~1e-4 in final F1). Same code, same seed: tiny differences are noise,
  not a bug. CPU vs. GPU and laptop vs. cluster differ more: compare only numbers from the same platform.
- **Round budget.** With R = 40 the LSTM's best round was in the last 10 rounds in 33–60 % of the runs (check
  `hit_round_limit` in the summary): use R ≥ 80 for the LSTM. FedBN with E = 3 hit the limit in 100 %. A method that
  learns slower (FedProx with large μ, DP, q-FedAvg, small `server-lr`) can look worse only because of the budget.
- **Selection metric.** FCN's validation loss rises after ~10 rounds while macro-F1 stays flat: selection is on
  validation macro-F1 with patience 10 applied to the full curve. Do not switch to loss.
- **The pooled metric hides site damage.** On S2 K = 5 FedAvg's pooled F1 equals centralised while its worst client is
  0.135. Always look at per-client numbers and users per client (small S2 clients have ~6 users: noisy).
- **Personalised FedBN uses the test user's site.** Realistic for a site's own users, but a different deployment from
  one global model; global BN ≈ FedAvg everywhere, so the gain is the local BN layers.
- **Fresh Adam per round** differs from centralised training (part of any FL gap may be optimiser state).
- **DP changes more than the noise.** It switches to equal client weights (`unit-weight`) and clips updates, so even a
  tiny noise multiplier is a different aggregation; compare with a no-noise run of the same setting. ε is computed
  only for FedAvg-style aggregation (fedavg, fedbn, fedprox), is client-level, and with K = 5 the noise is divided by
  5 only: expect poor utility at meaningful ε (≤ 10). ε grows with the rounds.
- **Robust aggregation drops information.** With honest non-IID clients (S2, S3) FedMedian / Krum treat a skewed
  site as an outlier; Krum keeps one client's model. Always run the robust strategy **without** attackers too, and
  report the worst client. Krum with K = 5 tolerates f = 1 (`num-malicious`).
- **Attack size.** `attack-inflate-examples` only matters for size-weighted aggregation (FedAvg family); median /
  trimmed mean / Krum ignore it. Label noise flips only training labels (val / test stay clean, `meta.json` counts
  the flips); a poisoner with honest weights is diluted by K − 1 honest clients unless it inflates its size.
- **Adaptive server optimisers need tuning** (`server-lr` for FedAdam / FedYogi, on validation); q-FedAvg barely moves
  with a small `qfedavg-client-lr`. A "worse" result at default settings is a tuning result, not a method result.
- **summarize_fl needs the centralised weights.** It re-evaluates `outputs/wisdm/**/model.pt` (not in git) on every
  client's test users: run it where `centralised_array` ran.
- **Partitions across machines.** Regenerated partitions are bit-identical in every array; `meta.json` JSD values can
  differ in the last 1–2 digits across numpy versions (irrelevant).

## Cluster budget (fep, account `student`)

At most 2 running jobs for the whole account (shared with other projects). One FL run FCN K = 5, 40 rounds ≈ 50 s on
an A100, LSTM ≈ 80 s; a 15-run array task on 2 GPUs 6.5–10 min; one centralised seed a few minutes. `dgxa100` is
often busy (use `ucsx` or `dgxh100`); haswell CPU is several times slower. A wrong job costs a slot and the time
until someone looks at it.
