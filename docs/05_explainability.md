# 5. Explainability: where and why does a model fail?

Two scripts in [`explainability/`](../explainability/), adapted from the course tutorial's error analysis and
permutation importance to our (N, 200, 3) accelerometer windows (10 s at 20 Hz, x / y / z axes). The
tutorial's Grad-CAM is for images and is not part of this repo.

Both read a **run folder** (`results.json` + `final_model.pt` for federated runs, `model.pt` for centralised
ones). The weights are not in git: run the scripts where the models were trained, e.g. on fep with
`bash Fep/submit.sh command python explainability/...`.

**Which data?** `--data val` (default) = all clients' validation users together: diagnose and compare here.
`--data client_<i>` = one client (with `--split train|val|test`). `--data test` = the fold's test users, only
for the final numbers of a model you already chose.

## Error analysis

```bash
python explainability/error_analysis.py --run outputs/wisdm_fl/s2_K5/fcn-fedavg/fold0/seed1
python explainability/error_analysis.py --run outputs/wisdm/fcn/fold0/seed1 --partitions data/partitions/wisdm/s2_K5/fold0
python explainability/error_analysis.py --run <run> --data client_3 --split train   # suspicious training windows
```

Prints per-class precision / recall / F1 with the class each one is most confused with, accuracy and
macro-F1 per client and the worst users, and the hardest windows (highest loss). Writes
`errors_<data>.png` (confusion matrix + recall per class), `hardest_<data>.png` (the x / y / z traces of
the hardest windows: look for mislabelled or ambiguous segments, e.g. a "Jogging" window that looks like
walking, or a phone lying still during "Standing") and `errors_<data>.json`.

For FedBN runs `final_model.pt` holds the **averaged** BatchNorm layers (the "global BN" model): the
clients' own BN layers are not saved.

## Permutation importance

```bash
python explainability/feature_importance.py --run <run>                                # x vs y vs z
python explainability/feature_importance.py --run <run> --group-by segment --segments 10   # where in the window
python explainability/feature_importance.py --run <run> --group-by axis-segment          # heat map
python explainability/feature_importance.py --run <run> --untrained                       # sanity check
```

One part of the input (an axis, a time segment, or both) is shuffled between windows, which breaks its
link with the activity; the drop in macro-F1 (and accuracy) measures how much the model relies on it,
averaged over `--repeats` shuffles (± std = the noise of the method itself).

Hypotheses worth checking: the axis along gravity should separate the static activities (Sitting /
Standing / LyingDown); time segments should matter about equally for the FCN (global average pooling over
time), while the LSTM (classifies from its last time step) may lean on the end of the window.
The **untrained** model must give importances close to 0: if not, the differences you see are noise.
Compare clients (`--data client_<i>`), centralised vs. federated models, and FedBN vs. FedAvg.
