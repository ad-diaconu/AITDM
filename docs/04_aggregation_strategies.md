# 4. Aggregation strategies, privacy and attacks (Stage 2)

Adapted from the course tutorial's
[docs/04_aggregation_strategies.md](https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL/blob/main/docs/04_aggregation_strategies.md)
to our WISDM app. The tutorial's DermaMNIST numbers are not repeated here: our numbers come from our
runs (`experiments/summarize_fl.py`).

After every round the server receives one model update per client and turns them into **one** global model.
That step is the *aggregation strategy*, the server's main lever for the trust dimensions of the project:

| Goal | Problem | Options in this repo |
|---|---|---|
| **Accuracy under non-IID data** | clients pull the model in different directions (*client drift*) | `fedprox`, `fedavgm`, `fedadam`, `fedyogi`; `fedbn` / `local-bn` (feature shift) |
| **Robustness** | one faulty or malicious client can wreck the average | `fedmedian`, `fedtrimmedavg`, `krum` |
| **Fairness** | the global model serves big or majority clients and neglects small ones | `qfedavg`, `loss-weighting='balanced'` |
| **Privacy** | the updates themselves can leak training data | DP wrapper (`dp-noise-multiplier > 0`) |

Everything is chosen at run time, no code changes. The strategies come from `flwr.serverapp.strategy` and are
built in `build_strategy()` in [`fl_app/aitdm_fl/server_app.py`](../fl_app/aitdm_fl/server_app.py); the
options and their defaults are listed in [`fl_app/pyproject.toml`](../fl_app/pyproject.toml).

## How to run them

A **reported** experiment (5 folds x 3 seeds) is one line of a run list, run on fep with `fl_array`
([Fep guide](../Fep/02_running_on_slurm.md#3-stage-2-strategies-privacy-attacks)); start from
[`experiments/runs/stage2_example.txt`](../experiments/runs/stage2_example.txt):

```text
--split s2 --clients 5 --model fcn --strategy fedprox
--split s2 --clients 5 --model fcn --strategy krum --config "malicious-clients='4' attack='sign-flip'" --tag signflip4
```

`--strategy` picks the aggregation, `--config` passes any other key of `pyproject.toml` (flwr syntax),
`--tag` names the variant. The output folder is `outputs/wisdm_fl/<split>_K<K>/<model>-<strategy>[-localbn][-tag]/`.
For a quick look at one fold: `bash Fep/submit.sh simulation data/partitions/wisdm/s2_K5/fold0 "strategy='krum'"`.

**Measuring the effect.** `experiments/summarize_fl.py` compares every method with `<model>-fedavg[-tag]` of
the same split and K (paired over the 15 fold x seed runs, Wilcoxon, Holm per experiment), and reports
per-client macro-F1, the **worst client** (fairness, a separate Holm family), per-user accuracy and the
communication cost. Report macro-F1 / balanced accuracy, never accuracy alone (WISDM is imbalanced).

---

## Notation

$K$ clients participate. In round $t$: $w_t$ is the global model; client $k$ has $n_k$ training windows,
$n = \sum_k n_k$, and returns its locally trained model $w_t^k$; its *update* is $\Delta^k = w_t^k - w_t$.

## 1. FedAvg and FedBN: the Stage 1 baselines

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \, w_t^k$$

Weighted average of the client models by their number of training windows (`num-examples`). Simple and
works well on IID data; weak under strong non-IID data, against a single malicious client, and large
clients dominate.

**FedBN** (`--strategy fedbn`, Li et al., 2021) aggregates like FedAvg but every client keeps its own
BatchNorm layers (statistics and affine parameters): they never leave the client and replace the averaged
values it receives. It targets *feature shift* (different phones, placements, people). `--local-bn` keeps BN
local with **any** other strategy (method name `...-localbn`). The LSTM has no BatchNorm, so FedBN = FedAvg
for it. Our server also evaluates the averaged-BN model (`...-globalbn` in the summary): what a new site
without its own BN statistics would get.

## 2. Strategies for non-IID data

**FedProx** (`fedprox`, Li et al., 2020): each client minimises its loss plus
$\frac{\mu}{2} \lVert w - w_t \rVert^2$. The server side is FedAvg; Flower sends `proximal-mu` to the clients and
`train()` in [`task.py`](../fl_app/aitdm_fl/task.py) adds the penalty. $\mu = 0$ is FedAvg.
`--config "proximal-mu=0.1"`.

**FedAvgM** (`fedavgm`, Hsu et al., 2019): momentum on the server, smoothing the zig-zag of heterogeneous
clients. `--config "server-momentum=0.9"`.

**FedAdam / FedYogi** (`fedadam`, `fedyogi`, Reddi et al., 2021): the averaged update is a pseudo-gradient for
Adam (Yogi) on the server, with learning rate `server-lr` (default 0.01). Often helps on heterogeneous
data but adds a hyperparameter: tune it on the validation users, never on the test users.

## 3. Robust (Byzantine-tolerant) aggregation

The mean has a breakdown point of 0: one extreme value can move it anywhere.

| Strategy | Rule | Tolerates | Paper |
|---|---|---|---|
| `fedmedian` | coordinate-wise **median** of the client models | < 50 % attackers | Yin et al., 2018 |
| `fedtrimmedavg` | per coordinate, drop the `trim-beta` largest and smallest values, average the rest | `trim-beta` on each side | Yin et al., 2018 |
| `krum` | pick the **single** client model closest to its $K - f - 2$ nearest neighbours | $f$ = `num-malicious`, needs $K > 2f + 2$ | Blanchard et al., 2017 |

With K = 5, Krum tolerates $f = 1$ (`num-malicious=1`, the default). **The price:** robust rules throw
information away. With honest but non-IID clients an "outlier" may simply be a site whose users do other
activities (our S2 / S3 splits), so robustness can cost accuracy and fairness: compare the worst client.

### Simulating attackers

Selected clients become attackers (`poison()` in [`client_app.py`](../fl_app/aitdm_fl/client_app.py)):

| `attack` | What the malicious client sends |
|---|---|
| `sign-flip` | $w_t - 5 \, \Delta^k$ (pushes the model 5x in the wrong direction) |
| `noise` | random Gaussian weights |
| `none` | its honestly trained weights (with wrong labels and / or an inflated size, below) |

`--config "malicious-clients='4' attack='sign-flip'"` (partition ids, comma-separated). In a deployment
job a single SuperNode can be flagged with `--node-config "data-path='...' malicious=true"`.

**Lying about the size.** FedAvg weights every client by the `num-examples` *the client itself reports*.
`attack-inflate-examples=100` makes the attackers claim 100x their size; FedMedian, FedTrimmedAvg and Krum
ignore the reported sizes.

**Data poisoning** (wrong labels instead of wrong weights): build partitions where some clients have
wrong training labels, and select them with `--partition-tag`:

```bash
python data_splitting/split_data.py --splits s2 --clients 5 --label-noise 0,0,0,0,0.5   # -> s2_K5-noise0-0-0-0-0.5
python experiments/run_seeds.py --split s2 --clients 5 --partition-tag noise0-0-0-0-0.5 --model fcn --strategy fedmedian
```

Only the training labels are flipped (the clients' validation and test users stay correct, so model
selection and evaluation are unaffected); `meta.json` records how many labels were flipped per client.
`--sensor-skew` gives every client a differently calibrated phone (gain, offset and noise per axis, client 0
clean): a feature-shift setting for FedBN.

## 4. Fairness

**q-FedAvg** (`qfedavg`, Li et al., 2020): clients with a higher loss get a larger weight, $\propto F_k^q$.
$q = 0$ is FedAvg; a larger $q$ pushes towards equal performance across clients at some cost in average
performance. `qfedavg-client-lr` (default 1.0) sets the step size (Flower estimates the Lipschitz
constant as $1 / \text{lr}$); a value as small as our Adam learning rate barely moves the model.

**Class-balanced loss** (`--config "loss-weighting='balanced'"`): every client weights its loss by the inverse
class frequency of its **own** training windows (no statistics leave the client). Helps rare activities
(Stairs, Sitting, Standing, LyingDown) under label skew.

## 5. Privacy: central differential privacy

`dp-noise-multiplier > 0` wraps **any** strategy in `DifferentialPrivacyServerSideFixedClipping`:

1. each client update is clipped to L2 norm $\lVert \Delta^k \rVert_2 \leq C$ (`dp-clipping-norm`),
2. Gaussian noise with standard deviation $\frac{\sigma C}{K}$ is added to the aggregate ($\sigma$ = `dp-noise-multiplier`),
3. clients are averaged with **equal** weights (`unit-weight`), as in DP-FedAvg.

For FedAvg-style aggregation (`fedavg`, `fedbn`, `fedprox`) Flower's Rényi-DP accountant computes the
privacy budget; $\varepsilon$ at $\delta$ = `dp-delta` is stored in `results.json` under `"privacy"`. Other
strategies get the noise without accounting (a `[DP]` message says so). The guarantee is **client-level**
(whether a whole site took part), not per user or per window.

The noise is divided by K = 5 only, so expect a hard privacy–utility trade-off; real DP federated learning
averages hundreds of clients per round, and $\varepsilon$ grows with every round (we run up to 40).
Ideas for the report: macro-F1 against $\varepsilon$, the clipping norm, fewer rounds, and local DP (DP-SGD on
every client, e.g. [Opacus](https://opacus.ai/)) for user-level privacy.

---

## Choosing a strategy (cheat sheet)

| Situation | Try first |
|---|---|
| IID or mildly non-IID, honest clients | `fedavg` |
| Strong label skew (S2, S3) | `fedprox`, `fedavgm`, `fedadam` (tune them) |
| Feature shift (different phones / people) | `fedbn`, `--local-bn` |
| Suspected faulty or malicious clients, noisy labels | `fedtrimmedavg`, `fedmedian`, `krum` |
| Small or unusual clients are neglected | `qfedavg`, `loss-weighting='balanced'` |
| Must hide which sites took part | DP wrapper ($\varepsilon$ is reported) |

**Baselines:** always compare with FedAvg and the centralised model on the same folds and seeds.

## Writing your own strategy

Subclass an existing strategy and override the step you want to change, then return it from
`build_strategy()` in `server_app.py` (keep passing `**common`: it carries our metric callbacks):

```python
from flwr.serverapp.strategy import FedAvg

class FedAvgWithLogging(FedAvg):
    """FedAvg that prints every client's training loss before aggregating."""

    def aggregate_train(self, server_round, replies):
        replies = list(replies)  # we iterate twice
        for msg in replies:
            if msg.has_content():
                m = msg.content["metrics"]
                print(f"round {server_round} | client {m['client']} | train_loss={m['train_loss']:.3f}")
        return super().aggregate_train(server_round, replies)
```

| Method | Runs | Typical use |
|---|---|---|
| `configure_train` | before each round | choose clients, per-round config (lr schedule, ...) |
| `aggregate_train` | after training | your own aggregation rule, outlier detection, reputation scores |
| `configure_evaluate` / `aggregate_evaluate` | evaluation phase | custom client-side metrics |

## References

* [McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), AISTATS 2017](https://arxiv.org/abs/1602.05629)
* [Li et al., *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*, ICLR 2021](https://arxiv.org/abs/2102.07623)
* [Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), MLSys 2020](https://arxiv.org/abs/1812.06127)
* [Hsu et al., *Measuring the Effects of Non-Identical Data Distribution* (FedAvgM), 2019](https://arxiv.org/abs/1909.06335)
* [Reddi et al., *Adaptive Federated Optimization* (FedAdam/FedYogi), ICLR 2021](https://arxiv.org/abs/2003.00295)
* [Yin et al., *Byzantine-Robust Distributed Learning* (median, trimmed mean), ICML 2018](https://arxiv.org/abs/1803.01498)
* [Blanchard et al., *Machine Learning with Adversaries* (Krum), NeurIPS 2017](https://arxiv.org/abs/1703.02757)
* [Li et al., *Fair Resource Allocation in Federated Learning* (q-FedAvg), ICLR 2020](https://arxiv.org/abs/1905.10497)
* [McMahan et al., *Learning Differentially Private Recurrent Language Models* (DP-FedAvg), ICLR 2018](https://arxiv.org/abs/1710.06963)
