# Contributing

This repo holds ML experiments: the WISDM centralised baseline, federated learning (FedAvg / FedBN) and the reporting site.
Most of what we change is not just code. It is data, splits, configs and results, so the git history has to answer
one question for anyone who reads it later: **which code and which data produced this number?**

The conventions below exist for that. They are lightweight, so follow them.

---

## 1. Workflow in short

1. Branch off an up-to-date `main` (see [§3 Branch names](#3-branch-names)).
2. Commit in small, single-purpose commits (see [§2 Commit messages](#2-commit-messages)).
3. Open a PR into `main`, wait for at least one review, then merge.
4. Delete the branch after merging.

Never push directly to `main`.

---

## 2. Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/) with **types adapted to ML work**.

### 2.1 Format

```
<type>(<scope>)<!>: <summary>

<body: what and why, optional>

<trailers, optional>
```

- **summary**: imperative mood, lower case, no trailing period, at most ~72 characters.
  `add fedprox strategy`, not `Added FedProx.` or `fedprox stuff`.
- **scope**: optional but recommended (see [§2.3](#23-scopes)).
- **`!`**: marks a **results-breaking** change (see [§2.4](#24-results-breaking-changes-)).

### 2.2 Types

| Type | Use it for | Example |
|---|---|---|
| `exp` | A new experiment, or changes to one: run matrix, hyperparameter grid, ablation | `exp(fl): add lstm-fedavg r80 run list` |
| `data` | Dataset download, preprocessing, windowing, splits, client partitions | `data(partitions): add s3_K5 label-skew split` |
| `model` | Model architecture (layers, init, normalisation) | `model(fcn): add dropout before classifier head` |
| `train` | Training loop, loss, optimiser, scheduler, FL strategy and aggregation | `train(fl): implement fedbn local batchnorm` |
| `eval` | Metrics, evaluation protocol, test-set handling | `eval: report per-client macro-f1 and worst client` |
| `results` | Committing produced outputs: `results.json`, `summary*.json`, report data | `results(fl): add s2_K5 fcn-fedavg-e3 runs` |
| `viz` | Plots, HTML reports, the `presented_outputs` site | `viz(site): add fl-vs-centralised gap chart` |
| `infra` | Cluster jobs, sbatch scripts, environment, dependencies, Flower config | `infra(fep): raise array job time limit to 4h` |
| `docs` | Experiment setup docs, `planning.md`, README, this file | `docs(fl): document non-iid split statistics` |
| `fix` | A bug fix in any of the above | `fix(data): stop test users leaking into fold 3 train` |
| `refactor` | A code change with **no** effect on outputs | `refactor(train): extract shared optimiser factory` |
| `perf` | Faster or leaner with identical outputs | `perf(data): cache windows as float16 npz` |
| `test` | Tests and sanity checks | `test(fl): assert K=1 fedavg matches centralised` |
| `chore` | Housekeeping: `.gitignore`, renames, cleanup | `chore: ignore slurm logs` |

If a change touches two types, it probably belongs in two commits.

### 2.3 Scopes

Use the part of the project that changed. Common scopes:

- **Pipelines**: `baseline`, `fl`
- **Data**: `wisdm`, `partitions`
- **Models**: `fcn`, `lstm`
- **Strategies**: `fedavg`, `fedbn`
- **Tooling**: `fep` (cluster / sbatch), `site`, `report`

Add a new scope when a new area appears. Keep it short and lower case.

### 2.4 Results-breaking changes (`!`)

In this repo, `!` does **not** mean a broken API. It means **results from before this commit can no longer be
compared with results from after it.** Typical causes:

- a change to preprocessing, windowing, normalisation or the label set
- a change to folds, test users or client partitions
- a change to a metric's definition
- a change to default hyperparameters that existing runs relied on

Add a `BREAKING-RESULTS:` trailer that says what is invalidated and what needs re-running:

```
data(wisdm)!: switch windows from 10 s to 5 s

Shorter windows double the sample count and match the setup in the reference paper.

BREAKING-RESULTS: all outputs/wisdm and outputs/wisdm_fl runs; re-run train.py and run_fl.py matrices
```

### 2.5 ML trailers

Add any of these at the end of the message, one per line, when they apply:

| Trailer | Meaning | Example |
|---|---|---|
| `Metrics:` | Headline numbers the commit produced or changed (mean ± std over folds × seeds) | `Metrics: test macro-F1 0.539 ± 0.023 (fcn), 0.532 ± 0.035 (lstm)` |
| `Config:` | The command, args or run-list file that reproduces it | `Config: wisdm_baseline/fep/fl_primary.txt` |
| `Code:` | For `results` commits: the commit hash of the code that produced the outputs | `Code: 746fb7d` |
| `Seeds:` / `Folds:` | Coverage when it is not the default (5 folds × 3 seeds) | `Seeds: 1,2` |
| `Refs:` | Research question, issue or PR | `Refs: RQ2, #4` |
| `BREAKING-RESULTS:` | See [§2.4](#24-results-breaking-changes-) | |

### 2.6 Full examples

```
train(fedbn): keep batchnorm statistics local to each client

FedBN aggregates every parameter except BatchNorm weights, biases and running stats.
LSTM has no BatchNorm, so fedbn is not run for it.

Refs: RQ2
```

```
results(fl): add primary matrix for s0/s1/s2 with K=5

Metrics: fcn-fedavg s0 <mean> ± <std>, fcn-fedbn s0 <mean> ± <std>
Config: wisdm_baseline/fep/fl_primary.txt
Code: 746fb7d
Refs: RQ1, RQ3
```

```
fix(eval): compute macro-f1 over classes present in the test fold

Classes missing from a fold were counted as F1 = 0 and pulled the mean down.
```

### 2.7 Rules for results

- Commit code and the results it produced **separately**: first `train`/`exp`/`fix`, then `results` with a `Code:` trailer.
- Do not commit results produced from uncommitted code.
- Never edit a `results.json` by hand. Re-run instead.
- Model weights (`*.pt`) and logs stay out of git (see `.gitignore`).

---

## 3. Branch names

### 3.1 Format

```
<owner>/<type>-<short-description>
```

- **owner**: your short handle (the same one every time), e.g. `robertv`, `sebi`, `andrei`.
- **type**: one of the commit types in [§2.2](#22-types). Use the type of the branch's **main** goal.
- **short-description**: kebab-case, 2 to 5 words, lower case, ASCII only. Put the experiment's key parameters in it
  when they matter (split, K, model, strategy).

### 3.2 Examples

| Branch | Purpose |
|---|---|
| `robertv/exp-fl-stage1` | First stage of federated experiments |
| `robertv/exp-fedprox-s2-k5` | FedProx on split S2 with 5 clients |
| `sebi/viz-output-comparisons` | Visualise outputs and comparisons |
| `andrei/data-s3-label-skew` | New label-skew partition |
| `andrei/fix-fold3-test-leak` | Fix test users leaking into training |
| `sebi/docs-contributing` | This file |

Avoid: `test`, `new-branch`, `robert-final-v2`, `Feature/FedProx`, names with spaces or diacritics.

### 3.3 Rules

- One branch = one experiment or one change. Do not pile unrelated runs onto one branch.
- Keep branches short-lived. Rebase onto or merge `main` before opening the PR if `main` has moved.
- A failed or abandoned experiment is still useful: open a PR with the results and a short note in
  the description, or keep the branch and say so in the team channel, rather than deleting it silently.

---

## 4. Pull requests

- **Title**: in commit-message format, e.g. `exp(fl): fedbn vs fedavg on s0-s3 with K=5`.
- **Description**: what changed, why, which RQ it answers, headline metrics and how to reproduce (command or run list).
- Mark results-breaking PRs clearly and list what has to be re-run.
- Update the relevant file in `docs/` when an experiment's setup changes.
