---
name: push
description: Commit and push the current changes following this repo's CONTRIBUTING.md (ML conventional commits, owner/type-description branches, code and results in separate commits). Use when the user runs /push or asks to commit and push their work.
argument-hint: "[optional hint, e.g. 'fedprox on s2' or 'pr' to also open a pull request]"
disable-model-invocation: true
---

# /push: commit and push by CONTRIBUTING.md

The user invoked this skill, so they are asking you to commit **and** push without confirming each step.
Stop and ask only in the cases listed under "When to stop and ask".

`CONTRIBUTING.md` at the repo root is the source of truth. **Read it in full first, every time.** If it disagrees
with anything below, CONTRIBUTING.md wins. Do not rely on a remembered copy of it.

Arguments: `$ARGUMENTS`
- Free text is a hint about what the work is (use it for the scope, summary and branch name).
- If it contains `pr`, also open a pull request at the end (step 7).

## 1. Look at the state

Run these and read the output:

```
git status --porcelain
git branch --show-current
git diff --stat HEAD
git diff HEAD
git log --oneline -10
git config --get aitdm.owner
```

For untracked files, look at what they are. Skim large data/results files instead of dumping them. If nothing has
changed and nothing is unpushed, say so and stop.

## 2. Check for things that must not be committed

Do not stage, and tell the user about:
- model weights (`*.pt`, `*.pth`, `*.ckpt`), `logs/`, `__pycache__/`, slurm `*.out` / `*.err`
- secrets (`.env`, tokens, keys)
- any single file over ~50 MB, unless it is a dataset or partition file of the kind already tracked under `data/`

If a `results.json` looks hand-edited (the diff changes a few numbers but no run was done), stop and ask.

## 3. Get on the right branch

Never commit on `main`.

- **Owner handle**: use `git config --get aitdm.owner`. If it is empty, look at the user's earlier branches
  (`git branch -a`, merged PR branch names in `git log --merges`). If you still cannot tell, ask once, then save it
  with `git config aitdm.owner <handle>` so you do not have to ask again.
- **If on `main`**: create `<owner>/<type>-<short-description>` from the changes (CONTRIBUTING.md §3), using the
  type of the main goal, kebab-case, 2 to 5 words, with split/K/model/strategy when they matter. Then
  `git switch -c <branch>`. Uncommitted changes move with you.
- **If on a feature branch**: stay on it, even if its name does not match the convention. If the changes clearly
  belong to a different experiment than the branch name says, mention it in the summary, but do not move them.

## 4. Plan the commits

Split the changes into single-purpose commits (CONTRIBUTING.md §2). For each one pick the type, scope and summary.

Rules that matter most here:
- **Code first, results second.** If both code (`*.py`, `*.sbatch`, run lists, templates) and outputs
  (`outputs/**`, `summary*.json`, `report.html`) changed, commit the code first, then the results in a separate
  `results(...)` commit with a `Code: <short hash of the code commit>` trailer.
- **Results trailers.** For a `results` commit, read the changed `summary*.json` (or the new `results.json` files)
  and add a `Metrics:` trailer with headline test macro-F1 as mean ± std. Use numbers you actually read from the
  files; never estimate them. Add `Config:` when a run list under `wisdm_baseline/fep/` or a command clearly
  produced the runs. Add `Seeds:` / `Folds:` only when coverage is not the default 5 × 3.
- **Results-breaking (`!`).** Consider it whenever the diff changes preprocessing, windowing, normalisation, labels,
  folds, test users, client partitions, metric definitions or default hyperparameters, e.g. `prepare_wisdm.py`,
  `make_fl_partitions.py`, the metric code in `train.py` / `run_fl.py`, or `data/**`. See "When to stop and ask".
- **`Refs:`** Add RQ numbers when the docs or the hint make the link clear (`docs/experiment_setup_*.md` list the RQs).
- Summary: imperative, lower case, no trailing period, at most ~72 characters. Add a body only when the *why* is
  not obvious from the summary.

## 5. Commit

For each planned commit, stage exactly its files (`git add <paths>`, never `git add -A` or `git add .`), then commit
with a heredoc so trailers and line breaks survive:

```
git commit -F - <<'EOF'
<type>(<scope>): <summary>

<body>

<trailers>
EOF
```

Do not use `--no-verify` or `--amend`. If a hook fails, fix the cause and make a new commit.
Run `git status` at the end to confirm nothing you meant to commit is left over.

## 6. Push

```
git push -u origin <branch>
```

Never push to `main` and never force-push. If the push is rejected because the remote branch moved, run
`git pull --rebase origin <branch>` once. If that has conflicts, stop and tell the user.

If `main` on the remote has moved ahead of the branch's base, mention it (CONTRIBUTING.md §3.3 asks for a rebase or
merge before the PR), but do not rebase unless the user asks.

## 7. Pull request (only if `pr` was passed)

Open it with `gh pr create --base main --head <branch>`:
- Title in commit-message format (CONTRIBUTING.md §4).
- Body: what changed, why, which RQ it answers, headline metrics, how to reproduce. Say clearly if it is
  results-breaking and what must be re-run.

If `gh` is missing or not logged in, print the title and body so the user can paste them.

## When to stop and ask

Ask (one short question, then carry on) only when:
- a change might be results-breaking and it is unclear whether it changes outputs (ask: `!` or not?);
- the owner handle cannot be found;
- something from step 2 was found and it is unclear whether the user meant to commit it;
- a push or rebase fails in a way you cannot safely resolve.

Everything else, decide yourself using CONTRIBUTING.md.

## Report

End with a short summary: branch name, each commit as `<hash> <subject>`, the push result (or PR link), and
anything you skipped or flagged.
