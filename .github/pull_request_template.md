## What

<!-- What changed and why, in 1-3 lines. Title in commit-message format, e.g. exp(fl): fedprox on s2 with K=5 -->

Fixes #
<!-- the planner issue, if there is one ("AITDM project planning"); otherwise delete the line -->

**Research question:** <!-- RQ1-RQ4 (docs/experiment_setup_federated.md) or the Stage 2 question, or n/a -->

## Results

<!-- Headline test macro-F1 as mean ± std over 5 folds x 3 seeds, paired difference to the reference
     (centralised / <model>-fedavg[-tag]) with wins/15 and Holm p, worst client if relevant; or "n/a". -->

## Reproduce

<!-- Run list (experiments/runs/<list>.txt) or command, the fep job ID(s), and the commit the runs used (COMMIT). -->

## Checklist

- [ ] Results are committed separately from the code, with `Code:` / `Config:` trailers; no hand-edited `results.json`
- [ ] Test users were not used for any decision (epoch / round, hyperparameters, choice of variant)
- [ ] Compared methods use the same splits, K, seeds and (fold, seed) pairs; only the intended thing differs
- [ ] Cluster results come from a clean commit (`COMMIT` without `-dirty`); no model weights or data committed
- [ ] Results-breaking? If yes, say what must be re-run:
- [ ] Docs updated (`docs/`, run lists) if an experiment's setup changed
