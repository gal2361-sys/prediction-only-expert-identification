# Prediction-Only Expert Identification via Disagreement Geometry

Code for the paper *A Framework for Prediction-Only Expert Identification in
Classification via Disagreement Geometry*.

The method identifies, for each class, a subset of "expert" models from a
heterogeneous pool using only the models' hard predictions on unlabeled data --
no ground-truth labels, validation performance, calibration, or model internals.
It builds a class-conditional disagreement geometry, selects a compact,
well-separated expert subset by a Dunn-index criterion, and aggregates the
experts by geometry-aware weighted voting. The repository also includes the
prediction-only failure diagnostics and the diagnostic-gated fallback.

## What you can reproduce, and from what

The pipeline has three stages joined by `data/`:

```
training/ --(GPU)--> data/predictions/*.json --(CPU)--> data/results/*.jsonl --> analysis/
```

- **From the shipped prediction JSONs** (`data/predictions/`): rerun the method
  and reproduce the results tables and the gating results. No raw datasets or
  retraining needed. Only 3 sample prediction files ship (one per modality); the
  full 24 are linked from `data/predictions/README.md`. The experiment notebooks
  run on whatever prediction files are present, so the 3 samples work out of the
  box.
- **From the shipped results JSONL** (`data/results/`): reproduce every figure
  and the detection table without rerunning the method.
- **Raw datasets are not shipped.** Regenerating predictions from scratch
  requires the training notebooks and the original data sources.

`data/results/failure_mode_points.json` is a small precomputed artifact used by
the failure-mode figures (the IMDb and CIFAR-10 centrality-vs-F1 plots); the
analysis notebook reads it directly, so those figures render without the
prediction files.

## Layout

```
src/               importable core (method, baselines, diagnostics, pipeline)
experiments/       thin notebooks that run the method over prediction JSONs
analysis/          notebooks that produce the figures, tables, and detection stats
training/          one notebook per modality that produces prediction JSONs
toy_example.ipynb  minimal worked example (no data needed)
figures_theory/    theory-validation figures (the Section 3 plots)
theorem3_sim_results_v5.json  cached Theorem 3 simulation results (loaded by theory_validation)
data/predictions/  prediction JSONs (input to experiments); see its README
data/results/      run records (input to analysis); see its README
requirements.txt
```

## src/ modules and where they appear in the paper

| Module | Contents | Paper |
| --- | --- | --- |
| `disagreement.py` | class-conditional set disagreement `d^(c)(m,m')` | Eq. (set disagreement), Sec. 4.1 |
| `dunn.py` | Dunn-index expert subset search | Eq. (Dunn subset), Sec. 4.2 |
| `voting.py` | expert-weighted voting (method); full-pool weighted and expert-majority votes (for analysis) | Sec. 4.3 |
| `baselines.py` | full-pool majority, Dawid--Skene EM, Spectral Meta-Learner | Sec. 5.3 baselines |
| `signals.py` | centrality dispersion (A1), under/over-bias (A2), separation gap | Sec. 7.4, App. bias signals |
| `gating.py` | signal-based class gating and capped full-ensemble fallback | Sec. 7.4 |
| `metrics.py` | per-class confusion and macro/micro F1 | Sec. 6 (macro-F1) |
| `io_utils.py` | prediction-JSON loader and evaluation-size schedule | App. data splits |
| `pipeline.py` | per-run loop tying the above together | Sec. 5 protocol |

## Notebooks and what they reproduce

| Notebook | Reproduces |
| --- | --- |
| `toy_example.ipynb` | the appendix worked example (expert sets E_1, E_2, E_3) |
| `experiments/run_experiment.ipynb` | the run records feeding the results tables |
| `experiments/run_gated_experiment.ipynb` | the gated run records (the fallback effect) |
| `analysis/results_tables.ipynb` | the results tables (vision, tabular, text) from the run records |
| `analysis/analysis_and_detection_plots.ipynb` | all Section 7 figures (centrality vs. quality, separation vs. rank, gap vs. gain, the IMDb / CIFAR-10 / Santander / QNLI failure-mode plots, and the A2 bias-vs-separation scatters) |
| `analysis/detection_quality.ipynb` | the failure-detection statistics table |
| `analysis/runtime_analysis.ipynb` | the per-dataset runtime table |
| `analysis/theory_validation.ipynb` | the Section 3 theory figures (ratio monotonicity, recovery vs. N, sample complexity, composition); loads the cached simulation results if present, recomputes otherwise |

## Running it

```bash
pip install -r requirements.txt
```

This installs the core stack (numpy, pandas, matplotlib), which is enough for the
experiment, gated, and analysis notebooks and the toy example. The training
notebooks need the heavier ML stack (commented at the bottom of
`requirements.txt`); uncomment it only to regenerate predictions from raw data.

Typical order:

1. `toy_example.ipynb` -- sanity check, no data required.
2. `experiments/run_experiment.ipynb` -- point it at `data/predictions/`,
   writes `data/results/experiment_records.jsonl`.
3. `experiments/run_gated_experiment.ipynb` -- writes
   `data/results/gated_records.jsonl`.
4. `analysis/` notebooks -- read the results JSONL and produce the results
   tables, figures, detection table, and runtime table.

The notebooks add `src/` to the path; run them from the repository root.

## Protocol notes

- Pools are fixed *a priori*: 15 models per modality (compositions in the paper
  appendix). Expert search uses subset sizes in `[3, 10]`.
- Evaluation sweeps 20 sizes per dataset with 3 trials each (60 runs), seeded by
  `s = 100 * size + trial`; subsamples are stratified without rebalancing.
- The training notebooks are one representative example per modality. Other
  datasets in a modality follow the same pool, sizing, and split protocol with a
  source-specific loader.
- Method timing covers expert search and weighted voting only; baselines and
  diagnostics are excluded from the reported runtime. Absolute times are
  indicative and vary with hardware and CPU load.

## Gating thresholds

The gated experiment exposes the detection thresholds as knobs, defaulting to the
values reported in the paper (centrality dispersion `< 0.035`; separation-gap
precondition `>= 0.535`; under-bias `>= 0.8`; over-bias `>= 2.0`), with the
fallback budget (beta in the paper) swept over {0.7, 0.8, 0.9, 1.0}, with 1.0 as the
reported setting. Tighten or loosen them to make detection stricter or looser.
