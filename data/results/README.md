# Results JSONL

Run records written by the experiment notebooks. Shipping these lets a reviewer
reproduce every figure and the detection table without rerunning the method.

## Files

- `experiment_records.jsonl` -- one record per (dataset, size, trial) from
  `experiments/run_experiment.ipynb`.
- `gated_records.jsonl` -- one record per (dataset, size, trial) from
  `experiments/run_gated_experiment.ipynb`.
- `dbcc_records.jsonl` -- one record per (dataset, size, trial) from
  `experiments/run_dbcc_baseline.ipynb` (the DBCC baseline, same subsamples and seeds).

## Dataset keys

The `dataset` field holds the key under which the prediction file was produced. One key
does not name its dataset: `HEPMASS` denotes the credit card fraud detection dataset
(OpenML dataset 1597: 284,807 transactions, 0.17% fraudulent, 28 principal components
with time and amount). The tabular training notebook failed to load HEPMASS by name and
fell back to this dataset; the split, the training and all records under this key are of
the credit card data. The key is kept unchanged so that the records remain joinable; the
analysis notebooks print it as "Credit Card Fraud".

## experiment_records.jsonl

Each line is one run record:

```json
{
  "dataset": "<name>", "size": <int>, "trial": <int>, "seed": <int>,
  "subsample_hash": "<hex>",                 // identifies the exact subsample
  "expert_partition": {"<class>": ["<model_id>", ...]},
  "dunn_scores": {"<class>": <float>},
  "macro_f1": {"experts_weighted": <float>, "all_majority": <float>,
               "em": <float>, "sml": <float>,
               "all_weighted": <float>, "experts_majority": <float>},
  "per_class_f1": {"<method>": {"<class>": <float>}},   // for the gap-vs-gain plot
  "per_model_f1": {"<class>": {"<model_id>": <float>}}, // for centrality/rank plots
  "per_model_centrality": {"<class>": {"<model_id>": <float>}},
  "signals": {"<class>": {"centrality_dispersion": <float>,
                          "under_bias": <float>, "over_bias": <float>,
                          "separation_gap": <float>}},
  "timings": {"dunn_search": <float>, "weighted_voting": <float>,
              "method_total": <float>}
}
```

## gated_records.jsonl

```json
{
  "dataset": "<name>", "size": <int>, "trial": <int>, "seed": <int>,
  "subsample_hash": "<hex>",
  "ungated_macro_f1": <float>,
  "gated_macro_f1": {"<delta>": <float>},        // swept over delta
  "gated_classes": {"<class>": ["A1" | "A2-under" | "A2-over" | "empty", ...]},
  "num_gated": <int>
}
```

The detection analysis joins the two files on (dataset, size, trial) and verifies
`subsample_hash` matches, so each (record, class) has both its per-class F1 (loss
label) and its gate decision.

## dbcc_records.jsonl

```json
{
  "dataset": "<name>", "size": <int>, "trial": <int>, "seed": <int>,
  "subsample_hash": "<hex>",                 // same subsample as experiment_records.jsonl
  "num_classes": <int>, "num_models": <int>,
  "macro_f1": {"dbcc": <float>, "dbcc_stage1": <float>},   // published DBCC; stage 1 only (no correction loop)
  "per_class_f1": {"dbcc": {"<class>": <float>}, "dbcc_stage1": {"<class>": <float>}},
  "dbcc_experts": {"<class>": ["<model_id>", ...]},        // DBCC's expert set per class
  "dbcc_em_iters": <int>,                                  // correction-loop iterations run (cap 20)
  "time": {"dbcc": <float>, "dbcc_stage1": <float>}
}
```

`analysis/dbcc_comparison.ipynb` joins this file with `experiment_records.jsonl` on
(dataset, size, seed) and asserts that `subsample_hash` matches on every run.
