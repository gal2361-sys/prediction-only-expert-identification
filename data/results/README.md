# Results JSONL

Run records written by the experiment notebooks. Shipping these lets a reviewer
reproduce every figure and the detection table without rerunning the method.

## Files

- `experiment_records.jsonl` -- one record per (dataset, size, trial) from
  `experiments/run_experiment.ipynb`.
- `gated_records.jsonl` -- one record per (dataset, size, trial) from
  `experiments/run_gated_experiment.ipynb`.

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
