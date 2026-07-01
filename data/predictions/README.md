# Prediction JSONs

One JSON per dataset, named `<dataset>_test_predictions_logits.json`, produced by
the training notebooks (`training/`) and consumed by the experiment pipeline
(`experiments/run_experiment.ipynb`).

Shipping these lets a reviewer reproduce the method results without raw datasets
or retraining: the method runs entirely off model predictions.

## Sample vs. full set

To keep the repository light, only **3 sample prediction files** are shipped
here, one per modality (a small dataset on which the method performs well in each
modality): **CIFAR-10** (vision), **MultiNLI** (text), and **uci_letter**
(tabular). The experiment notebooks glob whatever files are present, so they run
as-is on these three.

The **full set of all 24 prediction JSONs** (about 2 GB total) is available here:

**https://drive.google.com/drive/folders/15klyTUmpGv_FCXXfonW0UB5ZLMChNYea**

To reproduce the complete results and figures, download the 24 files into this
`data/predictions/` folder (alongside or replacing the samples) and rerun the
experiment and gated-experiment notebooks; everything downstream follows from the
regenerated run records.

## Schema

```json
{
  "dataset": "<name>",
  "num_classes": <int>,
  "y_test": [<int>, ...],       // true labels for the held-out test split
  "N_test_min": <int>,          // minimum evaluation size (coverage rule)
  "N_test_target": <int>,       // target evaluation size (10 * N_test_min)
  "models": {
    "<model_id>": {
      "y_pred": [<int>, ...],   // hard predicted labels (required)
      "logits": [[...], ...],   // raw scores, no softmax (optional)
      "metrics": { "...": ... } // per-model metrics (optional)
    },
    ...
  }
}
```

## Notes

- The method uses only `y_pred`. `logits` are optional and unused by the
  method; they are retained for reference and any score-based extension.
- `N_test_min = ceil(20 / p_min)` and `N_test_target = 10 * N_test_min`, where
  `p_min` is the minimum class prevalence. The pipeline sweeps 20 evenly spaced
  evaluation sizes in `[N_test_min, N_test_target]`.
- Model ids are arbitrary strings; the pool is fixed per modality (15 models).
- Raw datasets are intentionally not shipped; only predictions are.
