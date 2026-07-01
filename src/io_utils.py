"""
Input loading and evaluation-size utilities.

Loads the per-dataset prediction JSON produced by the training stage and defines
the test-size schedule used by the evaluation protocol. No fixed paths are baked
in: the location of the prediction files is supplied by the caller.

Prediction-JSON schema (one file per dataset)
---------------------------------------------
{
  "dataset": "<name>",
  "num_classes": <int>,
  "y_test": [<int>, ...],            # true labels for the test split
  "N_test_min": <int>,               # minimum evaluation size (coverage rule)
  "N_test_target": <int>,            # optional target evaluation size
  "models": {
     "<model_id>": {
         "y_pred": [<int>, ...],     # hard predicted labels
         "logits": [[...], ...],     # optional; only baselines that use scores need them
         "metrics": { ... }          # optional
     },
     ...
  }
}

The method itself uses only "y_pred"; "logits" are optional and consumed solely
by score-based baselines if present.
"""

import os
import json
import numpy as np


def load_dataset_predictions(json_path, enforce_num_models=None, sanity_check=True):
    """
    Load one dataset's prediction JSON.

    Parameters
    ----------
    json_path : str
        Path to the prediction JSON.
    enforce_num_models : int or None
        If set, warn when the model count differs and truncate to the first
        `enforce_num_models` (sorted) if there are too many.
    sanity_check : bool
        If True and logits are present, warn when argmax(logits) disagrees with
        the stored hard labels.

    Returns
    -------
    dict with keys:
        "dataset", "num_classes", "y_true",
        "logits"  : {model_id -> (N, C) array} (may be empty if absent),
        "predictions" : {model_id -> (N,) array},
        "metrics" : {model_id -> dict},
        "n_test_min", "n_test_target"
    """
    with open(json_path, "r") as f:
        obj = json.load(f)

    dataset_name = obj.get(
        "dataset",
        os.path.basename(json_path).replace("_test_predictions_logits.json", ""),
    )
    num_classes = int(obj["num_classes"])
    true_labels = np.asarray(obj["y_test"], dtype=np.int64)

    if "N_test_min" in obj:
        n_test_min = int(obj["N_test_min"])
        n_test_target = int(obj["N_test_target"]) if "N_test_target" in obj else 10 * n_test_min
    else:
        # Fallback: derive from the label distribution per the protocol,
        # N_test_min = ceil(20 / p_min), N_test_target = 10 * N_test_min.
        counts = np.bincount(true_labels, minlength=num_classes)
        p_min = counts[counts > 0].min() / len(true_labels)
        n_test_min = int(np.ceil(20.0 / p_min))
        n_test_target = 10 * n_test_min

    models_dict = obj["models"]
    model_names = sorted(models_dict.keys())

    if enforce_num_models is not None and len(model_names) != enforce_num_models:
        print(f"[WARN] {dataset_name}: found {len(model_names)} models, "
              f"expected {enforce_num_models}.")
        if len(model_names) > enforce_num_models:
            model_names = model_names[:enforce_num_models]
            print(f"       -> using first {enforce_num_models} models (sorted).")

    all_logits = {}
    all_predictions = {}
    model_metrics = {}

    for m in model_names:
        entry = models_dict[m]
        y_pred = np.asarray(entry["y_pred"], dtype=np.int64)
        all_predictions[m] = y_pred
        model_metrics[m] = entry.get("metrics", {})

        if "logits" in entry and entry["logits"] is not None:
            logits = np.asarray(entry["logits"], dtype=np.float64)
            all_logits[m] = logits
            if sanity_check:
                argmax_pred = logits.argmax(axis=1).astype(np.int64)
                if argmax_pred.shape == y_pred.shape:
                    mismatch = int(np.sum(argmax_pred != y_pred))
                    if mismatch > 0:
                        print(f"[WARN] {dataset_name} / {m}: y_pred differs from "
                              f"argmax(logits) on {mismatch} samples.")
                else:
                    print(f"[WARN] {dataset_name} / {m}: shape mismatch "
                          f"y_pred {y_pred.shape} vs argmax(logits) {argmax_pred.shape}.")

    return {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "y_true": true_labels,
        "logits": all_logits,
        "predictions": all_predictions,
        "metrics": model_metrics,
        "n_test_min": n_test_min,
        "n_test_target": n_test_target,
    }


def make_test_sizes(min_size, max_size, num_sizes=20):
    """
    Generate `num_sizes` evenly spaced, unique, integer test sizes in
    [min_size, max_size]. Used to sweep evaluation sizes per the protocol.
    """
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) > max_size ({max_size})")
    if min_size == max_size:
        return [int(min_size)]

    sizes = np.round(np.linspace(min_size, max_size, num=num_sizes)).astype(int)
    sizes = np.unique(sizes)
    sizes = sizes[(sizes >= min_size) & (sizes <= max_size)]
    return [int(s) for s in sizes]
