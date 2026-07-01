"""
Experiment pipeline.

Ties the method, baselines, signals, and scoring together into a single run over
one dataset. For each evaluation size and trial, it:

  1. draws a stratified-free random subsample of the test predictions,
  2. builds the class-conditional prediction sets,
  3. identifies experts (Dunn search) and times the method,
  4. forms expert-weighted predictions and the baselines,
  5. scores every method with macro-F1,
  6. computes the prediction-only diagnostic signals.

The result is a list of per-run records (one dict per size/trial), which is the
input consumed by the analysis stage. Timing covers the prediction method only
(expert search + weighted voting); diagnostics and baselines are timed
separately and excluded from the reported method runtime.
"""

import time
import random
import hashlib
import numpy as np
from collections import defaultdict

from dunn import find_dunn_experts
from voting import (
    experts_weighted_voting, all_weighted_voting, majority_voting_experts,
    predictions_to_vector,
)
from baselines import majority_voting_all_models, em_dawid_skene, sml_predict
from signals import class_signals, model_centrality
from disagreement import class_disagreement_matrix
from metrics import summarize_method_performance


def hash_indices(idxs):
    """Short, stable hash of a subsample index array (reproducibility check)."""
    return hashlib.sha1(np.asarray(idxs).tobytes()).hexdigest()[:12]


def build_subsample_indices(n_total, size, seed):
    """Draw `size` distinct indices from range(n_total) without replacement."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=size, replace=False)


def build_prediction_sets(subsample_idx, model_pool, predictions, num_classes):
    """
    Build the class-conditional prediction sets on a subsample:
        sets[class][model] = list of local indices (0..size-1) the model assigns
        to that class.

    This is the representation consumed by the method, voting, and signals.
    """
    sets = {c: defaultdict(list) for c in range(num_classes)}
    for m in model_pool:
        preds_sub = predictions[m][subsample_idx]
        for local_i, cls in enumerate(preds_sub):
            sets[int(cls)][m].append(int(local_i))
    return sets


def per_model_class_f1(subsample_idx, predictions, y_true, num_classes):
    """
    Class-conditional F1 of every individual model on the subsample.

    This is required by the analysis plots, which rank models by their
    per-class F1 (e.g. centrality-vs-quality, separation-vs-rank). It is not
    part of the method itself and is excluded from method timing.

    Returns
    -------
    dict : {class -> {model_id -> f1}}.
    """
    subs_true = y_true[subsample_idx]
    out = {c: {} for c in range(num_classes)}
    for m, full_preds in predictions.items():
        yp = full_preds[subsample_idx]
        per_class = summarize_method_performance(subs_true, yp, num_classes)["per_class"]
        for c in range(num_classes):
            out[c][str(m)] = float(per_class[c]["f1"])
    return out


def run_single_subsample(subsample_idx, predictions, logits, y_true, num_classes,
                         min_subset_size=3, max_subset_size=10, seed=0):
    """
    Run the method and baselines on one subsample and return a result record.

    Parameters
    ----------
    subsample_idx : np.ndarray
        Indices into the full test set selected for this run.
    predictions : dict
        {model_id -> full (N,) hard-label array}.
    logits : dict
        {model_id -> full (N, C) array} or empty; only used by score-based code.
    y_true : np.ndarray
        Full (N,) true-label array.
    num_classes : int
    min_subset_size, max_subset_size : int
        Dunn search subset-size range.
    seed : int
        Seed used for this run (for deterministic tie-breaking in voting).

    Returns
    -------
    dict : a single run record (method/baseline macro-F1, expert partition,
           per-class signals, timings).
    """
    random.seed(seed)
    np.random.seed(seed)

    size = len(subsample_idx)
    model_pool = list(predictions.keys())
    subs_true = y_true[subsample_idx]

    # Build prediction sets (not part of the timed method).
    pred_sets = build_prediction_sets(subsample_idx, model_pool, predictions, num_classes)

    timings = {}
    t_method = time.time()

    # --- Expert identification (Dunn search) ---
    t0 = time.time()
    expert_partition, _, dunn_scores = find_dunn_experts(
        pred_sets, min_subset_size=min_subset_size, max_subset_size=max_subset_size
    )
    timings["dunn_search"] = float(time.time() - t0)

    # --- Expert-weighted voting (the method's aggregation) ---
    t0 = time.time()
    expW_cls, _ = experts_weighted_voting(expert_partition, pred_sets)
    y_expert_weighted = predictions_to_vector(expW_cls, size)
    timings["weighted_voting"] = float(time.time() - t0)

    timings["method_total"] = float(time.time() - t_method)

    # --- Baselines (timed separately, excluded from method runtime) ---
    allM_cls = majority_voting_all_models(pred_sets)
    y_all_majority = predictions_to_vector(allM_cls, size)

    subs_preds = {m: predictions[m][subsample_idx] for m in model_pool}
    y_em, _, _, _ = em_dawid_skene(subs_preds, num_classes)
    y_sml, _, _, _ = sml_predict(subs_preds, num_classes)

    # --- Analysis-only aggregations (for the gap-vs-gain plot) ---
    allW_cls, _ = all_weighted_voting(pred_sets, num_classes)
    y_all_weighted = predictions_to_vector(allW_cls, size)
    expM_cls = majority_voting_experts(expert_partition, pred_sets)
    y_experts_majority = predictions_to_vector(expM_cls, size)

    # --- Scoring (macro-F1 headline; per-class F1 retained for analysis) ---
    performance = {
        "experts_weighted": summarize_method_performance(subs_true, y_expert_weighted, num_classes),
        "all_majority":     summarize_method_performance(subs_true, y_all_majority, num_classes),
        "em":               summarize_method_performance(subs_true, y_em, num_classes),
        "sml":              summarize_method_performance(subs_true, y_sml, num_classes),
        "all_weighted":     summarize_method_performance(subs_true, y_all_weighted, num_classes),
        "experts_majority": summarize_method_performance(subs_true, y_experts_majority, num_classes),
    }

    # Per-class F1 of each aggregation method (gap-vs-gain plot needs these).
    per_class_f1 = {
        method: {int(c): perf["per_class"][c]["f1"] for c in range(num_classes)}
        for method, perf in performance.items()
    }

    # --- Prediction-only diagnostic signals (per class) ---
    signals_per_class = {}
    for cls in range(num_classes):
        E_c = expert_partition.get(cls, ())
        signals_per_class[cls] = class_signals(pred_sets, E_c, cls, model_pool, num_classes)

    # --- Per-model class-conditional F1 and centrality (for the analysis plots) ---
    per_model_f1 = per_model_class_f1(subsample_idx, predictions, y_true, num_classes)
    per_model_cent = {}
    for cls in range(num_classes):
        sets_c = {m: pred_sets.get(cls, {}).get(m, []) for m in model_pool}
        dist_mat = class_disagreement_matrix(sets_c)
        cent = model_centrality(dist_mat)
        per_model_cent[int(cls)] = {str(model_pool[i]): float(cent[i]) for i in range(len(model_pool))}

    return {
        "size": int(size),
        "seed": int(seed),
        "subsample_hash": hash_indices(subsample_idx),
        "expert_partition": {int(c): list(map(str, v)) for c, v in expert_partition.items()},
        "dunn_scores": {int(c): float(v) for c, v in dunn_scores.items()},
        "macro_f1": {k: v["macro_f1"] for k, v in performance.items()},
        "per_class_f1": per_class_f1,
        "per_model_f1": {int(c): d for c, d in per_model_f1.items()},
        "per_model_centrality": per_model_cent,
        "signals": {int(c): s for c, s in signals_per_class.items()},
        "timings": timings,
    }


def run_dataset(loaded, test_sizes, trials_per_size=3,
                min_subset_size=3, max_subset_size=10):
    """
    Run the full size/trial sweep for one loaded dataset.

    Parameters
    ----------
    loaded : dict
        Output of io_utils.load_dataset_predictions.
    test_sizes : list of int
        Evaluation sizes to sweep (see io_utils.make_test_sizes).
    trials_per_size : int
        Independent random trials per size.
    min_subset_size, max_subset_size : int
        Dunn search subset-size range.

    Returns
    -------
    list of dict : one run record per (size, trial).
    """
    predictions = loaded["predictions"]
    logits = loaded["logits"]
    y_true = loaded["y_true"]
    num_classes = loaded["num_classes"]
    n_total = len(y_true)

    records = []
    for size in test_sizes:
        for trial in range(trials_per_size):
            seed = int(size * 100 + trial)
            subs_idx = build_subsample_indices(n_total, size, seed)
            rec = run_single_subsample(
                subs_idx, predictions, logits, y_true, num_classes,
                min_subset_size=min_subset_size, max_subset_size=max_subset_size,
                seed=seed,
            )
            rec["trial"] = int(trial)
            rec["dataset"] = loaded["dataset"]
            records.append(rec)
    return records
