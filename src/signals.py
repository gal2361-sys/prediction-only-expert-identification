"""
Prediction-only failure-detection signals.

These quantities are computed entirely from model predictions and the expert
partition, without ground-truth labels, and are used to flag regimes in which
expert-restricted aggregation is likely to be unreliable (A-type failures).

Two families are provided:

A1 -- geometric compression (weak separability).
    Centrality of a model in a class is one minus its average class-conditional
    disagreement with the other models,
        centrality_c(m) = 1 - mean_{m' != m} d^(c)(m, m').
    The A1 signal is the dispersion (standard deviation) of these centralities
    across the pool for a class. Low dispersion indicates a compressed geometry
    with little dynamic range, where expert selection is unstable.

A2 -- prediction-prevalence distortion (shared bias).
    Compares how often experts predict a class relative to non-experts. A large
    under- or over-bias indicates that expert agreement may be driven by a shared
    predictive tendency rather than genuine competence. These signals are used in
    conjunction with the geometric separation gap (provided here as a separate
    quantity); the decision rule that combines them, including any thresholds, is
    applied by the caller (see the gated-experiment notebook), so thresholds are
    intentionally not hard-coded in this module.

All functions take direct inputs (prediction sets, expert sets, a distance
matrix), matching the representations used elsewhere in the method.
"""

import numpy as np

from disagreement import class_disagreement_matrix

EPS = 1e-12


# ---------------------------------------------------------------------------
# A1: centrality and its dispersion
# ---------------------------------------------------------------------------

def model_centrality(distance_matrix):
    """
    Per-model centrality within a class:
        centrality(m) = 1 - mean_{m' != m} d(m, m').

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (M, M)
        Class-conditional disagreement matrix (diagonal ignored).

    Returns
    -------
    np.ndarray, shape (M,) : centrality score per model.
    """
    M = distance_matrix.shape[0]
    if M <= 1:
        return np.zeros(M, dtype=np.float64)
    mat = distance_matrix.astype(np.float64).copy()
    # Exclude the diagonal from each row's mean.
    off_diag_sum = mat.sum(axis=1) - np.diag(mat)
    mean_dist = off_diag_sum / float(M - 1)
    return 1.0 - mean_dist


def centrality_dispersion(distance_matrix):
    """
    A1 signal: standard deviation of per-model centralities for a class.

    Low values indicate a compressed disagreement geometry (limited dynamic
    range), associated with unstable expert selection. The flagging threshold is
    left to the caller.
    """
    cent = model_centrality(distance_matrix)
    if cent.size == 0:
        return float("nan")
    return float(np.std(cent))


# ---------------------------------------------------------------------------
# A2: prediction-prevalence distortion (under- / over-bias)
# ---------------------------------------------------------------------------

def class_prevalence(prediction_sets, cls, model_ids, num_classes):
    """
    Fraction of (model, sample) predictions assigned to class `cls`, over the
    given set of models.

        prevalence(cls) = sum_{m in model_ids} |S_{m,cls}|
                          --------------------------------------
                          sum_{k} sum_{m in model_ids} |S_{m,k}|

    Parameters
    ----------
    prediction_sets : dict
        Maps class -> {model id -> list of sample indices assigned to that class}.
    cls : int
    model_ids : iterable
        Models to aggregate over (e.g. experts or non-experts).
    num_classes : int

    Returns
    -------
    float : prevalence of `cls`, or NaN if these models made no predictions.
    """
    model_ids = list(model_ids)

    def count_for_class(c):
        cls_dict = prediction_sets.get(c, {}) or {}
        total = 0
        for m in model_ids:
            lst = cls_dict.get(m, None)
            if lst is not None:
                total += len(lst)
        return total

    denom = sum(count_for_class(k) for k in range(num_classes))
    if denom <= 0:
        return float("nan")
    return float(count_for_class(cls) / float(denom))


def _split_experts_nonexperts(expert_set, all_models):
    expert_set = set(expert_set)
    non_experts = [m for m in all_models if m not in expert_set]
    return list(expert_set), non_experts


def under_bias(prediction_sets, expert_set, cls, all_models, num_classes):
    """
    A2 under-bias signal for a class:
        UnderBias(c) = (prevalence_NE(c) - prevalence_E(c)) / (prevalence_E(c) + eps).

    Large positive values indicate experts assign class `c` substantially less
    often than non-experts.
    """
    experts, non_experts = _split_experts_nonexperts(expert_set, all_models)
    p_E = class_prevalence(prediction_sets, cls, experts, num_classes) if experts else float("nan")
    p_NE = class_prevalence(prediction_sets, cls, non_experts, num_classes) if non_experts else float("nan")
    if not np.isfinite(p_E) or not np.isfinite(p_NE):
        return float("nan")
    return float((p_NE - p_E) / (p_E + EPS))


def over_bias(prediction_sets, expert_set, cls, all_models, num_classes):
    """
    A2 over-bias signal for a class:
        OverBias(c) = (prevalence_E(c) - prevalence_NE(c)) / (prevalence_NE(c) + eps).

    Large positive values indicate experts assign class `c` substantially more
    often than non-experts, consistent with a shared over-prediction tendency.
    """
    experts, non_experts = _split_experts_nonexperts(expert_set, all_models)
    p_E = class_prevalence(prediction_sets, cls, experts, num_classes) if experts else float("nan")
    p_NE = class_prevalence(prediction_sets, cls, non_experts, num_classes) if non_experts else float("nan")
    if not np.isfinite(p_E) or not np.isfinite(p_NE):
        return float("nan")
    return float((p_E - p_NE) / (p_NE + EPS))


# ---------------------------------------------------------------------------
# Geometric separation gap (second input to the A2 decision rule)
# ---------------------------------------------------------------------------

def separation_gap(distance_matrix, expert_local_indices):
    """
    Geometric separation gap for an expert subset:
        gap = mean_{expert, non-expert} d  -  mean_{expert, expert} d.

    This is the quantity plotted on the separation axis in the analysis and used
    alongside the bias signals when deciding whether to flag a class. Combining
    it with a bias signal (and any thresholds) is the caller's responsibility.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (M, M)
    expert_local_indices : iterable of int
        Local (matrix) indices of the expert models.

    Returns
    -------
    float : inter-group minus intra-group mean disagreement, or NaN if undefined.
    """
    M = distance_matrix.shape[0]
    experts = list(expert_local_indices)
    non_experts = [i for i in range(M) if i not in set(experts)]

    intra = []
    for a in range(len(experts)):
        for b in range(a + 1, len(experts)):
            intra.append(distance_matrix[experts[a], experts[b]])

    inter = []
    for e in experts:
        for n in non_experts:
            inter.append(distance_matrix[e, n])

    if not intra or not inter:
        return float("nan")
    return float(np.mean(inter) - np.mean(intra))


# ---------------------------------------------------------------------------
# Convenience: compute the A1/A2 signals for one class from prediction sets
# ---------------------------------------------------------------------------

def class_signals(prediction_sets, expert_set, cls, all_models, num_classes):
    """
    Compute all prediction-only signals for a single class in one call.

    Returns
    -------
    dict with keys:
        'centrality_dispersion' : A1 signal
        'under_bias'            : A2 under-bias
        'over_bias'             : A2 over-bias
        'separation_gap'        : geometric separation gap for the expert set
    """
    sets_for_class = prediction_sets.get(cls, {}) or {}
    model_ids = sorted(sets_for_class.keys())
    dist_mat = class_disagreement_matrix(sets_for_class)

    local = {g: i for i, g in enumerate(model_ids)}
    expert_local = [local[g] for g in expert_set if g in local]

    return {
        "centrality_dispersion": centrality_dispersion(dist_mat),
        "under_bias": under_bias(prediction_sets, expert_set, cls, all_models, num_classes),
        "over_bias": over_bias(prediction_sets, expert_set, cls, all_models, num_classes),
        "separation_gap": separation_gap(dist_mat, expert_local),
    }
