"""
Diagnostic-gated aggregation.

Wraps the expert-weighted method with the prediction-only failure signals: for
each class, a gate decides whether expert identification looks unreliable, and
gated classes fall back to a capped full-ensemble vote instead of trusting the
expert subset.

The gate combines the signals from `signals.py` with thresholds. The thresholds
are exposed as arguments with defaults matching the values reported in the paper
(Section 7.4), so a user can tighten or loosen detection and observe the effect.

Reversion mechanism
-------------------
Non-gated classes keep their exact expert-weighted support. Gated classes
receive a full-ensemble fallback whose per-model support is capped at
`delta * W_bar / M`, where `W_bar` is the mean total expert-weight budget over
the non-gated classes and `M` is the pool size. Sweeping `delta` interpolates
between ignoring a gated class (delta = 0) and giving it a full-ensemble vote
comparable to a typical non-gated class's weight budget.
"""

import numpy as np

from disagreement import class_disagreement_matrix
from signals import model_centrality

# Default thresholds (Section 7.4 reported values).
DEFAULT_TH_CENTRALITY_DISP = 0.035   # A1: gate if centrality dispersion < this
DEFAULT_TH_SEP_GAP = 0.535           # A2 separation-gap precondition
DEFAULT_TH_UNDERBIAS = 0.8           # A2-under: under-bias >= this
DEFAULT_TH_OVERBIAS = 2.0            # A2-over: over-bias >= this

EPS = 1e-12


def _full_pool_centrality_dispersion(prediction_sets_for_class, model_pool):
    """
    Centrality dispersion over the full model pool for one class, including
    models that never predict the class (treated as empty prediction sets).
    Matches the A1 signal used in the analysis.
    """
    M = len(model_pool)
    if M < 2:
        return 0.0
    sets_for_class = {m: prediction_sets_for_class.get(m, []) for m in model_pool}
    dist_mat = class_disagreement_matrix(sets_for_class)
    cent = model_centrality(dist_mat)
    return float(np.std(cent))


def _prevalence(prediction_sets_for_class, models, size):
    """Mean per-model fraction of the subsample assigned to this class."""
    if not models or size <= 0:
        return 0.0
    vals = [len(prediction_sets_for_class.get(m, [])) / float(size) for m in models]
    return float(np.mean(vals))


def gate_class(cls, prediction_sets, expert_partition, model_pool, size,
               sep_gap, over_bias_value,
               th_centrality_disp=DEFAULT_TH_CENTRALITY_DISP,
               th_sep_gap=DEFAULT_TH_SEP_GAP,
               th_underbias=DEFAULT_TH_UNDERBIAS,
               th_overbias=DEFAULT_TH_OVERBIAS):
    """
    Decide whether to gate (distrust) class `cls`.

    Parameters
    ----------
    cls : int
    prediction_sets : dict
        {class -> {model id -> [sample indices]}}.
    expert_partition : dict
        {class -> tuple of expert model ids}.
    model_pool : list
        All model ids in the pool.
    size : int
        Subsample size (for prevalence).
    sep_gap : float
        Separation gap for this class (from signals.separation_gap).
    over_bias_value : float
        Over-bias for this class (from signals.over_bias).
    th_* : float
        Thresholds; defaults match the paper.

    Returns
    -------
    (gated : bool, fired : list of str)
        `fired` lists which signals tripped ("A1", "A2-under", "A2-over",
        or "empty" when the expert set is empty).
    """
    E_c = list(expert_partition.get(cls, ()))
    fired = []

    if len(E_c) == 0:
        return True, ["empty"]

    sets_c = prediction_sets.get(cls, {}) or {}

    # A1: full-pool centrality dispersion.
    disp = _full_pool_centrality_dispersion(sets_c, model_pool)
    if disp < th_centrality_disp:
        fired.append("A1")

    # A2: bias signals gated behind the separation-gap precondition.
    non_experts = [m for m in model_pool if m not in set(E_c)]
    p_E = _prevalence(sets_c, E_c, size)
    p_NE = _prevalence(sets_c, non_experts, size)
    under_bias_value = (p_NE - p_E) / (abs(p_E) + EPS)

    if sep_gap >= th_sep_gap and under_bias_value >= th_underbias:
        fired.append("A2-under")
    if sep_gap >= th_sep_gap and over_bias_value >= th_overbias:
        fired.append("A2-over")

    return (len(fired) > 0), fired


def gated_predictions(prediction_sets, expert_partition, expert_weights,
                      gate_flags, num_classes, size, model_pool, deltas):
    """
    Form gated predictions on the expert-weighted base.

    Non-gated classes contribute their exact expert-weighted support. Gated
    classes contribute a capped full-ensemble fallback (per-model support
    delta * W_bar / M). One label vector is returned per delta.

    Parameters
    ----------
    prediction_sets : dict
        {class -> {model id -> [sample indices]}}.
    expert_partition : dict
        {class -> tuple of expert model ids}.
    expert_weights : dict
        {class -> {model id -> weight}} from experts_weighted_voting.
    gate_flags : dict or list
        {class -> bool} indicating gated classes.
    num_classes : int
    size : int
    model_pool : list
    deltas : iterable of float
        Fallback budget multipliers to sweep.

    Returns
    -------
    dict : {delta -> label vector (length `size`, -1 for unassigned)}.
    """
    M = len(model_pool)

    nongated_budgets = [
        float(sum(expert_weights.get(c, {}).values()))
        for c in range(num_classes) if not gate_flags[c]
    ]
    W_bar = float(np.mean(nongated_budgets)) if nongated_budgets else float(M)

    out = {}
    for delta in deltas:
        gated_per_model = (delta * W_bar / M) if M > 0 else 0.0
        voting = {}
        for c in range(num_classes):
            if gate_flags[c]:
                for _, pred_list in (prediction_sets.get(c, {}) or {}).items():
                    for s_idx in pred_list:
                        d = voting.setdefault(s_idx, {})
                        d[c] = d.get(c, 0.0) + gated_per_model
            else:
                for g, w in expert_weights.get(c, {}).items():
                    for s_idx in prediction_sets.get(c, {}).get(g, []):
                        d = voting.setdefault(s_idx, {})
                        d[c] = d.get(c, 0.0) + float(w)

        y = np.full(size, -1, dtype=np.int64)
        for s_idx, scores in voting.items():
            best_c = max(scores.keys(), key=lambda cc: (scores[cc], -cc))
            y[int(s_idx)] = int(best_c)
        out[delta] = y
    return out
