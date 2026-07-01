"""
Class-conditional disagreement geometry.

For a fixed class c, each model m is represented by its class-c prediction set
S_{m,c} = { i : y_hat_{m,i} = c }, i.e. the indices of samples the model assigns
to class c. The disagreement between two models on class c is the normalized
symmetric difference of their prediction sets (a Jaccard-type distance):

    d^(c)(m, m') = |S_{m,c} XOR S_{m',c}| / |S_{m,c} UNION S_{m',c}|,

with the convention d^(c) = 1 when both prediction sets are empty (the union is
empty), since two models that never predict class c provide no evidence of
class-c competence and should not be grouped together.

This module builds the per-class pairwise distance matrix used by the expert
search (see dunn.py) and the weighted voting scheme (see voting.py).
"""

import numpy as np


def set_symmetric_difference(set_a, set_b):
    """Size of the symmetric difference between two index lists (treated as sets)."""
    s1 = set() if set_a is None else set(set_a)
    s2 = set() if set_b is None else set(set_b)
    return len(s1 ^ s2)


def set_union_size(set_a, set_b):
    """Size of the union between two index lists (treated as sets)."""
    s1 = set() if set_a is None else set(set_a)
    s2 = set() if set_b is None else set(set_b)
    return len(s1 | s2)


def class_disagreement_matrix(prediction_sets_for_class):
    """
    Build the pairwise class-conditional disagreement matrix for a single class.

    Parameters
    ----------
    prediction_sets_for_class : dict
        Maps a model id -> list of sample indices that the model assigns to this
        class (i.e. S_{m,c}). A value of None is treated as an empty set.

    Returns
    -------
    dist_mat : np.ndarray, shape (M, M)
        Symmetric matrix of class-conditional disagreement distances, where M is
        the number of models. Diagonal entries are set to 1.0 by convention and
        are not used by the subset search.

    Notes
    -----
    Models are ordered by sorted model id; row/column k corresponds to the k-th
    model in that sorted order. Prediction sets are materialized once per model
    rather than rebuilt inside the pairwise loop, so construction is linear in
    the number of samples.
    """
    model_ids = sorted(prediction_sets_for_class.keys())
    num_models = len(model_ids)
    dist_mat = np.zeros((num_models, num_models), dtype=np.float64)

    sets = [
        set(prediction_sets_for_class[g]) if prediction_sets_for_class[g] is not None else set()
        for g in model_ids
    ]

    for j in range(num_models):
        dist_mat[j, j] = 1.0
        for k in range(j + 1, num_models):
            union = len(sets[j] | sets[k])
            d = 1.0 if union == 0 else len(sets[j] ^ sets[k]) / float(union)
            dist_mat[j, k] = d
            dist_mat[k, j] = d

    return dist_mat
