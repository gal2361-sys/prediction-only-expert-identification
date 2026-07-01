"""
Dunn-index expert subset selection.

For each class, the expert set is the subset of models that maximizes a
Dunn-style compactness--separation ratio in the class-conditional disagreement
geometry:

    Dunn^(c)(A) =  min_{m in A, m' not in A} d^(c)(m, m')
                   --------------------------------------
                   max_{m, m' in A, m != m'} d^(c)(m, m')

The numerator measures separation of the candidate subset A from the rest of the
pool; the denominator measures the internal diameter of A. Maximizing the ratio
favors subsets that are internally consistent (small diameter) yet well
separated from the remaining models. Candidate subset sizes are restricted to a
range [r_min, r_max] to avoid degenerate solutions and bound the search cost.
"""

import numpy as np
from itertools import combinations

from disagreement import class_disagreement_matrix

# Default subset-size range used in the experiments.
DEFAULT_MIN_SUBSET_SIZE = 3
DEFAULT_MAX_SUBSET_SIZE = 10

# Small constant guarding against a zero internal diameter (perfect-consensus subset).
_DIAMETER_EPS = 1e-6


def model_total_intragroup_distance(dist_mat, members):
    """
    Sum of class-conditional distances from each member to the other members of
    the group, used to derive centrality-based voting weights (see voting.py).

    Parameters
    ----------
    dist_mat : np.ndarray
        Pairwise distance matrix (local indexing).
    members : iterable of int
        Local indices of the group members.

    Returns
    -------
    dict : local index -> summed within-group distance.
    """
    members = list(members)
    if not members:
        return {}
    if len(members) == 1:
        return {members[0]: 0.0}
    sub = dist_mat[np.ix_(members, members)].copy()
    np.fill_diagonal(sub, 0.0)
    sums = sub.sum(axis=1)
    return {members[i]: float(sums[i]) for i in range(len(members))}


def find_dunn_experts(prediction_sets, min_subset_size=DEFAULT_MIN_SUBSET_SIZE,
                      max_subset_size=DEFAULT_MAX_SUBSET_SIZE):
    """
    Identify the Dunn-maximizing expert subset for every class.

    Parameters
    ----------
    prediction_sets : dict
        Maps class -> {model id -> list of sample indices assigned to that class}.
    min_subset_size, max_subset_size : int
        Inclusive range of candidate subset sizes to search.

    Returns
    -------
    expert_partition : dict
        Maps class -> tuple of model ids selected as experts for that class.
    pairwise_distances : dict
        Maps a descriptive key -> the class-conditional distance for each model
        pair (retained for diagnostics and reproducibility).
    dunn_scores : dict
        Maps class -> the maximal Dunn score achieved for that class.
    """
    expert_partition = {}
    dunn_scores = {}
    pairwise_distances = {}

    for class_idx in prediction_sets.keys():
        disagreement_mat = class_disagreement_matrix(prediction_sets[class_idx])
        model_ids = sorted(prediction_sets[class_idx].keys())
        num_models = len(model_ids)

        # Record all pairwise distances for this class (diagnostics / reproducibility).
        for a_local in range(num_models):
            for b_local in range(a_local + 1, num_models):
                model_a = model_ids[a_local]
                model_b = model_ids[b_local]
                key = f"class{class_idx}_model{model_a}_model{model_b}_dist"
                pairwise_distances[key] = float(disagreement_mat[a_local, b_local])

        max_dunn = 0.0
        r_min = max(min_subset_size, 1)
        r_max = min(max_subset_size, num_models)

        for r in range(r_min, r_max + 1):
            for subgroup_local in combinations(range(num_models), r):
                # Skip any subset containing a model that never predicts this class.
                has_empty = any(
                    prediction_sets[class_idx][model_ids[l]] is None for l in subgroup_local
                )
                if has_empty:
                    continue

                max_diff_in = 0.0
                min_diff_between = np.inf

                for j_local in subgroup_local:
                    for k_local in range(num_models):
                        if j_local == k_local:
                            continue
                        dist = float(disagreement_mat[j_local, k_local])
                        if k_local not in subgroup_local:
                            min_diff_between = min(min_diff_between, dist)
                        else:
                            max_diff_in = max(max_diff_in, dist)

                if min_diff_between is np.inf:
                    continue
                if max_diff_in == 0.0:
                    max_diff_in = _DIAMETER_EPS

                dunn_index = float(min_diff_between / max_diff_in)
                if dunn_index >= max_dunn:
                    max_dunn = dunn_index
                    expert_partition[class_idx] = tuple(
                        model_ids[l] for l in subgroup_local
                    )

        dunn_scores[class_idx] = float(max_dunn)

    return expert_partition, pairwise_distances, dunn_scores
