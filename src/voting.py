"""
Expert-restricted aggregation.

Given the per-class expert sets, predictions are formed by restricting
aggregation to the selected experts in a class-specific manner. Each expert
contributes only to the class for which it was identified.

The primary scheme is geometry-aware weighted voting: an expert's weight for a
class decreases with its total class-conditional disagreement with the other
experts in the same group,

    w_m^(c) = 1 / (1 + sum_{m' in E_c} d^(c)(m, m')),

so that models lying more centrally in the expert subset (lower total
disagreement) carry more influence. For each sample, the predicted class is the
one with the largest accumulated expert weight. Baseline aggregators (full-pool
majority vote, EM, SML) live in baselines.py.
"""

import random
import numpy as np
from collections import defaultdict

from disagreement import class_disagreement_matrix
from dunn import model_total_intragroup_distance


def experts_weighted_voting(experts, prediction_sets):
    """
    Geometry-aware weighted voting restricted to the per-class expert sets.

    Parameters
    ----------
    experts : dict
        Maps class -> iterable of expert model ids for that class.
    prediction_sets : dict
        Maps class -> {model id -> list of sample indices assigned to that class}.

    Returns
    -------
    final_classification : dict
        Maps class -> list of sample indices assigned to that class by the vote.
    voting_weights : dict
        Maps class -> {model id -> centrality weight} (retained for diagnostics).
    """
    voting_dict = {}      # sample_idx -> {class -> accumulated weight}
    voting_weights = {}   # class -> {model id -> weight}

    for cls in prediction_sets.keys():
        if cls not in experts or not experts[cls]:
            continue

        model_ids = sorted(prediction_sets[cls].keys())
        global_to_local = {g: l for l, g in enumerate(model_ids)}

        valid_experts = [g for g in experts[cls] if g in global_to_local]
        if not valid_experts:
            continue

        dist_mat = class_disagreement_matrix(prediction_sets[cls])
        experts_local = [global_to_local[g] for g in valid_experts]
        within_group_dist = model_total_intragroup_distance(dist_mat, experts_local)

        voting_weights[cls] = {}
        for g in valid_experts:
            j_local = global_to_local[g]
            sum_within = float(within_group_dist.get(j_local, 0.0))
            weight = 1.0 / (1.0 + sum_within)
            voting_weights[cls][g] = float(weight)

            pred_list = prediction_sets[cls][g]
            if pred_list is None:
                continue
            for sample_idx in pred_list:
                if sample_idx not in voting_dict:
                    voting_dict[sample_idx] = {}
                voting_dict[sample_idx][cls] = voting_dict[sample_idx].get(cls, 0.0) + weight

    final_classification = {cls: [] for cls in experts.keys()}
    for sample_idx, class_scores in voting_dict.items():
        # Largest accumulated weight wins; deterministic tie-break by class id.
        chosen_cls = max(class_scores.keys(), key=lambda c: (class_scores[c], -c))
        final_classification[chosen_cls].append(int(sample_idx))

    return final_classification, voting_weights


def predictions_to_vector(class_to_indices, size, default=-1):
    """Convert a {class -> [sample indices]} map into a dense label vector."""
    y_pred = np.full(size, default, dtype=np.int64)
    for c, idxs in class_to_indices.items():
        for i in idxs:
            y_pred[int(i)] = int(c)
    return y_pred


def all_weighted_voting(prediction_sets, num_classes):
    """
    Full-pool weighted vote (the "all_weighted" curve in the gap-vs-gain plot).

    Equivalent to expert-weighted voting with every model treated as an expert
    for every class. Used only for analysis (the weighted-aggregation gain
    curve), not as a headline baseline.
    """
    all_models = sorted({
        m for cls_dict in prediction_sets.values() for m in cls_dict.keys()
    })
    full_partition = {c: tuple(all_models) for c in range(num_classes)}
    return experts_weighted_voting(full_partition, prediction_sets)


def majority_voting_experts(expert_partition, prediction_sets):
    """
    Unweighted majority vote restricted to the per-class expert sets (the
    "experts_majority" curve in the gap-vs-gain analysis). Ties are broken by
    full-pool agreement, then at random.
    """
    max_index = max(
        sample_idx
        for cls_dict in prediction_sets.values()
        for model_dict in cls_dict.values()
        for sample_idx in model_dict
    )
    sample_size = max_index + 1

    vote_counter = [defaultdict(int) for _ in range(sample_size)]
    for cls in expert_partition:
        for model in expert_partition[cls]:
            for sample_idx in prediction_sets[cls][model]:
                vote_counter[sample_idx][cls] += 1

    all_vote_counter = [defaultdict(int) for _ in range(sample_size)]
    for cls in prediction_sets:
        for model in prediction_sets[cls]:
            for sample_idx in prediction_sets[cls][model]:
                all_vote_counter[sample_idx][cls] += 1

    sample_votes = [-1] * sample_size
    for i in range(sample_size):
        if not vote_counter[i]:
            continue
        max_count = max(vote_counter[i].values())
        tied = [c for c, cnt in vote_counter[i].items() if cnt == max_count]
        if len(tied) == 1:
            sample_votes[i] = tied[0]
        elif all_vote_counter[i]:
            max_all = max(all_vote_counter[i][c] for c in tied)
            tied2 = [c for c in tied if all_vote_counter[i][c] == max_all]
            sample_votes[i] = random.choice(tied2)
        else:
            sample_votes[i] = random.choice(tied)

    final_classification = defaultdict(list)
    for idx, pred in enumerate(sample_votes):
        if pred != -1:
            final_classification[int(pred)].append(int(idx))
    return final_classification
