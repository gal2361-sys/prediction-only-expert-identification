"""
Prediction-only baseline aggregators.

These are the label-free baselines compared against expert-restricted
aggregation. All operate on hard predicted labels (or, where noted, model
outputs) without ground-truth labels, validation performance, or model
internals.

Currently implemented:
  - Dawid--Skene EM: latent-label / annotator-reliability estimation via
    expectation--maximization over hard predictions.
  - Spectral Meta-Learner (SML): competence estimation from the leading
    eigenvector of the pairwise agreement matrix, used as weighted-vote weights.
"""

import random
import numpy as np
from collections import defaultdict


def em_dawid_skene(preds_by_model, num_classes, max_iter=50, tol=1e-6,
                   init="mv", eps=1e-12, seed=123):
    """
    Classic Dawid--Skene EM for unsupervised multiclass label aggregation.

    Model:
      - latent true label z_i in {0..C-1}
      - each model m has a confusion matrix conf_m[y_obs, z_true]
      - class prior pi[z]

    E-step:  q_i(z) proportional to pi[z] * prod_m conf_m[y_{i,m}, z]
    M-step:  pi[z]            = mean_i q_i(z)
             conf_m[y, z]     = sum_i q_i(z) 1[y_{i,m}=y] / sum_i q_i(z)

    Parameters
    ----------
    preds_by_model : dict
        Maps model id -> 1D array of hard predicted labels (length N).
    num_classes : int
    max_iter, tol : EM stopping controls.
    init : {"mv", "uniform"}
        Posterior initialization: majority vote or uniform.
    eps : numerical-stability floor.
    seed : reserved for reproducibility.

    Returns
    -------
    y_pred_em : np.ndarray, shape (N,)
        Aggregated hard labels (argmax of the posterior).
    post : np.ndarray, shape (N, C)
        Posterior q_i(z).
    priors : np.ndarray, shape (C,)
        Estimated class priors.
    conf : dict
        Maps model id -> (C, C) confusion matrix conf[y_obs, z_true].
    """
    rng = np.random.default_rng(seed)  # noqa: F841  (reserved for reproducibility)

    model_names = sorted(preds_by_model.keys())
    M = len(model_names)
    if M == 0:
        raise ValueError("em_dawid_skene: no models provided.")
    N = len(preds_by_model[model_names[0]])
    C = int(num_classes)

    # Stack predictions: Y[m, i]
    Y = np.stack(
        [np.asarray(preds_by_model[m], dtype=np.int64) for m in model_names], axis=0
    )

    # Initialize posterior.
    if init == "uniform":
        post = np.full((N, C), 1.0 / C, dtype=np.float64)
    elif init == "mv":
        counts = np.zeros((N, C), dtype=np.int64)
        for mi in range(M):
            counts[np.arange(N), Y[mi]] += 1
        mv = counts.argmax(axis=1)
        post = np.full((N, C), eps, dtype=np.float64)
        post[np.arange(N), mv] = 1.0
        post = post / post.sum(axis=1, keepdims=True)
    else:
        raise ValueError("init must be 'mv' or 'uniform'")

    priors = np.maximum(post.mean(axis=0), eps)
    priors = priors / priors.sum()

    conf = {m: np.full((C, C), 1.0 / C, dtype=np.float64) for m in model_names}

    prev_ll = None
    for _ in range(int(max_iter)):
        # ----- M-step -----
        priors = np.maximum(post.mean(axis=0), eps)
        priors = priors / priors.sum()

        for mi, m in enumerate(model_names):
            num = np.zeros((C, C), dtype=np.float64)
            denom = np.zeros(C, dtype=np.float64)
            yi = Y[mi]
            for z in range(C):
                w = post[:, z]
                denom[z] = w.sum() + eps
                np.add.at(num[:, z], yi, w)
            A = num / denom.reshape(1, C)
            A = np.maximum(A, eps)
            A = A / A.sum(axis=0, keepdims=True)
            conf[m] = A

        # ----- E-step -----
        log_post = np.log(priors + eps).reshape(1, C).repeat(N, axis=0)
        for mi, m in enumerate(model_names):
            A = conf[m]
            yi = Y[mi]
            log_post += np.log(A[yi, :] + eps)

        maxlp = log_post.max(axis=1, keepdims=True)
        expv = np.exp(log_post - maxlp)
        post = expv / (expv.sum(axis=1, keepdims=True) + eps)

        ll = float(np.sum(maxlp + np.log(expv.sum(axis=1, keepdims=True) + eps)))
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    y_pred_em = post.argmax(axis=1).astype(np.int64)
    return y_pred_em, post, priors, conf


def sml_predict(preds_by_model, num_classes, eps=1e-12):
    """
    Spectral Meta-Learner (SML).

    Estimates per-model competence from the leading eigenvector of the pairwise
    agreement matrix, then aggregates predictions by competence-weighted voting.

    Parameters
    ----------
    preds_by_model : dict
        Maps model id -> 1D array of hard predicted labels (length N).
    num_classes : int
    eps : numerical-stability floor.

    Returns
    -------
    y_pred_sml : np.ndarray, shape (N,)
        Competence-weighted aggregated labels.
    sml_weights : dict
        Maps model id -> nonnegative normalized competence weight.
    agreement_matrix : np.ndarray, shape (M, M)
        Pairwise fraction-of-agreement matrix.
    leading_eigvec : np.ndarray, shape (M,)
        The normalized leading eigenvector used as weights.
    """
    model_names = sorted(preds_by_model.keys())
    M = len(model_names)
    if M == 0:
        raise ValueError("sml_predict: no models provided.")

    Y = np.stack(
        [np.asarray(preds_by_model[m], dtype=np.int64) for m in model_names], axis=0
    )
    N = Y.shape[1]
    C = int(num_classes)

    # Pairwise agreement matrix.
    A = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i, M):
            val = 1.0 if i == j else float(np.mean(Y[i] == Y[j]))
            A[i, j] = val
            A[j, i] = val

    # Leading eigenvector, with deterministic sign and nonnegative normalization.
    eigvals, eigvecs = np.linalg.eigh(A)
    v = np.asarray(eigvecs[:, np.argmax(eigvals)], dtype=np.float64)
    if np.sum(v) < 0:
        v = -v
    v = np.maximum(v, 0.0)
    if float(v.sum()) <= eps:
        v = np.ones(M, dtype=np.float64) / float(M)
    else:
        v = v / float(v.sum())

    sml_weights = {model_names[i]: float(v[i]) for i in range(M)}

    # Competence-weighted vote.
    scores = np.zeros((N, C), dtype=np.float64)
    for i in range(M):
        scores[np.arange(N), Y[i]] += v[i]

    y_pred_sml = scores.argmax(axis=1).astype(np.int64)
    return y_pred_sml, sml_weights, A, v


def majority_voting_all_models(prediction_sets):
    """
    Full-pool unweighted majority vote (the "All (Maj)" baseline).

    Unlike em_dawid_skene and sml_predict, which take a {model id -> label
    vector} mapping, this baseline consumes the class-conditional prediction-set
    format {class -> {model id -> [sample indices assigned to that class]}},
    matching the representation used by the expert-restricted method.

    Ties are broken at random.

    Returns
    -------
    final_classification : dict
        Maps class -> list of sample indices assigned to that class.
    """
    sample_size = 1 + max(
        sample_idx
        for cls_dict in prediction_sets.values()
        for model_dict in cls_dict.values()
        for sample_idx in model_dict
    )

    pool_votes = [defaultdict(int) for _ in range(sample_size)]
    for cls in prediction_sets:
        for model in prediction_sets[cls]:
            for sample_idx in prediction_sets[cls][model]:
                pool_votes[sample_idx][cls] += 1

    sample_votes = [-1] * sample_size
    for i in range(sample_size):
        if not pool_votes[i]:
            continue
        max_count = max(pool_votes[i].values())
        tied = [c for c, cnt in pool_votes[i].items() if cnt == max_count]
        sample_votes[i] = tied[0] if len(tied) == 1 else random.choice(tied)

    final_classification = defaultdict(list)
    for idx, pred in enumerate(sample_votes):
        if pred != -1:
            final_classification[int(pred)].append(int(idx))
    return final_classification
