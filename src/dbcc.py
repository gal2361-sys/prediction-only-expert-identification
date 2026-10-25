"""
Distance-Based Collaborative Classification (DBCC), Ghanaiem et al.,
Mathematics 11(23):4784, 2023, Algorithm 2.  Implemented from the paper for use
as a prediction-only baseline.

Inputs are exactly those of the method in this repository: a matrix of hard
labels, one per (model, sample), and nothing else.

Reading of the published procedure.  Decisions the paper leaves open are marked
DECISION; they are the only places where another reimplementation could differ.

Stage 1, expert identification, for each class j (Alg. 2, lines 3-20):
  - d(u,v | j) = |A_u xor A_v| / (|A_u| + |A_v|), A_u = samples model u assigns
    to class j (Eq. 3).  DECISION: both sets empty -> d = 0.
  - central classification xi_j = argmin_u sum_v d(u,v | j) (Eq. 2, line 9).
  - a_k = d(xi_j, k | j) (line 11).
  - cluster {a_k} into two groups by a two-component Gaussian mixture (Sec. 5:
    "we apply Gaussian mixture clustering and the expectation-maximization
    algorithm").  DECISION: the expert group is the component with the smaller
    mean (it contains the centre, a = 0, which line 15 gives weight 1); if all
    distances are equal every model is an expert.
  - weights: experts 1, others 0 (line 15).

Stage 2, labelling (lines 21-29): for each sample, each class scores the number
of its experts that voted for it; the label is the argmax ("relative majority
vote", Sec. 5).  DECISION: ties broken at random (the paper breaks majority-vote
ties at random, Sec. 6.2.1); a sample with an all-zero score vector (no expert
of any class voted for its own class) falls back to the full-pool majority vote.

Stage 3, EM correction (lines 30-33).  Published text: "M-step: from the
estimated correct classification, obtain normalized Hamming distances for all
agents.  E-step: estimate the correct classification by running steps 4-17 over
the obtained distances."  Reading adopted: the current estimate gamma* replaces
the central classification; for each class j the Eq. 3 distance of every model's
class-j set to gamma*'s class-j set is computed; the two-component split and
the 1/0 weights are recomputed from these distances; the stage-2 labelling is
recomputed.  Stop when fewer than n x 10^-3 labels change (Sec. 5: "of order
n x 10^-3") or after max_iter iterations (DECISION: 20; the paper gives none).

The implementation reproduces the per-class distance tables of the paper's
worked example (Table 1) exactly.  centre="nearest_pair" (the model whose
nearest neighbour is closest) is an alternative reading of the central
classification, kept for the record; the medoid reading of Eq. 2 is the
published one and the one used.
"""

import numpy as np
from sklearn.mixture import GaussianMixture


def _indicator(Y, j):
    """(M, N) float matrix: B[m, i] = 1 if model m assigns sample i to class j."""
    return (Y == j).astype(np.float64)


def pairwise_norm_hamming(B):
    """Eq. (3) for all model pairs from the indicator matrix of one class."""
    sizes = B.sum(axis=1)
    inter = B @ B.T
    symdiff = sizes[:, None] + sizes[None, :] - 2.0 * inter
    denom = sizes[:, None] + sizes[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(denom > 0, symdiff / denom, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def norm_hamming_to_reference(B, r):
    """Eq. (3) distance of every model's class-j set to a reference set r."""
    sizes = B.sum(axis=1)
    rs = float(r.sum())
    inter = B @ r
    symdiff = sizes + rs - 2.0 * inter
    denom = sizes + rs
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, symdiff / denom, 0.0)


def split_two_groups(a, seed=0):
    """Two-component GMM on 1-D distances; boolean mask of the expert group."""
    a = np.asarray(a, dtype=float)
    if np.allclose(a, a[0]):
        return np.ones_like(a, dtype=bool)
    gm = GaussianMixture(n_components=2, n_init=5, random_state=seed,
                         reg_covar=1e-6).fit(a.reshape(-1, 1))
    lab = gm.predict(a.reshape(-1, 1))
    low = int(np.argmin(gm.means_.ravel()))
    return lab == low


def _label_by_experts(Y, C, experts, rng, fallback):
    """Stage 2 vote. experts[j] is a boolean mask over models."""
    M, N = Y.shape
    scores = np.zeros((N, C))
    for j in range(C):
        ex = np.flatnonzero(experts[j])
        if len(ex):
            scores[:, j] = (Y[ex] == j).sum(axis=0)
    y = scores.argmax(axis=1).astype(np.int64)
    smax = scores.max(axis=1)
    # random tie-breaking
    ties = (scores == smax[:, None]).sum(axis=1) > 1
    for i in np.flatnonzero(ties):
        y[i] = rng.choice(np.flatnonzero(scores[i] == smax[i]))
    y[smax == 0] = fallback[smax == 0]
    return y


def dbcc_predict(preds_by_model, num_classes, em=True, max_iter=20, seed=0,
                 return_experts=False, centre="medoid"):
    """
    Parameters
    ----------
    preds_by_model : dict {model id -> (N,) int array of hard labels}
    num_classes : int
    em : bool            run the stage-3 correction loop (the published DBCC)
    max_iter : int       cap on stage-3 iterations
    centre : "medoid" (Eq. 2 as published) or "nearest_pair" (see module doc)

    Returns y (N,) int labels; with return_experts also {class -> [model ids]}
    and the number of stage-3 iterations run.
    """
    rng = np.random.default_rng(seed)
    names = sorted(preds_by_model)
    Y = np.stack([np.asarray(preds_by_model[m], dtype=np.int64) for m in names])
    M, N = Y.shape
    C = int(num_classes)
    Bs = [_indicator(Y, j) for j in range(C)]

    counts = np.zeros((N, C))
    for m in range(M):
        counts[np.arange(N), Y[m]] += 1
    fallback = counts.argmax(axis=1)

    # ---- stage 1 ----
    experts = []
    for j in range(C):
        D = pairwise_norm_hamming(Bs[j])
        if centre == "medoid":
            c = int(np.argmin(D.sum(axis=1)))
        else:
            c = int(np.argmin(np.min(D + np.eye(M) * 2.0, axis=1)))
        experts.append(split_two_groups(D[c], seed=seed))

    # ---- stage 2 ----
    y = _label_by_experts(Y, C, experts, rng, fallback)
    n_iter = 0

    # ---- stage 3 ----
    if em:
        tol = max(1, int(np.ceil(N * 1e-3)))
        for n_iter in range(1, max_iter + 1):
            new_experts = []
            for j in range(C):
                r = (y == j).astype(np.float64)
                a = norm_hamming_to_reference(Bs[j], r)
                new_experts.append(split_two_groups(a, seed=seed))
            y_new = _label_by_experts(Y, C, new_experts, rng, fallback)
            changed = int(np.sum(y_new != y))
            experts, y = new_experts, y_new
            if changed < tol:
                break

    if return_experts:
        return y, {j: [names[m] for m in np.flatnonzero(experts[j])] for j in range(C)}, n_iter
    return y
