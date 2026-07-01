"""
Evaluation metrics.

Helpers that turn a predicted label vector into the per-class and aggregate
scores reported in the paper. Macro-averaged F1 is the headline metric used in
all results tables; micro-F1 and accuracy are also returned for completeness.

Predictions use -1 as an "unassigned" sentinel (a sample no method placed in any
class); such entries are ignored when scoring.
"""

import numpy as np


def per_class_confusion(y_true, y_pred, num_classes, ignore_value=-1):
    """
    Per-class confusion counts and F1.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        True and predicted labels. Entries of y_pred equal to `ignore_value`
        are treated as unassigned and excluded.
    num_classes : int
    ignore_value : int
        Sentinel marking unassigned predictions.

    Returns
    -------
    dict : class -> {"tp", "fp", "fn", "tn", "f1", "acc"}.
    """
    y_pred = np.asarray(y_pred)
    mask = (y_pred != ignore_value)
    if mask.sum() == 0:
        return {
            c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "f1": 0.0, "acc": 0.0}
            for c in range(num_classes)
        }

    yt = np.asarray(y_true)[mask].astype(np.int64)
    yp = np.asarray(y_pred)[mask].astype(np.int64)
    N = len(yt)

    out = {}
    for c in range(num_classes):
        t_pos = (yt == c)
        p_pos = (yp == c)

        tp = int(np.sum(t_pos & p_pos))
        fp = int(np.sum(~t_pos & p_pos))
        fn = int(np.sum(t_pos & ~p_pos))
        tn = int(np.sum(~t_pos & ~p_pos))

        if tp == 0 and fp == 0 and fn == 0:
            f1 = 0.0
        else:
            denom_f1 = 2 * tp + fp + fn
            f1 = 2 * tp / denom_f1 if denom_f1 > 0 else 0.0

        acc = (tp + tn) / N if N > 0 else 0.0
        out[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": float(f1), "acc": float(acc)}
    return out


def summarize_method_performance(y_true, y_pred, num_classes, ignore_value=-1):
    """
    Aggregate scores for a predicted label vector.

    Returns
    -------
    dict with keys:
        "per_class"   : output of per_class_confusion
        "overall_acc" : multiclass accuracy over assigned samples
        "macro_f1"    : unweighted mean of per-class F1 (headline metric)
        "micro_f1"    : pooled-count F1
    """
    per_class = per_class_confusion(y_true, y_pred, num_classes, ignore_value)

    total_tp = sum(per_class[c]["tp"] for c in range(num_classes))
    total_fp = sum(per_class[c]["fp"] for c in range(num_classes))
    total_fn = sum(per_class[c]["fn"] for c in range(num_classes))

    y_pred = np.asarray(y_pred)
    mask = (y_pred != ignore_value)
    if int(np.sum(mask)) == 0:
        overall_acc = 0.0
    else:
        yt = np.asarray(y_true)[mask].astype(np.int64)
        yp = np.asarray(y_pred)[mask].astype(np.int64)
        overall_acc = float(np.mean(yp == yt))

    macro_f1 = float(np.mean([per_class[c]["f1"] for c in range(num_classes)]))
    denom_micro = 2 * total_tp + total_fp + total_fn
    micro_f1 = float(2 * total_tp / denom_micro) if denom_micro > 0 else 0.0

    return {
        "per_class": per_class,
        "overall_acc": overall_acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }
