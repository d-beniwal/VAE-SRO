"""Performance metrics for pair-CSRO predictions.

All metrics operate on the *present* pairs only (upper triangle of the binary
mask), matching the convention used throughout the paper.
"""
from __future__ import annotations

import numpy as np

from .model import ELEMENTS


def iter_present_pairs(mask):
    """Yield (row, col) indices of present pairs in the upper triangle."""
    triu = np.triu(mask, 1)
    rows, cols = np.where(triu == 1)
    yield from zip(rows, cols)


def collect_pair_values(Y_true, Y_pred, masks):
    """Flatten true/pred CSRO over all present pairs across a set of alloys.

    Returns
    -------
    (y_true, y_pred, pair_labels, alloy_index) : parallel 1-D arrays/lists.
    """
    yt, yp, labels, alloy_idx = [], [], [], []
    for k, (Yt, Yp, m) in enumerate(zip(Y_true, Y_pred, masks)):
        for r, c in iter_present_pairs(m):
            a, b = sorted([ELEMENTS[r], ELEMENTS[c]])
            yt.append(Yt[r, c]); yp.append(Yp[r, c])
            labels.append(f"{a}-{b}"); alloy_idx.append(k)
    return np.array(yt), np.array(yp), labels, alloy_idx


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def classification_stats(y_true, y_pred):
    """Ordering (<0) vs clustering (>0) classification metrics.

    Sign convention: negative CSRO = ordering, positive = clustering.
    Returns a dict with accuracy, precision, recall (for ordering), counts, and
    the confusion matrix ``[[TO, FC],[FO, TC]]``.
    """
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    order_true = yt < 0; order_pred = yp < 0
    TO = int(np.sum(order_true & order_pred))     # true ordering
    FC = int(np.sum(order_true & ~order_pred))    # ordering called clustering
    FO = int(np.sum(~order_true & order_pred))    # clustering called ordering
    TC = int(np.sum(~order_true & ~order_pred))   # true clustering
    n = len(yt)
    acc = (TO + TC) / n if n else float("nan")
    prec = TO / (TO + FO) if (TO + FO) else float("nan")
    rec = TO / (TO + FC) if (TO + FC) else float("nan")
    misclassified = int(np.sum((yt * yp) < 0))
    return dict(accuracy=acc, precision=prec, recall=rec,
                confusion=[[TO, FC], [FO, TC]], n=n,
                n_ordering=int(np.sum(order_true)), n_clustering=int(np.sum(~order_true)),
                misclassified=misclassified)


def summarise(y_true, y_pred):
    """Convenience: MAE + classification accuracy in one dict."""
    s = classification_stats(y_true, y_pred)
    s["mae"] = mae(y_true, y_pred)
    return s
