import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef
)

def tune_threshold_mcc(y_true, y_proba, n_thresholds=501):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    best_thr = 0.5
    best_mcc = -1.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr

    return best_thr, best_mcc


def safe_roc_auc(y_true, y_proba):
    # roc_auc_score crashes if only one class is present
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba)


def safe_pr_auc(y_true, y_proba):
    # PR curve works even if one class exists, but it's meaningless.
    # We'll still return it, but you can treat it cautiously.
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)
