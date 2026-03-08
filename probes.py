import numpy as np
import wandb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from metrics import tune_threshold_mcc, safe_roc_auc, safe_pr_auc


def run_probes(x: np.ndarray, y_label: np.ndarray, config, feature_set_name: str, label_name: str):
    if config.debug:
        print(f"\n[DEBUG MODE] Skipping CV for {feature_set_name} | {label_name}")
        # Simple train/test split or just train on all for debug
        X_train, X_test = x, x
        y_train, y_test = y_label, y_label
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            random_state=config.rng,
            class_weight='balanced',
            max_iter=config.logistic_regression_max_iter
        )
        # If only one class is present in debug, fit will fail. 
        # But with 2 images it's likely.
        try:
            clf.fit(X_train, y_train)
            y_test_proba = clf.predict_proba(X_test)[:, 1]
            chosen_thr = 0.5
            y_test_pred = (y_test_proba >= chosen_thr).astype(int)
            
            roc = safe_roc_auc(y_test, y_test_proba)
            print(f"  Debug ROC-AUC: {roc}")
        except ValueError as e:
            print(f"  Could not fit model in debug: {e}")
        return

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.rng)

    fold_rocs = []
    fold_prs = []
    fold_mccs = []
    fold_f1s = []
    fold_baccs = []
    fold_thresholds = []
    fold_confusions = []

    print(f"\n==============================")
    print(f"Feature Set: {feature_set_name} | Label: {label_name}")
    print(f"Positives: {np.sum(y_label)} / {len(y_label)} ({np.mean(y_label) * 100:.2f}%)")
    print(f"==============================")

    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y_label), start=1):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y_label[train_idx], y_label[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            random_state=config.rng,
            class_weight='balanced',
            max_iter=config.logistic_regression_max_iter
        )
        clf.fit(X_train, y_train)

        inner_skf = StratifiedKFold(n_splits=config.inner_cv_splits, shuffle=True, random_state=config.rng)
        inner_thresholds = []

        for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            inner_clf = LogisticRegression(
                random_state=config.rng,
                class_weight='balanced',
                max_iter=config.logistic_regression_max_iter
            )
            inner_clf.fit(X_inner_train, y_inner_train)

            y_val_proba = inner_clf.predict_proba(X_val)[:, 1]
            thr, _ = tune_threshold_mcc(y_val, y_val_proba)
            inner_thresholds.append(thr)

        chosen_thr = float(np.median(inner_thresholds))
        fold_thresholds.append(chosen_thr)

        y_test_proba = clf.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= chosen_thr).astype(int)

        cm = confusion_matrix(y_test, y_test_pred)
        fold_confusions.append(cm)

        roc = safe_roc_auc(y_test, y_test_proba)
        pr = safe_pr_auc(y_test, y_test_proba)
        mcc = matthews_corrcoef(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        bacc = balanced_accuracy_score(y_test, y_test_pred)

        fold_rocs.append(roc)
        fold_prs.append(pr)
        fold_mccs.append(mcc)
        fold_f1s.append(f1)
        fold_baccs.append(bacc)

        print(f"\nFold {fold}:")
        print(f"  Threshold: {chosen_thr:.3f}")
        print(f"  Confusion:\n{cm}")
        print(f"  ROC-AUC: {roc}")
        print(f"  PR-AUC:  {pr:.4f}")
        print(f"  MCC:     {mcc:.4f}")
        print(f"  F1:      {f1:.4f}")
        print(f"  BAcc:    {bacc:.4f}")

    # --- Summaries ---
    def mean_std(arr):
        arr = np.array(arr, dtype=float)
        return np.nanmean(arr), np.nanstd(arr)

    roc_mean, roc_std = mean_std(fold_rocs)
    pr_mean, pr_std = mean_std(fold_prs)
    mcc_mean, mcc_std = mean_std(fold_mccs)
    f1_mean, f1_std = mean_std(fold_f1s)
    bacc_mean, bacc_std = mean_std(fold_baccs)

    wandb.log({
        f"{label_name}/{feature_set_name}_mean_roc_auc": roc_mean,
        f"{label_name}/{feature_set_name}_std_roc_auc": roc_std,
        f"{label_name}/{feature_set_name}_mean_pr_auc": pr_mean,
        f"{label_name}/{feature_set_name}_std_pr_auc": pr_std,
        f"{label_name}/{feature_set_name}_mean_mcc": mcc_mean,
        f"{label_name}/{feature_set_name}_std_mcc": mcc_std,
        f"{label_name}/{feature_set_name}_mean_f1": f1_mean,
        f"{label_name}/{feature_set_name}_std_f1": f1_std,
        f"{label_name}/{feature_set_name}_mean_bacc": bacc_mean,
        f"{label_name}/{feature_set_name}_std_bacc": bacc_std,
        f"{label_name}/{feature_set_name}_median_threshold": np.median(fold_thresholds),
    })

    print(f"\n--- CV Summary for {label_name} | {feature_set_name} ---")
    print(f"Thresholds (median across folds): {np.median(fold_thresholds):.3f}")
    print(f"Threshold spread: min={np.min(fold_thresholds):.3f}, max={np.max(fold_thresholds):.3f}")
    print(f"ROC-AUC: {roc_mean:.4f} ± {roc_std:.4f}")
    print(f"PR-AUC:  {pr_mean:.4f} ± {pr_std:.4f}")
    print(f"MCC:     {mcc_mean:.4f} ± {mcc_std:.4f}")
    print(f"F1:      {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"BAcc:    {bacc_mean:.4f} ± {bacc_std:.4f}")
