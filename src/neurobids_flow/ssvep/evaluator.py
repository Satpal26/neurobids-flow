"""
NeuroBIDS-Flow — SSVEP Evaluator
==================================
Computes standard BCI evaluation metrics:
  - Classification accuracy
  - Information Transfer Rate (ITR) in bits/min
  - Confusion matrix
  - Per-class precision / recall / F1
  - Cross-validation results (stratified k-fold)

Usage:
    from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
    ev = SSVEPEvaluator(n_classes=4, epoch_duration=2.0, gap_duration=0.5)
    results = ev.evaluate(clf, X, y)
    ev.print_report(results)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from sklearn.model_selection import StratifiedKFold


# ── ITR formula (ITRB) ─────────────────────────────────────────────────────────
def itr_bits_per_trial(n_classes: int, accuracy: float) -> float:
    """
    Compute ITR in bits per trial (Wolpaw formula).

    Parameters
    ----------
    n_classes : int
        Number of SSVEP target frequencies.
    accuracy : float
        Classification accuracy in [0, 1].

    Returns
    -------
    bits_per_trial : float
    """
    if n_classes < 2:
        return 0.0
    p = np.clip(accuracy, 1e-9, 1 - 1e-9)
    q = 1.0 - p
    q_per = q / (n_classes - 1) if n_classes > 1 else 0.0
    if q_per <= 0:
        return math.log2(n_classes)
    bpt = math.log2(n_classes) + p * math.log2(p) + q * math.log2(q_per)
    return max(0.0, bpt)


def itr_bits_per_minute(
    n_classes: int,
    accuracy: float,
    epoch_duration: float,
    gap_duration: float = 0.5,
) -> float:
    """
    Convert ITR from bits/trial to bits/min.

    Parameters
    ----------
    n_classes : int
    accuracy : float
    epoch_duration : float
        Length of one SSVEP epoch in seconds.
    gap_duration : float
        Inter-trial gap in seconds (default 0.5s).
    """
    bpt = itr_bits_per_trial(n_classes, accuracy)
    trial_time = epoch_duration + gap_duration
    return bpt * (60.0 / trial_time)


# ── Result container ───────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    accuracy: float
    itr_bpt: float
    itr_bpm: float
    n_classes: int
    n_epochs: int
    epoch_duration: float
    confusion_matrix: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    cv_fold_accuracies: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    method: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "accuracy": round(self.accuracy, 4),
            "itr_bpt": round(self.itr_bpt, 4),
            "itr_bpm": round(self.itr_bpm, 2),
            "n_classes": self.n_classes,
            "n_epochs": self.n_epochs,
            "epoch_duration": self.epoch_duration,
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "cv_fold_accuracies": [round(a, 4) for a in self.cv_fold_accuracies],
        }


# ── Confusion matrix helpers ───────────────────────────────────────────────────
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[i] + recall[i]
        f1[i] = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0
    return precision, recall, f1


# ── Main Evaluator ─────────────────────────────────────────────────────────────
class SSVEPEvaluator:
    """
    Standard SSVEP BCI evaluation suite.

    Parameters
    ----------
    n_classes : int
        Number of SSVEP targets.
    epoch_duration : float
        Duration of each epoch in seconds.
    gap_duration : float
        Inter-trial gap in seconds (for ITR calculation).
    n_splits : int
        Number of folds for cross-validation.
    """

    def __init__(
        self,
        n_classes: int,
        epoch_duration: float = 2.0,
        gap_duration: float = 0.5,
        n_splits: int = 5,
    ):
        self.n_classes = n_classes
        self.epoch_duration = epoch_duration
        self.gap_duration = gap_duration
        self.n_splits = n_splits

    def evaluate(
        self,
        clf: Any,
        X: np.ndarray,
        y: np.ndarray,
        method_name: str = "unknown",
        run_cv: bool = True,
    ) -> EvalResult:
        """
        Full evaluation: overall accuracy + ITR + confusion matrix + CV.

        Parameters
        ----------
        clf : classifier with .predict(X) method
            For TRCA: must also have .fit(X, y). For CCA/FBCCA: training-free.
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,) — integer class labels (0-based)
        method_name : str
        run_cv : bool
            If True, run stratified k-fold CV.

        Returns
        -------
        EvalResult
        """
        needs_fit = hasattr(clf, "fit") and hasattr(clf, "_fitted")

        # ── Full dataset eval ──────────────────────────────────────────────────
        if needs_fit:
            clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = float(np.mean(y_pred == y))
        bpt = itr_bits_per_trial(self.n_classes, acc)
        bpm = itr_bits_per_minute(
            self.n_classes, acc, self.epoch_duration, self.gap_duration
        )
        cm = confusion_matrix(y, y_pred, self.n_classes)
        prec, rec, f1 = precision_recall_f1(cm)

        # ── Cross-validation ──────────────────────────────────────────────────
        fold_accs: list[float] = []
        if run_cv and len(np.unique(y)) >= 2:
            n_splits = min(self.n_splits, len(y) // self.n_classes)
            n_splits = max(2, n_splits)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                try:
                    if needs_fit:
                        clf.fit(X_tr, y_tr)
                    fold_preds = clf.predict(X_te)
                    fold_accs.append(float(np.mean(fold_preds == y_te)))
                except Exception:
                    fold_accs.append(0.0)

        cv_mean = float(np.mean(fold_accs)) if fold_accs else acc
        cv_std = float(np.std(fold_accs)) if fold_accs else 0.0

        return EvalResult(
            accuracy=acc,
            itr_bpt=bpt,
            itr_bpm=bpm,
            n_classes=self.n_classes,
            n_epochs=len(y),
            epoch_duration=self.epoch_duration,
            confusion_matrix=cm,
            per_class_precision=prec,
            per_class_recall=rec,
            per_class_f1=f1,
            cv_fold_accuracies=fold_accs,
            cv_mean=cv_mean,
            cv_std=cv_std,
            method=method_name,
        )

    def print_report(self, result: EvalResult, stim_freqs: list[float] | None = None) -> None:
        """Print a formatted evaluation report."""
        width = 62
        print("=" * width)
        print(f"  SSVEP Evaluation Report — {result.method}")
        print("=" * width)
        print(f"  Method          : {result.method}")
        print(f"  Epochs          : {result.n_epochs}")
        print(f"  Classes         : {result.n_classes}")
        print(f"  Epoch duration  : {result.epoch_duration:.1f}s")
        print("─" * width)
        print(f"  Accuracy        : {result.accuracy:.4f}  ({result.accuracy*100:.1f}%)")
        print(f"  ITR (bits/trial): {result.itr_bpt:.4f}")
        print(f"  ITR (bits/min)  : {result.itr_bpm:.2f}")
        print(f"  Chance level    : {1.0/result.n_classes:.4f}  ({100/result.n_classes:.1f}%)")
        print("─" * width)
        if result.cv_fold_accuracies:
            fold_str = "  ".join(f"{a:.3f}" for a in result.cv_fold_accuracies)
            print(f"  CV folds        : {fold_str}")
            print(f"  CV mean ± std   : {result.cv_mean:.4f} ± {result.cv_std:.4f}")
        print("─" * width)
        print("  Confusion Matrix:")
        freqs = stim_freqs if stim_freqs else list(range(result.n_classes))
        header = "        " + "  ".join(f"{str(f):>6}" for f in freqs)
        print(header)
        for i, row in enumerate(result.confusion_matrix):
            label = f"{str(freqs[i]):>6}" if i < len(freqs) else f"{i:>6}"
            row_str = "  ".join(f"{v:>6}" for v in row)
            print(f"  {label}  {row_str}")
        print("─" * width)
        print("  Per-class metrics:")
        print(f"  {'Class':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        for i, (p, r, f) in enumerate(
            zip(result.per_class_precision, result.per_class_recall, result.per_class_f1)
        ):
            label = str(freqs[i]) if i < len(freqs) else str(i)
            print(f"  {label:>8}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}")
        print("=" * width)