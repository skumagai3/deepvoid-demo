"""Voxel-level evaluation metrics, all derived from one confusion matrix.

Computing everything from the confusion matrix makes full 512^3 volumes cheap
(one ``np.bincount``) and guarantees that every reported number is consistent.
Results agree with scikit-learn (see tests/test_core.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                     where: Optional[np.ndarray] = None) -> np.ndarray:
    """Rows = true class, columns = predicted class. ``where`` is an optional
    boolean mask selecting the voxels to score (e.g. the held-out region)."""
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch {np.shape(y_true)} vs {np.shape(y_pred)}")
    if where is not None:
        w = np.asarray(where, dtype=bool).ravel()
        t, p = t[w], p[w]
    if t.size and (t.min() < 0 or t.max() >= num_classes or p.min() < 0 or p.max() >= num_classes):
        raise ValueError("labels outside [0, num_classes)")
    return np.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def per_class(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    precision = _safe_div(tp, cm.sum(axis=0))
    recall = _safe_div(tp, cm.sum(axis=1))
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def mcc(cm: np.ndarray) -> float:
    """Multiclass Matthews correlation coefficient (Gorodkin 2004); equals
    sklearn.metrics.matthews_corrcoef."""
    cm = cm.astype(np.float64)
    t = cm.sum(axis=1)
    p = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    num = c * s - t @ p
    den = np.sqrt((s ** 2 - p @ p) * (s ** 2 - t @ t))
    return float(num / den) if den > 0 else 0.0


def summarize(cm: np.ndarray, names: Sequence[str]) -> Dict[str, object]:
    pc = per_class(cm)
    total = cm.sum()
    out: Dict[str, object] = {
        "n_voxels": int(total),
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "balanced_accuracy": float(pc["recall"].mean()),
        "f1_macro": float(pc["f1"].mean()),
        "f1_micro": float(np.trace(cm) / total) if total else 0.0,
        "mcc": mcc(cm),
        "void_f1": float(pc["f1"][0]),
        "confusion_matrix": cm.tolist(),
        "true_fraction": (cm.sum(axis=1) / max(total, 1)).tolist(),
        "pred_fraction": (cm.sum(axis=0) / max(total, 1)).tolist(),
    }
    for i, name in enumerate(names):
        out[f"{name}_precision"] = float(pc["precision"][i])
        out[f"{name}_recall"] = float(pc["recall"][i])
        out[f"{name}_f1"] = float(pc["f1"][i])
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, names: List[str],
             where: Optional[np.ndarray] = None) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, len(names), where=where)
    return summarize(cm, names)


def format_report(s: Dict[str, object], names: Sequence[str]) -> str:
    lines = [f"voxels scored: {s['n_voxels']:,}",
             f"MCC {s['mcc']:.4f} | void F1 {s['void_f1']:.4f} | balanced acc "
             f"{s['balanced_accuracy']:.4f} | macro F1 {s['f1_macro']:.4f}",
             f"{'class':<10}{'prec':>8}{'recall':>8}{'F1':>8}{'true %':>9}{'pred %':>9}"]
    for i, n in enumerate(names):
        lines.append(f"{n:<10}{s[f'{n}_precision']:>8.3f}{s[f'{n}_recall']:>8.3f}"
                     f"{s[f'{n}_f1']:>8.3f}{100 * s['true_fraction'][i]:>9.2f}"
                     f"{100 * s['pred_fraction'][i]:>9.2f}")
    return "\n".join(lines)
