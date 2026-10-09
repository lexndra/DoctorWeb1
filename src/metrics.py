"""
Модуль вычисления метрик из статьи:
1. Потокенные метрики: Accuracy, Weighted Precision, Weighted Recall, Weighted F1, MCC, Cohen's Kappa.
2. Метрики локализации границ:
   - Boundary MAE (Mean Absolute Error) — среднее абсолютное отклонение координаты границы в токенах (M4GT).
   - F1@K (при K = 3) — точность обнаружения множественных границ перехода (TriBERT).
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)


def extract_boundaries(seq: List[int]) -> List[int]:
    """Находит индексы точек перехода, где y_t != y_{t-1}."""
    return [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]


def compute_boundary_mae(true_seqs: List[List[int]], pred_seqs: List[List[int]]) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE) локализации первой границы перехода в токенах.
    Если границы в последовательности нет, координатой считается длина последовательности.
    """
    errors = []
    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        true_b = next((i for i in range(1, len(true_seq)) if true_seq[i] != true_seq[i - 1]), len(true_seq))
        pred_b = next((i for i in range(1, len(pred_seq)) if pred_seq[i] != pred_seq[i - 1]), len(pred_seq))
        errors.append(abs(true_b - pred_b))
    return float(np.mean(errors)) if errors else 0.0


def compute_f1_at_k(
    true_seqs: List[List[int]],
    pred_seqs: List[List[int]],
    k: int = 3,
    tolerance: int = 2,
) -> float:
    """
    Вычисляет F1@K (при K=3) для мульти-граничной сегментации.
    Предсказанная граница считается совпавшей с истинной, если она находится в пределах `tolerance` токенов.
    """
    scores = []
    for true_seq, pred_seq in zip(true_seqs, pred_seqs):
        gt_bounds = extract_boundaries(true_seq)
        pred_bounds = extract_boundaries(pred_seq)[:k]

        if not gt_bounds and not pred_bounds:
            scores.append(1.0)
            continue
        if not gt_bounds or not pred_bounds:
            scores.append(0.0)
            continue

        matched = 0
        used_gt = set()
        for pb in pred_bounds:
            for idx, gb in enumerate(gt_bounds):
                if idx not in used_gt and abs(pb - gb) <= tolerance:
                    matched += 1
                    used_gt.add(idx)
                    break

        f1 = (2.0 * matched) / (len(pred_bounds) + len(gt_bounds))
        scores.append(f1)

    return float(np.mean(scores)) if scores else 0.0


def evaluate_sequences(true_seqs: List[List[int]], pred_seqs: List[List[int]]) -> Dict[str, float]:
    """Считает полный набор классификационных и граничных метрик статьи."""
    flat_true = np.concatenate([np.array(s, dtype=np.int64) for s in true_seqs if len(s) > 0])
    flat_pred = np.concatenate([np.array(s, dtype=np.int64) for s in pred_seqs if len(s) > 0])

    if len(flat_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "kappa": 0.0,
            "boundary_mae": 0.0,
            "f1_at_3": 0.0,
        }

    return {
        "accuracy": float(accuracy_score(flat_true, flat_pred)),
        "precision": float(precision_score(flat_true, flat_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(flat_true, flat_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(flat_true, flat_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(flat_true, flat_pred)),
        "kappa": float(cohen_kappa_score(flat_true, flat_pred)),
        "boundary_mae": compute_boundary_mae(true_seqs, pred_seqs),
        "f1_at_3": compute_f1_at_k(true_seqs, pred_seqs, k=3, tolerance=2),
    }
