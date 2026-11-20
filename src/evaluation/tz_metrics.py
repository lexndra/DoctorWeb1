"""Метрики качества по таблице 3 ТЗ (п. 4.1.11) и калибровка порогов на валидационной выборке (п. 4.1.3.2)."""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.text_utils import char_mask, merge_intervals


def safe_auc(y, s) -> Optional[float]:
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def fpr_at(y, s, thr) -> Optional[float]:
    """thr — число или массив порогов той же длины (порог по языку записи)."""
    y, s = np.asarray(y), np.asarray(s)
    thr = np.broadcast_to(np.asarray(thr, dtype=float), s.shape)
    neg = y == 0
    if neg.sum() == 0:
        return None
    return float((s[neg] >= thr[neg]).mean())


def prf_at(y, s, thr) -> Dict[str, float]:
    y = np.asarray(y)
    p = (np.asarray(s) >= np.asarray(thr, dtype=float)).astype(int)
    return {
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "accuracy": float((y == p).mean()),
    }


def bootstrap_auc_ci(y, s, iters: int = 1000, seed: int = 42) -> Dict[str, float]:
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "width": float(hi - lo), "iters": len(vals)}


def calibrate_doc_threshold(y, s, is_marketing_human, max_fpr: float = 0.20) -> Dict[str, float]:
    """Порог документа: максимум F1 при FPR ≤ max_fpr на всех человеческих письмах и отдельно
    на легитимных рассылках. Если ограничения недостижимы — максимум F1 без ограничений."""
    y, s, mk = np.asarray(y), np.asarray(s), np.asarray(is_marketing_human)
    cands = np.unique(np.round(s, 4))
    best, best_any = None, None
    for t in cands:
        f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
        fpr_h = (s[y == 0] >= t).mean() if (y == 0).any() else 0
        fpr_m = (s[mk] >= t).mean() if mk.any() else 0
        if best_any is None or f1 > best_any[1]:
            best_any = (t, f1, fpr_h, fpr_m)
        if fpr_h <= max_fpr and fpr_m <= max_fpr and (best is None or f1 > best[1]):
            best = (t, f1, fpr_h, fpr_m)
    chosen, constrained = (best, True) if best else (best_any, False)
    return {"threshold": float(chosen[0]), "val_f1": float(chosen[1]), "val_fpr_human": float(chosen[2]),
            "val_fpr_marketing": float(chosen[3]), "constraints_met": constrained, "max_fpr": max_fpr}


def calibrate_by_language(y, s, langs, is_marketing_human, max_fpr: float = 0.20, min_human: int = 30) -> Dict:
    """Общий порог + отдельные пороги по языкам (калибровочные параметры, подбираются только на val).
    Для языка с малым числом человеческих текстов используется общий порог."""
    y, s, langs, mk = map(np.asarray, (y, s, langs, is_marketing_human))
    glob = calibrate_doc_threshold(y, s, mk, max_fpr)
    by_lang, details = {}, {}
    for lang in sorted(set(langs.tolist())):
        idx = langs == lang
        if (y[idx] == 0).sum() < min_human or len(set(y[idx].tolist())) < 2:
            by_lang[lang] = glob["threshold"]
            details[lang] = {"fallback_to_global": True}
            continue
        c = calibrate_doc_threshold(y[idx], s[idx], mk[idx], max_fpr)
        by_lang[lang] = c["threshold"]
        details[lang] = c
    return {"global": glob, "by_lang": by_lang, "by_lang_details": details}


def thresholds_for(langs, cal: Dict) -> np.ndarray:
    """Массив порогов для записей по их языку (по калибровочному файлу)."""
    base = float(cal["doc_threshold"])
    by = cal.get("doc_threshold_by_lang") or {}
    return np.array([float(by.get(l, base)) for l in langs])


def predicted_intervals(sentences, thr: float) -> List[List[int]]:
    return merge_intervals([(x.start, x.end) for x in sentences if x.prob >= thr])


def char_prf(texts: Sequence[str], gold: Sequence[Sequence], pred: Sequence[Sequence]) -> Dict[str, float]:
    """Микро-усреднённые precision/recall/F1 по символам (без учёта пробелов)."""
    tp = fp = fn = 0
    for t, g, p in zip(texts, gold, pred):
        ws = np.array([not c.isspace() for c in t], dtype=bool)
        gm, pm = char_mask(len(t), g) & ws, char_mask(len(t), p) & ws
        tp += int((gm & pm).sum())
        fp += int((~gm & pm).sum())
        fn += int((gm & ~pm).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def calibrate_sentence_threshold(texts, gold, sentence_lists, grid=None) -> Dict[str, float]:
    grid = grid if grid is not None else np.round(np.arange(0.2, 0.91, 0.025), 3)
    best = None
    for t in grid:
        pred = [predicted_intervals(sl, t) for sl in sentence_lists]
        f1 = char_prf(texts, gold, pred)["f1"]
        if best is None or f1 > best[1]:
            best = (float(t), f1)
    return {"threshold": best[0], "val_char_f1": best[1]}


# ----------------------------------------------------------------------------- маскировка AI-признаков

_HOMO = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c"})


def mask_zero_width(text: str, rng: random.Random) -> str:
    return re.sub(r" ", lambda m: " ​" if rng.random() < 0.3 else " ", text)


def mask_homoglyphs(text: str, rng: random.Random) -> str:
    return "".join(ch.translate(_HOMO) if rng.random() < 0.15 else ch for ch in text)


def mask_typos(text: str, rng: random.Random) -> str:
    chars = list(text)
    for i in range(1, len(chars) - 1):
        if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < 0.03:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def mask_lowercase_casual(text: str, rng: random.Random) -> str:
    """Имитация «очеловечивания»: убрать формульные приветствия/концовки, понизить регистр начала строк."""
    lines = text.splitlines()
    out = []
    for l in lines:
        if re.match(r"\s*(dear|hello|hi|best regards|kind regards|warm regards|sincerely|с уважением|уважаем)", l, re.I):
            continue
        out.append(l[:1].lower() + l[1:] if l and rng.random() < 0.5 else l)
    return "\n".join(out)


MASKINGS = {
    "zero_width": mask_zero_width,
    "homoglyphs": mask_homoglyphs,
    "typos": mask_typos,
    "casual_rewrite": mask_lowercase_casual,
}
