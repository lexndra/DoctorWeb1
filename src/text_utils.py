"""Общие функции работы с текстом: нормализация (п. 4.1.2.1 ТЗ), разбиение на предложения
с символьными границами, операции над интервалами."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple

ZERO_WIDTH_RE = re.compile(r"[​-‏⁠-⁤﻿­]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_TYPO = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"', "«": '"', "»": '"', "„": '"',
    "—": "-", "–": "-", "−": "-", "…": "...", " ": " ",
})


def normalize_text(text: str) -> str:
    """Нормализация для извлечения признаков: NFKC, удаление невидимых символов, унификация типографики
    (кавычки/тире), замена URL, адресов и цифр на маркеры. Убирает артефакты, не связанные с авторством
    (например, ASCII-кавычки старых писем против «умных» кавычек генератора)."""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.translate(_TYPO)
    text = _URL_RE.sub(" URL ", text)
    text = _EMAIL_RE.sub(" EMAIL ", text)
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"')\]]*|$)", re.M)


def sentence_spans(text: str, min_len: int = 3) -> List[Tuple[int, int]]:
    """Предложения/строки с символьными границами [start, end) в исходном тексте."""
    spans = []
    for m in _SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= min_len and re.search(r"[^\W\d_]{2}", text[s:e]):
            spans.append((s, e))
    return spans


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def span_ai_fraction(span: Tuple[int, int], intervals: Sequence[Sequence[int]]) -> float:
    n = span[1] - span[0]
    if n <= 0:
        return 0.0
    return min(1.0, sum(overlap(span, (s, e)) for s, e in intervals) / n)


def merge_intervals(intervals: Iterable[Sequence[int]], gap: int = 2) -> List[List[int]]:
    out: List[List[int]] = []
    for s, e in sorted((int(a), int(b)) for a, b in intervals):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def char_mask(length: int, intervals: Iterable[Sequence[int]]):
    import numpy as np

    m = np.zeros(length, dtype=bool)
    for s, e in intervals:
        m[max(0, int(s)): min(length, int(e))] = True
    return m
