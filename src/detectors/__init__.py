"""Методы оценки вероятности генерации с единым программным интерфейсом (п. 4.5.1 ТЗ, «модульность»).

Каждый детектор реализует `score(text, subject="") -> DetectorOutput`.
"""
from src.detectors.base import DetectorOutput, SentenceScore  # noqa: F401
