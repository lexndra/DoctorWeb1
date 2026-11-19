from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SentenceScore:
    start: int
    end: int
    prob: float


@dataclass
class DetectorOutput:
    """Единый результат любого метода оценки."""

    prob: float                                   # вероятность генерации документа 0..1
    sentences: List[SentenceScore] = field(default_factory=list)  # оценки предложений с границами
    signals: Dict[str, float] = field(default_factory=dict)       # значения ключевых сигналов
    contributions: Dict[str, float] = field(default_factory=dict)  # вклад признаков в логит
    method: str = ""
