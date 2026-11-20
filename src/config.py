"""Настройки системы (п. 4.1.4.2, 4.1.8.1, 4.5.2.2 ТЗ). Читаются из переменных окружения / файла .env.

Порог по умолчанию берётся из файла калибровки (models/calibration.json), подобранного на val;
переменная DW1_THRESHOLD его переопределяет.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # python-dotenv не обязателен
    pass


def _f(name: str, default: Optional[float]) -> Optional[float]:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


@dataclass
class Settings:
    method: str = field(default_factory=lambda: os.getenv("DW1_METHOD", "features"))  # features | lm | combined
    threshold: Optional[float] = field(default_factory=lambda: _f("DW1_THRESHOLD", None))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("DW1_MODELS_DIR", str(ROOT / "models"))))
    lm_checkpoint: Optional[str] = field(default_factory=lambda: os.getenv("DW1_LM_CHECKPOINT") or os.getenv("MODEL_CHECKPOINT"))
    weight_features: float = field(default_factory=lambda: _f("DW1_WEIGHT_FEATURES", 0.5))
    weight_lm: float = field(default_factory=lambda: _f("DW1_WEIGHT_LM", 0.5))
    weight_images: float = field(default_factory=lambda: _f("DW1_WEIGHT_IMAGES", 0.2))
    prompt_leak_floor: float = field(default_factory=lambda: _f("DW1_PROMPT_LEAK_FLOOR", 0.9))
    enable_images: bool = field(default_factory=lambda: os.getenv("DW1_ENABLE_IMAGES", os.getenv("ENABLE_IMAGES", "0")) == "1")

    def calibration(self, method: Optional[str] = None) -> dict:
        name = {"features": "calibration.json", "lm": "lm_calibration.json", "combined": "combined_calibration.json"}[method or self.method]
        p = self.models_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def resolved_threshold(self, method: Optional[str] = None, lang: Optional[str] = None) -> float:
        """Порог решения: DW1_THRESHOLD_<ЯЗЫК> → DW1_THRESHOLD → порог языка из калибровки → общий порог из калибровки."""
        if lang:
            v = _f(f"DW1_THRESHOLD_{lang.upper()}", None)
            if v is not None:
                return v
        if self.threshold is not None:
            return self.threshold
        cal = self.calibration(method)
        if lang and lang in (cal.get("doc_threshold_by_lang") or {}):
            return float(cal["doc_threshold_by_lang"][lang])
        return float(cal.get("doc_threshold", 0.5))
