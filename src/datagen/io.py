"""Чтение выборок датасета.

Публичная часть датасета (data/dataset/<версия>/) хранится в репозитории. Письма Заказчика (кейс) по п. 4.5.2 ТЗ
не покидают контур: они лежат отдельно в data/private/<версия>/ (в .gitignore) и подмешиваются при наличии.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PRIVATE_ROOT = Path("data/private")


def read_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def load_split(dataset_dir: Path, split: str, include_private: bool = True) -> List[Dict]:
    dataset_dir = Path(dataset_dir)
    recs = read_jsonl(dataset_dir / f"{split}.jsonl")
    if include_private:
        recs += read_jsonl(PRIVATE_ROOT / dataset_dir.name / f"{split}.jsonl")
    return recs
