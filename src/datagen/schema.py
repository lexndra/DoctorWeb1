"""Схема разметки датасета (п. 4.1.2.4 ТЗ).

Каждая запись — одно письмо/текст в формате JSON Lines.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

SCHEMA_VERSION = "1.0"

SCHEMA: Dict[str, Dict[str, str]] = {
    "id": {"type": "str", "desc": "Уникальный идентификатор записи"},
    "group_id": {"type": "str", "desc": "Группа (исходное письмо и все его производные); разбиение выполняется по группам, чтобы исключить утечки"},
    "subject": {"type": "str", "desc": "Тема письма (может быть пустой для текстов не из почты)"},
    "text": {"type": "str", "desc": "Видимый текст письма (тело)"},
    "label": {"type": "int", "desc": "Бинарная метка: 1 — сгенерировано ИИ (полностью или частично), 0 — написано человеком"},
    "label_name": {"type": "str", "desc": "human | ai | mixed"},
    "generation_method": {"type": "str", "desc": "human | full | partial_rewrite | partial_insert | partial_append | variation | external_ai | external_mixed"},
    "generator_model": {"type": "str|null", "desc": "Модель-генератор (null для человека)"},
    "language": {"type": "str", "desc": "ru | en"},
    "email_type": {"type": "str", "desc": "corporate | marketing | text (не письмо: внешний корпус текстов)"},
    "source": {"type": "str", "desc": "Происхождение: enron_aeslc | enron_newsletters | llmtrace_detection | dw1_generated"},
    "ai_zones": {"type": "list[str]", "desc": "Перечень зон письма, содержащих сгенерированный текст (subject, body)"},
    "ai_char_intervals": {"type": "list[[int,int]]", "desc": "Символьные интервалы сгенерированных фрагментов в поле text, [start, end)"},
    "split": {"type": "str", "desc": "train | val | test"},
}


def make_id(*parts: str) -> str:
    h = hashlib.sha1("␟".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def detect_lang(text: str) -> str:
    cyr = len(re.findall(r"[а-яёА-ЯЁ]", text))
    lat = len(re.findall(r"[a-zA-Z]", text))
    return "ru" if cyr > lat else "en"


def validate_record(rec: Dict) -> List[str]:
    """Проверка записи на соответствие схеме. Возвращает список ошибок."""
    errors = []
    for key in SCHEMA:
        if key == "split":
            continue
        if key not in rec:
            errors.append(f"нет поля {key}")
    if errors:
        return errors
    text = rec["text"]
    if not isinstance(text, str) or len(text.strip()) < 40:
        errors.append("слишком короткий текст")
    if rec["label"] not in (0, 1):
        errors.append("label не 0/1")
    if rec["label"] == 0 and rec["ai_char_intervals"]:
        errors.append("у человеческого текста есть AI-интервалы")
    if rec["label"] == 1 and not rec["ai_char_intervals"]:
        errors.append("у AI-текста нет интервалов")
    for s, e in rec["ai_char_intervals"]:
        if not (0 <= s < e <= len(text)):
            errors.append(f"интервал вне текста: {s},{e}")
    if rec["language"] not in ("ru", "en"):
        errors.append("неизвестный язык")
    return errors
