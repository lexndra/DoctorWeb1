"""Структурные сигналы письма: маскировка (скрытый текст, невидимые символы, подмена ссылок)
и утечка промпта генератора.

Маскировка сама по себе не доказывает генерацию ИИ (адаптивные рассылки тоже прячут блоки),
поэтому в итоговый индекс она по умолчанию НЕ входит — только в объяснение для аналитика.
Утечка промпта (в письме остались инструкции для LLM: «Output only the HTML», «=== Prompt brief ===»,
«Hard constraint: …») — прямое свидетельство генерации; при её обнаружении индекс поднимается до
DW1_PROMPT_LEAK_FLOOR (по умолчанию 0.9).
"""
from __future__ import annotations

import re
from typing import Dict, List

from src.text_utils import ZERO_WIDTH_RE

PROMPT_LEAK_PATTERNS = [
    r"===\s*(prompt|final output|technical structure|user instructions|creative)",
    r"\bprompt brief\b",
    r"\bhard (constraint|rules?)\b",
    r"\boutput only (the|a)\b",
    r"\breturn only the (complete )?html\b",
    r"\bdo not add commentary\b",
    r"\bmarkdown fences\b",
    r"\bas an ai( language model)?\b",
    r"\bhere is (the|your) (html|email|rewritten)\b",
    r"\bpersonalization tokens?\b",
    r"\bmerge (tags?|fields?)\b",
    r"\binbox[- ]placement hidden text\b",
    r"\brun seed\b",
    r"\bmailer variation tokens?\b",
    r"```html",
    r"\bвыведи только\b",
    r"\bне добавляй пояснени",
    r"\bкак языковая модель\b",
]
_LEAK_RE = re.compile("|".join(PROMPT_LEAK_PATTERNS), re.I)


def find_prompt_leaks(*texts: str, limit: int = 5) -> List[str]:
    hits: List[str] = []
    for t in texts:
        for m in _LEAK_RE.finditer(t or ""):
            s = max(0, m.start() - 40)
            hits.append((t[s: m.end() + 40]).replace("\n", " ").strip())
            if len(hits) >= limit:
                return hits
    return hits


def masking_signals(html_analysis, raw_text: str) -> Dict[str, object]:
    sig: Dict[str, object] = {
        "zero_width_chars_count": len(ZERO_WIDTH_RE.findall(raw_text or "")),
        "hidden_elements_count": 0,
        "hidden_text_chars": 0,
        "suspicious_links_count": 0,
        "forms_count": 0,
        "password_fields_count": 0,
        "iframes_count": 0,
    }
    if html_analysis is not None:
        sig.update({
            "hidden_elements_count": html_analysis.hidden_elements_count,
            "hidden_text_chars": sum(len(t) for t in html_analysis.hidden_texts),
            "suspicious_links_count": html_analysis.suspicious_links_count,
            "forms_count": html_analysis.forms_count,
            "password_fields_count": html_analysis.password_fields_count,
            "iframes_count": html_analysis.iframes_count,
        })
    return sig
