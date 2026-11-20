"""Загрузка исходных корпусов и приведение их к схеме датасета.

Источники (все открытые):
  * Yale-LILY/aeslc — деловая переписка Enron (2000–2002 гг., заведомо написано людьми), EN, corporate.
  * corbt/enron-emails — сырые письма Enron; из них отбираются внешние рассылки/ньюслеттеры
    (легитимный маркетинг, написанный людьми до появления LLM), EN, marketing.
  * iitolstykh/LLMTrace_detection — RU/EN тексты человека, ИИ (≈37 современных моделей) и смешанные
    с символьной разметкой ИИ-фрагментов (Apache-2.0).

Результат: data/raw/{enron_corporate,enron_marketing,llmtrace}.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from src.datagen.schema import make_id

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

_FWD_RE = re.compile(
    r"(-{3,}\s*Original Message\s*-{3,}|-{3,}\s*Forwarded by|^From:\s|^To:\s|^Sent:\s|^cc:\s|^Subject:\s|^>)",
    re.I | re.M,
)


def clean_enron_body(body: str) -> str:
    """Удаляет цитируемую/пересланную часть, HTML-теги и служебные маркеры."""
    m = _FWD_RE.search(body)
    if m:
        body = body[: m.start()]
    body = re.sub(r"<[^>]{1,200}>", " ", body)
    body = re.sub(r"\[IMAGE\]", " ", body, flags=re.I)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def human_record(text: str, subject: str, language: str, email_type: str, source: str) -> Dict:
    rid = make_id(source, subject, text)
    return {
        "id": rid,
        "group_id": rid,
        "subject": subject,
        "text": text,
        "label": 0,
        "label_name": "human",
        "generation_method": "human",
        "generator_model": None,
        "language": language,
        "email_type": email_type,
        "source": source,
        "ai_zones": [],
        "ai_char_intervals": [],
    }


def load_enron_corporate(n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("Yale-LILY/aeslc", split="train")
    rows = []
    for r in ds:
        body = clean_enron_body(r["email_body"] or "")
        sents = re.split(r"(?<=[.!?])\s+", body)
        if 200 <= len(body) <= 2000 and len(sents) >= 3:
            rows.append(human_record(body, (r["subject_line"] or "").strip(), "en", "corporate", "enron_aeslc"))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_enron_marketing(n: int, seed: int, scan_rows: int = 120000, per_sender: int = 12) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("corbt/enron-emails", split="train", streaming=True)
    by_sender: Counter = Counter()
    rows, seen = [], set()
    for i, r in enumerate(ds):
        if i >= scan_rows:
            break
        sender = (r.get("from") or "").lower()
        body = r.get("body") or ""
        if sender.endswith("@enron.com") or not sender:
            continue
        if not re.search(r"unsubscribe|newsletter|to be removed from|remove you from|opt[- ]out", body, re.I):
            continue
        dom = sender.split("@")[-1]
        if by_sender[dom] >= per_sender:
            continue
        text = clean_enron_body(body)
        # отрезаем стандартный футер отписки — он есть только у старых рассылок и стал бы «подсказкой» модели
        text = re.split(r"(?im)^.*(unsubscribe|to be removed|opt[- ]out|you are receiving this).*$", text)[0].strip()
        if not (300 <= len(text) <= 3000):
            continue
        key = text[:200]
        if key in seen:
            continue
        seen.add(key)
        by_sender[dom] += 1
        rows.append(human_record(text, (r.get("subject") or "").strip(), "en", "marketing", "enron_newsletters"))
    random.Random(seed).shuffle(rows)
    log.info("Рассылки Enron: найдено %d (из %d отправителей)", len(rows), len(by_sender))
    return rows[:n]


LLMTRACE_TYPES = {"question", "review", "short_form", "story", "news", "article", "factual"}


def load_llmtrace(per_cell: Dict[str, int], seed: int) -> List[Dict]:
    """per_cell: {"train": N, "validation": N, "test": N} — сколько брать на каждую пару (язык, метка)."""
    from datasets import load_dataset

    out = []
    for split, n in per_cell.items():
        ds = load_dataset("iitolstykh/LLMTrace_detection", split=split)
        cells: Dict[tuple, List[Dict]] = {}
        for r in ds:
            if r["data_type"] not in LLMTRACE_TYPES:
                continue
            text = r["text"] or ""
            if not (150 <= len(text) <= 3000):
                continue
            lang = "ru" if r["lang"] == "ru" else "en"
            cells.setdefault((lang, r["label"]), []).append(r)
        rng = random.Random(seed)
        for (lang, lab), items in cells.items():
            rng.shuffle(items)
            for r in items[:n]:
                ivs = [[int(s), int(e)] for s, e in (r["ai_char_intervals"] or [])]
                rid = make_id("llmtrace", split, r["text"])
                out.append({
                    "id": rid,
                    "group_id": f"llmtrace_topic_{r['topic_id']}",
                    "subject": "",
                    "text": r["text"],
                    "label": 0 if lab == "human" else 1,
                    "label_name": lab,
                    "generation_method": {"human": "human", "ai": "external_ai", "mixed": "external_mixed"}[lab],
                    "generator_model": r["model"],
                    "language": lang,
                    "email_type": "text",
                    "source": "llmtrace_detection",
                    "ai_zones": ["body"] if lab != "human" else [],
                    "ai_char_intervals": ivs,
                    "llmtrace_split": {"validation": "val"}.get(split, split),
                    "data_type": r["data_type"],
                })
    return out


def load_case(case_dir: Path, n: int, seed: int) -> List[Dict]:
    """Письма кейса Заказчика из папки samples_prompt — по пояснению к данным они сгенерированы LLM по промпту
    (метка 1). Письма с утечкой промпта исключаются (в них текст инструкции, а не письма). Группа — домен
    отправителя: письма одной кампании не попадают одновременно в train и test.
    Папки samples_spam / samples_gray не размечены по авторству и в обучение не входят."""
    from src.eml_parser import parse_eml
    from src.signals import find_prompt_leaks
    from src.text_utils import ZERO_WIDTH_RE

    files = sorted(p for p in case_dir.rglob("samples_prompt/*") if p.is_file())
    rows = []
    for f in files:
        try:
            d = parse_eml(f, analyze_images=False)
        except Exception as e:
            log.warning("кейс: не удалось разобрать %s: %s", f.name, e)
            continue
        text = ZERO_WIDTH_RE.sub("", d["visible_text"] or "")
        text = re.sub(r"[ \t ]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()[:3500]
        hidden = list(d["html_analysis"].hidden_texts) if d["html_analysis"] else []
        if len(text) < 150 or find_prompt_leaks(text, d["subject"] or "", *hidden):
            continue
        sender = (d["from"] or "").lower()
        m = re.search(r"@([\w.-]+)", sender)
        dom = m.group(1) if m else sender or f.name
        lang = "ru" if len(re.findall(r"[а-яё]", text.lower())) > len(re.findall(r"[a-z]", text.lower())) else "en"
        rid = make_id("case", f.name)
        rows.append({
            "id": rid, "group_id": f"case_{dom}", "subject": (d["subject"] or "").strip(), "text": text,
            "label": 1, "label_name": "ai", "generation_method": "external_ai", "generator_model": "unknown (кейс Заказчика)",
            "language": lang, "email_type": "marketing", "source": "customer_case", "ai_zones": ["body"],
            "ai_char_intervals": [[0, len(text)]], "case_file": f.name,
        })
    random.Random(seed).shuffle(rows)
    # не более per_group писем на кампанию (домен), чтобы одна рассылка не доминировала
    per_group, cnt, capped = 40, Counter(), []
    for r in rows:
        if cnt[r["group_id"]] < per_group:
            cnt[r["group_id"]] += 1
            capped.append(r)
    log.info("Кейс: пригодных писем samples_prompt %d (из %d файлов), после ограничения %d на кампанию: %d (%d кампаний)",
             len(rows), len(files), per_group, len(capped), len(cnt))
    return capped[:n]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Загрузка исходных корпусов")
    ap.add_argument("--case-dir", type=Path, default=Path("data/case"))
    ap.add_argument("--n-case", type=int, default=700)
    ap.add_argument("--only-case", action="store_true")
    ap.add_argument("--n-corporate", type=int, default=1500)
    ap.add_argument("--n-marketing", type=int, default=700)
    ap.add_argument("--llmtrace-train", type=int, default=1200)
    ap.add_argument("--llmtrace-val", type=int, default=250)
    ap.add_argument("--llmtrace-test", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.case_dir.exists():
        n = write_jsonl(RAW_DIR / "customer_case.jsonl", load_case(args.case_dir, args.n_case, args.seed))
        log.info("customer_case: %d", n)
    if args.only_case:
        return
    n = write_jsonl(RAW_DIR / "enron_corporate.jsonl", load_enron_corporate(args.n_corporate, args.seed))
    log.info("enron_corporate: %d", n)
    n = write_jsonl(RAW_DIR / "enron_marketing.jsonl", load_enron_marketing(args.n_marketing, args.seed))
    log.info("enron_marketing: %d", n)
    n = write_jsonl(
        RAW_DIR / "llmtrace.jsonl",
        load_llmtrace({"train": args.llmtrace_train, "validation": args.llmtrace_val, "test": args.llmtrace_test}, args.seed),
    )
    log.info("llmtrace: %d", n)


if __name__ == "__main__":
    main()
