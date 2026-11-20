"""Генерация синтетических писем локальной языковой моделью (п. 4.1.2.3 ТЗ).

Режимы:
  full     — полностью сгенерированное письмо на тему исходного человеческого письма (тема сохраняется,
             чтобы модель-детектор не училась различать темы вместо авторства);
  partial  — частичная правка письма человека с символьной разметкой: rewrite (переписать предложение),
             insert (вставить новое предложение), append (дописать концовку);
  variation — вариация уже сгенерированного письма (перефразирование).
Контроль качества: удаление преамбул/тем/плейсхолдеров, проверка длины и языка, валидация по схеме.
Запуск повторяемый: уже выполненные задания пропускаются (файл дописывается).

Пример:
    python -m src.datagen.generate --full-corporate 300 --full-marketing 200 --partial 450 --variation 60
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.datagen.llm_client import LLMClient
from src.datagen.schema import detect_lang, make_id, validate_record

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/generated/generated.jsonl")

PERSONAS_EN = [
    "a project manager", "a financial analyst", "an HR specialist", "a sales director", "a lawyer",
    "an IT administrator", "an operations lead", "an executive assistant", "a trader", "a contracts specialist",
]
TONES_EN = ["concise and direct", "friendly and warm", "formal", "neutral and businesslike", "slightly informal"]
NAMES_EN = ["Mark", "Susan", "David", "Karen", "Jeff", "Linda", "Tom", "Sara", "Mike", "Julie", "Chris", "Anne"]


# ----------------------------------------------------------------------------- QC

_PREAMBLE_RE = re.compile(
    r"^\s*(sure|certainly|of course|here(?:'s| is| are)|below is|конечно|вот|ниже)[^\n]*\n+", re.I
)
_SUBJECT_RE = re.compile(r"^\s*(\*\*)?(subject|тема)\s*:\s*(.*?)(\*\*)?\s*$", re.I | re.M)
_PLACEHOLDER_RE = re.compile(r"\[[^\]\n]{1,40}\]|\{\{[^}]*\}\}|<[A-Z][a-z ]+>")


def clean_generation(text: str, names: List[str]) -> Tuple[str, Optional[str]]:
    """Возвращает (очищенный текст, тема или None)."""
    text = text.strip().strip("`").strip()
    text = _PREAMBLE_RE.sub("", text, count=1)
    subject = None
    m = _SUBJECT_RE.search(text[:300])
    if m:
        subject = m.group(3).strip()
        text = (text[: m.start()] + text[m.end():]).strip()
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # markdown bold
    rng = random.Random(text[:50])
    text = _PLACEHOLDER_RE.sub(lambda _m: rng.choice(names), text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, subject


def qc_ok(text: str, lang: str, min_len: int = 120, max_len: int = 3500) -> Optional[str]:
    if not (min_len <= len(text) <= max_len):
        return f"длина {len(text)}"
    if detect_lang(text) != lang:
        return "язык не совпадает"
    if re.search(r"as an ai|language model|я — языковая модель|как ии", text, re.I):
        return "самораскрытие модели"
    if has_foreign_script(text):
        return "посторонняя письменность (CJK)"
    return None


def has_foreign_script(text: str) -> bool:
    return bool(re.search(r"[぀-ヿ㐀-鿿가-힯]", text))


# ----------------------------------------------------------------------------- sentence utils

_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.M)


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for m in _SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= 25:
            spans.append((s, e))
    return spans


# ----------------------------------------------------------------------------- tasks

def task_full(llm: LLMClient, seed: Dict, kind: str, rng: random.Random) -> Optional[Dict]:
    persona, tone = rng.choice(PERSONAS_EN), rng.choice(TONES_EN)
    words = rng.choice([80, 120, 160, 220])
    gist = seed["text"][:400].replace("\n", " ")
    if kind == "corporate":
        prompt = (
            f"Write a work email as {persona} at an energy trading company. Tone: {tone}. About {words} words.\n"
            f"Subject of the email: \"{seed['subject'] or 'work matter'}\".\n"
            f"It should deal with a similar situation as this note (do not copy it): \"{gist}\"\n"
            "Use concrete realistic details and real-sounding first names. Do NOT use placeholders in brackets. "
            "Output only the email body, without a subject line and without any explanations."
        )
    else:
        prompt = (
            f"Write a marketing newsletter email from a company to its subscribers. Tone: {tone}. About {words + 60} words.\n"
            f"Subject: \"{seed['subject'] or 'news and offers'}\".\n"
            f"Similar in topic to this newsletter (do not copy it): \"{gist}\"\n"
            "Plain text only, no HTML, no placeholders in brackets, no unsubscribe footer. "
            "Output only the email body without a subject line and without explanations."
        )
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=700)
    text, subj = clean_generation(raw, NAMES_EN)
    err = qc_ok(text, "en")
    if err:
        log.info("QC отклонил full (%s)", err)
        return None
    return {
        "id": make_id("full", seed["id"], kind),
        "group_id": seed["group_id"],
        "subject": seed["subject"],
        "text": text,
        "label": 1,
        "label_name": "ai",
        "generation_method": "full",
        "generator_model": llm.model,
        "language": "en",
        "email_type": seed["email_type"],
        "source": "dw1_generated",
        "ai_zones": ["body"],
        "ai_char_intervals": [[0, len(text)]],
        "parent_id": seed["id"],
        "gen_params": {"persona": persona, "tone": tone, "words": words},
    }


def task_partial(llm: LLMClient, seed: Dict, method: str, rng: random.Random) -> Optional[Dict]:
    text = seed["text"]
    spans = sentence_spans(text)
    if len(spans) < 3:
        return None
    ru = seed["language"] == "ru"
    if method == "rewrite":
        idx = rng.randrange(1, len(spans))
        s, e = spans[idx]
        target = text[s:e]
        prompt = (
            (f"Вот текст:\n\"\"\"{text}\"\"\"\n\nПерепиши ТОЛЬКО это предложение другими словами, сохранив смысл "
             f"и стиль окружающего текста, сделай его немного развёрнутее:\n\"{target}\"\n"
             "Выведи только новое предложение, без кавычек и пояснений.")
            if ru else
            (f"Here is an email:\n\"\"\"{text}\"\"\"\n\nRewrite ONLY this sentence in different words, keeping its meaning "
             f"and fitting the surrounding text, make it a bit more elaborate:\n\"{target}\"\n"
             "Output only the new sentence, without quotes or explanations.")
        )
    elif method == "insert":
        idx = rng.randrange(0, len(spans) - 1)
        s = e = spans[idx][1]
        prompt = (
            (f"Вот текст:\n\"\"\"{text}\"\"\"\n\nНапиши 1–2 новых предложения, которые органично продолжают мысль после фрагмента "
             f"\"{text[spans[idx][0]:spans[idx][1]]}\". Выведи только новые предложения, без пояснений.")
            if ru else
            (f"Here is an email:\n\"\"\"{text}\"\"\"\n\nWrite 1-2 new sentences that naturally continue the thought right after "
             f"\"{text[spans[idx][0]:spans[idx][1]]}\". Output only the new sentences, no explanations.")
        )
    else:  # append
        s = e = spans[-1][1]
        prompt = (
            (f"Вот текст:\n\"\"\"{text}\"\"\"\n\nДопиши к нему 2–3 предложения завершения в том же стиле. "
             "Выведи только новые предложения, без пояснений.")
            if ru else
            (f"Here is an email:\n\"\"\"{text}\"\"\"\n\nAdd 2-3 closing sentences to it in the same style. "
             "Output only the new sentences, no explanations.")
        )
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=250)
    new, _ = clean_generation(raw, NAMES_EN)
    new = new.strip().strip('"«»').strip()
    new = re.sub(r"\s*\n\s*", " ", new)
    if not (15 <= len(new) <= 700) or detect_lang(new) != seed["language"] or has_foreign_script(new):
        log.info("QC отклонил partial: %r", new[:80])
        return None
    if method == "rewrite":
        out = text[:s] + new + text[e:]
        iv = [s, s + len(new)]
    else:
        sep = " "
        out = text[:s] + sep + new + text[e:]
        iv = [s + len(sep), s + len(sep) + len(new)]
    return {
        "id": make_id("partial", seed["id"], method),
        "group_id": seed["group_id"],
        "subject": seed["subject"],
        "text": out,
        "label": 1,
        "label_name": "mixed",
        "generation_method": f"partial_{method}",
        "generator_model": llm.model,
        "language": seed["language"],
        "email_type": seed["email_type"],
        "source": "dw1_generated",
        "ai_zones": ["body"],
        "ai_char_intervals": [iv],
        "parent_id": seed["id"],
        **({"llmtrace_split": seed["llmtrace_split"]} if "llmtrace_split" in seed else {}),
    }


def task_variation(llm: LLMClient, parent: Dict, rng: random.Random) -> Optional[Dict]:
    prompt = (
        "Rewrite the following email so that it says the same thing with different wording and structure. "
        "Output only the new email body.\n\n" + parent["text"]
    )
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=1.0, max_tokens=700)
    text, _ = clean_generation(raw, NAMES_EN)
    if qc_ok(text, "en"):
        return None
    return {
        **{k: parent[k] for k in ("group_id", "subject", "language", "email_type")},
        "id": make_id("variation", parent["id"]),
        "text": text,
        "label": 1,
        "label_name": "ai",
        "generation_method": "variation",
        "generator_model": llm.model,
        "source": "dw1_generated",
        "ai_zones": ["body"],
        "ai_char_intervals": [[0, len(text)]],
        "parent_id": parent["id"],
    }


# ----------------------------------------------------------------------------- driver

def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Генерация синтетических писем локальной LLM")
    ap.add_argument("--full-corporate", type=int, default=300)
    ap.add_argument("--full-marketing", type=int, default=200)
    ap.add_argument("--partial-corporate", type=int, default=200)
    ap.add_argument("--partial-marketing", type=int, default=100)
    ap.add_argument("--partial-ru", type=int, default=150)
    ap.add_argument("--variation", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    llm = LLMClient()
    if not llm.ping():
        raise SystemExit(f"LLM недоступна по адресу {llm.base_url}. Запустите локальную модель (например, `ollama serve`).")

    corp = read_jsonl(RAW_DIR / "enron_corporate.jsonl")
    mkt = read_jsonl(RAW_DIR / "enron_marketing.jsonl")
    ru_h = [r for r in read_jsonl(RAW_DIR / "llmtrace.jsonl") if r["language"] == "ru" and r["label"] == 0]

    rng = random.Random(args.seed)
    tasks: List[Tuple[str, callable]] = []
    for s in corp[: args.full_corporate]:
        tasks.append((make_id("full", s["id"], "corporate"), lambda s=s: task_full(llm, s, "corporate", random.Random(s["id"]))))
    for s in mkt[: args.full_marketing]:
        tasks.append((make_id("full", s["id"], "marketing"), lambda s=s: task_full(llm, s, "marketing", random.Random(s["id"]))))
    methods = ["rewrite", "insert", "append"]
    pc = corp[args.full_corporate: args.full_corporate + args.partial_corporate]
    pm = mkt[args.full_marketing: args.full_marketing + args.partial_marketing]
    pr = ru_h[: args.partial_ru]
    for i, s in enumerate(pc + pm + pr):
        m = methods[i % 3]
        tasks.append((make_id("partial", s["id"], m), lambda s=s, m=m: task_partial(llm, s, m, random.Random(s["id"]))))

    # перемешиваем, чтобы при досрочной остановке в данных были все типы заданий
    random.Random(args.seed).shuffle(tasks)
    done = {r["id"] for r in read_jsonl(args.out)}
    todo = [(tid, fn) for tid, fn in tasks if tid not in done]
    log.info("Заданий: %d, уже выполнено: %d, осталось: %d", len(tasks), len(tasks) - len(todo), len(todo))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0, ok, fail = time.time(), 0, 0
    with args.out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fn): tid for tid, fn in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                log.error("Ошибка задания %s: %s", futs[fut], e)
                rec = None
            if rec is not None and not validate_record(rec):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                ok += 1
            else:
                fail += 1
            if i % 20 == 0:
                el = time.time() - t0
                log.info("Прогресс %d/%d (ok=%d, отклонено=%d), %.1f c/письмо", i, len(todo), ok, fail, el / i)

    # вариации — от уже сгенерированных полных писем
    gen = read_jsonl(args.out)
    done = {r["id"] for r in gen}
    parents = [r for r in gen if r["generation_method"] == "full"][: args.variation]
    with args.out.open("a", encoding="utf-8") as f:
        for p in parents:
            if make_id("variation", p["id"]) in done:
                continue
            try:
                rec = task_variation(llm, p, rng)
            except Exception as e:
                log.error("Ошибка вариации: %s", e)
                continue
            if rec and not validate_record(rec):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Готово за %.1f мин", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
