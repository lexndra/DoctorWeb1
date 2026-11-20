"""Сборка размеченного датасета DW1 (п. 4.1.2.2, 4.1.9.2 ТЗ).

Шаги:
  1. Объединение источников (data/raw/*.jsonl) и сгенерированных писем (data/generated/generated.jsonl).
  2. Валидация каждой записи по схеме (src/datagen/schema.py), отбраковка невалидных.
  3. Дедупликация похожих писем: точные дубли по нормализованному тексту и близкие дубли
     (косинус TF-IDF символьных n-грамм >= порога) — оставляется первая запись.
  4. Разбиение train/val/test по группам (исходное письмо и все его производные попадают в одну выборку);
     для LLMTrace сохраняется оригинальное разбиение авторов корпуса.
  5. Сохранение: data/dataset/<version>/{train,val,test}.jsonl, schema.json, manifest.json и архив ZIP.

Запуск:  python -m src.datagen.build_dataset --version v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.datagen.schema import SCHEMA, SCHEMA_VERSION, validate_record

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def dedup(records: List[Dict], threshold: float) -> List[Dict]:
    """Удаление точных и близких дубликатов. Сравниваются только записи с одинаковой меткой и
    из разных групп (производные одного письма — это не утечка, они остаются в одной выборке)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    seen, uniq = set(), []
    for r in records:
        k = hashlib.md5(_norm(r["text"]).encode()).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    exact_removed = len(records) - len(uniq)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform([r["text"] for r in uniq])
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    drop = set()
    for i in range(len(uniq)):
        if i in drop:
            continue
        for d, j in zip(dist[i][1:], idx[i][1:]):
            if j <= i or j in drop:
                continue
            if 1 - d >= threshold and uniq[i]["group_id"] != uniq[j]["group_id"]:
                drop.add(j)
    out = [r for i, r in enumerate(uniq) if i not in drop]
    log.info("Дедупликация: точных дублей %d, близких %d (порог %.2f)", exact_removed, len(drop), threshold)
    return out


def assign_splits(records: List[Dict], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> None:
    groups = sorted({r["group_id"] for r in records if "llmtrace_split" not in r})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    gsplit = {g: ("train" if i < a else "val" if i < b else "test") for i, g in enumerate(groups)}
    for r in records:
        r["split"] = r.get("llmtrace_split") or gsplit[r["group_id"]]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Сборка датасета DW1")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--generated", type=Path, default=Path("data/generated/generated.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/dataset"))
    ap.add_argument("--dedup-threshold", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records: List[Dict] = []
    for name in ("enron_corporate", "enron_marketing", "llmtrace", "customer_case"):
        records += read_jsonl(args.raw_dir / f"{name}.jsonl")
    gen = read_jsonl(args.generated)
    records += gen
    log.info("Всего записей до проверки: %d (сгенерировано: %d)", len(records), len(gen))

    # группы llmtrace: производные (partial_ru) наследуют llmtrace_split родителя — уже записано генератором
    valid, rejected = [], Counter()
    for r in records:
        errs = validate_record(r)
        if errs:
            rejected[errs[0]] += 1
        else:
            valid.append(r)
    log.info("Отбраковано по схеме: %d %s", sum(rejected.values()), dict(rejected))

    valid = dedup(valid, args.dedup_threshold)
    assign_splits(valid, seed=args.seed)

    # проверка отсутствия утечек: ни одна группа не встречается в двух выборках
    g2s = defaultdict(set)
    for r in valid:
        g2s[r["group_id"]].add(r["split"])
    leaks = [g for g, s in g2s.items() if len(s) > 1]
    assert not leaks, f"утечка групп между выборками: {leaks[:5]}"

    out = args.out_dir / args.version
    priv = Path("data/private") / args.version  # письма Заказчика — только локально (п. 4.5.2 ТЗ)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val", "test"):
        part = [r for r in valid if r["split"] == split]
        public = [r for r in part if r["source"] != "customer_case"]
        private = [r for r in part if r["source"] == "customer_case"]
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in public:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if private:
            priv.mkdir(parents=True, exist_ok=True)
            with (priv / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for r in private:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter((r["language"], r["email_type"], r["label_name"]) for r in part)
        stats[split] = {"total": len(part), "public": len(public), "private_customer_case": len(private),
                        "by_lang_type_label": {"/".join(k): v for k, v in sorted(c.items())}}

    (out / "schema.json").write_text(json.dumps({"version": SCHEMA_VERSION, "fields": SCHEMA}, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.jsonl"))}
    manifest = {
        "dataset_version": args.version,
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "dedup_threshold": args.dedup_threshold,
        "split_ratios_own_data": [0.7, 0.15, 0.15],
        "sources": {
            "enron_aeslc": "Yale-LILY/aeslc (деловая переписка Enron, человек)",
            "enron_newsletters": "corbt/enron-emails, внешние рассылки 2000–2002 гг. (маркетинг, человек)",
            "llmtrace_detection": "iitolstykh/LLMTrace_detection (RU/EN, человек/ИИ/смешанные, Apache-2.0)",
            "dw1_generated": "сгенерировано локальной LLM (src/datagen/generate.py)",
            "customer_case": "архив кейса Заказчика, папка samples_prompt (письма, сгенерированные LLM по промпту)",
        },
        "generator_models": dict(Counter(r["generator_model"] for r in valid if r["generator_model"]).most_common(50)),
        "rejected_by_schema": dict(rejected),
        "stats": stats,
        "sha256": hashes,
        "private_part": f"data/private/{args.version} (письма кейса Заказчика, в репозиторий не входят)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zpath = args.out_dir / f"dw1_dataset_{args.version}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, arcname=f"{args.version}/{p.name}")
    log.info("Датасет сохранён: %s (%s); архив %s", out, {k: (v["public"], v["private_customer_case"]) for k, v in stats.items()}, zpath)


if __name__ == "__main__":
    main()
