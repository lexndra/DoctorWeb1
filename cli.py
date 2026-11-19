#!/usr/bin/env python3
"""Интерфейс командной строки DW-1 (п. 4.8.1 ТЗ).

Команды:
  prepare-data   загрузка исходных корпусов (Enron, LLMTrace) → data/raw
  generate       генерация писем локальной LLM (OpenAI-совместимый API) → data/generated
  build-dataset  валидация, дедупликация, разбиение train/val/test, архив → data/dataset/<версия>
  train          обучение метода на признаках + калибровка порогов на val → models/
  evaluate       расчёт всех метрик ТЗ на test → results/metrics/
  analyze        анализ одного письма (EML / HTML / текст): индекс, вердикт, объяснение, JSON
  batch          пакетный анализ каталога: JSON по каждому письму + сводный JSON

Примеры:
  python cli.py analyze --eml samples/ai_phishing_evasion.eml --output report.json
  python cli.py batch --dir data/case --out-dir results/batch
  python cli.py evaluate --methods features,lm
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

GREEN, RED, BOLD, RESET = "\033[1;32m", "\033[1;31m", "\033[1m", "\033[0m"
log = logging.getLogger("dw1")


def _settings(args):
    from src.config import Settings

    st = Settings()
    if getattr(args, "method", None):
        st.method = args.method
    if getattr(args, "threshold", None) is not None:
        st.threshold = args.threshold
    if getattr(args, "images", False):
        st.enable_images = True
    return st


def cmd_analyze(args):
    from src.pipeline import EmailAnalysisPipeline

    pipe = EmailAnalysisPipeline(settings=_settings(args))
    t0 = time.time()
    if args.eml:
        report = pipe.analyze_email(args.eml)
    elif args.html:
        report = pipe.analyze_direct_text(text="", subject=args.subject, html_content=Path(args.html).read_text(encoding="utf-8", errors="replace"),
                                          file_id=Path(args.html).stem)
    else:
        report = pipe.analyze_direct_text(text=args.text, subject=args.subject)
    color = RED if report.overall_ai_index >= report.threshold else GREEN
    print(f"\n{BOLD}Индекс AI-генерации:{RESET} {color}{report.overall_ai_index:.3f}{RESET} (порог {report.threshold:.3f})")
    print(f"{BOLD}Вердикт:{RESET} {color}{report.verdict}{RESET}")
    print(f"{BOLD}Объяснение:{RESET}\n{report.explanation}")
    print(f"\nВремя анализа: {time.time() - t0:.2f} c")
    if args.output:
        Path(args.output).write_text(report.to_json(), encoding="utf-8")
        print(f"JSON сохранён: {args.output}")


def cmd_batch(args):
    from src.pipeline import EmailAnalysisPipeline

    src_dir, out_dir = Path(args.dir), Path(args.out_dir)
    if not src_dir.is_dir():
        sys.exit(f"Каталог не найден: {src_dir}")
    files = sorted(p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() not in {".zip", ".pdf", ".json", ".txt", ".md"}
                   and not p.name.startswith("."))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "emails").mkdir(exist_ok=True)
    pipe = EmailAnalysisPipeline(settings=_settings(args))
    log.info("Пакетный анализ: %d файлов, метод %s, порог %.3f", len(files), pipe.method, pipe.threshold)
    summary, errors, ai = [], [], 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        rel = f.relative_to(src_dir)
        out_file = out_dir / "emails" / (str(rel).replace("/", "__") + ".json")
        try:
            if args.resume and out_file.exists():
                d = json.loads(out_file.read_text(encoding="utf-8"))
            else:
                d = pipe.analyze_email(f).to_dict()
                d["file_path"] = str(rel)
                out_file.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            is_ai = d["overall_ai_index"] >= d["threshold"]
            ai += is_ai
            summary.append({"file": str(rel), "folder": rel.parts[0] if len(rel.parts) > 1 else "", "overall_ai_index": d["overall_ai_index"],
                            "verdict": d["verdict"], "ai_zones": d["ai_zones_detected"], "prompt_leakage": bool(d.get("prompt_leakage")),
                            "hidden_elements": d["html_signals"].get("hidden_elements_count", 0)})
        except Exception as e:  # повреждённое письмо не прерывает пакет (п. 4.1.1.3)
            log.warning("Ошибка разбора %s: %s", rel, e)
            errors.append({"file": str(rel), "error": str(e)[:300]})
        if i % 200 == 0:
            log.info("  %d/%d (%.1f c)", i, len(files), time.time() - t0)
    el = time.time() - t0
    by_folder = {}
    for s in summary:
        b = by_folder.setdefault(s["folder"] or "(корень)", {"total": 0, "ai": 0, "prompt_leakage": 0})
        b["total"] += 1
        b["ai"] += s["overall_ai_index"] >= pipe.threshold
        b["prompt_leakage"] += s["prompt_leakage"]
    res = {"directory": str(src_dir), "method": pipe.method, "threshold": pipe.threshold, "total_files": len(files),
           "processed": len(summary), "errors_count": len(errors), "ai_count": int(ai), "human_count": len(summary) - int(ai),
           "elapsed_seconds": round(el, 2), "sec_per_email": round(el / max(1, len(files)), 4), "by_folder": by_folder,
           "errors": errors, "emails": summary}
    (out_dir / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОбработано {len(summary)}/{len(files)} (ошибок {len(errors)}) за {el:.1f} c; AI: {ai}, человек: {len(summary) - ai}")
    for k, v in by_folder.items():
        print(f"  {k}: {v['ai']}/{v['total']} признаны AI, утечка промпта: {v['prompt_leakage']}")
    print(f"Сводный JSON: {out_dir / 'summary.json'}; по письмам: {out_dir / 'emails'}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="DW-1: выявление AI-assisted происхождения писем")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="загрузить исходные корпуса")
    g = sub.add_parser("generate", help="сгенерировать письма локальной LLM")
    g.add_argument("rest", nargs=argparse.REMAINDER)
    b = sub.add_parser("build-dataset", help="собрать датасет")
    b.add_argument("--version", default="v1")
    t = sub.add_parser("train", help="обучить и откалибровать метод на признаках")
    t.add_argument("--dataset", default="data/dataset/v1")
    e = sub.add_parser("evaluate", help="метрики ТЗ на test")
    e.add_argument("--dataset", default="data/dataset/v1")
    e.add_argument("--methods", default="features")

    for name in ("analyze", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--method", choices=["features", "lm", "combined"], default=None, help="метод оценки (по умолчанию DW1_METHOD)")
        p.add_argument("--threshold", type=float, default=None, help="порог (по умолчанию — из калибровки)")
        p.add_argument("--images", action="store_true", help="включить анализ изображений (OCR + детектор)")
        if name == "analyze":
            grp = p.add_mutually_exclusive_group(required=True)
            grp.add_argument("--eml", "-e")
            grp.add_argument("--html")
            grp.add_argument("--text", "-t")
            p.add_argument("--subject", "-s", default=None)
            p.add_argument("--output", "-o", default=None)
        else:
            p.add_argument("--dir", "-d", required=True)
            p.add_argument("--out-dir", "-o", default="results/batch")
            p.add_argument("--resume", action="store_true", help="пропускать уже обработанные письма")

    args = ap.parse_args()
    if args.command == "prepare-data":
        from src.datagen import sources

        sys.argv = ["sources"]
        sources.main()
    elif args.command == "generate":
        from src.datagen import generate

        sys.argv = ["generate"] + [a for a in args.rest if a != "--"]
        generate.main()
    elif args.command == "build-dataset":
        from src.datagen import build_dataset

        sys.argv = ["build_dataset", "--version", args.version]
        build_dataset.main()
    elif args.command == "train":
        from src.train_detector import run

        run(Path(args.dataset), _settings(args).models_dir)
    elif args.command == "evaluate":
        from src.evaluation import run as ev

        sys.argv = ["evaluate", "--dataset", args.dataset, "--methods", args.methods]
        ev.main()
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()
