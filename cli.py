#!/usr/bin/env python3
"""CLI-интерфейс аналитического модуля детекции AI-генерации писем (Doctor Web DW-1).

Поддерживаемые команды:
  1. analyze  — анализ текста, HTML или EML письма с цветным отчетом в консоли и экспортом в JSON.
  2. batch    — пакетная обработка каталога писем (до 1941+ писем) с сохранением сводного отчета.

Примеры запуска:
  python cli.py analyze --text "Ваш текст для проверки"
  python cli.py analyze --html email.html
  python cli.py analyze --eml letter.eml --output result.json
  python cli.py batch --dir ./emails_folder/ --output summary.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.pipeline import EmailAnalysisPipeline


# Цветовая палитра ANSI для терминала
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_single_report(report) -> None:
    """Выводит красивый структурированный отчет по письму в терминал."""
    is_ai = "AI" in report.verdict
    badge_color = RED if is_ai else GREEN

    print("\n" + "=" * 65)
    print(f" {BOLD}АНАЛИЗ СОДЕРЖИМОГО:{RESET} {report.file_id}")
    print("=" * 65)
    if report.metadata.get("subject"):
        print(f"{BOLD}Тема:{RESET}       {report.metadata.get('subject')}")
    if report.metadata.get("sender"):
        print(f"{BOLD}От кого:{RESET}    {report.metadata.get('sender')}")
    if report.metadata.get("date"):
        print(f"{BOLD}Дата:{RESET}       {report.metadata.get('date')}")
    print("-" * 65)

    print(f"{BOLD}ИТОГОВЫЙ ВЕРДИКТ:{RESET}   {badge_color}{report.verdict}{RESET}")
    print(
        f"{BOLD}ИНДЕКС AI-ГЕНЕРАЦИИ:{RESET} {badge_color}{report.overall_ai_index:.1%}{RESET} "
        f"(порог: {report.threshold:.2f})"
    )

    print(f"\n{BOLD}РАЗБИВКА ПО ЗОНАМ ПИСЬМА:{RESET}")
    print(f"  {'Зона':<12} | {'Слов':<6} | {'ИИ-слов':<8} | {'Вероятность ИИ':<16} | {'Статус'}")
    print("  " + "-" * 60)
    for z in report.zones:
        z_badge = f"{RED}ИИ{RESET}" if z.is_ai else f"{GREEN}Человек{RESET}"
        print(
            f"  {z.name:<12} | {z.words_count:<6} | {z.ai_words_count:<8} | "
            f"{z.ai_probability:>14.1%}  | {z_badge}"
        )

    # Изображения
    if report.images:
        print(f"\n{BOLD}ВЛОЖЕННЫЕ ИЗОБРАЖЕНИЯ ({len(report.images)}):{RESET}")
        for img in report.images:
            img_badge = f"{RED}ИИ{RESET}" if img.is_ai else f"{GREEN}Человек{RESET}"
            print(f"  • {img.file_name}: {img_badge} (вероятность: {img.ai_probability:.1%})")
            if img.has_qr:
                print(f"    {YELLOW}[QR-код обнаружен]{RESET} Ссылка: {', '.join(img.qr_codes)}")
            if img.has_text:
                print(f"    {CYAN}[OCR текст]:{RESET} {img.ocr_text[:80]}...")

    # HTML и сокрытие
    html_sig = report.html_signals
    if html_sig and (html_sig.get("hidden_elements_count", 0) > 0 or html_sig.get("suspicious_links_count", 0) > 0):
        print(f"\n{YELLOW}{BOLD}СИГНАЛЫ HTML-СОКРЫТИЯ И ОБХОДА ФИЛЬТРОВ:{RESET}")
        if html_sig.get("hidden_elements_count", 0) > 0:
            print(f"  • Скрытых элементов: {html_sig.get('hidden_elements_count')}")
            for ht in html_sig.get("hidden_texts", []):
                print(f"    Скрытый текст: {ht}")
        if html_sig.get("suspicious_links_count", 0) > 0:
            print(f"  • Подмена ссылок (phishing/spoofing): {html_sig.get('suspicious_links_count')}")

    # Заключение для SOC
    print(f"\n{BOLD}ОБЪЯСНЕНИЕ ДЛЯ АНАЛИТИКА SOC (на русском языке):{RESET}")
    print("-" * 65)
    print(report.explanation)
    print("=" * 65 + "\n")


def run_analyze(args, pipeline: EmailAnalysisPipeline):
    t0 = time.time()
    report = None

    if args.eml:
        eml_path = Path(args.eml)
        if not eml_path.exists():
            print(f"Ошибка: файл не найден: {eml_path}")
            sys.exit(1)
        try:
            report = pipeline.analyze_email(eml_path)
        except NotImplementedError as e:
            print(f"\n[!] {e}\n")
            sys.exit(1)

    elif args.html:
        html_path = Path(args.html)
        if not html_path.exists():
            print(f"Ошибка: файл не найден: {html_path}")
            sys.exit(1)
        html_content = html_path.read_text(encoding="utf-8")
        report = pipeline.analyze_direct_text(
            text="",
            subject=args.subject,
            html_content=html_content,
        )

    elif args.text:
        report = pipeline.analyze_direct_text(
            text=args.text,
            subject=args.subject,
        )

    elapsed = time.time() - t0
    if report:
        print_single_report(report)
        print(f"Время анализа: {elapsed:.2f} сек.")

        if args.output:
            out_path = Path(args.output)
            out_path.write_text(report.to_json(), encoding="utf-8")
            print(f"[+] JSON-отчет успешно сохранен в: {out_path}")


def run_batch(args, pipeline: EmailAnalysisPipeline):
    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        print(f"Ошибка: каталог не найден: {dir_path}")
        sys.exit(1)

    eml_files = list(dir_path.glob("*.eml")) + list(dir_path.glob("*.EML"))
    if not eml_files:
        print(f"В каталоге {dir_path} не найдено .eml файлов.")
        sys.exit(0)

    print(f"\n[Batch] Старт пакетной обработки {len(eml_files)} писем...")
    results = []
    ai_count = 0
    human_count = 0
    t0 = time.time()

    for i, f in enumerate(eml_files, 1):
        try:
            r = pipeline.analyze_email(f)
            results.append(r.to_dict())
            if "AI" in r.verdict:
                ai_count += 1
            else:
                human_count += 1
            print(f"[{i}/{len(eml_files)}] {f.name:<30} -> {r.verdict} ({r.overall_ai_index:.1%})")
        except NotImplementedError as e:
            print(f"\n[!] Ошибка: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[{i}/{len(eml_files)}] {f.name:<30} -> ОШИБКА РАЗБОРА: {e}")
            results.append({
                "file_id": f.name,
                "error": str(e),
                "is_corrupted": True
            })

    elapsed = time.time() - t0
    avg_speed = len(eml_files) / elapsed if elapsed > 0 else 0.0

    print("\n" + "=" * 55)
    print(f" {BOLD}ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ КАТАЛОГА{RESET}")
    print("=" * 55)
    print(f" Всего обработано писем:     {len(eml_files)}")
    print(f" Признано сгенерированными: {RED}{ai_count}{RESET} ({ai_count/len(eml_files)*100:.1f}%)")
    print(f" Признано написанными людьми: {GREEN}{human_count}{RESET} ({human_count/len(eml_files)*100:.1f}%)")
    print(f" Общее время обработки:      {elapsed:.1f} сек. ({avg_speed:.1f} писем/сек)")
    print("=" * 55)

    if args.output:
        out_file = Path(args.output)
        summary = {
            "total_processed": len(eml_files),
            "ai_count": ai_count,
            "human_count": human_count,
            "elapsed_seconds": round(elapsed, 2),
            "emails": results
        }
        out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[+] Сводный отчет сохранен в: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Doctor Web DW-1: Аналитический модуль выявления AI-генерации в письмах"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Анализ текста, HTML или EML письма")
    group = p_analyze.add_mutually_exclusive_group(required=True)
    group.add_argument("--eml", "-e", type=str, help="Путь к EML файлу")
    group.add_argument("--html", type=str, help="Путь к HTML файлу")
    group.add_argument("--text", "-t", type=str, help="Текст для проверки")

    p_analyze.add_argument("--subject", "-s", type=str, default=None, help="Тема письма (опционально)")
    p_analyze.add_argument("--output", "-o", type=str, default=None, help="Путь для сохранения JSON отчета")
    p_analyze.add_argument("--threshold", type=float, default=0.40, help="Порог AI-индекса (по умолч.: 0.40)")
    p_analyze.add_argument("--checkpoint", "-c", type=str, default=None, help="Путь к .pt файлу модели")
    p_analyze.add_argument("--no-images", action="store_true", help="Отключить анализ изображений")

    # batch
    p_batch = subparsers.add_parser("batch", help="Пакетный анализ каталога EML писем")
    p_batch.add_argument("--dir", "-d", type=str, required=True, help="Каталог с файлами .eml")
    p_batch.add_argument("--output", "-o", type=str, default="batch_summary.json", help="Выходной сводный JSON")
    p_batch.add_argument("--threshold", type=float, default=0.40, help="Порог AI-индекса (по умолч.: 0.40)")
    p_batch.add_argument("--checkpoint", "-c", type=str, default=None, help="Путь к .pt файлу модели")
    p_batch.add_argument("--no-images", action="store_true", help="Отключить анализ изображений")

    args = parser.parse_args()

    pipeline = EmailAnalysisPipeline(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        enable_images=(not args.no_images),
    )

    if args.command == "analyze":
        run_analyze(args, pipeline)
    elif args.command == "batch":
        run_batch(args, pipeline)


if __name__ == "__main__":
    main()
