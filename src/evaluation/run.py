"""Расчёт всех метрик таблицы 3 ТЗ одной командой (принцип «проверяемость»).

Оценка выполняется только на тестовой выборке; пороги берутся из файлов калибровки, подобранных на val.
Для метода lm/combined, если файла калибровки нет, порог подбирается здесь же — на val (не на test).

Выход: results/metrics/metrics_<method>.json, roc_<method>.png, report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import Settings
from src.evaluation.tz_metrics import (
    MASKINGS, bootstrap_auc_ci, calibrate_doc_threshold, calibrate_sentence_threshold, char_prf, fpr_at,
    predicted_intervals, prf_at, safe_auc,
)
from src.text_utils import overlap

log = logging.getLogger(__name__)

TZ_REQUIREMENTS = [
    ("roc_auc", "ROC-AUC по бинарной метке", ">=", 0.70),
    ("f1", "F1-мера на калиброванном пороге", ">=", 0.70),
    ("fpr_human", "FPR на письмах, написанных человеком", "<=", 0.20),
    ("fpr_marketing", "FPR на легитимных маркетинговых рассылках", "<=", 0.20),
    ("roc_auc_partial", "ROC-AUC: частично сгенерированные vs человек", ">=", 0.70),
    ("char_f1_partial", "Локализация фрагментов, F1 по символам (частично сгенерированные)", ">=", 0.60),
    ("zone_recall", "Полнота выявления сгенерированных зон", None, None),
    ("masking_auc_min", "Сохранность ROC-AUC при маскировке (минимум по сценариям)", None, None),
    ("roc_auc_ci_width", "Ширина 95% ДИ ROC-AUC (бутстреп, 1000 итераций)", None, None),
]


def read_jsonl(p: Path) -> List[Dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def score_all(detector, records):
    outs = []
    t0 = time.time()
    for i, r in enumerate(records, 1):
        outs.append(detector.score(r["text"]))
        if i % 500 == 0:
            log.info("  оценено %d/%d (%.1f c)", i, len(records), time.time() - t0)
    return outs


def ensure_calibration(method: str, detector, settings: Settings, val: List[Dict]) -> Dict:
    cal = settings.calibration(method)
    if cal:
        return cal
    log.info("Калибровка метода %s на val (%d записей)", method, len(val))
    outs = score_all(detector, val)
    y = np.array([r["label"] for r in val])
    s = np.array([o.prob for o in outs])
    mk = np.array([r["label"] == 0 and r["email_type"] == "marketing" for r in val])
    dc = calibrate_doc_threshold(y, s, mk)
    li = [i for i, r in enumerate(val) if r["label"] == 1]
    sc = calibrate_sentence_threshold([val[i]["text"] for i in li], [val[i]["ai_char_intervals"] for i in li], [outs[i].sentences for i in li])
    cal = {"method": method, "doc_threshold": dc["threshold"], "sentence_threshold": sc["threshold"], "calibrated_on": "val",
           "val_metrics": {"roc_auc": safe_auc(y, s), **{k: v for k, v in dc.items() if k != "threshold"}, "char_f1": sc["val_char_f1"]}}
    name = {"lm": "lm_calibration.json", "combined": "combined_calibration.json"}.get(method, "calibration.json")
    (settings.models_dir / name).write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    return cal


def evaluate_method(method: str, detector, settings: Settings, test: List[Dict], val: List[Dict], out_dir: Path, seed: int = 42) -> Dict:
    cal = ensure_calibration(method, detector, settings, val)
    thr, sthr = float(cal["doc_threshold"]), float(cal.get("sentence_threshold", 0.5))
    log.info("[%s] оценка на test (%d), порог %.4f, порог локализации %.3f", method, len(test), thr, sthr)
    t0 = time.time()
    outs = score_all(detector, test)
    sec_per_doc = (time.time() - t0) / max(1, len(test))
    y = np.array([r["label"] for r in test])
    s = np.array([o.prob for o in outs])
    human = y == 0
    mk = np.array([r["email_type"] == "marketing" for r in test])
    mixed = np.array([r["label_name"] == "mixed" for r in test])

    m: Dict[str, object] = {"method": method, "n_test": len(test), "threshold": thr, "sentence_threshold": sthr,
                            "calibration_source": cal.get("calibrated_on", "val"), "sec_per_text": round(sec_per_doc, 4)}
    m["roc_auc"] = safe_auc(y, s)
    m["roc_auc_ci95"] = bootstrap_auc_ci(y, s, 1000, seed)
    m["roc_auc_ci_width"] = m["roc_auc_ci95"]["width"]
    m.update(prf_at(y, s, thr))
    m["fpr_human"] = fpr_at(y, s, thr)
    hm = human & mk
    m["fpr_marketing"] = float((s[hm] >= thr).mean()) if hm.any() else None
    m["n_marketing_human"] = int(hm.sum())
    sel = human | mixed
    m["roc_auc_partial"] = safe_auc(y[sel], s[sel])

    li = [i for i in range(len(test)) if mixed[i]]
    pred_iv = {i: predicted_intervals(outs[i].sentences, sthr) for i in range(len(test))}
    m["char_partial"] = char_prf([test[i]["text"] for i in li], [test[i]["ai_char_intervals"] for i in li], [pred_iv[i] for i in li])
    m["char_f1_partial"] = m["char_partial"]["f1"]
    ai_idx = [i for i in range(len(test)) if y[i] == 1]
    m["char_all_ai"] = char_prf([test[i]["text"] for i in ai_idx], [test[i]["ai_char_intervals"] for i in ai_idx], [pred_iv[i] for i in ai_idx])
    hit = [any(overlap(tuple(p), tuple(g)) > 0 for p in pred_iv[i] for g in test[i]["ai_char_intervals"]) for i in ai_idx]
    m["zone_recall"] = float(np.mean(hit)) if hit else None

    # маскировка AI-признаков: изменяем только AI-тексты теста
    rng = random.Random(seed)
    masking = {}
    for name, fn in MASKINGS.items():
        ms = s.copy()
        for i in ai_idx:
            ms[i] = detector.score(fn(test[i]["text"], rng)).prob
        masking[name] = {"roc_auc": safe_auc(y, ms), "recall": prf_at(y, ms, thr)["recall"]}
    m["masking"] = masking
    m["masking_auc_min"] = min(v["roc_auc"] for v in masking.values())

    # разрезы
    def cut(key):
        res = {}
        for val_ in sorted({str(r.get(key)) for r in test}):
            idx = np.array([str(r.get(key)) == val_ for r in test])
            res[val_] = {"n": int(idx.sum()), "roc_auc": safe_auc(y[idx], s[idx]), **prf_at(y[idx], s[idx], thr),
                         "fpr": fpr_at(y[idx], s[idx], thr)}
        return res

    m["by_language"] = cut("language")
    m["by_email_type"] = cut("email_type")
    m["by_source"] = cut("source")
    # срез без «разницы эпох»: письма Enron (человек) против писем LLM, сгенерированных на темы тех же писем
    em = np.array([r["source"] in ("enron_aeslc", "enron_newsletters", "dw1_generated") and r["language"] == "en" for r in test])
    m["email_matched"] = {"n": int(em.sum()), "roc_auc": safe_auc(y[em], s[em]), **prf_at(y[em], s[em], thr),
                          "fpr": fpr_at(y[em], s[em], thr)} if em.any() else None
    cs = np.array([r["source"] == "customer_case" for r in test])
    m["customer_case_recall"] = float((s[cs] >= thr).mean()) if cs.any() else None
    m["customer_case_n"] = int(cs.sum())
    m["recall_by_generation_method"] = {g: float((s[np.array([r["generation_method"] == g for r in test])] >= thr).mean())
                                        for g in sorted({r["generation_method"] for r in test}) if g != "human"}

    m["tz_compliance"] = {}
    for key, title, op, lim in TZ_REQUIREMENTS:
        v = m.get(key)
        ok = None if (op is None or v is None) else (v >= lim if op == ">=" else v <= lim)
        m["tz_compliance"][key] = {"title": title, "value": v, "requirement": f"{op} {lim}" if op else "не предъявляется", "pass": ok}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{method}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_roc(y, s, mixed, human, m, out_dir / f"roc_{method}.png")
    np.save(out_dir / f"scores_{method}.npy", s)
    return m


def plot_roc(y, s, mixed, human, m, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    f, t, _ = roc_curve(y, s)
    ax.plot(f, t, lw=2, label=f"Все AI vs человек (AUC={m['roc_auc']:.3f})")
    sel = human | mixed
    if len(set(y[sel].tolist())) == 2:
        f2, t2, _ = roc_curve(y[sel], s[sel])
        ax.plot(f2, t2, lw=2, ls="--", label=f"Частично сгенерир. vs человек (AUC={m['roc_auc_partial']:.3f})")
    ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1)
    ax.axvline(0.2, color="#c33", lw=1, ls=":", label="FPR = 0.20 (ТЗ)")
    ax.scatter([m["fpr_human"]], [m["recall"]], color="#c33", zorder=5, label=f"Рабочая точка (порог {m['threshold']:.2f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(f"ROC — метод {m['method']}, test n={m['n_test']}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def performance_check(settings: Settings, case_dir: Path, n: int = 300) -> Dict:
    """Скорость метода на основе признаков на реальных письмах кейса (табл. 2 ТЗ)."""
    from src.pipeline import EmailAnalysisPipeline

    files = sorted(p for p in case_dir.rglob("*") if p.is_file() and not p.name.endswith((".zip", ".pdf")))[:n]
    if not files:
        return {}
    pipe = EmailAnalysisPipeline(settings=settings)
    t0 = time.time()
    ok = 0
    for f in files:
        try:
            pipe.analyze_email(f)
            ok += 1
        except Exception:
            pass
    el = time.time() - t0
    per = el / len(files)
    return {"emails": len(files), "parsed_ok": ok, "sec_per_email": round(per, 4),
            "projected_sec_per_1000": round(per * 1000, 1),
            "req_single_email_le_1s": per <= 1.0, "req_1000_emails_le_300s": per * 1000 <= 300}


def write_report(results: Dict[str, Dict], perf: Dict, manifest: Dict, train_params: Dict, out: Path):
    L = ["# Отчёт о метриках DW-1 (п. 4.1.11 ТЗ)", "",
         f"Датасет: `{manifest.get('dataset_version')}`; тест: {next(iter(results.values()))['n_test']} записей. "
         "Пороги подобраны на val, тест для настройки не использовался.", ""]
    for method, m in results.items():
        L += [f"## Метод `{method}`", "", "| Метрика (табл. 3 ТЗ) | Значение | Требование | Статус |", "|---|---:|---:|:---:|"]
        for key, c in m["tz_compliance"].items():
            v = c["value"]
            vs = "—" if v is None else f"{v:.3f}"
            st = "—" if c["pass"] is None else ("PASS" if c["pass"] else "**FAIL**")
            L.append(f"| {c['title']} | {vs} | {c['requirement']} | {st} |")
        L += ["", f"Precision {m['precision']:.3f}, recall {m['recall']:.3f}, accuracy {m['accuracy']:.3f}; "
                  f"порог {m['threshold']:.3f}; 95% ДИ ROC-AUC [{m['roc_auc_ci95']['low']:.3f}; {m['roc_auc_ci95']['high']:.3f}].", "",
              "Разрез по языку:", "", "| Язык | n | ROC-AUC | F1 | FPR |", "|---|---:|---:|---:|---:|"]
        for k, v in m["by_language"].items():
            L.append(f"| {k} | {v['n']} | {fmt(v['roc_auc'])} | {v['f1']:.3f} | {fmt(v['fpr'])} |")
        L += ["", "Разрез по источнику:", "", "| Источник | n | ROC-AUC | F1 | FPR |", "|---|---:|---:|---:|---:|"]
        for k, v in m["by_source"].items():
            L.append(f"| {k} | {v['n']} | {fmt(v['roc_auc'])} | {v['f1']:.3f} | {fmt(v['fpr'])} |")
        if m.get("email_matched"):
            e = m["email_matched"]
            L += ["", f"Срез «письма, согласованные по теме и эпохе» (Enron человек vs LLM-письма на те же темы): n={e['n']}, "
                      f"ROC-AUC {fmt(e['roc_auc'])}, F1 {e['f1']:.3f}, FPR {fmt(e['fpr'])}."]
        if m.get("customer_case_recall") is not None:
            L += ["", f"Recall на отложенных письмах кейса Заказчика (samples_prompt, кампании не пересекаются с train): "
                      f"{m['customer_case_recall']:.3f} (n={m['customer_case_n']})."]
        L += ["", "Recall по способу генерации: " + ", ".join(f"{k} {v:.2f}" for k, v in m["recall_by_generation_method"].items()), "",
              "Маскировка AI-признаков (ROC-AUC): " + ", ".join(f"{k} {v['roc_auc']:.3f}" for k, v in m["masking"].items()), "",
              f"![ROC](roc_{method}.png)", ""]
    if perf:
        L += ["## Производительность (табл. 2 ТЗ)", "",
              f"- Письма кейса: {perf['emails']}, среднее время анализа одного письма: {perf['sec_per_email']} c (требование ≤ 1 c)",
              f"- Прогноз на 1000 писем: {perf['projected_sec_per_1000']} c (требование ≤ 300 c)"]
        if train_params:
            L.append(f"- Обучение и калибровка: {train_params.get('total_seconds_with_calibration')} c на CPU (требование ≤ 600 c)")
        L.append("")
    out.write_text("\n".join(L), encoding="utf-8")


def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Расчёт метрик ТЗ на тестовой выборке")
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset/v1"))
    ap.add_argument("--methods", default="features", help="через запятую: features,lm,combined")
    ap.add_argument("--out", type=Path, default=Path("results/metrics"))
    ap.add_argument("--case-dir", type=Path, default=Path("data/case"))
    ap.add_argument("--perf-n", type=int, default=300)
    ap.add_argument("--max-test", type=int, default=0, help="ограничить test (для LM на CPU); 0 — весь")
    args = ap.parse_args()

    from src.pipeline import build_detector

    from src.datagen.io import load_split

    test, val = load_split(args.dataset, "test"), load_split(args.dataset, "val")
    if args.max_test:
        rng = random.Random(0)
        test = rng.sample(test, min(args.max_test, len(test)))
        val = rng.sample(val, min(args.max_test, len(val)))
    results = {}
    for method in [x.strip() for x in args.methods.split(",") if x.strip()]:
        st = Settings()
        st.method = method
        try:
            det = build_detector(st)
        except Exception as e:
            log.error("Метод %s недоступен: %s", method, e)
            continue
        results[method] = evaluate_method(method, det, st, test, val, args.out)
    st = Settings()
    st.method = "features"
    perf = performance_check(st, args.case_dir, args.perf_n) if args.case_dir.exists() else {}
    (args.out / "performance.json").write_text(json.dumps(perf, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8")) if (args.dataset / "manifest.json").exists() else {}
    tp_path = st.models_dir / "training_params.json"
    tp = json.loads(tp_path.read_text(encoding="utf-8")) if tp_path.exists() else {}
    if results:
        write_report(results, perf, manifest, tp, args.out / "report.md")
        log.info("Отчёт: %s", args.out / "report.md")


if __name__ == "__main__":
    main()
