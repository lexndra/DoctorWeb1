#!/usr/bin/env python
"""Независимая проверка детекторов DW-1 на сторонних бенчмарках (SeqXGPT, LLMTrace_classification,
Defactify, NTIRE).

Принципы (чтобы результаты можно было защищать):
  * случайная выборка с фиксированным seed (не «первые N строк»);
  * пороги берутся ТОЛЬКО из файлов калибровки, подобранных на нашем val
    (models_dir/calibration.json, models_dir/lm_calibration.json); на данных бенчмарков ничего
    не подбирается. Переменная DW1_THRESHOLD здесь намеренно игнорируется;
  * если калибровки для метода нет, отчитываемся пороговой-независимой ROC-AUC и метриками при
    фиксированном пороге 0.5 с явной пометкой;
  * ROC-AUC сопровождается бутстрэп-интервалом 95 %;
  * для изображений порог 0.5 задан априори.

Запуск:
    .venv/bin/python run_academic_validation.py --models-dir models --with-lm
Быстрый запуск без LM:  .venv/bin/python run_academic_validation.py --models-dir models_dev
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("academic_validation")

SEQXGPT_REPO = "TheItCrOw/SeqXGPT"
LLMTRACE_REPO = "iitolstykh/LLMTrace_classification"
DEFACTIFY_REPO = "Rajarshi-Roy-research/Defactify_Image_Dataset"
NTIRE_REPO = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
IMAGE_MODEL = "umm-maybe/AI-image-detector"
DEFACTIFY_GENERATORS = {0: "real (MS-COCO)", 1: "SD 2.1", 2: "SDXL", 3: "SD 3", 4: "DALL-E 3", 5: "Midjourney v6"}


# ----------------------------------------------------------------------------- метрики
def roc_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> Optional[List[float]]:
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2 or n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def threshold_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {"threshold": thr, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "accuracy": acc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


def full_metrics(y, s, thr: float, n_boot: int, seed: int) -> Dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    return {"roc_auc": roc_auc(y, s), "roc_auc_ci95": bootstrap_auc_ci(y, s, n_boot, seed), **threshold_metrics(y, s, thr)}


def recall_by_group(y, s, groups, thr: float) -> Dict[str, Dict]:
    """Для каждой группы: recall на ИИ-объектах и FPR на человеческих/реальных (если такие есть в группе)."""
    out: Dict[str, Dict] = {}
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        d: Dict = {"n": int(m.sum()), "mean_score": float(s[m].mean())}
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.any():
            d["n_ai"], d["recall"] = int(pos.sum()), float((s[pos] >= thr).mean())
        if neg.any():
            d["n_human"], d["fpr"] = int(neg.sum()), float((s[neg] >= thr).mean())
        out[str(g)] = d
    return out


# ----------------------------------------------------------------------------- графики
def plot_roc(path: Path, title: str, curves: Dict[str, tuple]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, s) in curves.items():
        y, s = np.asarray(y), np.asarray(s, float)
        if len(set(y.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="случайный (0.5)")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(path: Path, title: str, m: Dict, labels=("человек", "ИИ")):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1], [f"пред. {labels[0]}", f"пред. {labels[1]}"])
    ax.set_yticks([0, 1], [f"ист. {labels[0]}", f"ист. {labels[1]}"])
    ax.set_title(f"{title}\nпорог={m['threshold']:.4g}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- детекторы текста
class TextMethod:
    def __init__(self, name: str, scorer: Callable[[str], float], threshold: float, threshold_source: str, info: Dict):
        self.name, self.scorer, self.threshold, self.threshold_source, self.info = name, scorer, threshold, threshold_source, info


def load_text_methods(args) -> List[TextMethod]:
    from src.config import Settings
    from src.pipeline import build_detector

    methods: List[TextMethod] = []
    models_dir = Path(args.models_dir).resolve()
    names = ["features"] + (["lm"] if args.with_lm else [])
    for name in names:
        cal_file = models_dir / ("calibration.json" if name == "features" else "lm_calibration.json")
        st = Settings()
        st.method, st.models_dir, st.threshold = name, models_dir, None
        if name == "lm":
            if args.lm_checkpoint:
                st.lm_checkpoint = args.lm_checkpoint
            os.environ["DW1_LM_DEVICE"] = args.lm_device  # LMDetector читает устройство из окружения
        t0 = time.time()
        det = build_detector(st)
        info = {"load_seconds": round(time.time() - t0, 1)}
        if name == "lm":
            info["checkpoint"] = Path(str(getattr(det, "checkpoint", ""))).name
            info["device"] = str(getattr(det, "device", ""))
        else:
            info["model_file"] = f"{models_dir.name}/feature_model.json"
            tp = models_dir / "training_params.json"
            if tp.exists():
                p = json.loads(tp.read_text(encoding="utf-8"))
                info.update({k: p.get(k) for k in ("model", "dataset_version", "model_sha256")})
        if cal_file.exists():
            cal = json.loads(cal_file.read_text(encoding="utf-8"))
            thr = float(cal["doc_threshold"])
            vm = cal.get("val_metrics", {})
            info["own_val_metrics"] = {k: round(vm[k], 4) for k in ("roc_auc", "val_f1", "val_fpr_human") if k in vm}
            src = f"{cal_file.name}: doc_threshold, подобран на собственном val ({cal.get('calibrated_on', 'val')})"
        else:
            thr = 0.5
            src = f"файла {cal_file.name} нет → фиксированный порог 0.5 (НЕ калиброван); основная метрика — ROC-AUC"
        max_chars = args.max_chars

        def scorer(text: str, _d=det) -> float:
            return float(_d.score(text[:max_chars]).prob)

        methods.append(TextMethod(name, scorer, thr, src, info))
        log.info("метод %s загружен, порог %.4f (%s)", name, thr, src)
    return methods


def score_texts(methods: List[TextMethod], texts: List[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for m in methods:
        t0 = time.time()
        vals = []
        for i, t in enumerate(texts):
            vals.append(m.scorer(t))
            if (i + 1) % 100 == 0:
                log.info("  %s: %d/%d (%.0f с)", m.name, i + 1, len(texts), time.time() - t0)
        out[m.name] = vals
        m.info["last_seconds"] = round(time.time() - t0, 1)
    return out


def sample_balanced(df, label_col: str, n_per_class: Dict, seed: int, strata_col: Optional[str] = None):
    """Простая случайная выборка без возвращения (при strata_col — равные доли по страте внутри класса)."""
    import pandas as pd

    parts = []
    for lab, n in n_per_class.items():
        sub = df[df[label_col] == lab]
        if strata_col is None:
            parts.append(sub.sample(n=min(n, len(sub)), random_state=seed))
        else:
            groups = sorted(sub[strata_col].unique())
            k = n // len(groups)
            for j, g in enumerate(groups):
                gsub = sub[sub[strata_col] == g]
                parts.append(gsub.sample(n=min(k + (1 if j < n - k * len(groups) else 0), len(gsub)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def evaluate_text_bench(bench: str, df, methods: List[TextMethod], args, out_dir: Path,
                        group_cols: List[str], slice_col: Optional[str] = None) -> Dict:
    texts = df["text"].tolist()
    y = df["y"].to_numpy()
    scores = score_texts(methods, texts)
    res: Dict = {"n": int(len(df)), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()), "methods": {}}
    for m in methods:
        s = np.asarray(scores[m.name])
        r = {"threshold_source": m.threshold_source, "info": m.info,
             "overall": full_metrics(y, s, m.threshold, args.bootstrap, args.seed)}
        if slice_col:
            r["by_" + slice_col] = {}
            for v in sorted(df[slice_col].unique()):
                mask = (df[slice_col] == v).to_numpy()
                r["by_" + slice_col][str(v)] = full_metrics(y[mask], s[mask], m.threshold, args.bootstrap, args.seed)
        for gc in group_cols:
            r["by_" + gc] = recall_by_group(y, s, df[gc].astype(str).to_numpy(), m.threshold)
        res["methods"][m.name] = r
        plot_confusion(out_dir / f"confusion_{m.name}.png", f"{bench}: {m.name}", r["overall"])
    plot_roc(out_dir / "roc_curve.png", f"{bench} (n={len(df)})", {m.name: (y, scores[m.name]) for m in methods})
    if slice_col:
        for v in sorted(df[slice_col].unique()):
            mask = (df[slice_col] == v).to_numpy()
            plot_roc(out_dir / f"roc_curve_{v}.png", f"{bench} [{v}] (n={mask.sum()})",
                     {m.name: (y[mask], np.asarray(scores[m.name])[mask]) for m in methods})
    preds = []
    keep = [c for c in df.columns if c not in ("text",)]
    for i in range(len(df)):
        row = {c: (df[c].iloc[i].item() if hasattr(df[c].iloc[i], "item") else df[c].iloc[i]) for c in keep}
        row["text_preview"] = texts[i][:200]
        for m in methods:
            row[f"score_{m.name}"] = round(scores[m.name][i], 6)
            row[f"pred_{m.name}"] = int(scores[m.name][i] >= m.threshold)
        preds.append(row)
    (out_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- SeqXGPT
def run_seqxgpt(args, methods, out_dir: Path, tmp: Path) -> Dict:
    import pandas as pd
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{SEQXGPT_REPO}/data/train-00000-of-00001.parquet", "rb") as fh:
        full = pd.read_parquet(io.BytesIO(fh.read()))  # 33 МБ, только в памяти
    pool = {"rows": int(len(full)), "label_counts": full["label"].value_counts().to_dict(),
            "agent_counts": full["agent"].value_counts().to_dict(), "lang": full["lang"].value_counts().to_dict()}
    # Наблюдение о составе: доля ИИ-текстов, начало которых совпадает с началом какого-то человеческого текста
    human_prefixes = set(full.loc[full.label == "human", "text"].str[:80])
    ai = full[full.label == "ai"]
    pool["ai_texts_sharing_80char_prefix_with_human_text"] = float(ai["text"].str[:80].isin(human_prefixes).mean())
    half = args.n_seqxgpt // 2
    # для ИИ — равные доли генераторов
    df_h = sample_balanced(full, "label", {"human": half}, args.seed)
    df_a = sample_balanced(full, "label", {"ai": half}, args.seed, strata_col="agent")
    df = pd.concat([df_h, df_a]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df["y"] = (df["label"] == "ai").astype(int)
    df = df[["id", "text", "label", "agent", "source", "y"]]
    del full
    res = evaluate_text_bench("SeqXGPT", df, methods, args, out_dir, group_cols=["agent"])
    res["pool"] = pool
    res["sampling"] = (f"весь файл {SEQXGPT_REPO} ({pool['rows']} строк, 33 МБ, единственный сплит train) прочитан по HTTP в "
                       f"память; простая случайная выборка без возвращения seed={args.seed}: {half} human + "
                       f"{half} ai (поровну по 5 генераторам)")
    return res


# ----------------------------------------------------------------------------- LLMTrace classification
def run_llmtrace(args, methods, out_dir: Path, tmp: Path) -> Dict:
    import pandas as pd
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{LLMTRACE_REPO}/test.jsonl", "rb", block_size=16 * 2**20) as fh:
        full = pd.read_json(io.BytesIO(fh.read()), lines=True, dtype=False)  # 318 МБ, только в памяти, на диск не пишется
    full = full[full["label"].isin(["ai", "human"]) & full["text"].astype(str).str.strip().astype(bool)]
    pool = {"rows": int(len(full)), "by_lang_label": {f"{k[0]}/{k[1]}": int(v) for k, v in full.groupby(["lang", "label"]).size().items()}}
    langs = sorted(full["lang"].unique())
    per = args.n_llmtrace // (2 * len(langs))
    parts = [sample_balanced(full[full.lang == lg], "label", {"human": per, "ai": per}, args.seed) for lg in langs]
    df = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df["y"] = (df["label"] == "ai").astype(int)
    df = df[["lang", "label", "model", "data_type", "prompt_type", "topic_id", "text", "y"]]
    del full
    res = evaluate_text_bench("LLMTrace_classification test", df, methods, args, out_dir,
                              group_cols=["data_type", "model"], slice_col="lang")
    res["pool"] = pool
    res["sampling"] = (f"файл test.jsonl ({LLMTRACE_REPO}, 318 МБ) прочитан по HTTP в память целиком (на диск не сохраняется); "
                       f"простая случайная выборка без возвращения seed={args.seed}: по {per} human и {per} ai для каждого "
                       f"языка ({', '.join(langs)})")
    return res


# ----------------------------------------------------------------------------- классификатор изображений
_CLF = None


def score_images(blobs: List[bytes]):
    """Вероятность метки 'artificial' классификатора umm-maybe/AI-image-detector (как в src/image_analyzer.py,
    но без OCR/EXIF/QR — только классификатор)."""
    global _CLF
    import torch
    from PIL import Image
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else -1)
    if _CLF is None:
        _CLF = pipeline("image-classification", model=IMAGE_MODEL, device=device)
    scores, t0 = [], time.time()
    for i, b in enumerate(blobs):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        preds = _CLF(img, top_k=None)
        sc = next((p["score"] for p in preds if p["label"].lower() == "artificial"), None)
        if sc is None:
            raise RuntimeError(f"Нет метки 'artificial' в выходе модели: {preds}")
        scores.append(float(sc))
        if (i + 1) % 100 == 0:
            log.info("  image: %d/%d (%.0f с)", i + 1, len(blobs), time.time() - t0)
    return scores, {"model": IMAGE_MODEL, "device": str(device), "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- Defactify
def run_defactify(args, out_dir: Path) -> Dict:
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import HfApi, HfFileSystem
    from PIL import Image
    from transformers import pipeline

    rng = np.random.default_rng(args.seed)
    fs = HfFileSystem()
    files = sorted(f.rfilename for f in HfApi().dataset_info(DEFACTIFY_REPO).siblings
                   if f.rfilename.startswith("data/test-") and f.rfilename.endswith(".parquet"))
    pfs = {f: pq.ParquetFile(fs.open(f"datasets/{DEFACTIFY_REPO}/{f}", block_size=8 * 2**20)) for f in files}
    groups = [(f, g) for f in files for g in range(pfs[f].metadata.num_row_groups)]
    total_rows = sum(pfs[f].metadata.num_rows for f in files)
    chosen = [groups[i] for i in sorted(rng.choice(len(groups), size=min(args.defactify_row_groups, len(groups)), replace=False))]
    log.info("Defactify: случайно выбранные группы строк %s из %d", chosen, len(groups))
    import pandas as pd

    tabs = []
    for f, g in chosen:
        t = pfs[f].read_row_group(g, columns=["Image", "Label_A", "Label_B"]).to_pandas()
        t["shard"], t["row_group"], t["row_in_group"] = f, g, np.arange(len(t))
        tabs.append(t)
    cand = pd.concat(tabs, ignore_index=True)
    del tabs
    pool = {"test_rows_total": int(total_rows), "row_groups_total": len(groups), "row_groups_read": [f"{f}#{g}" for f, g in chosen],
            "candidate_rows": int(len(cand)), "candidate_label_b": {DEFACTIFY_GENERATORS[int(k)]: int(v) for k, v in cand["Label_B"].value_counts().sort_index().items()}}
    half = args.n_defactify // 2
    df_r = sample_balanced(cand, "Label_A", {0: half}, args.seed)
    df_a = sample_balanced(cand, "Label_A", {1: half}, args.seed, strata_col="Label_B")
    df = pd.concat([df_r, df_a]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    del cand

    scores, img_info = score_images([df["Image"].iloc[i]["bytes"] for i in range(len(df))])
    y = df["Label_A"].to_numpy().astype(int)
    s = np.asarray(scores)
    thr = 0.5
    gen_names = df["Label_B"].map(lambda v: DEFACTIFY_GENERATORS[int(v)]).to_numpy()
    r = {"threshold_source": "фиксированный априорный порог 0.5 (score метки 'artificial'), на Defactify не подбирался",
         "info": img_info,
         "overall": full_metrics(y, s, thr, args.bootstrap, args.seed),
         "by_generator": recall_by_group(y, s, gen_names, thr)}
    for g in DEFACTIFY_GENERATORS.values():
        if g == DEFACTIFY_GENERATORS[0]:
            continue
        m = (gen_names == g) | (y == 0)
        if (gen_names == g).any():
            r["by_generator"][g]["roc_auc_vs_real"] = roc_auc(y[m], s[m])
    plot_roc(out_dir / "roc_curve.png", f"Defactify test (n={len(df)})", {IMAGE_MODEL.split('/')[-1]: (y, s)})
    plot_confusion(out_dir / "confusion_image.png", "Defactify: image classifier", r["overall"], labels=("реальное", "ИИ"))
    preds = [{"shard": df["shard"].iloc[i], "row_group": int(df["row_group"].iloc[i]), "row_in_group": int(df["row_in_group"].iloc[i]),
              "label_a": int(y[i]), "label_b": int(df["Label_B"].iloc[i]), "generator": gen_names[i],
              "score_artificial": round(scores[i], 6), "pred": int(scores[i] >= thr)} for i in range(len(df))]
    (out_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"n": int(len(df)), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()), "pool": pool, "methods": {"image": r},
            "sampling": (f"полный test-сплит (~{total_rows} строк, ~3.7 ГБ) не скачивается: по HTTP читаются {len(chosen)} "
                         f"случайно выбранные (seed={args.seed}) группы строк parquet из {len(groups)} (≈{pool['candidate_rows']} "
                         f"строк, только в памяти); внутри них простая случайная выборка: {half} реальных + {half} ИИ "
                         f"(поровну по 5 генераторам)")}


# ----------------------------------------------------------------------------- NTIRE
def _zip_member_bytes(fs, path: str, zi) -> bytes:
    """Чтение одного файла из удалённого zip по HTTP-диапазону (без скачивания архива целиком)."""
    import zlib

    slack = 30 + len(zi.filename.encode()) + 512  # локальный заголовок + запас под extra-поле: обычно хватает одного запроса
    blob = fs.cat_file(path, start=zi.header_offset, end=zi.header_offset + slack + zi.compress_size)
    n_name, n_extra = int.from_bytes(blob[26:28], "little"), int.from_bytes(blob[28:30], "little")
    off = 30 + n_name + n_extra
    data = blob[off: off + zi.compress_size]
    if len(data) < zi.compress_size:
        start = zi.header_offset + off
        data = fs.cat_file(path, start=start, end=start + zi.compress_size)
    if zi.compress_type == 0:
        return data
    if zi.compress_type == 8:
        return zlib.decompress(data, -15)
    raise RuntimeError(f"Неподдерживаемое сжатие {zi.compress_type}")


def run_ntire(args, out_dir: Path) -> Dict:
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from huggingface_hub import HfApi, HfFileSystem

    info = HfApi().dataset_info(NTIRE_REPO)
    shards = sorted(f.rfilename for f in info.siblings if f.rfilename.startswith("shard_") and f.rfilename.endswith(".zip"))
    rng = np.random.default_rng(args.seed)
    shard = shards[int(rng.integers(0, len(shards)))]
    fs = HfFileSystem()
    path = f"datasets/{NTIRE_REPO}/{shard}"
    z = zipfile.ZipFile(fs.open(path, block_size=2**20))  # читается только центральный каталог архива
    lab_name = next(n for n in z.namelist() if n.endswith("labels.csv"))
    labels = pd.read_csv(io.BytesIO(z.read(lab_name)))
    pool = {"repo_gated": getattr(info, "gated", None), "shards": shards, "shard_used": shard, "shard_rows": int(len(labels)),
            "shard_label_counts": {str(k): int(v) for k, v in labels["label"].value_counts().sort_index().items()}}
    half = args.n_ntire // 2
    df = sample_balanced(labels, "label", {0: half, 1: half}, args.seed)
    prefix = lab_name.rsplit("/", 1)[0] + "/images/"
    infos = [z.getinfo(prefix + n) for n in df["image_name"]]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        blobs = list(ex.map(lambda zi: _zip_member_bytes(fs, path, zi), infos))
    dl_secs, dl_mb = round(time.time() - t0, 1), sum(map(len, blobs)) / 2**20
    scores, img_info = score_images(blobs)
    del blobs
    img_info.update({"download_seconds": dl_secs, "download_mb": round(dl_mb, 1)})
    y, s, thr = df["label"].to_numpy().astype(int), np.asarray(scores), 0.5
    r = {"threshold_source": "фиксированный априорный порог 0.5 (score метки 'artificial'), на NTIRE не подбирался",
         "info": img_info, "overall": full_metrics(y, s, thr, args.bootstrap, args.seed)}
    plot_roc(out_dir / "roc_curve.png", f"NTIRE train {shard} (n={len(df)})", {IMAGE_MODEL.split('/')[-1]: (y, s)})
    plot_confusion(out_dir / "confusion_image.png", "NTIRE: image classifier", r["overall"], labels=("реальное", "ИИ"))
    preds = [{"shard": shard, "image_name": df["image_name"].iloc[i], "label": int(y[i]), "score_artificial": round(scores[i], 6),
              "pred": int(scores[i] >= thr)} for i in range(len(df))]
    (out_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"n": int(len(df)), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()), "pool": pool, "methods": {"image": r},
            "sampling": (f"архивы не скачиваются целиком (6 zip по 11–21 ГБ): случайно (seed={args.seed}) выбран {shard}, по HTTP "
                         f"прочитаны его оглавление и labels.csv ({len(labels)} изображений), затем простая случайная выборка "
                         f"{half} реальных + {half} сгенерированных; загружено ≈{dl_mb:.0f} МБ, только в памяти")}


# ----------------------------------------------------------------------------- отчёт
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def ci(m):
    c = m.get("roc_auc_ci95")
    return "—" if not c else f"[{c[0]:.3f}; {c[1]:.3f}]"


def auc_verdict(m) -> str:
    a, c = m.get("roc_auc"), m.get("roc_auc_ci95")
    if a is None:
        return "ROC-AUC не определена"
    if c and c[1] < 0.5:
        return f"ROC-AUC {a:.3f} — **хуже случайного угадывания** (весь 95 % ДИ ниже 0.5): на этом наборе метод в среднем ставит ИИ-текстам более низкие оценки, чем человеческим"
    if c and c[0] <= 0.5 <= c[1]:
        return f"ROC-AUC {a:.3f} — статистически **не отличается от случайного** (95 % ДИ содержит 0.5)"
    if a < 0.7:
        return f"ROC-AUC {a:.3f} — различение выше случайного, но слабое"
    if a < 0.85:
        return f"ROC-AUC {a:.3f} — умеренное различение"
    return f"ROC-AUC {a:.3f} — хорошее различение"


def thr_verdict(m) -> str:
    s = f"при пороге {m['threshold']:.4g}: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}, FPR {m['fpr']:.3f}"
    if m["fpr"] > 0.5:
        s += " — **FPR выше 50 %: большинство человеческих текстов помечается как ИИ, такой режим неприемлем**, несмотря на значение F1"
    elif m["fpr"] > 0.2:
        s += " — FPR выше ограничения 20 %, заданного при калибровке на нашем val"
    if m["recall"] < 0.2:
        s += " — детектор почти ничего не находит"
    return s


def row(bench, slice_, method, thr_src, m):
    return (f"| {bench} | {slice_} | {method} | {m['n']} ({m['n_pos']}/{m['n_neg']}) | {fmt(m['roc_auc'])} | {ci(m)} | "
            f"{m['threshold']:.4g}{'' if 'val' in thr_src else ' *'} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['fpr'])} |")


def write_report(summary: Dict, out: Path):
    b = summary["benchmarks"]
    L: List[str] = []
    dev = summary["models_dir_is_dev"]
    L += ["# Академическая валидация DW-1 на сторонних бенчмарках", ""]
    if dev:
        L += [f"> **ВНИМАНИЕ: предварительный прогон на dev-моделях (`{summary['models_dir']}`).** Цифры ниже не являются "
              "итоговыми; итоговый отчёт формируется повторным запуском на финальных моделях.", ""]
    L += [f"Дата запуска: {summary['started']}; длительность: {summary['runtime_seconds'] / 60:.1f} мин; seed = {summary['seed']}; "
          f"каталог моделей: `{summary['models_dir']}`.", "",
          "Команда: `" + summary["command"] + "`", "", "## 1. Сводная таблица", "",
          "Положительный класс — «сгенерировано ИИ». n указано как всего (ИИ/человек). ROC-AUC не зависит от порога; "
          "95 % ДИ — перцентильный бутстрэп (" + str(summary["bootstrap"]) + " повторов). "
          "Порог со звёздочкой * — фиксированный априорный 0.5 (не калибровался ни на нашем val, ни на бенчмарке), см. раздел 2.", "",
          "| Бенчмарк | Срез | Метод | n | ROC-AUC | 95 % ДИ | Порог | Precision | Recall | F1 | FPR |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, title in (("seqxgpt", "SeqXGPT"), ("llmtrace", "LLMTrace_classification (test)"), ("defactify", "Defactify (test)"), ("ntire", "NTIRE 2026 (train, 1 шард)")):
        r = b.get(key)
        if not r or "methods" not in r:
            L.append(f"| {title} | — | — | — | {'не запускался' if not r else 'не оценивался (ошибка, см. раздел 3)'} | — | — | — | — | — | — |")
            continue
        for mn, mr in r["methods"].items():
            L.append(row(title, "все", mn, mr["threshold_source"], mr["overall"]))
            for lg, lm in mr.get("by_lang", {}).items():
                L.append(row(title, lg, mn, mr["threshold_source"], lm))
    L += ["", "## 2. Методология", "",
          f"* **Случайная выборка с фиксированным seed = {summary['seed']}** (а не первые N строк потока). Способ выборки для каждого набора указан ниже.",
          "* **Пороги выбраны априори.** Для метода `features` используется `doc_threshold` из `calibration.json`, подобранный на "
          "валидационной части *нашего* корпуса (ограничение FPR ≤ 20 %). Для `lm` — `doc_threshold` из `lm_calibration.json`, "
          "если он есть; если нет — фиксированный порог 0.5, помеченный как некалиброванный, и основной метрикой считается ROC-AUC. "
          "Для изображений — фиксированный порог 0.5 по вероятности метки `artificial`. **На данных бенчмарков ничего не подбиралось** "
          "(ни пороги, ни веса), переменная `DW1_THRESHOLD` скриптом игнорируется.",
          f"* Тексты длиннее {summary['max_chars']} символов обрезаются до первых {summary['max_chars']} символов (одинаково для всех методов).",
          "* Все сторонние данные — только для оценки; в обучении `features` использовался собственный корпус (см. `training_params.json`), "
          "в который входит `iitolstykh/LLMTrace_detection` (другой набор тех же авторов, см. ниже).",
          "* Данные бенчмарков не сохраняются на диск: SeqXGPT и LLMTrace читаются по HTTP в память; Defactify и NTIRE — "
          "выборочно (отдельные группы строк parquet / отдельные файлы zip-архива по HTTP-диапазонам).", ""]
    for key, mr in summary.get("text_methods", {}).items():
        L.append(f"* Метод `{key}`: порог {mr['threshold']:.4g} — {mr['threshold_source']}. Сведения: `{json.dumps(mr['info'], ensure_ascii=False)}`")
        ov = mr["info"].get("own_val_metrics", {})
        if ov.get("roc_auc") is not None:
            L.append(f"  * Для сравнения: ROC-AUC этого метода на *нашем* val (из файла калибровки) = {ov['roc_auc']:.3f}"
                     + (" — **уже на собственных данных не выше случайного уровня**, поэтому от бенчмарков нельзя ожидать лучшего результата."
                        if ov["roc_auc"] <= 0.55 else "."))
    L += ["", "## 3. Результаты и интерпретация по наборам", ""]

    # SeqXGPT
    r = b.get("seqxgpt")
    L += ["### 3.1 SeqXGPT (`TheItCrOw/SeqXGPT`)", "",
          "**Что это за данные.** Англоязычный набор (все строки `lang = en-EN`), единственный сплит `train`, 38 304 текста: "
          "6 400 человеческих и ≈31 900 сгенерированных моделями GPT-2, GPT-J, GPT-Neo, LLaMA и GPT-3 (`gpt3re`). "
          "Источник — SeqXGPT-Bench и связанные с ним наборы (поле `source`). Это не письма; генераторы — относительно "
          "старые модели 2019–2023 гг.", ""]
    if r and "methods" in r:
        share = r["pool"].get("ai_texts_sharing_80char_prefix_with_human_text")
        L += [f"**Выборка.** {r['sampling']}.", ""]
        if share is not None:
            L += [f"**Наблюдение о составе (факт, измерено скриптом):** у {share * 100:.0f} % ИИ-текстов первые 80 символов совпадают "
                  "с началом какого-либо человеческого текста набора. Т. е. значительная часть «ИИ»-документов, по-видимому, "
                  "начинается с человеческого фрагмента и продолжена моделью (в SeqXGPT-Bench изначально решалась задача "
                  "разметки на уровне предложений). Документная метка `ai` в таком случае означает «содержит ИИ-фрагмент».", ""]
        for mn, mr in r["methods"].items():
            L.append(f"* `{mn}`: {auc_verdict(mr['overall'])}; {thr_verdict(mr['overall'])}.")
            L.append("  Recall по генераторам: " + ", ".join(
                f"{g} {v['recall']:.2f}" for g, v in mr["by_agent"].items() if "recall" in v) + ".")
        L += ["", "Требование ТЗ — работа с RU и EN текстами, поэтому английский набор является обязательной, а не «посторонней» проверкой. "
              "Модель `lm` построена на mDeBERTa-v3 — многоязычной модели; `razdel` используется только для разбиения на слова/предложения "
              "и не делает модель русскоязычной.", "",
              "*Гипотезы (не проверялись):* (а) смешанные человек+ИИ документы снижают документную оценку; "
              "(б) стиль старых генераторов (GPT-2/Neo/J) отличается от современных LLM, на которых строился наш корпус; "
              "(в) научные аннотации и отзывы — далёкий от писем домен.", ""]
    else:
        L += [f"Не выполнено: {r}", ""]

    # LLMTrace
    r = b.get("llmtrace")
    L += ["### 3.2 LLMTrace classification (`iitolstykh/LLMTrace_classification`, сплит test)", "",
          "**Что это за данные.** Корпус авторов LLMTrace для бинарной классификации «человек / ИИ» на русском и английском "
          "(поля `lang`, `label`, `model`, `data_type` — жанр: новости, статьи, вопросы, поэзия и т. п.). "
          "Сплит test — 318 МБ jsonl.", "",
          "**Связь с нашими обучающими данными (честно).** Наш корпус включает `iitolstykh/LLMTrace_detection` — *другой* набор тех же "
          "авторов (задача детекции/разметки интервалов). Здесь оценивается `LLMTrace_classification`, т. е. другое подмножество, "
          "не использованное в обучении. Однако наборы собраны одной командой по схожей методике, поэтому возможны пересечения "
          "по темам, источникам человеческих текстов и генераторам; полной независимости гарантировать нельзя, и результат "
          "может быть оптимистичнее, чем на полностью внешних данных. Пересечение текстов скриптом не проверялось.", ""]
    if r and "methods" in r:
        L += [f"**Выборка.** {r['sampling']}. Размер пула после фильтрации: `{json.dumps(r['pool']['by_lang_label'], ensure_ascii=False)}`.", ""]
        for mn, mr in r["methods"].items():
            L.append(f"* `{mn}`, все: {auc_verdict(mr['overall'])}; {thr_verdict(mr['overall'])}.")
            for lg, lm in mr.get("by_lang", {}).items():
                L.append(f"  * `{lg}`: {auc_verdict(lm)}; {thr_verdict(lm)}.")
            fpr_dt = ", ".join(f"{g} {v['fpr']:.2f} (n={v['n_human']})" for g, v in mr["by_data_type"].items() if "fpr" in v)
            rec_dt = ", ".join(f"{g} {v['recall']:.2f} (n={v['n_ai']})" for g, v in mr["by_data_type"].items() if "recall" in v)
            L.append(f"  * FPR по жанрам (человеческие тексты): {fpr_dt}.")
            L.append(f"  * Recall по жанрам (ИИ-тексты): {rec_dt}.")
        L += ["", "Подробности по жанрам и генераторам — в `llmtrace/metrics.json`.", ""]
    else:
        L += [f"Не выполнено: {r}", ""]

    # Defactify
    r = b.get("defactify")
    L += ["### 3.3 Defactify (`Rajarshi-Roy-research/Defactify_Image_Dataset`, сплит test)", "",
          "**Что это за данные.** ≈96 тыс. изображений: реальные фотографии MS-COCO и изображения, сгенерированные по тем же подписям "
          "моделями Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3 и Midjourney v6. Колонки: `Caption`, `Image`, "
          "`Label_A` (0 — реальное, 1 — ИИ), `Label_B` (0 — реальное, 1 — SD 2.1, 2 — SDXL, 3 — SD 3, 4 — DALL-E 3, 5 — Midjourney). "
          "Сплиты: train ≈42 тыс., validation ≈9 тыс., test ≈45 тыс. (≈7.5 ГБ всего). Предыдущая версия отчёта ошибочно описывала "
          "этот набор как мемы/фотомонтаж без диффузионных артефактов — это неверно.", "",
          f"**Детектор.** Классификатор изображений `{IMAGE_MODEL}` (тот же, что в `src/image_analyzer.py`); для бенчмарка вызывается "
          "только классификатор (без OCR/EXIF/QR), оценка — вероятность метки `artificial`, порог 0.5 задан априори.", ""]
    if r and "methods" in r:
        mr = r["methods"]["image"]
        L += [f"**Выборка.** {r['sampling']}. В исходных parquet-файлах классы и генераторы перемешаны внутри групп строк "
              f"(состав прочитанных групп: `{json.dumps(r['pool']['candidate_label_b'], ensure_ascii=False)}`); выборка не охватывает "
              "все группы строк test-сплита — это ограничение ради объёма скачивания.", "",
              f"* {auc_verdict(mr['overall'])}; {thr_verdict(mr['overall'])}.", "", "| Генератор | n | Recall при 0.5 | ROC-AUC против реальных | Средняя оценка |", "|---|---|---|---|---|"]
        for g, v in mr["by_generator"].items():
            rate = f"FPR {v['fpr']:.3f}" if "fpr" in v else f"{v['recall']:.3f}"
            L.append(f"| {g} | {v['n']} | {rate} | {fmt(v.get('roc_auc_vs_real'))} | {v['mean_score']:.3f} |")
        n_gen = min(v["n"] for g, v in mr["by_generator"].items() if "recall" in v)
        L += ["", "*Гипотеза (не проверялась):* генераторы SD 3, DALL-E 3 и Midjourney v6 вышли в 2023–2024 гг.; если "
              "классификатор `umm-maybe/AI-image-detector` обучался на изображениях более ранних генераторов, на них возможна "
              f"более низкая полнота. Таблица выше позволяет это сопоставить, но при ≈{n_gen} изображениях на генератор "
              "погрешность оценок велика.", ""]
    else:
        L += [f"Не выполнено: {r}", ""]

    r = b.get("ntire", {})
    L += ["### 3.4 NTIRE (`deepfakesMSU/NTIRE-RobustAIGenDetection-train`)", "",
          "**Что это за данные.** Обучающая часть соревнования NTIRE 2026 «Robust AI-Generated Image Detection in the Wild» "
          "(CVPR 2026 Workshops; организаторы — лаборатория МГУ). ≈277 тыс. реальных и сгенерированных изображений с "
          "дополнительными «in-the-wild» преобразованиями (кадрирование, масштабирование, сжатие, размытие); 6 zip-шардов по ~50 тыс., "
          "по README распределение в шардах одинаковое; разметка — `labels.csv` (0 — реальное, 1 — сгенерированное), "
          "сведений о генераторе для каждого изображения нет.", "",
          ]
    if r and "methods" in r:
        mr = r["methods"]["image"]
        L += ["**Доступ.** В предыдущей версии отчёта набор числился закрытым (gated, «нужен HF-токен»). На момент данного "
              f"запуска Hugging Face API сообщает `gated = {r['pool'].get('repo_gated')}`, файлы прочитаны без токена.", ""]
        L += [f"**Выборка.** {r['sampling']}. Состав шарда: `{json.dumps(r['pool']['shard_label_counts'])}`.", "",
              f"* {auc_verdict(mr['overall'])}; {thr_verdict(mr['overall'])}.", "",
              "*Гипотеза (не проверялась):* преобразования «in-the-wild» (сжатие, размытие, ресайз) специально задуманы "
              "организаторами как помеха для детекторов и могут влиять на результат классификатора.", ""]
    else:
        why = "в этом прогоне не запускался" if not r else f"ошибка доступа/обработки: `{r.get('error')}`"
        L += [f"**Не оценивался** ({why}). Никаких выводов по этому набору не делается.", ""]
    L += [
          "## 4. Ограничения", "",
          "* Небольшие выборки (сотни объектов): доверительные интервалы ROC-AUC широкие, различия в пределах ДИ не следует интерпретировать.",
          "* Ни один из наборов не состоит из электронных писем; это проверка переносимости на чужие домены, а не оценка в целевом сценарии.",
          "* LLMTrace_classification может пересекаться по темам/источникам с LLMTrace_detection, использованным в обучении.",
          "* Метрики при пороге зависят от того, насколько распределение оценок на чужом домене совпадает с нашим val; "
          "мы сознательно не перекалибровываем порог, чтобы не допустить утечки.", "",
          "## 5. Файлы", "",
          "* `summary.json` — все метрики; `<набор>/metrics.json`, `<набор>/predictions.json` — метрики и предсказания по каждому объекту;",
          "* `<набор>/roc_curve*.png`, `<набор>/confusion_*.png` — ROC-кривые и матрицы ошибок;",
          ""]
    out.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=os.getenv("DW1_MODELS_DIR", str(ROOT / "models")))
    ap.add_argument("--with-lm", action="store_true", help="оценить также метод lm (TNC-DBGC*, нужен torch)")
    ap.add_argument("--lm-checkpoint", default=os.getenv("DW1_LM_CHECKPOINT"))
    ap.add_argument("--lm-device", default=os.getenv("DW1_LM_DEVICE", "cpu"),
                    help="cpu (по умолчанию: на Apple M-серии оказался в ~2.7 раза быстрее mps для этой модели) | mps | cuda")
    ap.add_argument("--only", nargs="+", choices=["seqxgpt", "llmtrace", "defactify", "ntire"],
                    default=["seqxgpt", "llmtrace", "defactify", "ntire"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seqxgpt", type=int, default=300)
    ap.add_argument("--n-llmtrace", type=int, default=400, help="всего; поровну по языкам и классам")
    ap.add_argument("--n-defactify", type=int, default=300)
    ap.add_argument("--n-ntire", type=int, default=300)
    ap.add_argument("--defactify-row-groups", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "results" / "academic_validation"))
    ap.add_argument("--report-only", action="store_true", help="только пересобрать отчёт .md из существующего summary.json")
    args = ap.parse_args()
    if args.report_only:
        write_report(json.loads((Path(args.out) / "summary.json").read_text(encoding="utf-8")),
                     Path(args.out) / "ACADEMIC_VALIDATION_REPORT.md")
        return

    import huggingface_hub.hf_file_system  # noqa: F401  (ранний импорт: fsspec)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for h in logging.getLogger().handlers:
        h.addFilter(lambda rec: not rec.getMessage().startswith("HTTP Request"))
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except Exception:  # noqa: BLE001
        pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    summary: Dict = {"started": datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
                     "models_dir": str(Path(args.models_dir)), "models_dir_is_dev": "dev" in Path(args.models_dir).name,
                     "command": " ".join(["python", "run_academic_validation.py"] + sys.argv[1:]),
                     "bootstrap": args.bootstrap, "max_chars": args.max_chars,
                     "python": platform.python_version(), "platform": platform.platform(), "benchmarks": {}}

    text_benches = [b for b in ("seqxgpt", "llmtrace") if b in args.only]
    methods = load_text_methods(args) if text_benches else []
    summary["text_methods"] = {m.name: {"threshold": m.threshold, "threshold_source": m.threshold_source, "info": m.info} for m in methods}

    with tempfile.TemporaryDirectory(prefix="dw1_acad_") as tmpd:
        tmp = Path(tmpd)
        runners = {"seqxgpt": lambda d: run_seqxgpt(args, methods, d, tmp),
                   "llmtrace": lambda d: run_llmtrace(args, methods, d, tmp),
                   "defactify": lambda d: run_defactify(args, d),
                   "ntire": lambda d: run_ntire(args, d)}
        for bench in [b for b in ("seqxgpt", "llmtrace", "defactify", "ntire") if b in args.only]:
            d = out / bench
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            log.info("=== %s ===", bench)
            t0 = time.time()
            try:
                res = runners[bench](d)
                res["seconds"] = round(time.time() - t0, 1)
            except Exception as e:  # noqa: BLE001
                log.exception("%s: ошибка", bench)
                res = {"error": f"{type(e).__name__}: {e}"}
            (d / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["benchmarks"][bench] = res

    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out / "ACADEMIC_VALIDATION_REPORT.md")
    log.info("Готово за %.1f мин. Отчёт: %s", summary["runtime_seconds"] / 60, out / "ACADEMIC_VALIDATION_REPORT.md")
    for bench, r in summary["benchmarks"].items():
        for mn, mr in r.get("methods", {}).items():
            o = mr["overall"]
            print(f"{bench:10s} {mn:9s} AUC={fmt(o['roc_auc'])} CI={ci(o)} thr={o['threshold']:.4g} "
                  f"P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} FPR={o['fpr']:.3f}")


if __name__ == "__main__":
    main()
