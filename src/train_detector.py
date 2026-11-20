"""Обучение и калибровка метода на основе признаков (п. 4.1.3 ТЗ).

  * обучение — только на train;
  * подбор порогов (документ, локализация) — только на val, результат в models/calibration.json;
  * test здесь не используется;
  * модель и параметры обучения — в JSON: models/feature_model.json, models/training_params.json.

Запуск:  python cli.py train   (или python -m src.train_detector --dataset data/dataset/v1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np

from src.detectors.feature_model import FeatureDetector
from src.evaluation.tz_metrics import calibrate_by_language, calibrate_doc_threshold, calibrate_sentence_threshold, safe_auc

log = logging.getLogger(__name__)


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def run(dataset: Path, models_dir: Path, max_fpr: float = 0.2, seed: int = 42) -> dict:
    t0 = time.time()
    from src.datagen.io import load_split

    train, val = load_split(dataset, "train"), load_split(dataset, "val")
    log.info("train=%d val=%d", len(train), len(val))
    model = FeatureDetector().fit(train, seed=seed)
    t_train = time.time() - t0

    outs = [model.score(r["text"]) for r in val]
    y = np.array([r["label"] for r in val])
    s = np.array([o.prob for o in outs])
    mk = np.array([r["label"] == 0 and r["email_type"] == "marketing" for r in val])
    lang_cal = calibrate_by_language(y, s, [r["language"] for r in val], mk, max_fpr=max_fpr)
    doc_cal = lang_cal["global"]
    loc_idx = [i for i, r in enumerate(val) if r["label"] == 1]
    sent_cal = calibrate_sentence_threshold(
        [val[i]["text"] for i in loc_idx], [val[i]["ai_char_intervals"] for i in loc_idx], [outs[i].sentences for i in loc_idx]
    )
    calibration = {
        "method": "features",
        "doc_threshold": doc_cal["threshold"],
        "doc_threshold_by_lang": lang_cal["by_lang"],
        "sentence_threshold": sent_cal["threshold"],
        "zone_threshold": doc_cal["threshold"],
        "explanation_threshold": doc_cal["threshold"],
        "calibrated_on": "val",
        "val_metrics": {"roc_auc": safe_auc(y, s), **{k: v for k, v in doc_cal.items() if k != "threshold"},
                        "char_f1": sent_cal["val_char_f1"], "by_lang": lang_cal["by_lang_details"]},
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "feature_model.json")
    (models_dir / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8")) if (dataset / "manifest.json").exists() else {}
    params = {
        "model": "features (TF-IDF sentence LR + stylometric document LR)",
        "dataset": str(dataset), "dataset_version": manifest.get("dataset_version"), "dataset_sha256": manifest.get("sha256"),
        "seed": seed, "max_fpr_constraint": max_fpr, "model_params": model.params,
        "train_seconds": round(t_train, 1), "total_seconds_with_calibration": round(time.time() - t0, 1),
        "python": platform.python_version(), "platform": platform.platform(),
        "model_sha256": hashlib.sha256((models_dir / "feature_model.json").read_bytes()).hexdigest(),
    }
    (models_dir / "training_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Калибровка (val): %s", json.dumps(calibration["val_metrics"], ensure_ascii=False))
    log.info("Сохранено в %s; обучение+калибровка %.1f c", models_dir, time.time() - t0)
    return calibration


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset/v1"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--max-fpr", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.dataset, a.models_dir, a.max_fpr, a.seed)


if __name__ == "__main__":
    main()
