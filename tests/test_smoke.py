"""Быстрые проверки ключевых компонентов: схема датасета, метрики, локализация, конвейер.

    python -m pytest -q tests
"""
import random
from pathlib import Path

import numpy as np
import pytest

from src.datagen.schema import validate_record
from src.evaluation.tz_metrics import calibrate_doc_threshold, char_prf, MASKINGS
from src.signals import find_prompt_leaks
from src.text_utils import merge_intervals, normalize_text, sentence_spans

ROOT = Path(__file__).resolve().parent.parent


def _rec(**kw):
    base = dict(id="x", group_id="g", subject="", text="Hello there. This is a test email with enough text.", label=0,
                label_name="human", generation_method="human", generator_model=None, language="en", email_type="corporate",
                source="t", ai_zones=[], ai_char_intervals=[])
    base.update(kw)
    return base


def test_schema_validation():
    assert validate_record(_rec()) == []
    assert validate_record(_rec(label=1, label_name="ai")) != []  # у AI должны быть интервалы
    assert validate_record(_rec(ai_char_intervals=[[0, 5]])) != []  # у человека интервалов нет
    assert validate_record(_rec(label=1, ai_char_intervals=[[0, 999]])) != []  # выход за границы


def test_char_f1():
    t = "aaaa bbbb cccc"
    assert char_prf([t], [[[0, 4]]], [[[0, 4]]])["f1"] == 1.0
    assert char_prf([t], [[[0, 4]]], [[[10, 14]]])["f1"] == 0.0
    r = char_prf([t], [[[0, 9]]], [[[0, 4]]])
    assert r["precision"] == 1.0 and 0 < r["recall"] < 1


def test_threshold_respects_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    s = np.r_[rng.normal(0.3, 0.15, 200), rng.normal(0.7, 0.15, 200)]
    cal = calibrate_doc_threshold(y, s, np.zeros(400, bool), max_fpr=0.2)
    assert cal["constraints_met"] and cal["val_fpr_human"] <= 0.2


def test_sentences_and_intervals():
    t = "Hi Tom.\nPlease send the report. Thanks!"
    spans = sentence_spans(t)
    assert [t[s:e] for s, e in spans] == ["Hi Tom.", "Please send the report.", "Thanks!"]
    assert merge_intervals([[0, 5], [6, 10], [20, 25]]) == [[0, 10], [20, 25]]


def test_normalization_removes_artifacts():
    assert normalize_text("It’s “ok” — call 555-1234​") == "It's \"ok\" - call 000-0000"


def test_prompt_leak_detection():
    assert find_prompt_leaks("=== Prompt brief === Create a fresh HTML email")
    assert not find_prompt_leaks("Hi team, the meeting is moved to Friday.")


def test_maskings_change_text():
    rng = random.Random(0)
    t = "Dear customer, we are excited to share our new offer with you today."
    for fn in MASKINGS.values():
        assert isinstance(fn(t, rng), str)


@pytest.mark.skipif(not (ROOT / "models" / "feature_model.json").exists(), reason="модель не обучена")
def test_pipeline_end_to_end(tmp_path):
    from src.pipeline import EmailAnalysisPipeline

    p = EmailAnalysisPipeline()
    rep = p.analyze_email(ROOT / "samples" / "ai_phishing_evasion.eml")
    d = rep.to_dict()
    for key in ("file_id", "metadata", "overall_ai_index", "verdict", "zones", "images", "explanation", "method"):
        assert key in d
    assert 0.0 <= d["overall_ai_index"] <= 1.0
    for z in d["zones"]:
        assert 0 <= z["start_char"] <= z["end_char"]
    rep2 = p.analyze_direct_text("Привет! Скинь, пожалуйста, отчёт до пятницы.", subject="отчёт")
    assert rep2.verdict
