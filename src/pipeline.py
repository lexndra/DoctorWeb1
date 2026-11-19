"""Конвейер анализа письма (п. 4.1.12 ТЗ):
разбор EML → извлечение признаков/зон → оценка вероятности генерации → объяснение → JSON.

Методы оценки (DW1_METHOD): features (по умолчанию, CPU, обучаемая интерпретируемая модель),
lm (пословная языковая модель TNC-DBGC*), combined (взвешенная сумма). Формат JSON единый для всех методов.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config import Settings
from src.detectors.base import DetectorOutput, SentenceScore
from src.detectors.feature_model import FEATURE_DESCRIPTIONS_RU
from src.evaluation.tz_metrics import predicted_intervals
from src.signals import find_prompt_leaks, masking_signals
from src.text_utils import ZERO_WIDTH_RE, overlap

log = logging.getLogger(__name__)

ZONE_NAMES_RU = {"subject": "тема", "greeting": "приветствие", "body": "основной текст", "signoff": "подпись/концовка"}


@dataclass
class AISpan:
    text: str
    start_char: int
    end_char: int
    probability: float = 0.0


@dataclass
class ZoneAnalysisResult:
    name: str
    text: str
    start_char: int
    end_char: int
    words_count: int
    ai_words_count: int
    ai_probability: float
    is_ai: bool
    ai_spans: List[AISpan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["ai_probability"] = round(self.ai_probability, 4)
        return d


@dataclass
class ImageAnalysisResult:
    filename: str
    ai_probability: Optional[float]
    is_ai: Optional[bool]
    has_qr: bool = False
    ocr_text: str = ""
    reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EmailDetectionReport:
    file_id: str
    file_path: str
    metadata: Dict[str, str]
    overall_ai_index: float
    verdict: str
    threshold: float
    ai_zones_detected: List[str]
    zones: List[ZoneAnalysisResult]
    images: List[ImageAnalysisResult]
    html_signals: Dict[str, object]
    evasion_signals: Dict[str, object]
    explanation: str
    reasons: List[str]
    method: str
    # поля, добавленные с сохранением обратной совместимости (п. 4.1.6.3)
    text_ai_probability: float = 0.0
    key_signals: Dict[str, float] = field(default_factory=dict)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    sentence_scores: List[Dict[str, object]] = field(default_factory=list)
    prompt_leakage: List[str] = field(default_factory=list)
    explanation_threshold: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "overall_ai_index": round(self.overall_ai_index, 4),
            "verdict": self.verdict,
            "threshold": round(self.threshold, 4),
            "ai_zones_detected": self.ai_zones_detected,
            "zones": [z.to_dict() for z in self.zones],
            "images": [i.to_dict() for i in self.images],
            "html_signals": self.html_signals,
            "evasion_signals": self.evasion_signals,
            "explanation": self.explanation,
            "reasons": self.reasons,
            "method": self.method,
            "text_ai_probability": round(self.text_ai_probability, 4),
            "key_signals": self.key_signals,
            "feature_contributions": self.feature_contributions,
            "sentence_scores": self.sentence_scores,
            "prompt_leakage": self.prompt_leakage,
            "explanation_threshold": round(self.explanation_threshold, 4),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def build_detector(settings: Settings):
    """Создаёт детектор согласно настройкам (единый интерфейс score())."""
    from src.detectors.feature_model import FeatureDetector

    method = settings.method
    fpath = settings.models_dir / "feature_model.json"
    if method in ("features", "combined"):
        if not fpath.exists():
            raise FileNotFoundError(f"Нет обученной модели {fpath}. Выполните: python cli.py train")
        feat = FeatureDetector.load(fpath)
        if method == "features":
            return feat
    from src.detectors.lm_model import CombinedDetector, LMDetector

    lm = LMDetector(settings.lm_checkpoint)
    if method == "lm":
        return lm
    return CombinedDetector(feat, lm, settings.weight_features, settings.weight_lm)


class EmailAnalysisPipeline:
    def __init__(self, settings: Optional[Settings] = None, detector=None, threshold: Optional[float] = None,
                 enable_images: Optional[bool] = None, **_ignored):
        self.settings = settings or Settings()
        if threshold is not None:
            self.settings.threshold = threshold
        if enable_images is not None:
            self.settings.enable_images = enable_images
        self.detector = detector or build_detector(self.settings)
        self.method = getattr(self.detector, "method_name", self.settings.method)
        cal = self.settings.calibration(self.method)
        self.threshold = self.settings.resolved_threshold(self.method)
        self.sentence_threshold = float(cal.get("sentence_threshold", 0.5))
        self.explanation_threshold = float(cal.get("explanation_threshold", self.threshold))

    # ------------------------------------------------------------------ images
    def _analyze_images(self, images: List[dict]) -> List[ImageAnalysisResult]:
        if not images:
            return []
        if not self.settings.enable_images:
            return [ImageAnalysisResult(filename=i.get("filename") or f"image_{n}", ai_probability=None, is_ai=None,
                                        error="анализ изображений отключён (DW1_ENABLE_IMAGES=0)")
                    for n, i in enumerate(images, 1)]
        out = []
        try:
            from src.image_analyzer import analyze_image
        except Exception as e:
            return [ImageAnalysisResult(filename=i.get("filename") or "image", ai_probability=None, is_ai=None,
                                        error=f"модуль изображений недоступен: {e}") for i in images]
        for n, item in enumerate(images, 1):
            name = item.get("filename") or f"image_{n}"
            try:
                a = analyze_image(item["data"])
                p = a.get("ai_score")
                reasons = []
                if p is not None and p >= 0.5:
                    reasons.append(f"детектор сгенерированных изображений: {p:.2f}")
                if not a.get("exif"):
                    reasons.append("нет EXIF-метаданных камеры")
                if a.get("c2pa"):
                    reasons.append(f"C2PA: {a['c2pa']}")
                out.append(ImageAnalysisResult(name, p, (p >= 0.5) if p is not None else None, a.get("has_qr", False),
                                               (a.get("ocr_text") or "")[:500], reasons))
            except Exception as e:
                out.append(ImageAnalysisResult(name, None, None, error=str(e)[:200]))
        return out

    # ------------------------------------------------------------------ core
    def _analyze(self, file_id: str, file_path: str, subject: str, visible_text: str, metadata: Dict[str, str],
                 html_analysis=None, images: Optional[List[dict]] = None, raw_text: str = "") -> EmailDetectionReport:
        from src.eml_parser import split_text_zones

        visible_text = ZERO_WIDTH_RE.sub("", visible_text or "")
        subject = (subject or "").strip()
        zr = split_text_zones(subject, visible_text)
        full_text, zones = zr["text"], zr["zones"]
        body_offset = (len(subject) + 1) if subject and visible_text.strip() else (0 if not subject else len(full_text))
        body_text = full_text[body_offset:] if body_offset < len(full_text) else ""

        det: DetectorOutput = self.detector.score(body_text) if body_text.strip() else DetectorOutput(prob=0.0, method=self.method)
        sentences = [SentenceScore(s.start + body_offset, s.end + body_offset, s.prob) for s in det.sentences]
        if subject:
            sub_out = self.detector.score(subject)
            sp = max((x.prob for x in sub_out.sentences), default=sub_out.prob)
            sentences.insert(0, SentenceScore(0, len(subject), float(sp)))

        ai_intervals = predicted_intervals(sentences, self.sentence_threshold)
        zone_results: List[ZoneAnalysisResult] = []
        for name in ("subject", "greeting", "body", "signoff"):
            b = zones.get(name)
            if not b:
                continue
            zs, ze = b
            ztext = full_text[zs:ze]
            num = den = 0.0
            for s in sentences:
                ov = overlap((s.start, s.end), (zs, ze))
                num += ov * s.prob
                den += ov
            zprob = num / den if den else 0.0
            spans = []
            for s, e in ai_intervals:
                a, c = max(s, zs), min(e, ze)
                if c > a:
                    pr = max((x.prob for x in sentences if overlap((x.start, x.end), (a, c)) > 0), default=0.0)
                    spans.append(AISpan(full_text[a:c], a, c, round(pr, 4)))
            words = len(re.findall(r"\w+", ztext))
            ai_words = sum(len(re.findall(r"\w+", sp.text)) for sp in spans)
            zone_results.append(ZoneAnalysisResult(name, ztext, zs, ze, words, ai_words, zprob,
                                                   zprob >= self.sentence_threshold, spans))

        img_results = self._analyze_images(images or [])
        img_scores = [i.ai_probability for i in img_results if i.ai_probability is not None]
        text_prob = det.prob
        index = text_prob
        if img_scores:
            w = self.settings.weight_images
            index = (1 - w) * text_prob + w * max(img_scores)
        hidden = list(html_analysis.hidden_texts) if html_analysis is not None else []
        leaks = find_prompt_leaks(full_text, *hidden)
        if leaks:
            index = max(index, self.settings.prompt_leak_floor)
        index = max(0.0, min(1.0, index))
        is_ai = index >= self.threshold
        verdict = "Сгенерировано ИИ (AI-Generated)" if is_ai else "Написано человеком (Human-Written)"

        mask = masking_signals(html_analysis, raw_text or visible_text)
        html_signals = {k: mask[k] for k in ("hidden_elements_count", "suspicious_links_count", "forms_count",
                                              "password_fields_count", "iframes_count")}
        html_signals["hidden_texts"] = [h[:300] for h in hidden[:5]]
        evasion = {"zero_width_chars_count": mask["zero_width_chars_count"], "hidden_text_chars": mask["hidden_text_chars"]}

        reasons = self._reasons(det, zone_results, img_results, leaks, mask, sentences, full_text)
        ai_zones = [z.name for z in zone_results if z.is_ai]
        expl = (
            f"Письмо отнесено к категории «{verdict}»: итоговый индекс AI-генерации {index:.2f} "
            f"(порог {self.threshold:.2f}, метод: {self.method}; оценка текста {text_prob:.2f}).\n"
        )
        if ai_zones:
            expl += "Зоны с признаками генерации: " + ", ".join(ZONE_NAMES_RU.get(z, z) for z in ai_zones) + ".\n"
        else:
            expl += "Ни одна зона не превысила порог локализации.\n"
        expl += "Ключевые факторы:\n" + "\n".join(f"• {r}" for r in reasons)

        return EmailDetectionReport(
            file_id=file_id, file_path=file_path, metadata=metadata, overall_ai_index=index, verdict=verdict,
            threshold=self.threshold, ai_zones_detected=ai_zones, zones=zone_results, images=img_results,
            html_signals=html_signals, evasion_signals=evasion, explanation=expl, reasons=reasons,
            method=self.method, text_ai_probability=text_prob, key_signals=det.signals,
            feature_contributions=det.contributions,
            sentence_scores=[{"start_char": s.start, "end_char": s.end, "probability": round(s.prob, 4),
                              "text": full_text[s.start:s.end][:300]} for s in sentences],
            prompt_leakage=leaks, explanation_threshold=self.explanation_threshold,
        )

    def _reasons(self, det, zones, images, leaks, mask, sentences, full_text) -> List[str]:
        r: List[str] = []
        if leaks:
            r.append(f"В письме остались инструкции для языковой модели (утечка промпта), например: «{leaks[0][:120]}»")
        for z in zones:
            if z.is_ai:
                r.append(f"Зона «{ZONE_NAMES_RU.get(z.name, z.name)}»: вероятность генерации {z.ai_probability:.0%}, "
                         f"выделено фрагментов: {len(z.ai_spans)}")
        if det.contributions:
            # признаки, сильнее всего сдвинувшие оценку в сторону вердикта (коэффициент × стандартизованное значение)
            sign = 1 if det.prob >= self.threshold else -1
            top = sorted(((k, c) for k, c in det.contributions.items() if c * sign > 0.1), key=lambda kv: -abs(kv[1]))[:3]
            for k, c in top:
                direction = "в сторону ИИ" if c > 0 else "в сторону человека"
                r.append(f"{FEATURE_DESCRIPTIONS_RU.get(k, k).capitalize()}: {det.signals.get(k)} (вклад {c:+.2f}, {direction})")
        top_s = sorted(sentences, key=lambda s: -s.prob)[:1]
        if top_s and top_s[0].prob >= self.sentence_threshold:
            s = top_s[0]
            r.append(f"Наиболее «машинное» предложение ({s.prob:.0%}): «{full_text[s.start:s.end][:140]}»")
        for im in images:
            if im.is_ai:
                r.append(f"Изображение {im.filename}: вероятность генерации {im.ai_probability:.0%} ({'; '.join(im.reasons)})")
        if mask["hidden_elements_count"]:
            r.append(f"В HTML {mask['hidden_elements_count']} скрытых элементов ({mask['hidden_text_chars']} симв. скрытого текста) — "
                     "признак маскировки/шаблонной рассылки; в индекс не входит")
        if mask["suspicious_links_count"]:
            r.append(f"Подозрительных ссылок (текст ≠ адрес): {mask['suspicious_links_count']}")
        if mask["zero_width_chars_count"]:
            r.append(f"Невидимых символов нулевой ширины: {mask['zero_width_chars_count']}")
        if not r:
            r.append("Явных признаков генерации не обнаружено")
        return r

    # ------------------------------------------------------------------ public API
    def analyze_email(self, eml_path: str | Path) -> EmailDetectionReport:
        from src.eml_parser import parse_eml

        path = Path(eml_path)
        if not path.exists():
            raise FileNotFoundError(f"EML файл '{path}' не найден")
        data = parse_eml(path, analyze_images=False)
        if not (data["visible_text"] or "").strip() and not (data["subject"] or "").strip():
            raise ValueError("в письме нет ни темы, ни текстовой части")
        meta = {"subject": data["subject"], "sender": data["from"], "date": data["date"]}
        return self._analyze(path.stem, str(path), data["subject"], data["visible_text"], meta,
                             data["html_analysis"], data["images"], raw_text=(data["plain_text"] or "") + (data["html"] or ""))

    def analyze_direct_text(self, text: str = "", subject: Optional[str] = None, html_content: Optional[str] = None,
                            file_id: str = "direct_input") -> EmailDetectionReport:
        html_analysis = None
        visible = text or ""
        if html_content:
            from src.html_analyzer import analyze_html

            html_analysis = analyze_html(html_content)
            if not visible.strip():
                visible = html_analysis.visible_text
        meta = {"subject": subject or "", "sender": "", "date": ""}
        return self._analyze(file_id, file_id, subject or "", visible, meta, html_analysis, [],
                             raw_text=(text or "") + (html_content or ""))
