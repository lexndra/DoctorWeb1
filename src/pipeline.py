"""Главный конвейер принятия решений (Decision Engine & Pipeline).

Объединяет:
1. Модель токенизации и поиска границ DeBERTa-BiGRU-CRF (текстовые зоны).
2. Анализатор изображений, QR и метаданных (image_analyzer).
3. Анализатор HTML-разметки и анти-сокрытия (html_analyzer).
4. Расчет композитного индекса AI-генерации (0.0..1.0) и вердикта.
5. Формирование русскоязычного объяснения для аналитика SOC (п. 4.1.5 ТЗ).
6. Сериализацию в единый формат JSON (п. 4.1.6 ТЗ).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from razdel import tokenize
from transformers import AutoTokenizer

from src.html_analyzer import analyze_html, HtmlAnalysisResult
from src.model import DeBERTaBiGRUCRFTagger


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_default_checkpoint() -> str:
    """Ищет файл весов tnc_dbgc_coat_best.pt в стандартных путях."""
    candidates = [
        Path("tnc_dbgc_coat_best.pt"),
        Path("/Users/roman/Library/CloudStorage/GoogleDrive-korchaginroma511@gmail.com/Мой диск/doctor_web/coat_segmentation/tnc_dbgc_coat_best.pt"),
        Path("/Users/roman/Desktop/DoctorWeb1/tnc_dbgc_coat_best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "tnc_dbgc_coat_best.pt"


# Контракт данных для EML-парсера (задача сокомандника)
@dataclass
class EmailZone:
    """Зона письма с символьными границами."""
    name: str  # 'subject', 'greeting', 'body', 'cta', 'signoff'
    text: str
    start_char: int
    end_char: int


@dataclass
class ParsedEmail:
    """Контракт разобранного EML письма для передачи в Pipeline."""
    file_id: str
    file_path: str
    subject: str
    sender: str
    recipient: str
    date: str
    headers: Dict[str, str]
    full_text: str
    zones: List[EmailZone]
    zero_width_chars_count: int = 0
    html_raw: Optional[str] = None
    html_analysis: Optional[HtmlAnalysisResult] = None
    extracted_images: List[Path] = field(default_factory=list)
    precomputed_image_results: List[ImageAnalysisResult] = field(default_factory=list)
    is_corrupted: bool = False
    parse_warnings: List[str] = field(default_factory=list)


@dataclass
class AISpan:
    """Сгенерированный ИИ фрагмент с символьными границами."""
    text: str
    start_char: int
    end_char: int


@dataclass
class ZoneAnalysisResult:
    """Результат анализа одной зоны письма."""
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
        return {
            "name": self.name,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "words_count": self.words_count,
            "ai_words_count": self.ai_words_count,
            "ai_probability": round(self.ai_probability, 4),
            "is_ai": self.is_ai,
            "ai_spans": [asdict(s) for s in self.ai_spans],
        }


@dataclass
class ImageAnalysisResult:
    """Результат проверки одного изображения письма."""
    file_name: str
    ai_probability: float
    is_ai: bool
    has_qr: bool
    qr_codes: List[str]
    has_text: bool
    ocr_text: str
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "file_name": self.file_name,
            "ai_probability": round(self.ai_probability, 4),
            "is_ai": self.is_ai,
            "has_qr": self.has_qr,
            "qr_codes": self.qr_codes,
            "has_text": self.has_text,
            "ocr_text": self.ocr_text,
            "details": self.details,
        }


@dataclass
class EmailDetectionReport:
    """Финальный отчет анализа электронного письма (п. 4.1.6 ТЗ)."""
    file_id: str
    file_path: str
    metadata: Dict[str, str]

    # Итоговый вердикт
    overall_ai_index: float
    verdict: str  # "Сгенерировано ИИ (AI-Generated)" или "Написано человеком (Human-Written)"
    threshold: float

    # Анализ по зонам письма
    zones: List[ZoneAnalysisResult]
    ai_zones_detected: List[str]

    # Мультимодальные данные
    images: List[ImageAnalysisResult]
    html_signals: Dict[str, object]
    evasion_signals: Dict[str, object]

    # Объяснение для аналитика SOC (п. 4.1.5 ТЗ)
    explanation: str
    reasons: List[str]

    method: str = "TNC-DBGC* + Vision + HTML Evasion (Ensemble)"

    def to_dict(self) -> Dict[str, object]:
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "overall_ai_index": round(self.overall_ai_index, 4),
            "verdict": self.verdict,
            "threshold": self.threshold,
            "ai_zones_detected": self.ai_zones_detected,
            "zones": [z.to_dict() for z in self.zones],
            "images": [img.to_dict() for img in self.images],
            "html_signals": self.html_signals,
            "evasion_signals": self.evasion_signals,
            "explanation": self.explanation,
            "reasons": self.reasons,
            "method": self.method,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class EmailAnalysisPipeline:
    """Главный класс конвейера анализа писем на генеративный ИИ."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        threshold: float = 0.40,
        enable_images: bool = True,
    ):
        self.device = device or get_default_device()
        self.threshold = threshold
        self.enable_images = enable_images
        self.ckpt_path = Path(checkpoint_path or find_default_checkpoint())

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Файл весов модели '{self.ckpt_path}' не найден!")

        print(f"[Pipeline] Загрузка модели DeBERTa-BiGRU-CRF на {self.device}...")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model_name = ckpt.get("model_name", "microsoft/mdeberta-v3-base")
        self.hidden_dim = ckpt.get("hidden_dim", 256)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = DeBERTaBiGRUCRFTagger(
            model_name=self.model_name,
            num_labels=2,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self._image_analyzer_available = False
        if self.enable_images:
            try:
                import src.image_analyzer as img_module
                self.img_module = img_module
                self._image_analyzer_available = True
            except Exception as e:
                print(f"[Pipeline] Предупреждение: Модуль изображений недоступен: {e}")

    @torch.no_grad()
    def _predict_words(self, words: List[str]) -> List[int]:
        """Прогоняет слова через DeBERTa-BiGRU-CRF и возвращает предсказания (0=человек, 1=ИИ)."""
        if not words:
            return []

        all_preds: List[int] = []

        # Чанкование по 250 слов во избежание обрезки на 512 токенах
        for chunk_start in range(0, len(words), 250):
            chunk_words = words[chunk_start : chunk_start + 250]
            encoding = self.tokenizer(
                chunk_words,
                is_split_into_words=True,
                max_length=512,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            subword_preds: List[int] = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]

            word_ids = encoding.word_ids(batch_index=0)
            seen_words = set()
            chunk_word_preds: List[int] = []

            for w_id, p_label in zip(word_ids, subword_preds):
                if w_id is not None and w_id not in seen_words:
                    seen_words.add(w_id)
                    chunk_word_preds.append(p_label)

            all_preds.extend(chunk_word_preds[: len(chunk_words)])

        return all_preds

    def _analyze_zone(self, zone: EmailZone) -> ZoneAnalysisResult:
        """Анализирует текст одной зоны письма."""
        tokens = list(tokenize(zone.text))
        words = [t.text for t in tokens if t.text.strip()]

        if not words:
            return ZoneAnalysisResult(
                name=zone.name,
                text=zone.text,
                start_char=zone.start_char,
                end_char=zone.end_char,
                words_count=0,
                ai_words_count=0,
                ai_probability=0.0,
                is_ai=False,
            )

        preds = self._predict_words(words)
        ai_words_count = sum(preds)
        ai_prob = (ai_words_count / len(preds)) if preds else 0.0
        is_ai = ai_prob >= 0.35  # Порог локализации зоны

        # Поиск непрерывных интервалов (спанов) сгенерированного текста
        ai_spans: List[AISpan] = []
        current_span_words: List[str] = []
        span_start_rel: Optional[int] = None

        word_tokens = [t for t in tokens if t.text.strip()]
        for t, pred in zip(word_tokens, preds):
            if pred == 1:
                if span_start_rel is None:
                    span_start_rel = t.start
                current_span_words.append(t.text)
            else:
                if current_span_words and span_start_rel is not None:
                    span_text = " ".join(current_span_words)
                    ai_spans.append(
                        AISpan(
                            text=span_text,
                            start_char=zone.start_char + span_start_rel,
                            end_char=zone.start_char + t.start,
                        )
                    )
                    current_span_words = []
                    span_start_rel = None

        if current_span_words and span_start_rel is not None:
            span_text = " ".join(current_span_words)
            ai_spans.append(
                AISpan(
                    text=span_text,
                    start_char=zone.start_char + span_start_rel,
                    end_char=zone.end_char,
                )
            )

        return ZoneAnalysisResult(
            name=zone.name,
            text=zone.text,
            start_char=zone.start_char,
            end_char=zone.end_char,
            words_count=len(words),
            ai_words_count=ai_words_count,
            ai_probability=ai_prob,
            is_ai=is_ai,
            ai_spans=ai_spans,
        )

    def _analyze_images(self, image_paths: List[Path]) -> List[ImageAnalysisResult]:
        """Прогоняет изображения через модуль Алёны."""
        results: List[ImageAnalysisResult] = []
        if not self._image_analyzer_available or not image_paths:
            return results

        for img_path in image_paths:
            try:
                res = self.img_module.analyze_image(img_path)

                ai_preds = res.get("ai_detection", [])
                ai_prob = 0.0
                if isinstance(ai_preds, list):
                    for p in ai_preds:
                        label = str(p.get("label", "")).lower()
                        if label in ("artificial", "ai", "generated"):
                            ai_prob = float(p.get("score", 0.0))

                qr_codes = res.get("qr_codes", [])
                ocr_text = str(res.get("ocr_text", "")).strip()

                results.append(
                    ImageAnalysisResult(
                        file_name=img_path.name,
                        ai_probability=ai_prob,
                        is_ai=(ai_prob >= 0.50),
                        has_qr=bool(qr_codes),
                        qr_codes=qr_codes,
                        has_text=bool(ocr_text),
                        ocr_text=ocr_text,
                        details=res,
                    )
                )
            except Exception as e:
                print(f"[Pipeline] Ошибка анализа изображения {img_path.name}: {e}")

        return results

    def _build_soc_explanation(
        self,
        overall_index: float,
        verdict: str,
        zone_results: List[ZoneAnalysisResult],
        img_results: List[ImageAnalysisResult],
        html_signals: Dict[str, object],
        evasion_signals: Dict[str, object],
    ) -> Tuple[str, List[str]]:
        """Генерирует связное русскоязычное объяснение для аналитика SOC (п. 4.1.5 ТЗ)."""
        reasons: List[str] = []

        # Текстовые причины
        for z in zone_results:
            if z.is_ai:
                reasons.append(
                    f"В зоне '{z.name}' выявлена высокая вероятность машинной генерации "
                    f"({z.ai_probability:.1%}, найдено {len(z.ai_spans)} сгенерированных фрагментов)."
                )

        # Причины по изображениям
        for img in img_results:
            if img.is_ai:
                reasons.append(
                    f"Во вложении '{img.file_name}' обнаружены визуальные признаки генерации ИИ "
                    f"(оценка классификатора: {img.ai_probability:.1%})."
                )
            if img.has_qr:
                reasons.append(
                    f"В изображении '{img.file_name}' найден QR-код с переходом: {', '.join(img.qr_codes)}."
                )

        # HTML причины
        hidden_cnt = html_signals.get("hidden_elements_count", 0)
        if hidden_cnt > 0:
            reasons.append(
                f"В HTML-разметке обнаружено {hidden_cnt} скрытых элементов "
                f"(стили display:none, font-size:0 или атрибут hidden) для маскировки спама."
            )

        susp_links = html_signals.get("suspicious_links_count", 0)
        if susp_links > 0:
            reasons.append(
                f"Обнаружено {susp_links} подозрительных ссылок с несовпадением видимого и реального домена (фишинг/спуфинг)."
            )

        # Невидимые символы
        zw_cnt = evasion_signals.get("zero_width_chars_count", 0)
        if zw_cnt > 0:
            reasons.append(
                f"В тексте найдено {zw_cnt} скрытых невидимых символов (Zero-Width/Bidi), используемых для обхода фильтров."
            )

        if not reasons:
            reasons.append("Текст письма, разметка и медиафайлы соответствуют естественному стилю человека.")

        summary = (
            f"Письмо отнесено к категории '{verdict}' с итоговым индексом AI-генерации {overall_index:.2f} "
            f"(порог классификации: {self.threshold:.2f})."
        )

        full_explanation = f"{summary}\n\nКлючевые факторы:\n" + "\n".join(f"• {r}" for r in reasons)
        return full_explanation, reasons

    def analyze_parsed_email(self, parsed: ParsedEmail) -> EmailDetectionReport:
        """Анализирует письмо на основе переданного объекта ParsedEmail (от EML-парсера)."""
        # 1. Анализ зон текста моделью DeBERTa
        zone_results: List[ZoneAnalysisResult] = []
        total_words = 0
        total_ai_words = 0

        for z in parsed.zones:
            zr = self._analyze_zone(z)
            zone_results.append(zr)
            total_words += zr.words_count
            total_ai_words += zr.ai_words_count

        text_ai_score = (total_ai_words / total_words) if total_words > 0 else 0.0

        # 2. Анализ изображений
        if parsed.precomputed_image_results:
            img_results = parsed.precomputed_image_results
        else:
            img_results = self._analyze_images(parsed.extracted_images)
        max_img_score = max([img.ai_probability for img in img_results], default=0.0)

        # 3. HTML и Evasion сигналы
        html_signals: Dict[str, object] = {}
        if parsed.html_analysis:
            html_signals = {
                "hidden_elements_count": parsed.html_analysis.hidden_elements_count,
                "hidden_texts": parsed.html_analysis.hidden_texts[:5],
                "suspicious_links_count": parsed.html_analysis.suspicious_links_count,
                "forms_count": parsed.html_analysis.forms_count,
                "password_fields_count": parsed.html_analysis.password_fields_count,
                "iframes_count": parsed.html_analysis.iframes_count,
            }

        evasion_signals = {
            "zero_width_chars_count": parsed.zero_width_chars_count,
            "is_corrupted": parsed.is_corrupted,
            "warnings": parsed.parse_warnings,
        }

        # 4. Расчет композитного индекса AI
        evasion_penalty = 0.0
        hidden_count = html_signals.get("hidden_elements_count", 0)
        if hidden_count > 0:
            evasion_penalty += 0.18 + min(0.06, hidden_count * 0.02)
        if html_signals.get("suspicious_links_count", 0) > 0:
            evasion_penalty += 0.18
        if parsed.zero_width_chars_count > 0:
            evasion_penalty += 0.15

        # Штраф при локализации явных сгенерированных фрагментов в ключевых зонах
        has_ai_zone = any(z.is_ai for z in zone_results)
        if has_ai_zone:
            evasion_penalty += 0.10

        if img_results:
            overall_index = 0.60 * text_ai_score + 0.25 * max_img_score + evasion_penalty
        else:
            overall_index = 0.75 * text_ai_score + evasion_penalty

        overall_index = max(0.0, min(1.0, overall_index))
        verdict = (
            "Сгенерировано ИИ (AI-Generated)"
            if overall_index >= self.threshold
            else "Написано человеком (Human-Written)"
        )

        # 5. Генерация объяснения для SOC
        ai_zones_detected = [z.name for z in zone_results if z.is_ai]
        explanation, reasons = self._build_soc_explanation(
            overall_index=overall_index,
            verdict=verdict,
            zone_results=zone_results,
            img_results=img_results,
            html_signals=html_signals,
            evasion_signals=evasion_signals,
        )

        return EmailDetectionReport(
            file_id=parsed.file_id,
            file_path=parsed.file_path,
            metadata={
                "subject": parsed.subject,
                "sender": parsed.sender,
                "recipient": parsed.recipient,
                "date": parsed.date,
            },
            overall_ai_index=overall_index,
            verdict=verdict,
            threshold=self.threshold,
            zones=zone_results,
            ai_zones_detected=ai_zones_detected,
            images=img_results,
            html_signals=html_signals,
            evasion_signals=evasion_signals,
            explanation=explanation,
            reasons=reasons,
        )

    def analyze_email(self, eml_path: str | Path) -> EmailDetectionReport:
        """Парсит EML через модуль src.eml_parser и запускает мультимодальный анализ."""
        path = Path(eml_path)
        if not path.exists():
            raise FileNotFoundError(f"EML файл '{path}' не найден!")

        from src.eml_parser import parse_eml
        data = parse_eml(path)

        # 1. Извлечение зон письма с символьными оффсетами
        zones: List[EmailZone] = []
        zoned_text = data.get("zoned_text", "")
        raw_zones = data.get("zones", {})

        for z_name in ("subject", "greeting", "body", "signoff"):
            bounds = raw_zones.get(z_name)
            if bounds is not None and len(bounds) == 2:
                s, e = bounds
                z_text = zoned_text[s:e]
                if z_text.strip():
                    zones.append(EmailZone(name=z_name, text=z_text, start_char=s, end_char=e))

        # Fallback: если зоны не выделились, берем visible_text как body
        if not zones:
            fallback_text = data.get("visible_text") or data.get("plain_text") or zoned_text or ""
            zones.append(EmailZone(name="body", text=fallback_text, start_char=0, end_char=len(fallback_text)))

        # 2. Извлечение результатов анализа изображений
        precomputed_images: List[ImageAnalysisResult] = []
        for item in data.get("image_results", []):
            if "analysis" in item:
                analysis = item["analysis"]
                img_name = item.get("filename") or f"image_{item.get('image_number', 1)}"
                ai_prob = float(analysis.get("ai_score", 0.0))
                qr_codes = analysis.get("qr_codes", [])
                ocr_text = str(analysis.get("ocr_text", "")).strip()
                precomputed_images.append(
                    ImageAnalysisResult(
                        file_name=img_name,
                        ai_probability=ai_prob,
                        is_ai=(ai_prob >= 0.50),
                        has_qr=bool(qr_codes),
                        qr_codes=qr_codes,
                        has_text=bool(ocr_text),
                        ocr_text=ocr_text,
                        details=analysis,
                    )
                )

        # 3. Подсчет zero-width символов в тексте письма и HTML
        zw_pattern = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")
        text_for_zw = (zoned_text or "") + (data.get("html") or "")
        zw_count = len(zw_pattern.findall(text_for_zw))

        parsed = ParsedEmail(
            file_id=path.stem,
            file_path=str(path),
            subject=data.get("subject", ""),
            sender=data.get("from", ""),
            recipient="",
            date=data.get("date", ""),
            headers={},
            full_text=zoned_text,
            zones=zones,
            zero_width_chars_count=zw_count,
            html_raw=data.get("html"),
            html_analysis=data.get("html_analysis"),
            precomputed_image_results=precomputed_images,
        )
        return self.analyze_parsed_email(parsed)

    def analyze_direct_text(
        self,
        text: str,
        subject: Optional[str] = None,
        html_content: Optional[str] = None,
        image_paths: Optional[List[Path]] = None,
    ) -> EmailDetectionReport:
        """Прямой анализ текста/HTML/изображений без необходимости в EML-файле."""
        zones: List[EmailZone] = []
        full_text = ""
        offset = 0

        if subject:
            zones.append(EmailZone(name="subject", text=subject, start_char=0, end_char=len(subject)))
            full_text = subject + "\n\n"
            offset = len(full_text)

        html_analysis = None
        visible_body = text
        if html_content:
            html_analysis = analyze_html(html_content)
            visible_body = html_analysis.visible_text or text

        full_text += visible_body
        zones.append(EmailZone(name="body", text=visible_body, start_char=offset, end_char=offset + len(visible_body)))

        zw_pattern = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")
        text_for_zw = (full_text or "") + (html_content or "")
        zw_count = len(zw_pattern.findall(text_for_zw))

        parsed = ParsedEmail(
            file_id="direct_input",
            file_path="memory",
            subject=subject or "",
            sender="",
            recipient="",
            date="",
            headers={},
            full_text=full_text,
            zones=zones,
            zero_width_chars_count=zw_count,
            html_raw=html_content,
            html_analysis=html_analysis,
            extracted_images=image_paths or [],
        )

        return self.analyze_parsed_email(parsed)
