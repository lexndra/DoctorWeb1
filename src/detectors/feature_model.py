"""Метод оценки на основе признаков (базовая интерпретируемая модель, п. 4.1.3.5 ТЗ).

Двухуровневая схема:
  1. Модель предложений — логистическая регрессия над TF-IDF словесных (1–2) и символьных (2–4) n-грамм
     нормализованного текста. Даёт вероятность генерации каждого предложения → локализация фрагментов
     с символьными границами.
  2. Документная модель — логистическая регрессия над именованными признаками: агрегаты оценок
     предложений и стилометрия (длина/вариативность предложений, лексическое разнообразие, пунктуация,
     шаблонные приветствия/концовки, «канцелярит» генераторов). Коэффициенты × стандартизованные значения
     дают вклад каждого признака в итог (объяснимость, п. 4.1.5.3).

Чтобы документная модель не переобучалась на оценках предложений, на обучающей выборке они считаются
out-of-fold (GroupKFold по группам писем).
Модель целиком сохраняется в JSON (п. 4.1.3.3).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.detectors.base import DetectorOutput, SentenceScore
from src.text_utils import normalize_text, sentence_spans, span_ai_fraction

log = logging.getLogger(__name__)

CLICHES = [
    # EN
    "i hope this email finds you well", "i hope this message finds you well", "i hope you are doing well",
    "i hope you're doing well", "please don't hesitate", "please do not hesitate", "feel free to reach out",
    "looking forward to", "thank you for your attention", "thank you for your understanding", "we are excited",
    "we're excited", "we are thrilled", "we're thrilled", "delve", "furthermore", "additionally", "moreover",
    "ensure", "seamless", "streamline", "leverage", "valued", "exclusive", "elevate", "embark", "tailored",
    "comprehensive", "crucial", "vital", "in conclusion", "it is important to note", "rest assured",
    "at your earliest convenience", "warm regards", "best regards", "kind regards", "stay tuned", "don't miss",
    # RU
    "надеюсь, у вас всё хорошо", "надеюсь, у вас все хорошо", "не стесняйтесь", "буду рад", "будем рады",
    "с нетерпением", "кроме того", "более того", "таким образом", "в заключение", "важно отметить",
    "следует отметить", "в целом", "обеспечить", "уникальн", "эксклюзивн", "комплексн", "ключев",
    "с уважением", "благодарим за", "мы рады", "в рамках",
]
GREETING_RE = re.compile(r"^\s*(dear|hello|hi|hey|good (morning|afternoon)|уважаем|здравствуй|добрый (день|вечер)|привет)\b", re.I)
CLOSING_RE = re.compile(r"(best|kind|warm) regards|sincerely|thank you|thanks|с уважением|спасибо|всего доброго", re.I)

DOC_FEATURES = [
    "sent_p_mean", "sent_p_max", "sent_frac_ai", "sent_p_std",
    "log_n_words", "avg_sent_len", "cv_sent_len",
    "ttr", "avg_word_len", "long_word_ratio", "punct_ratio", "comma_per_sent", "exclam_ratio",
    "question_ratio", "upper_ratio", "digit_ratio", "newline_per_sent", "bullet_ratio",
    "formulaic_greeting", "formulaic_closing", "cliche_per_100w", "repeat_bigram_ratio",
    "lowercase_sent_start", "double_space_ratio",
]

FEATURE_DESCRIPTIONS_RU = {
    "sent_p_mean": "средняя оценка генерации по предложениям",
    "sent_p_max": "наиболее «машинное» предложение",
    "sent_p_min": "наименее «машинное» предложение",
    "sent_p_q75": "доля предложений с высокой оценкой (верхний квартиль)",
    "sent_frac_ai": "доля предложений, похожих на сгенерированные",
    "sent_p_std": "неоднородность оценок предложений (признак частичной правки)",
    "log_n_sent": "число предложений",
    "log_n_words": "длина текста",
    "avg_sent_len": "средняя длина предложения",
    "std_sent_len": "разброс длины предложений",
    "cv_sent_len": "вариативность длины предложений (у людей выше)",
    "ttr": "лексическое разнообразие",
    "avg_word_len": "средняя длина слова",
    "long_word_ratio": "доля длинных слов",
    "punct_ratio": "плотность пунктуации",
    "comma_per_sent": "число запятых на предложение",
    "exclam_ratio": "восклицательные знаки",
    "question_ratio": "вопросительные знаки",
    "upper_ratio": "доля заглавных букв",
    "digit_ratio": "доля цифр",
    "newline_per_sent": "разбиение на строки/абзацы",
    "bullet_ratio": "маркированные списки",
    "formulaic_greeting": "формульное приветствие",
    "formulaic_closing": "формульная концовка",
    "cliche_per_100w": "шаблонные обороты, характерные для генераторов",
    "repeat_bigram_ratio": "повторы словосочетаний",
    "lowercase_sent_start": "предложения со строчной буквы (небрежность, характерная для людей)",
    "double_space_ratio": "двойные пробелы и опечатки набора",
}


def doc_features(text: str, spans: Sequence[Tuple[int, int]], sent_probs: Sequence[float]) -> np.ndarray:
    norm = normalize_text(text)
    low = norm.lower()
    words = re.findall(r"\w+", low)
    nw = max(1, len(words))
    sents = [text[s:e] for s, e in spans] or [text]
    slen = np.array([len(re.findall(r"\w+", s)) for s in sents], dtype=float)
    p = np.array(sent_probs if len(sent_probs) else [0.0], dtype=float)
    nchar = max(1, len(norm))
    letters = re.findall(r"[^\W\d_]", norm)
    bigrams = list(zip(words, words[1:]))
    lines = [l for l in text.splitlines() if l.strip()]
    f = {
        "sent_p_mean": p.mean(),
        "sent_p_max": p.max(),
        "sent_p_min": p.min(),
        "sent_p_q75": float(np.quantile(p, 0.75)),
        "sent_frac_ai": float((p >= 0.5).mean()),
        "sent_p_std": p.std(),
        "log_n_sent": math.log1p(len(sents)),
        "log_n_words": math.log1p(nw),
        "avg_sent_len": slen.mean(),
        "std_sent_len": slen.std(),
        "cv_sent_len": slen.std() / (slen.mean() + 1e-6),
        "ttr": len(set(words[:200])) / max(1, len(words[:200])),
        "avg_word_len": sum(map(len, words)) / nw,
        "long_word_ratio": sum(len(w) >= 9 for w in words) / nw,
        "punct_ratio": len(re.findall(r"[^\w\s]", norm)) / nchar,
        "comma_per_sent": norm.count(",") / len(sents),
        "exclam_ratio": norm.count("!") / len(sents),
        "question_ratio": norm.count("?") / len(sents),
        "upper_ratio": sum(c.isupper() for c in letters) / max(1, len(letters)),
        "digit_ratio": len(re.findall(r"\d", text)) / max(1, len(text)),
        "newline_per_sent": text.count("\n") / len(sents),
        "bullet_ratio": sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", l)) for l in lines) / max(1, len(lines)),
        "formulaic_greeting": float(bool(GREETING_RE.search(text[:80]))),
        "formulaic_closing": float(bool(CLOSING_RE.search(text[-120:]))),
        "cliche_per_100w": 100 * sum(low.count(c) for c in CLICHES) / nw,
        "repeat_bigram_ratio": 1 - len(set(bigrams)) / max(1, len(bigrams)),
        "lowercase_sent_start": sum(bool(re.match(r"[a-zа-яё]", s.strip())) for s in sents) / len(sents),
        "double_space_ratio": len(re.findall(r"\S  +\S| ,| \.", text)) / len(sents),
    }
    return np.array([f[k] for k in DOC_FEATURES], dtype=float)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class FeatureDetector:
    method_name = "features"

    def __init__(self):
        self.word_vec = None
        self.char_vec = None
        self.sent_coef: Optional[np.ndarray] = None
        self.sent_intercept: float = 0.0
        self.doc_mean: Optional[np.ndarray] = None
        self.doc_scale: Optional[np.ndarray] = None
        self.doc_coef: Optional[np.ndarray] = None
        self.doc_intercept: float = 0.0
        self.params: Dict = {}

    # ------------------------------------------------------------------ sentence level
    def _sent_X(self, sents: List[str]):
        from scipy.sparse import hstack

        norm = [normalize_text(s) for s in sents]
        return hstack([self.word_vec.transform(norm), self.char_vec.transform(norm)]).tocsr()

    def sentence_probs(self, sents: List[str]) -> np.ndarray:
        if not sents:
            return np.zeros(0)
        X = self._sent_X(sents)
        return _sigmoid(X @ self.sent_coef + self.sent_intercept)

    # ------------------------------------------------------------------ inference
    def score(self, text: str, subject: str = "") -> DetectorOutput:
        spans = sentence_spans(text)
        sents = [text[s:e] for s, e in spans]
        sp = self.sentence_probs(sents) if sents else np.zeros(0)
        x = doc_features(text, spans, sp)
        z = (x - self.doc_mean) / self.doc_scale
        contrib = z * self.doc_coef
        prob = float(_sigmoid(contrib.sum() + self.doc_intercept))
        return DetectorOutput(
            prob=prob,
            sentences=[SentenceScore(s, e, float(p)) for (s, e), p in zip(spans, sp)],
            signals={k: round(float(v), 4) for k, v in zip(DOC_FEATURES, x)},
            contributions={k: round(float(c), 4) for k, c in zip(DOC_FEATURES, contrib)},
            method=self.method_name,
        )

    # ------------------------------------------------------------------ training
    @staticmethod
    def _explode(records: List[Dict]):
        sents, labels, doc_idx, spans_all = [], [], [], []
        for i, r in enumerate(records):
            spans = sentence_spans(r["text"])
            spans_all.append(spans)
            for sp in spans:
                sents.append(r["text"][sp[0]:sp[1]])
                labels.append(int(span_ai_fraction(sp, r["ai_char_intervals"]) >= 0.5))
                doc_idx.append(i)
        return sents, np.array(labels), np.array(doc_idx), spans_all

    def _fit_sentence_model(self, sents: List[str], y: np.ndarray, C: float):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        norm = [normalize_text(s) for s in sents]
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True, lowercase=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=60000, sublinear_tf=True, lowercase=True)
        X = hstack([wv.fit_transform(norm), cv.fit_transform(norm)]).tocsr()
        clf = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=200)
        clf.fit(X, y)
        return wv, cv, clf

    def fit(self, records: List[Dict], n_folds: int = 3, C_sent: float = 2.0, C_doc: float = 1.0, seed: int = 42):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        t0 = time.time()
        sents, y_sent, doc_idx, spans_all = self._explode(records)
        groups = np.array([records[i]["group_id"] for i in doc_idx])
        log.info("Обучение: документов %d, предложений %d (AI-предложений %.1f%%)", len(records), len(sents), 100 * y_sent.mean())

        # out-of-fold оценки предложений для обучения документной модели
        oof = np.zeros(len(sents))
        for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(sents, y_sent, groups)):
            wv, cv, clf = self._fit_sentence_model([sents[i] for i in tr], y_sent[tr], C_sent)
            self.word_vec, self.char_vec = wv, cv
            self.sent_coef, self.sent_intercept = clf.coef_.ravel(), float(clf.intercept_[0])
            oof[te] = self.sentence_probs([sents[i] for i in te])
            log.info("  фолд %d/%d готов (%.0f c)", k + 1, n_folds, time.time() - t0)

        wv, cv, clf = self._fit_sentence_model(sents, y_sent, C_sent)
        self.word_vec, self.char_vec = wv, cv
        self.sent_coef, self.sent_intercept = clf.coef_.ravel(), float(clf.intercept_[0])

        X_doc = np.zeros((len(records), len(DOC_FEATURES)))
        for i, r in enumerate(records):
            X_doc[i] = doc_features(r["text"], spans_all[i], oof[doc_idx == i])
        y_doc = np.array([r["label"] for r in records])
        scaler = StandardScaler().fit(X_doc)
        dclf = LogisticRegression(C=C_doc, class_weight="balanced", max_iter=2000)
        dclf.fit(scaler.transform(X_doc), y_doc)
        self.doc_mean, self.doc_scale = scaler.mean_, np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        self.doc_coef, self.doc_intercept = dclf.coef_.ravel(), float(dclf.intercept_[0])
        self.params = {
            "n_docs": len(records), "n_sentences": len(sents), "n_folds": n_folds, "C_sent": C_sent, "C_doc": C_doc,
            "seed": seed, "word_ngram": [1, 2], "char_ngram": [2, 4], "word_max_features": 40000,
            "char_max_features": 60000, "train_seconds": round(time.time() - t0, 1),
        }
        log.info("Обучение завершено за %.1f c", time.time() - t0)
        return self

    # ------------------------------------------------------------------ JSON
    @staticmethod
    def _vec_to_json(v) -> Dict:
        return {
            "analyzer": v.analyzer, "ngram_range": list(v.ngram_range), "lowercase": v.lowercase,
            "sublinear_tf": v.sublinear_tf,
            "vocabulary": {t: int(i) for t, i in v.vocabulary_.items()}, "idf": v.idf_.round(6).tolist(),
        }

    @staticmethod
    def _vec_from_json(d: Dict):
        from sklearn.feature_extraction.text import TfidfVectorizer

        v = TfidfVectorizer(analyzer=d["analyzer"], ngram_range=tuple(d["ngram_range"]), lowercase=d["lowercase"],
                            sublinear_tf=d["sublinear_tf"], vocabulary=d["vocabulary"])
        v.idf_ = np.array(d["idf"])
        return v

    def save(self, path: str | Path) -> None:
        data = {
            "format": "dw1-feature-model", "version": 1, "doc_features": DOC_FEATURES,
            "word_vec": self._vec_to_json(self.word_vec), "char_vec": self._vec_to_json(self.char_vec),
            "sent_coef": self.sent_coef.round(6).tolist(), "sent_intercept": self.sent_intercept,
            "doc_mean": self.doc_mean.tolist(), "doc_scale": self.doc_scale.tolist(),
            "doc_coef": self.doc_coef.tolist(), "doc_intercept": self.doc_intercept, "params": self.params,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureDetector":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        assert d.get("format") == "dw1-feature-model", "неизвестный формат модели"
        assert d["doc_features"] == DOC_FEATURES, "набор признаков модели не совпадает с кодом"
        m = cls()
        m.word_vec, m.char_vec = cls._vec_from_json(d["word_vec"]), cls._vec_from_json(d["char_vec"])
        m.sent_coef, m.sent_intercept = np.array(d["sent_coef"]), float(d["sent_intercept"])
        m.doc_mean, m.doc_scale = np.array(d["doc_mean"]), np.array(d["doc_scale"])
        m.doc_coef, m.doc_intercept = np.array(d["doc_coef"]), float(d["doc_intercept"])
        m.params = d.get("params", {})
        return m
