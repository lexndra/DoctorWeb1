"""Метод оценки с помощью языковой модели: пословный тэггер TNC-DBGC* (mDeBERTa-v3 + BiGRU + CRF, src/model.py).

Оценка документа — доля слов, помеченных как сгенерированные; оценка предложения — доля таких слов
в предложении. Веса: models/tnc_dbgc_llmtrace.pt (дообучение на LLMTrace, см. kaggle/) или
исходный tnc_dbgc_coat_best.pt (CoAT, 2023). torch/transformers импортируются лениво.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from src.detectors.base import DetectorOutput, SentenceScore
from src.text_utils import sentence_spans

log = logging.getLogger(__name__)

DEFAULT_CANDIDATES = [
    "models/tnc_dbgc_llmtrace.pt",
    "tnc_dbgc_llmtrace.pt",
    "models/tnc_dbgc_coat_best.pt",
    "tnc_dbgc_coat_best.pt",
]


def find_lm_checkpoint(explicit: Optional[str] = None) -> Optional[Path]:
    from src.config import ROOT

    for c in ([explicit] if explicit else []) + DEFAULT_CANDIDATES:
        p = Path(c)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return p
    return None


class LMDetector:
    method_name = "lm"

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer

        from src.model import DeBERTaBiGRUCRFTagger

        path = find_lm_checkpoint(checkpoint)
        if path is None:
            raise FileNotFoundError("Веса языковой модели не найдены (models/tnc_dbgc_llmtrace.pt или tnc_dbgc_coat_best.pt)")
        import os

        device = device or os.getenv("DW1_LM_DEVICE")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model_name = ckpt.get("model_name", "microsoft/mdeberta-v3-base")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = DeBERTaBiGRUCRFTagger(model_name=self.model_name, num_labels=2, hidden_dim=ckpt.get("hidden_dim", 256))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        self.checkpoint = path
        self.torch = torch
        log.info("LM-детектор загружен: %s на %s", path.name, self.device)

    def _predict_words(self, words: List[str]) -> List[int]:
        preds: List[int] = []
        with self.torch.no_grad():
            for i in range(0, len(words), 250):
                chunk = words[i: i + 250]
                enc = self.tokenizer(chunk, is_split_into_words=True, max_length=512, truncation=True, return_tensors="pt")
                out = self.model(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device))[0]
                seen, wp = set(), []
                for w_id, p in zip(enc.word_ids(0), out):
                    if w_id is not None and w_id not in seen:
                        seen.add(w_id)
                        wp.append(int(p))
                wp += [0] * (len(chunk) - len(wp))  # обрезанные токенизатором слова
                preds.extend(wp[: len(chunk)])
        return preds

    def score(self, text: str, subject: str = "") -> DetectorOutput:
        words: List[Tuple[int, int]] = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        preds = self._predict_words([text[s:e] for s, e in words]) if words else []
        spans = sentence_spans(text)
        sents = []
        for s, e in spans:
            inside = [p for (ws, we), p in zip(words, preds) if ws >= s and we <= e + 1]
            sents.append(SentenceScore(s, e, float(sum(inside) / len(inside)) if inside else 0.0))
        prob = float(sum(preds) / len(preds)) if preds else 0.0
        return DetectorOutput(prob=prob, sentences=sents, signals={"lm_ai_word_fraction": round(prob, 4)}, method=self.method_name)


class CombinedDetector:
    """Комбинированный метод: взвешенная сумма оценок признаковой и языковой моделей (п. 4.1.3.4)."""

    method_name = "combined"

    def __init__(self, features, lm, w_features: float, w_lm: float):
        self.features, self.lm = features, lm
        tot = (w_features + w_lm) or 1.0
        self.wf, self.wl = w_features / tot, w_lm / tot

    def score(self, text: str, subject: str = "") -> DetectorOutput:
        a, b = self.features.score(text, subject), self.lm.score(text, subject)
        bmap = {(x.start, x.end): x.prob for x in b.sentences}
        sents = [SentenceScore(x.start, x.end, self.wf * x.prob + self.wl * bmap.get((x.start, x.end), b.prob)) for x in a.sentences]
        return DetectorOutput(
            prob=self.wf * a.prob + self.wl * b.prob,
            sentences=sents,
            signals={**a.signals, **b.signals, "features_prob": round(a.prob, 4), "lm_prob": round(b.prob, 4)},
            contributions=a.contributions,
            method=self.method_name,
        )
