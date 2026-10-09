"""
Архитектура TNC-DBGC* (Transformer + BiGRU + CRF с оптимизациями) из статьи:
Fine-Grained Detection of AI-Generated Text Using Sentence-Level Segmentation (arXiv:2509.17830v2).

Состав модели:
1. Предобученный Transformer-энкодер (по умолчанию microsoft/mdeberta-v3-base для русского языка).
2. Dynamic Dropout — адаптивный коэффициент отсева по мере обучения.
3. Двунаправленный рекуррентный блок BiGRU + LayerNorm + нелинейная проекция.
4. Линейный классификатор с инициализацией весов по методу Ксавье (Xavier Uniform).
5. Линейно-цепной слой условных случайных полей (CRF):
   - При обучении минимизирует отрицательное логарифмическое правдоподобие: L = -(S(x, y) - log Z(x)).
   - При инференсе декодирует оптимальную последовательность меток алгоритмом Витерби.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class DynamicDropout(nn.Module):
    """
    Динамический Dropout, плавно адаптирующий вероятность отсева p
    от initial_p до max_p по мере прохождения эпох/шагов обучения.
    """

    def __init__(self, initial_p: float = 0.1, max_p: float = 0.3):
        super().__init__()
        self.initial_p = initial_p
        self.max_p = max_p
        self.current_p = initial_p

    def set_progress(self, progress: float) -> None:
        clamped = min(max(progress, 0.0), 1.0)
        self.current_p = self.initial_p + (self.max_p - self.initial_p) * clamped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(x, p=self.current_p, training=self.training)


class DeBERTaBiGRUCRFTagger(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        num_labels: int = 2,
        hidden_dim: int = 256,
        num_gru_layers: int = 2,
        dropout_initial: float = 0.1,
        dropout_max: float = 0.3,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.transformer.float()
        transformer_hidden = self.transformer.config.hidden_size

        # Динамический отсев на выходе трансформера и после проекционного слоя
        self.dyn_dropout1 = DynamicDropout(dropout_initial, dropout_max)
        self.dyn_dropout2 = DynamicDropout(dropout_initial, dropout_max)

        # Двунаправленная рекуррентная сеть BiGRU для улавливания межпредложенческих переходов стиля
        self.gru = nn.GRU(
            input_size=transformer_hidden,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_max if num_gru_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.act = nn.ReLU()

        # Линейный классификатор (оценки эмиссии ψ_t для каждого токена)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        # Инициализация Ксавье (Xavier Uniform) для стабилизации дисперсии градиентов перед CRF
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # Линейно-цепной слой CRF (матрица переходов φ(y_t, y_{t+1}))
        self.crf = CRF(num_labels, batch_first=True)

    def update_dropout(self, progress: float) -> None:
        self.dyn_dropout1.set_progress(progress)
        self.dyn_dropout2.set_progress(progress)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dyn_dropout1(outputs.last_hidden_state.float())

        gru_out, _ = self.gru(seq_out)
        gru_out = self.layer_norm(gru_out)
        gru_out = self.dyn_dropout2(self.act(self.proj(gru_out)))

        emissions = self.classifier(gru_out)  # [B, L, num_labels]
        mask = attention_mask.bool()

        if labels is not None:
            # torchcrf требует метки в диапазоне [0, num_labels - 1], поэтому -100 на [CLS]/[SEP] заменяем на 0
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            log_likelihood = self.crf(emissions, crf_labels, mask=mask, reduction="mean")
            return -log_likelihood
        else:
            # Декодирование алгоритмом Витерби
            predictions: List[List[int]] = self.crf.decode(emissions, mask=mask)
            return predictions
