"""
Настройка оптимизатора с послойным распадом скорости обучения (LLRD — Layer-wise Learning Rate Decay)
в соответствии с разделом 4.2 и 7 статьи (1e-6 -> 5e-6 -> 1e-5 -> 1e-4).
"""

import torch


def build_llrd_optimizer(model, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """
    Формирует группы параметров для AdamW с возрастающим learning rate от нижних слоев к верхним:
    - Embeddings трансформера: 1e-6
    - Нижняя половина слоев энкодера трансформера: 5e-6
    - Верхняя половина слоев энкодера трансформера: 1e-5
    - Слои BiGRU, LayerNorm, Linear Projection, Classifier и CRF: 1e-4
    """
    encoder_layers = list(model.transformer.encoder.layer)
    mid = len(encoder_layers) // 2

    optimizer_grouped_parameters = [
        {
            "params": model.transformer.embeddings.parameters(),
            "lr": 1e-6,
            "weight_decay": weight_decay,
        },
        {
            "params": model.transformer.encoder.layer[:mid].parameters(),
            "lr": 5e-6,
            "weight_decay": weight_decay,
        },
        {
            "params": model.transformer.encoder.layer[mid:].parameters(),
            "lr": 1e-5,
            "weight_decay": weight_decay,
        },
        {
            "params": model.gru.parameters(),
            "lr": 1e-4,
            "weight_decay": weight_decay,
        },
        {
            "params": model.layer_norm.parameters(),
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
        {
            "params": model.proj.parameters(),
            "lr": 1e-4,
            "weight_decay": weight_decay,
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-4,
            "weight_decay": weight_decay,
        },
        {
            "params": model.crf.parameters(),
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5, weight_decay=weight_decay)
