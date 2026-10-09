"""
PyTorch Dataset для потокенной сегментации гибридных текстов из готовых CSV-файлов CoAT_Hybrid.
Никакой генерации на лету — просто читает готовые колонки `hybrid_text` и `word_labels` из CSV
и выравнивает метки слов по субтокенам трансформера через `encoding.word_ids()`.
"""

import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset


class HybridCoATDataset(Dataset):
    def __init__(self, csv_path: str | Path, tokenizer, max_length: int = 512):
        df = pd.read_csv(csv_path)
        self.texts: List[str] = df["hybrid_text"].astype(str).tolist()
        self.word_labels_list: List[List[int]] = [json.loads(lbl) for lbl in df["word_labels"].tolist()]
        self.author_seqs: List[str] = df["author_seq"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        words: List[str] = self.texts[idx].split()
        word_labels: List[int] = self.word_labels_list[idx]

        # Токенизация списка слов с сохранением привязки субтокенов к индексам слов
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()
        token_labels: List[int] = []
        for w_id in word_ids:
            if w_id is None or w_id >= len(word_labels):
                # Спецтокены [CLS], [SEP] получают метку -100
                token_labels.append(-100)
            else:
                token_labels.append(word_labels[w_id])

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(token_labels, dtype=torch.long),
        }


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Динамический паддинг батча до максимальной длины последовательности в текущем батче.
    Для pad-позиций в labels проставляется -100, а в attention_mask — 0.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids_batch.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_mask_batch.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_batch.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_batch),
        "attention_mask": torch.stack(attention_mask_batch),
        "labels": torch.stack(labels_batch),
    }
