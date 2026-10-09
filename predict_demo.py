"""
Демонстрационный скрипт инференса обученной модели TNC-DBGC*:
Принимает произвольный русскоязычный текст, предсказывает автора каждого слова
алгоритмом Витерби и выводит точки перехода (границы Человек <-> ИИ) с ANSI-подсветкой.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import AutoTokenizer
from razdel import tokenize

from src.model import DeBERTaBiGRUCRFTagger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def segment_text(
    text: str,
    model: DeBERTaBiGRUCRFTagger,
    tokenizer,
    device: torch.device,
    max_len: int = 512,
) -> Tuple[List[str], List[int], List[int]]:
    model.eval()
    words = [t.text for t in tokenize(text) if t.text.strip()]
    if not words:
        return [], [], []

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    subword_preds: List[int] = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    word_ids = encoding.word_ids(batch_index=0)
    word_preds: List[int] = []
    seen_words = set()
    for w_id, pred_label in zip(word_ids, subword_preds):
        if w_id is not None and w_id not in seen_words:
            seen_words.add(w_id)
            word_preds.append(pred_label)

    valid_words = words[: len(word_preds)]
    boundaries = [i for i in range(1, len(word_preds)) if word_preds[i] != word_preds[i - 1]]
    return valid_words, word_preds, boundaries


def main():
    parser = argparse.ArgumentParser(description="Сегментация текста на фрагменты Человека и ИИ")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tnc_dbgc_coat_best.pt",
        help="Путь к .pt файлу обученной модели (по умолчанию: tnc_dbgc_coat_best.pt)",
    )
    parser.add_argument("--text", type=str, required=True, help="Входной текст для анализа")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Ошибка: файл модели '{ckpt_path}' еще не создан!")
        print("Сначала запусти обучение командой: python3 -m src.train")
        return

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "microsoft/mdeberta-v3-base")
    hidden_dim = ckpt.get("hidden_dim", 256)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeBERTaBiGRUCRFTagger(model_name=model_name, num_labels=2, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    words, preds, boundaries = segment_text(args.text, model, tokenizer, device)
    ai_ratio = (sum(preds) / len(preds) * 100.0) if preds else 0.0

    print(f"\n==========================================")
    print(f"Доля ИИ-контента: {ai_ratio:.1f}% ({sum(preds)} из {len(preds)} слов)")
    print(f"Индексы слов-границ перехода: {boundaries}")
    print(f"Легенда: \033[42;30m ЧЕЛОВЕК (0) \033[0m  \033[41;97m ИИ / НЕЙРОСЕТЬ (1) \033[0m")
    print(f"==========================================\n")

    colored = []
    for w, lbl in zip(words, preds):
        if lbl == 1:
            colored.append(f"\033[41;97m{w}\033[0m")
        else:
            colored.append(f"\033[42;30m{w}\033[0m")
    print(" ".join(colored) + "\n")


if __name__ == "__main__":
    main()
