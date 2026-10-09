"""
Главный скрипт обучения и валидации модели TNC-DBGC* (Transformer + BiGRU + CRF)
на готовом склеенном датасете из папки /Users/roman/Desktop/doctor_web/CoAT_Hybrid.

Особенности для удобной работы на MacBook (MPS):
- Автоматически сохраняет чекпойнт каждые 50 шагов и при нажатии Ctrl+C,
  чтобы файл `tnc_dbgc_coat_best.pt` всегда существовал на диске.
- Позволяет задать `--max-train-samples` (по умолчанию 1200 для быстрого обучения на ноутбуке,
  либо 12049 для полного датасета).
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.dataset import HybridCoATDataset, custom_collate_fn
from src.model import DeBERTaBiGRUCRFTagger
from src.optim import build_llrd_optimizer
from src.metrics import evaluate_sequences


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: DeBERTaBiGRUCRFTagger, model_name: str, hidden_dim: int, metrics: dict, save_path: str):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "val_metrics": metrics,
        },
        save_path,
    )


def train_one_epoch(
    model: DeBERTaBiGRUCRFTagger,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    model_name: str,
    hidden_dim: int,
    save_path: str,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(enumerate(loader), total=num_batches, desc=f"Epoch {epoch_idx + 1}/{total_epochs} [Train]")
    for step, batch in pbar:
        global_progress = (epoch_idx + (step / max(num_batches, 1))) / max(total_epochs, 1)
        model.update_dropout(global_progress)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"crf_loss": f"{loss.item():.2f}", "drop_p": f"{model.dyn_dropout1.current_p:.2f}"})

        # Промежуточное автосохранение каждые 50 батчей, чтобы чекпойнт всегда был на диске
        if (step + 1) % 50 == 0:
            save_checkpoint(model, model_name, hidden_dim, {}, save_path)

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: DeBERTaBiGRUCRFTagger,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    true_seqs: List[List[int]] = []
    pred_seqs: List[List[int]] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        predictions: List[List[int]] = model(input_ids=input_ids, attention_mask=attention_mask)

        for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
            true_len = int(mask_seq.sum().item())
            valid_labels = label_seq[:true_len].cpu().tolist()
            valid_preds = pred_seq[:true_len]

            filtered_true = []
            filtered_pred = []
            for t_lbl, p_lbl in zip(valid_labels, valid_preds):
                if t_lbl != -100:
                    filtered_true.append(t_lbl)
                    filtered_pred.append(p_lbl)

            if filtered_true:
                true_seqs.append(filtered_true)
                pred_seqs.append(filtered_pred)

    return evaluate_sequences(true_seqs, pred_seqs)


def main():
    parser = argparse.ArgumentParser(description="Обучение модели TNC-DBGC* на готовом датасете CoAT_Hybrid")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/roman/Desktop/doctor_web/CoAT_Hybrid",
        help="Путь к папке с готовыми файлами train.csv, val.csv, test.csv",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/mdeberta-v3-base",
        help="Предобученный трансформер (microsoft/mdeberta-v3-base или ai-forever/ruRoberta-large)",
    )
    parser.add_argument("--max-train-samples", type=int, default=1200, help="Сколько примеров брать из train.csv (12049 для всех)")
    parser.add_argument("--max-val-samples", type=int, default=250, help="Сколько примеров брать из val.csv (2527 для всех)")
    parser.add_argument("--max-test-samples", type=int, default=250, help="Сколько примеров брать из test.csv (2560 для всех)")
    parser.add_argument("--max-len", type=int, default=512, help="Максимальная длина последовательности токенов")
    parser.add_argument("--batch-size", type=int, default=8, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=3, help="Число эпох обучения (в статье: 3)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Размерность скрытого состояния BiGRU")
    parser.add_argument("--save-path", type=str, default="tnc_dbgc_coat_best.pt", help="Путь сохранения лучшей модели")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1/3] Загрузка токенизатора ({args.model_name}) и готовых CSV из {data_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = partial(custom_collate_fn, pad_token_id=tokenizer.pad_token_id or 0)

    full_train = HybridCoATDataset(data_dir / "train.csv", tokenizer, max_length=args.max_len)
    full_val = HybridCoATDataset(data_dir / "val.csv", tokenizer, max_length=args.max_len)
    full_test = HybridCoATDataset(data_dir / "test.csv", tokenizer, max_length=args.max_len)

    train_dataset = Subset(full_train, range(min(args.max_train_samples, len(full_train))))
    val_dataset = Subset(full_val, range(min(args.max_val_samples, len(full_val))))
    test_dataset = Subset(full_test, range(min(args.max_test_samples, len(full_test))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = get_device()
    print(f"[2/3] Инициализация модели на устройстве: {device}")
    print(f"      Примеров: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    model = DeBERTaBiGRUCRFTagger(
        model_name=args.model_name,
        num_labels=2,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = build_llrd_optimizer(model, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_kappa = -float("inf")
    print("[3/3] Старт обучения (можно остановить в любой момент через Ctrl+C — модель сохранится)...")
    try:
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch_idx=epoch,
                total_epochs=args.epochs,
                model_name=args.model_name,
                hidden_dim=args.hidden_dim,
                save_path=args.save_path,
                grad_clip=1.0,
            )
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"\nЭпоха {epoch + 1}/{args.epochs} | Train CRF Loss: {train_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                f"Val MCC: {val_metrics['mcc']:.4f} | Val Kappa: {val_metrics['kappa']:.4f} | "
                f"Boundary MAE: {val_metrics['boundary_mae']:.2f} | F1@3: {val_metrics['f1_at_3']:.4f}"
            )

            if val_metrics["kappa"] >= best_kappa:
                best_kappa = val_metrics["kappa"]
                save_checkpoint(model, args.model_name, args.hidden_dim, val_metrics, args.save_path)
                print(f"  --> Сохранен лучший чекпойнт ({args.save_path}) с Kappa = {best_kappa:.4f}")
    except KeyboardInterrupt:
        print(f"\n[!] Обучение остановлено пользователем. Сохраняем текущую модель в {args.save_path}...")
        save_checkpoint(model, args.model_name, args.hidden_dim, {}, args.save_path)

    print("\nОценка сохраненной модели на тестовой выборке...")
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    print("=== ИТОГОВЫЕ МЕТРИКИ НА ТЕСТЕ ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
