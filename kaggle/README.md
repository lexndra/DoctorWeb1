# Дообучение модели на LLMTrace_detection в Kaggle

Ноутбук `finetune_llmtrace.ipynb` дообучает нашу модель TNC-DBGC* (mDeBERTa-v3 + BiGRU + CRF) на датасете
[`iitolstykh/LLMTrace_detection`](https://huggingface.co/datasets/iitolstykh/LLMTrace_detection) (ru + eng)
и считает метрики по протоколу заказчика: порог подбирается только на validation, test используется один раз в конце.
Ноутбук самодостаточный, репозиторий на Kaggle не нужен.

## 1. (Опционально) Загрузить текущий чекпойнт как Kaggle Dataset

Нужно, если хотите дообучать модель, уже обученную на CoAT (`INIT_FROM = "checkpoint"`, по умолчанию).

1. kaggle.com → **Datasets** → **New Dataset**.
2. Перетащите файл `tnc_dbgc_coat_best.pt` (≈1.1 ГБ), название, например, `tnc-dbgc-coat`, видимость **Private** → **Create**.
3. Путь к файлу в ноутбуке будет `/kaggle/input/tnc-dbgc-coat/tnc_dbgc_coat_best.pt`. Если slug получился другим,
   ничего страшного: ноутбук сам найдёт любой `*.pt` в `/kaggle/input`. Можно и явно указать путь в `CHECKPOINT_PATH`.

Если хотите обучать с нуля от `microsoft/mdeberta-v3-base`, пропустите этот шаг и поставьте `INIT_FROM = "base"`.

## 2. Создать ноутбук

1. kaggle.com → **Code** → **New Notebook**.
2. **File → Import Notebook** → выберите `kaggle/finetune_llmtrace.ipynb` из репозитория.
3. Правая панель **Session options** (или *Settings*):
   - **Accelerator → GPU T4 x2** (используется одна карта). P100 не рекомендуется: свежие образы Kaggle с PyTorch 2.x
     могут не поддерживать её архитектуру (sm_60).
   - **Internet → On** (нужен для загрузки датасета и `mdeberta-v3-base` с Hugging Face). Для этого аккаунт Kaggle
     должен быть подтверждён по телефону.
4. Если делали шаг 1: **Add Input** → вкладка *Your Datasets* → выберите `tnc-dbgc-coat`.
5. При необходимости поменяйте параметры в ячейке **КОНФИГУРАЦИЯ** (см. ниже).

## 3. Запуск

- Лучше всего: **Save Version → Save & Run All (Commit)**. Тогда ноутбук выполняется в фоне (лимит 12 часов),
  браузер можно закрыть. Результаты появятся во вкладке **Output** у сохранённой версии.
- Можно и интерактивно (**Run All**), но тогда вкладку нельзя надолго оставлять без активности, а файлы из
  `/kaggle/working` пропадут после остановки сессии, если их не скачать.

Первая ячейка доустанавливает `pytorch-crf` (и `sentencepiece`/`protobuf`, если их нет) и печатает информацию о GPU.
Ячейка с моделью сама проверяет правильность вычисления маргиналов CRF.

### Ожидаемое время (T4, параметры по умолчанию)

| Этап | Время |
|---|---|
| Установка, загрузка датасета и модели, токенизация | 5–10 мин |
| Обучение, 2 эпохи × ~20 000 документов (+ оценка на val после каждой) | 50–100 мин |
| Калибровка на val (3 000) и оценка на полном test (13 588) + bootstrap | 10–15 мин |
| **Итого** | **≈1–2 ч** (с запасом в пределах 3 ч) |

Если не укладывается по времени: `EPOCHS = 1`, `MAX_TRAIN_SAMPLES = 10000`, `MAX_TEST_SAMPLES = 5000`.
`EVAL_BASELINE = True` добавляет оценку исходной модели до дообучения (+15–25 мин), чтобы получить сравнение «до/после».

## Основные параметры (ячейка КОНФИГУРАЦИЯ)

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `INIT_FROM` | `"checkpoint"` | `"checkpoint"` — с `tnc_dbgc_coat_best.pt`; `"base"` — с `microsoft/mdeberta-v3-base` |
| `CHECKPOINT_PATH` | `/kaggle/input/tnc-dbgc-coat/tnc_dbgc_coat_best.pt` | при отсутствии ищется любой `*.pt` в `/kaggle/input` |
| `MAX_TRAIN_SAMPLES` / `MAX_VAL_SAMPLES` / `MAX_TEST_SAMPLES` | 20000 / 3000 / все | случайная подвыборка (seed) |
| `EPOCHS`, `BATCH_SIZE`, `MAX_LEN` | 2, 8, 512 | |
| `LR_EMB, LR_LOW, LR_HIGH, LR_HEAD` | 1e-6, 5e-6, 1e-5, 1e-4 | послойные learning rate, как в `src/optim.py` |
| `FP16` | `True` | смешанная точность (torch.cuda.amp) |
| `TRAIN_MAX_CHUNKS_PER_DOC` / `EVAL_MAX_CHUNKS_PER_DOC` | 2 / 4 | длинные тексты режутся на окна по 512 токенов |
| `DOC_SCORE` | `"marginal"` | score документа: средняя P(AI) по словам или `"viterbi_frac"` — доля AI-слов |
| `MAX_FPR` | 0.20 | ограничение FPR на human при выборе порога на validation |
| `SEED` | 42 | |

## 4. Что скачать и куда положить

Во вкладке **Output** (у версии ноутбука или в правой панели интерактивной сессии) скачайте:

| Файл на Kaggle | Куда положить в репозитории |
|---|---|
| `tnc_dbgc_llmtrace.pt` (≈1.1 ГБ) | `models/tnc_dbgc_llmtrace.pt` |
| `calibration.json` | `results/lm_kaggle_calibration.json` |
| `metrics.json` | `results/lm_metrics.json` |
| `training_params.json` | `results/lm_training_params.json` (по желанию) |
| `roc_curve_llmtrace.png` | `results/lm_roc_curve.png` (по желанию) |

Папку `models/` при необходимости создайте. Чекпойнт в том же формате, что и у `src/train.py`
(`model_state_dict`, `model_name`, `hidden_dim`, `val_metrics`). Пайплайн находит `models/tnc_dbgc_llmtrace.pt` автоматически
(метод `lm` / `combined`, переменная `DW1_METHOD`).

Порог для писем в пайплайне подбирается заново на val-выборке нашего датасета писем командой
`python evaluate.py --methods features,lm,combined` (файл `models/lm_calibration.json` создаётся автоматически).
Калибровка из Kaggle (`threshold`, score типа `marginal`) относится к LLMTrace и хранится только как результат эксперимента.

## Что внутри metrics.json

- `calibration` — выбранный порог и метрики на validation (F1, FPR и т.д.), флаг `fallback_used`, если ограничение FPR
  выполнить не удалось.
- `test.overall` и `test.by_lang.{ru,eng}`:
  `roc_auc` + `roc_auc_ci95` (bootstrap, 1000 итераций), `roc_auc_mixed_vs_human`, `roc_auc_ai_vs_human`,
  `doc_precision` / `doc_recall` / `doc_f1` / `fpr_human` при откалиброванном пороге,
  `char_localization_ai_mixed` / `char_localization_mixed` / `char_localization_all` (микро-F1 по символам),
  `word_accuracy`, `word_level` (те же пословные метрики, что в `src/metrics.py`).
- `validation` — те же метрики на validation, `training_history` — метрики по эпохам.

Символьная локализация считается по непробельным символам: предсказанные AI-символы — символы слов, помеченных AI,
эталон — символы внутри `ai_char_intervals`. Слова за пределами `EVAL_MAX_CHUNKS_PER_DOC` окон (очень длинные тексты)
в метриках не учитываются; доля покрытых слов печатается при подготовке данных.
