# DoctorWeb1 (DW-1): оценка AI-assisted происхождения электронных писем

Студенческий проект НГУ по кейсу компании «Доктор Веб» (ТЗ DW-1 v5).
Прототип аналитического модуля: на вход письмо EML (или каталог), на выход JSON с итоговым индексом
AI-генерации, вердиктом, вероятностями по зонам письма, выделенными сгенерированными фрагментами
(символьные границы), оценками изображений и объяснением на русском языке для аналитика SOC.

> Результаты носят рекомендательный характер (п. 3.2 ТЗ). Все цифры ниже воспроизводятся командами
> из раздела «Воспроизведение» и сохраняются в `results/`. Ограничения перечислены отдельно: читайте их вместе с метриками.

---

## Содержание
1. [Что умеет система](#что-умеет-система)
2. [Архитектура](#архитектура)
3. [Методы оценки](#методы-оценки)
4. [Датасет DW1](#датасет-dw1)
5. [Метрики качества (п. 4.1.11 ТЗ)](#метрики-качества)
6. [Каталог писем Заказчика (1941 письмо)](#каталог-писем-заказчика)
7. [Языковая модель TNC-DBGC* и дообучение](#языковая-модель)
8. [Валидация на внешних бенчмарках](#валидация-на-внешних-бенчмарках)
9. [Ограничения](#ограничения)
10. [Установка и запуск](#установка-и-запуск)
11. [Воспроизведение (датасет → обучение → метрики)](#воспроизведение)
12. [Формат JSON](#формат-json)
13. [Структура репозитория](#структура-репозитория)
14. [Что исправлено по сравнению с прошлой версией](#что-исправлено)

---

## Что умеет система

| Требование ТЗ | Статус | Где |
|---|---|---|
| 4.1.1 Приём EML и каталога, извлечение темы, отправителя, text/plain, text/html, изображений; пропуск повреждённых писем | ✅ | `src/eml_parser.py`, `cli.py batch` |
| 4.1.1.4 Полный каталог кейса (1941 письмо) без ошибок | ✅ 1941/1941, 0 ошибок | [раздел ниже](#каталог-писем-заказчика) |
| 4.1.2 Предобработка (очистка HTML, невидимые символы, типографика) | ✅ | `src/html_analyzer.py`, `src/text_utils.py` |
| 4.1.2.2 Выборки train/val/test, дедупликация, без утечек | ✅ разбиение по группам, проверка отсутствия пересечений | `src/datagen/build_dataset.py` |
| 4.1.2.3 Генерация датасета: полностью сгенерированные письма, частичная правка с символьной разметкой, вариации, контроль качества, валидация | ✅ локальная LLM через OpenAI-совместимый API | `src/datagen/generate.py` |
| 4.1.2.4 Схема разметки | ✅ | `src/datagen/schema.py`, `data/dataset/v1/schema.json` |
| 4.1.3 Обучение, калибровка порогов только на val, модели и параметры в JSON | ✅ | `src/train_detector.py`, `models/*.json` |
| 4.1.3.4 Метод на основе языковой модели / комбинированный с настраиваемыми весами | ✅ `DW1_METHOD=lm/combined`, `DW1_WEIGHT_*` | `src/detectors/lm_model.py` |
| 4.1.4 Индекс 0..1, вердикт по порогу из настроек, вероятности зон с границами, JSON по письму и сводный по каталогу | ✅ | `src/pipeline.py` |
| 4.1.4.6 Изображения: вероятность генерации, признаки, учёт в индексе | ✅ (включается `DW1_ENABLE_IMAGES=1`) | `src/image_analyzer.py` |
| 4.1.5 Объяснение: зоны, ключевые сигналы, оценки предложений, текст на русском | ✅ | `EmailDetectionReport.explanation` |
| 4.1.6 Единый JSON, обратная совместимость полей | ✅ | [формат](#формат-json) |
| 4.1.10 Производительность метода на признаках | ✅ см. метрики | `results/metrics/performance.json` |
| 4.1.11 Метрики качества | см. [таблицу](#метрики-качества) | `results/metrics/report.md` |
| 4.5 Python ≥ 3.11, Linux/Windows/macOS, Docker (непривилегированный пользователь), `.env` | ✅ | `Dockerfile`, `.env.example` |
| 4.5.2 Письма Заказчика не покидают контур | ✅ генерация на локальной LLM; письма кейса в репозиторий не входят | `data/private/` в `.gitignore` |

Графический интерфейс по ТЗ не требуется; дополнительно есть веб-страница для демонстрации (`app.py`).

---

## Архитектура

Конвейер по п. 4.1.12 ТЗ: **разбор → признаки → оценка → объяснение → сериализация**, плюс отдельный контур
подготовки данных, обучения и оценки качества.

```
L0  Источники      EML/каталог/архив кейса · открытые корпуса (Enron, LLMTrace) · локальная LLM (Ollama, OpenAI API)
L1  Разбор         src/eml_parser.py (MIME, HTML, изображения) · src/html_analyzer.py (видимый/скрытый текст, ссылки)
                   зоны: тема / приветствие / основной текст / подпись · src/datagen/* (генерация и разбиение датасета)
L2  Признаки,      src/detectors/feature_model.py — TF-IDF-модель предложений + стилометрическая модель документа
    модели         src/detectors/lm_model.py      — TNC-DBGC* (mDeBERTa-v3 + BiGRU + CRF), комбинированный метод
                   src/signals.py                 — маскировка (скрытый текст, zero-width, подмена ссылок), утечка промпта
                   src/image_analyzer.py          — детектор сгенерированных изображений, EXIF/C2PA, OCR, QR
L3  Результат      src/pipeline.py — индекс, вердикт, зоны с символьными границами, объяснение, JSON
L4  Интерфейсы     cli.py (prepare-data, generate, build-dataset, train, evaluate, analyze, batch) · Python API · Docker · app.py
L5  Оценка         src/evaluation/ — метрики табл. 3, бутстреп, сценарии маскировки, отчёты JSON/MD/PNG, журналирование
```

---

## Методы оценки

Все методы имеют единый интерфейс `score(text) → DetectorOutput(prob, sentences, signals, contributions)`
и дают одинаковый JSON.

**`features` (по умолчанию, CPU).** Интерпретируемая базовая модель (п. 4.1.3.5 ТЗ):
1. *Модель предложений*: логистическая регрессия на TF-IDF словесных (1–2) и символьных (2–4) n-грамм
   нормализованного текста. Оценка каждого предложения даёт локализацию фрагментов с символьными границами.
2. *Документная модель*: логистическая регрессия на 24 именованных признаках. Это агрегаты оценок предложений
   и стилометрия: вариативность длины предложений, лексическое разнообразие, пунктуация, формульные приветствия
   и концовки, шаблонные обороты генераторов, следы небрежного набора. Вклад признака равен произведению коэффициента
   на стандартизованное значение и выводится в объяснение (`feature_contributions`).
   На train оценки предложений считаются out-of-fold (GroupKFold), чтобы документная модель не переобучалась.

Перед извлечением признаков текст нормализуется: NFKC, удаление невидимых символов, унификация кавычек и тире,
замена URL, адресов и цифр на маркеры. Так модель не учится отличать «умные» кавычки генератора от ASCII-кавычек
старых писем.

**`lm`.** Пословный тэггер TNC-DBGC* (`src/model.py`). Используется `models/tnc_dbgc_llmtrace.pt` (после дообучения,
см. [ниже](#языковая-модель)) или исходный чекпоинт CoAT. Порог подбирается на val нашего датасета.

**`combined`.** `w_f · p_features + w_lm · p_lm`, веса задаются в `.env` (`DW1_WEIGHT_FEATURES`, `DW1_WEIGHT_LM`).

**Дополнительные сигналы письма:**
- *Утечка промпта.* В письме остались инструкции для LLM («=== Prompt brief ===», «Output only the HTML»,
  «Hard constraint: …»). Это прямое свидетельство генерации, поэтому индекс поднимается минимум до `DW1_PROMPT_LEAK_FLOOR` (0.9).
- *Маскировка* (скрытые элементы, символы нулевой ширины, подмена ссылок, формы с паролем) выводится в объяснение
  и JSON, но **в индекс не входит**: адаптивные легитимные рассылки тоже прячут блоки CSS-стилями, и это признак
  спама, а не генерации ИИ.
- *Изображения* (если включены): `index = (1 − w) · p_text + w · max(p_image)`, `w = DW1_WEIGHT_IMAGES`.

---

## Датасет DW1

Собирается командами `prepare-data → generate → build-dataset`. Версия `v1`: файлы `data/dataset/v1/`, архив
`data/dataset/dw1_dataset_v1.zip`, контрольные суммы и статистика в `manifest.json`.

**Источники:**

| Источник | Что это | Метка |
|---|---|---|
| `enron_aeslc` | деловая переписка Enron, 2000–2002 гг. ([Yale-LILY/aeslc](https://huggingface.co/datasets/Yale-LILY/aeslc)) | человек |
| `enron_newsletters` | внешние рассылки, полученные сотрудниками Enron в 2000–2002 гг. ([corbt/enron-emails](https://huggingface.co/datasets/corbt/enron-emails)); футер отписки удалён | человек (легитимный маркетинг) |
| `dw1_generated` | письма, сгенерированные локальной `qwen2.5:7b` на **те же темы**, что и письма Enron: полностью сгенерированные, частичные правки (rewrite / insert / append) с символьной разметкой, вариации | ИИ / смешанные |
| `llmtrace_detection` | RU/EN тексты человека, ИИ (≈37 современных моделей: GPT-4o, o3, GigaChat, YandexGPT, Qwen, Llama-3.3 и др.) и смешанные с символьной разметкой ([iitolstykh/LLMTrace_detection](https://huggingface.co/datasets/iitolstykh/LLMTrace_detection), Apache-2.0). Сохранено авторское разбиение | человек / ИИ / смешанные |
| `customer_case` | письма из папки `samples_prompt` архива кейса (по пояснению Заказчика сгенерированы LLM по промпту), не более 40 на кампанию. **В репозиторий не входят** (`data/private/`) | ИИ |

**Схема записи** (п. 4.1.2.4): `label` (0/1), `label_name` (human/ai/mixed), `generation_method`, `generator_model`,
`language`, `email_type` (corporate/marketing/text), `source`, `ai_zones`, `ai_char_intervals`, `group_id`, `split`.

**Разбиение без утечек:** исходное письмо и все его производные (генерации на его тему, правки, вариации) имеют общий
`group_id` и попадают в одну выборку; письма одной кампании кейса тоже. Дубликаты удаляются точно и по косинусной
близости TF-IDF ≥ 0.9. Пересечение групп между выборками проверяется при сборке.

DATASET_STATS_PLACEHOLDER

---

## Метрики качества

`python evaluate.py --methods features` → `results/metrics/report.md`, `metrics_features.json`, `roc_features.png`.
Порог подобран на val (максимум F1 при FPR ≤ 0.20 на всех человеческих письмах и отдельно на рассылках),
тест для настройки не использовался.

METRICS_PLACEHOLDER

---

## Каталог писем Заказчика

`python cli.py batch --dir data/case/Кейс_1/emails --out-dir results/batch` (сами письма в репозиторий не входят).

CASE_PLACEHOLDER

---

## Языковая модель

Модель TNC-DBGC* (mDeBERTa-v3-base + Dynamic Dropout + BiGRU + CRF, `src/model.py`) ранее обучалась на корпусе
CoAT (перефразы и переводы генераторами прошлого поколения: ruGPT-3, mT5 и др.).

LM_PLACEHOLDER

**Дообучение на современных генераторах.** Ноутбук `kaggle/finetune_llmtrace.ipynb` дообучает модель на LLMTrace_detection
(RU/EN, посимвольная разметка, ≈37 моделей 2023–2025 гг.) на бесплатном GPU Kaggle. Порог подбирается только на validation.
Инструкция: [`kaggle/README.md`](kaggle/README.md). После дообучения веса кладутся в `models/tnc_dbgc_llmtrace.pt`,
и `python evaluate.py --methods features,lm,combined` пересчитывает метрики на нашем тесте.

---

## Валидация на внешних бенчмарках

`python run_academic_validation.py`. Подробный отчёт: [`results/academic_validation/ACADEMIC_VALIDATION_REPORT.md`](results/academic_validation/ACADEMIC_VALIDATION_REPORT.md).

ACADEMIC_PLACEHOLDER

---

## Ограничения

LIMITATIONS_PLACEHOLDER

---

## Установка и запуск

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # метод на признаках, CLI, датасет, веб
pip install -r requirements-lm.txt         # опционально: языковая модель (torch, transformers)
pip install -r requirements-images.txt     # опционально: анализ изображений (OCR, детектор)
cp .env.example .env                       # настройки: метод, порог, веса, LLM
```

Обученная модель (`models/feature_model.json`, `models/calibration.json`) уже лежит в репозитории.

```bash
# одно письмо: индекс, вердикт, объяснение; полный JSON по запросу
python cli.py analyze --eml samples/ai_phishing_evasion.eml --output report.json
python cli.py analyze --text "Коллеги, напоминаю про отчёт до пятницы." --subject "Отчёт"
python cli.py analyze --html email.html --method combined

# каталог: JSON по каждому письму + сводный JSON; --resume для повторного запуска
python cli.py batch --dir path/to/emails --out-dir results/batch

# веб-интерфейс http://127.0.0.1:5000
uvicorn app:app --port 5000
```

Python API:
```python
from src.pipeline import EmailAnalysisPipeline
report = EmailAnalysisPipeline().analyze_email("letter.eml")
print(report.overall_ai_index, report.verdict)
print(report.to_json())
```

Docker (отдельный образ анализа, запуск от непривилегированного пользователя):
```bash
docker build -t dw1 .
docker run --rm -v "$PWD/samples:/in:ro" -v "$PWD/out:/out" dw1 analyze --eml /in/ai_phishing_evasion.eml --output /out/r.json
docker run --rm -v /path/to/emails:/in:ro -v "$PWD/out:/out" dw1 batch --dir /in --out-dir /out
```
Варианты с языковой моделью и изображениями описаны в шапке `Dockerfile`.

---

## Воспроизведение

```bash
python cli.py prepare-data                 # Enron, LLMTrace (+ письма кейса из data/case, если есть) → data/raw
ollama serve & ollama pull qwen2.5:7b      # локальная LLM (данные не покидают контур)
python cli.py generate                     # генерация → data/generated (повторный запуск продолжает с места остановки)
python cli.py build-dataset --version v1   # валидация, дедупликация, разбиение, архив
python cli.py train                        # обучение + калибровка на val → models/ (≈1–2 мин на CPU)
python evaluate.py --methods features      # все метрики ТЗ на test → results/metrics/
python -m pytest -q tests                  # smoke-тесты
```
Фиксируются: версия и SHA-256 файлов датасета (`manifest.json`), seed, параметры обучения
(`models/training_params.json`), калибровка (`models/calibration.json`), отчёты о метриках.

---

## Формат JSON

Поля прошлой версии сохранены (п. 4.1.6.3), новые добавлены.

```json
{
  "file_id": "letter",
  "metadata": {"subject": "...", "sender": "...", "date": "..."},
  "overall_ai_index": 0.83,
  "verdict": "Сгенерировано ИИ (AI-Generated)",
  "threshold": 0.49,
  "ai_zones_detected": ["body"],
  "zones": [
    {"name": "body", "text": "...", "start_char": 42, "end_char": 610, "words_count": 95, "ai_words_count": 61,
     "ai_probability": 0.71, "is_ai": true,
     "ai_spans": [{"text": "...", "start_char": 120, "end_char": 260, "probability": 0.93}]}
  ],
  "images": [{"filename": "banner.png", "ai_probability": 0.12, "is_ai": false, "reasons": [], "has_qr": false}],
  "html_signals": {"hidden_elements_count": 2, "suspicious_links_count": 1, "hidden_texts": ["..."]},
  "evasion_signals": {"zero_width_chars_count": 3, "hidden_text_chars": 83},
  "explanation": "Письмо отнесено к категории «Сгенерировано ИИ»: итоговый индекс 0.83 ...",
  "reasons": ["Зона «основной текст»: вероятность генерации 71% ...", "..."],
  "method": "features",
  "text_ai_probability": 0.83,
  "key_signals": {"cv_sent_len": 0.21, "cliche_per_100w": 3.1, "...": 0},
  "feature_contributions": {"sent_p_mean": 1.2, "...": 0},
  "sentence_scores": [{"start_char": 42, "end_char": 118, "probability": 0.35, "text": "..."}],
  "prompt_leakage": [],
  "explanation_threshold": 0.49
}
```

---

## Структура репозитория

```
cli.py                    команды CLI (п. 4.8.1)
evaluate.py               метрики ТЗ одной командой
app.py, static/, templates/  демонстрационный веб-интерфейс
src/pipeline.py           конвейер анализа письма
src/eml_parser.py, src/html_analyzer.py, src/image_analyzer.py   разбор письма, HTML, изображений
src/detectors/            методы оценки (features, lm, combined)
src/datagen/              источники, генерация (LLM), схема, сборка датасета
src/evaluation/           метрики табл. 3, калибровка, маскировка, отчёты
src/model.py, src/train.py   архитектура TNC-DBGC* и её обучение на CoAT
kaggle/                   ноутбук дообучения LM на GPU
models/                   обученная модель (JSON), калибровка, параметры обучения
data/dataset/v1/          публичная часть датасета (+ ZIP)
results/                  метрики, отчёты, ROC, пакетный прогон, внешние бенчмарки
tests/                    smoke-тесты
```

---

## Что исправлено

Преподаватель справедливо указал, что прежнее описание не соответствовало действительности. Что было не так и что сделано:

| Было в прошлой версии | Что было не так | Сейчас |
|---|---|---|
| «Все критерии ТЗ PASS: ROC-AUC 0.86, F1 0.80, FPR 0.20» | Считалось на 20 письмах, написанных вручную прямо в `evaluate.py`. В каждое «ИИ-письмо» были вставлены скрытый текст или подменённые ссылки, а индекс складывался из штрафов за эти признаки. Проверялся детектор HTML-маскировки, а не генерации. Порог 0.40 подогнан | Датасет с разбиением train/val/test, порог только на val, все метрики на test одной командой |
| «F1@3 локализации = 0.627» | Число было вписано в код константой | Посимвольная F1 по ТЗ, считается на test |
| «1941 письмо, 100% samples_prompt выявлено» | В сохранённом логе нет разбивки по папкам. Почти все письма были признаны ИИ из-за штрафа за скрытые элементы (они есть в 1915 из 1941 писем), текстовая модель сама по себе отметила около 21% | Пакетный прогон с разбивкой по папкам, маскировка в индекс не входит |
| «LLMTrace: [VALIDATED]» при ROC-AUC 0.349 | Хуже случайного угадывания. Порог подбирался на тех же данных, F1 0.62 получался при FPR 98% | Честная внешняя валидация с порогом, выбранным заранее |
| «Defactify — фотожабы и мемы без диффузионных артефактов» | Неверно: в Defactify реальные фото COCO против изображений SD 2.1/SDXL/SD3/DALL-E 3/Midjourney v6 | Описание исправлено, метрики пересчитаны |
| «SeqXGPT: модель только для русского из-за razdel и BiGRU» | Неверно: mDeBERTa многоязычная, razdel только делит текст на слова; к тому же ТЗ требует RU+EN | Обе языковые части оцениваются |
| «Метрики CoAT_Hybrid на тестовом сете 2560 текстов» | Скрипт по умолчанию учил на 1200 и тестировал на 250 текстах; какой прогон дал цифры, неизвестно | Перенесено в `results/legacy/` как историческое |
| Модель CoAT как основной детектор | На современных генераторах ROC-AUC ≈ 0.5 (см. раздел о языковой модели) | Основной метод — обучаемая модель на признаках; LM дообучается на LLMTrace |
| Не было: датасета писем, генератора, калибровки, метода на признаках, CLI-команд подготовки/обучения/метрик | Требования ТЗ 4.1.2, 4.1.3, 4.8.1, 7.10 (MVP) | Реализовано |
| Docker от root, без веб-части | Требование 4.5.2.2 | Непривилегированный пользователь, отдельный образ анализа |
