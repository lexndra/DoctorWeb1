# DW-1: выявление AI-assisted происхождения писем — образ для запуска CLI и веб-интерфейса.
#
# Сборка (по умолчанию — только CPU-метод на признаках, без torch):
#   docker build -t dw1 .
# С языковой моделью (torch CPU + transformers) и/или анализом изображений:
#   docker build -t dw1-lm --build-arg WITH_LM=1 .
#   docker build -t dw1-full --build-arg WITH_LM=1 --build-arg WITH_IMAGES=1 .
#
# Перед сборкой в models/ должны лежать feature_model.json и calibration.json (python cli.py train).
#
# Анализ одного письма из смонтированного каталога (только чтение) с сохранением JSON:
#   docker run --rm -v "$PWD/samples:/in:ro" -v "$PWD/out:/out" dw1 \
#       analyze --eml /in/ai_phishing_evasion.eml --output /out/result.json
# Пакетный анализ каталога писем (сводка /out/summary.json и отчёты /out/emails/*.json):
#   docker run --rm -v /path/to/emails:/in:ro -v "$PWD/out:/out" dw1 batch --dir /in --out-dir /out
# Метод LM (образ с WITH_LM=1, веса монтируются отдельно, в образ не входят):
#   docker run --rm -v "$PWD/models/tnc_dbgc_llmtrace.pt:/app/models/tnc_dbgc_llmtrace.pt:ro" \
#       -v "$PWD/samples:/in:ro" dw1-lm analyze --method lm --eml /in/legitimate_human.eml
# Веб-интерфейс на http://localhost:5000:
#   docker run --rm -p 5000:5000 --entrypoint uvicorn dw1 app:app --host 0.0.0.0 --port 5000
# Настройки (DW1_METHOD, DW1_THRESHOLD, ...) передаются через -e или --env-file .env (см. .env.example).
#
# Каталог /out должен быть доступен на запись пользователю с UID 10001 (или запускайте с --user "$(id -u)").

FROM python:3.12-slim

ARG WITH_LM=0
ARG WITH_IMAGES=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# Системные библиотеки нужны только OpenCV/EasyOCR (анализ изображений)
RUN if [ "$WITH_IMAGES" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements.txt requirements-lm.txt requirements-images.txt ./
RUN pip install -r requirements.txt \
    && if [ "$WITH_LM" = "1" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lm.txt; \
    fi \
    && if [ "$WITH_IMAGES" = "1" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-images.txt; \
    fi

# Код, веб-интерфейс, обученные модели (JSON; *.pt исключены в .dockerignore) и примеры
COPY src/ ./src/
COPY cli.py evaluate.py app.py README.md ./
COPY static/ ./static/
COPY templates/ ./templates/
COPY models/ ./models/
COPY samples/ ./samples/

# Запуск от непривилегированного пользователя (п. 4.5.2.2 ТЗ)
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /out && chown app:app /out
USER app

EXPOSE 5000

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
