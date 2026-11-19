"""Веб-интерфейс для EmailAnalysisPipeline.

Запуск: python app.py  (или uvicorn app:app --host 0.0.0.0 --port 5000)
Сайт: http://127.0.0.1:5000
Настройки метода и порога — через .env / переменные окружения (см. .env.example).
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".eml"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LOGGER = logging.getLogger(__name__)

# Модель должна загружаться один раз, а не при каждом запросе.
_pipeline = None
_pipeline_init_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_init_lock:
            if _pipeline is None:
                # Импорт внутри функции: главная страница доступна даже без ML-библиотек.
                from src.pipeline import EmailAnalysisPipeline

                # Метод, порог, каталог моделей и анализ изображений задаются через .env / окружение
                # (DW1_METHOD, DW1_THRESHOLD, DW1_MODELS_DIR, DW1_ENABLE_IMAGES; см. .env.example).
                _pipeline = EmailAnalysisPipeline()
    return _pipeline


def _model_status() -> dict:
    """Какие файлы моделей нужны выбранному методу и найдены ли они (без загрузки модели)."""
    from src.config import Settings

    settings = Settings()
    files = {}
    if settings.method in ("features", "combined"):
        for name in ("feature_model.json", "calibration.json"):
            files[name] = (settings.models_dir / name).is_file()
    if settings.method in ("lm", "combined"):
        from src.detectors.lm_model import find_lm_checkpoint

        ckpt = find_lm_checkpoint(settings.lm_checkpoint)
        files["lm_checkpoint"] = ckpt is not None
    try:
        threshold = round(settings.resolved_threshold(), 4)
    except Exception:
        threshold = None
    return {
        "method": settings.method,
        "threshold": threshold,
        "models_dir": settings.models_dir.name,
        "model_files": files,
        "model_file_found": all(files.values()) if files else False,
        "images_enabled": settings.enable_images,
    }


def _safe_filename(name: str) -> str:
    """Не позволяем имени загруженного файла содержать каталоги."""
    return name.replace("\\", "/").split("/")[-1]


def _decode_text(raw: bytes) -> str:
    # UTF-8 (в том числе BOM) — основной формат. Старые письма иногда в cp1251.
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def create_app(config: dict | None = None) -> FastAPI:
    app = FastAPI(title="DoctorWeb1", docs_url=None, redoc_url=None)
    app.state.config = config or {}
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(BASE_DIR / "templates/index.html", media_type="text/html")

    @app.get("/api/health")
    def health():
        status = _model_status()
        status["pipeline_loaded"] = _pipeline is not None
        return status

    @app.post("/api/analyze")
    def analyze(file: UploadFile | None = File(default=None)):
        if file is None or not file.filename:
            return error_response("Выберите файл для анализа.", 400)

        filename = _safe_filename(file.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return error_response("Поддерживаются файлы .eml, .html, .htm, .txt и .md.", 415)

        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return error_response("Файл слишком большой. Максимальный размер — 20 МБ.", 413)
        if not raw:
            return error_response("Загруженный файл пуст.", 422)

        try:
            # В тестах можно передать лёгкую подмену реальной модели.
            provider = app.state.config.get("PIPELINE_PROVIDER", get_pipeline)
            pipeline = provider()
        except FileNotFoundError as exc:
            LOGGER.warning("Файлы модели не найдены: %s", exc)
            return error_response(
                "Обученная модель не найдена (models/feature_model.json и models/calibration.json "
                "для метода features или веса .pt для метода lm). Выполните: python cli.py train "
                "или укажите каталог моделей в переменной DW1_MODELS_DIR.",
                503,
            )
        except (ImportError, ModuleNotFoundError):
            LOGGER.exception("Не удалось импортировать зависимости модели")
            return error_response(
                "Не установлены зависимости выбранного метода. Выполните pip install -r requirements.txt "
                "(для DW1_METHOD=lm|combined также requirements-lm.txt).",
                503,
            )
        except Exception:
            LOGGER.exception("Не удалось загрузить модель")
            return error_response("Не удалось запустить нейросеть. Подробности смотрите в консоли сервера.", 503)

        try:
            started = time.perf_counter()
            # Не допускаем параллельных инференсов одного экземпляра модели.
            with _inference_lock:
                if suffix == ".eml":
                    # EML-парсер принимает путь; файл удаляется сразу после анализа.
                    with tempfile.TemporaryDirectory(prefix="doctorweb-") as temp_dir:
                        eml_path = Path(temp_dir) / "mail.eml"
                        eml_path.write_bytes(raw)
                        report = pipeline.analyze_email(eml_path)
                elif suffix in {".html", ".htm"}:
                    report = pipeline.analyze_direct_text(text="", html_content=_decode_text(raw))
                else:
                    report = pipeline.analyze_direct_text(text=_decode_text(raw))

            result = report.to_dict()
            # Отчёт не должен выдавать временный путь на сервере.
            result["file_id"] = filename
            result["file_path"] = filename
            result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            return result
        except (UnicodeError, ValueError) as exc:
            return error_response(f"Не удалось прочитать документ: {str(exc)[:200]}", 422)
        except Exception:
            LOGGER.exception("Ошибка анализа %s", filename)
            return error_response("Ошибка анализа файла. Подробности смотрите в консоли сервера.", 500)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000)
