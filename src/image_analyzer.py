
from pathlib import Path
from io import BytesIO
from email import policy
from email.parser import BytesParser
from functools import lru_cache
import hashlib
import json
import shutil
import subprocess
import tempfile
import copy

import cv2
import easyocr
import numpy as np
from PIL import Image, ExifTags
from transformers import pipeline


@lru_cache(maxsize=1)     #создаем и сохраняем OCR для распоз текста
def get_ocr_reader():
    return easyocr.Reader(["ru", "en"], gpu=False)


@lru_cache(maxsize=1)
def get_ai_detector():
    return pipeline(   #классификатор
        "image-classification",
        model="umm-maybe/AI-image-detector",
        device=-1
    )

#кэш результатов одинаковых изображений
image_cache = {}


#открытие картинки из файла или байтов
def open_image(image_source):
    if isinstance(image_source, (str, Path)):
        raw = Path(image_source).read_bytes()
    else:
        raw = bytes(image_source)

    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("Изображение больше 15 МБ")

    image = Image.open(BytesIO(raw))
    image.load()

    if image.width * image.height > 25_000_000:
        raise ValueError("Слишком большое разрешение")

    return image, raw


# QR-коды поиск
def detect_qr(image):
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detector = cv2.QRCodeDetector()
    result = []

    success, data, points, _ = detector.detectAndDecodeMulti(image)

    if success:
        for code in data:
            if code:
                result.append(code)

    if not result:
        data, points, _ = detector.detectAndDecode(image)

        if data:
            result.append(data)

    return result


#распознавание текста EasyOCR
def extract_text(image):
    image = np.array(image.convert("RGB"))

    result = get_ocr_reader().readtext(
        image,
        detail=0
    )

    return " ".join(result)

#проверка картинки нейросетью
def detect_ai_image(image):
    image = image.convert("RGB")

    predictions = get_ai_detector()(image)

    return predictions


#проверка метаданных EXIF
def check_exif(image):
    exif = image.getexif() #метданные
    result = {}

    for tag, value in exif.items():
        name = ExifTags.TAGS.get(tag, str(tag))

        if name in ("Make", "Model", "Software", "DateTime"):
            result[name] = str(value)

    return {
        "has_exif": bool(exif),
        "metadata": result
    }


#проверка признаков JPEG-сжатия
def check_compression(image):
    if image.format != "JPEG":
        return {
            "applicable": False,
            "blockiness_ratio": None
        }

    gray = np.array(image.convert("L"), dtype=np.float32)

    if min(gray.shape) < 16:
        return {
            "applicable": False,
            "blockiness_ratio": None
        }

    #разность яркости соседних пикселей
    diff = np.abs(np.diff(gray, axis=1))
    #границы блоков JPEG размером 8 пикселей
    boundary = diff[:, 7::8]

    mask = np.ones(diff.shape[1], dtype=bool)
    mask[7::8] = False
    inside = diff[:, mask]

    if boundary.size == 0 or inside.size == 0:
        return {
            "applicable": False,
            "blockiness_ratio": None
        }

    ratio = float(
        boundary.mean() / (inside.mean() + 1e-6)
    )

    return {
        "applicable": True,
        "blockiness_ratio": round(ratio, 4)
    }


#проверка C2PA через внешнюю утилиту
def check_c2pa(raw, image_format):
    if shutil.which("c2patool") is None:
        return {
            "status": "not_checked",
            "reason": "c2patool не установлен"
        }

    suffixes = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "TIFF": ".tiff"
    }

    suffix = suffixes.get(image_format)

    if suffix is None:
        return {
            "status": "unsupported_format"
        }

    with tempfile.TemporaryDirectory() as folder:
        temp_path = Path(folder) / ("image" + suffix)
        temp_path.write_bytes(raw)

        try:
            process = subprocess.run(
                ["c2patool", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=15
            )

            if process.returncode != 0:
                return {
                    "status": "error",
                    "message": process.stderr[:500]
                }

            result = json.loads(process.stdout)

            if not isinstance(result, dict):
                return {
                    "status": "error",
                    "message": "Некорректный формат отчёта"
                }

            return {
                "status": "checked",
                "active_manifest": result.get("active_manifest"),
                "validation_status": result.get("validation_status"),
                "manifest_count": len(result.get("manifests") or {})
            }

        except (subprocess.TimeoutExpired, ValueError) as error:
            return {
                "status": "error",
                "message": str(error)
            }


#основная функция проверками
def analyze_image(image_source):
    image, raw = open_image(image_source)

    #создаём хеш изображения
    image_hash = hashlib.sha256(raw).hexdigest()

    #проверка кэша
    if image_hash in image_cache:
        cached = copy.deepcopy(image_cache[image_hash])
        cached["cached"] = True
        return cached

    qr_codes = detect_qr(image)
    text = extract_text(image)
    ai_result = detect_ai_image(image)
    ai_score = get_ai_score(ai_result)

    exif = check_exif(image)
    compression = check_compression(image)
    c2pa = check_c2pa(raw, image.format)

    result = {
        "sha256": image_hash,
        "width": image.width,
        "height": image.height,
        "format": image.format,

        "has_qr": len(qr_codes) > 0,
        "qr_count": len(qr_codes),
        "qr_codes": qr_codes,

        "has_text": len(text.strip()) > 0,
        "ocr_text": text,

        "ai_detection": ai_result,
        "ai_score": ai_score,

        "exif": exif,
        "compression": compression,
        "c2pa": c2pa,
        "cached": False
    }

    #запоминаем результат
    image_cache[image_hash] = copy.deepcopy(result)

    return result

#оценка AI от 0 до 1
def get_ai_score(predictions):
    for prediction in predictions:
        if prediction["label"].lower() == "artificial":
            return float(prediction["score"])

    return None

#достаём изображения из электронного письма
def extract_images_from_eml(eml_path):
    eml_path = Path(eml_path)

    if not eml_path.exists():
        raise FileNotFoundError(f"Письмо не найдено: {eml_path}")

    with eml_path.open("rb") as file:
        message = BytesParser(
            policy=policy.default
        ).parse(file)

    images = []

    #перебираем MIME-части письма
    for part in message.walk():
        if part.get_content_maintype() != "image":
            continue

        raw = part.get_payload(decode=True)

        if not raw:
            continue

        images.append({
            "filename": part.get_filename(),
            "content_id": part.get("Content-ID"),
            "content_type": part.get_content_type(),
            "disposition": part.get_content_disposition(),
            "data": raw
        })

    return images


#анализируем все изображения из письма
def analyze_eml(eml_path):
    images = extract_images_from_eml(eml_path)

    results = []

    for number, item in enumerate(images, start=1):
        try:
            result = analyze_image(item["data"])

            result["filename"] = item["filename"]
            result["content_id"] = item["content_id"]
            result["content_type"] = item["content_type"]
            result["disposition"] = item["disposition"]

        except Exception as error:
            result = {
                "filename": item["filename"],
                "error": str(error)
            }

        result["image_number"] = number
        results.append(result)

    return {
        "eml_file": str(eml_path),
        "image_count": len(images),
        "images": results
    }


#сохранение результатов в JSON
def save_json(result, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.file)

    if path.suffix.lower() == ".eml":
        result = analyze_eml(path)
    else:
        result = analyze_image(path)

    print(json.dumps(result, ensure_ascii=False, indent=4))

    if args.output:
        save_json(result, args.output)
