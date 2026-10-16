from pathlib import Path

import cv2
import easyocr
from PIL import Image
from transformers import pipeline


ocr_reader = easyocr.Reader(["ru", "en"], gpu=False)

ai_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")


def detect_qr(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return []

    detector = cv2.QRCodeDetector()

    data, points, _ = detector.detectAndDecode(image)

    result = []

    if data:
        result.append(data)

    return result


def extract_text(image_path):
    result = ocr_reader.readtext(str(image_path), detail=0)
    return " ".join(result)


def detect_ai_image(image_path):
    image = Image.open(image_path).convert("RGB")
    predictions = ai_detector(image)
    return predictions


def analyze_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {image_path}"
        )

    qr_codes = detect_qr(image_path)

    text = extract_text(image_path)

    ai_result = detect_ai_image(image_path)

    return {
        "file": str(image_path),

        "has_qr": len(qr_codes) > 0,
        "qr_count": len(qr_codes),
        "qr_codes": qr_codes,

        "has_text": len(text.strip()) > 0,
        "ocr_text": text,

        "ai_detection": ai_result
    }


