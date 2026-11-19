from email import policy
from email.parser import BytesParser
from pathlib import Path
from src.html_analyzer import analyze_html, print_report
import re


def split_text_zones(subject, visible_text):
    subject = subject.strip()
    visible_text = visible_text.strip()

    if subject and visible_text:
        full_text = subject + "\n" + visible_text
        offset = len(subject) + 1
    else:
        full_text = subject or visible_text
        offset = len(subject) if subject else 0

    zones = {
        "subject": [0, len(subject)] if subject else None,
        "greeting": None,
        "body": None,
        "signoff": None
    }

    lines = [
        m for m in re.finditer(r"[^\r\n]+", visible_text)
        if m.group().strip()
    ]

    greetings = (
        "привет", "здравствуйте", "добрый день",
        "доброе утро", "добрый вечер",
        "hello", "hi!", "hi,", "dear "
    )

    signoffs = (
        "с уважением", "всего доброго",
        "best regards", "kind regards",
        "regards", "sincerely"
    )

    def make_zone(start, end):
        while start < end and full_text[start].isspace():
            start += 1
        while end > start and full_text[end - 1].isspace():
            end -= 1
        return [start, end] if start < end else None

    body_start = 0
    body_end = len(visible_text)

    if lines:
        first = lines[0]
        if first.group().strip().lower().startswith(greetings):
            zones["greeting"] = make_zone(
                offset + first.start(),
                offset + first.end()
            )
            body_start = first.end()

        last = lines[-1]

        if last.group().strip().lower().startswith(signoffs):
            signoff_start = last.start()
        elif len(lines) >= 2 and lines[-2].group().strip().lower().startswith(signoffs):
            signoff_start = lines[-2].start()
        else:
            signoff_start = None

        if signoff_start is not None:
            zones["signoff"] = make_zone(
                offset + signoff_start,
                offset + last.end()
            )
            body_end = signoff_start

    zones["body"] = make_zone(
        offset + body_start,
        offset + body_end
    )

    return {
        "text": full_text,
        "zones": zones
    }

def analyze_email_images(images):
    if not images:
        return []

    from src.image_analyzer import analyze_image

    results = []

    for number, item in enumerate(images, start=1):
        try:
            analysis = analyze_image(item["data"])

            results.append({
                "image_number": number,
                "filename": item["filename"],
                "content_id": item["content_id"],
                "analysis": analysis
            })

        except Exception as error:
            results.append({
                "image_number": number,
                "filename": item["filename"],
                "error": str(error)
            })

    return results

def parse_eml(path: str | Path) -> dict:
    """Прочитать .eml: заголовки, HTML, обычный текст, вложения и изображения."""
    path = Path(path)
    with path.open("rb") as file:
        message = BytesParser(policy=policy.default).parse(file)

    html_part = message.get_body(preferencelist=("html",))
    plain_part = message.get_body(preferencelist=("plain",))

    html = html_part.get_content() if html_part is not None else ""
    plain_text = plain_part.get_content() if plain_part is not None else ""
    html_analysis = analyze_html(html) if html else None

    visible_text = (
        html_analysis.visible_text
        if html_analysis and html_analysis.visible_text.strip()
        else plain_text
    )
    zones_result = split_text_zones(
        str(message.get("Subject", "")),
        visible_text
    )
    images = []
    attachments = []

    for part in message.walk():
        if part.is_multipart():
            continue

        content_type = part.get_content_type()
        disposition = part.get_content_disposition()
        filename = part.get_filename()
        data = part.get_payload(decode=True) or b""

        if part.get_content_maintype() == "image" and data:
            images.append({
                "filename": filename,
                "content_id": str(part.get("Content-ID", "")).strip("<>") or None,
                "content_type": content_type,
                "disposition": disposition,
                "data": data,
            })

        if disposition == "attachment":
            attachments.append({
                "filename": filename,
                "content_type": content_type,
                "size": len(data),
            })
    image_results = analyze_email_images(images)
    return {
        "subject": str(message.get("Subject", "")),
        "from": str(message.get("From", "")),
        "date": str(message.get("Date", "")),
        "html": html,
        "plain_text": plain_text,
        "images": images,
        "image_results": image_results,
        "attachments": attachments,
        "html_analysis": html_analysis,
        "visible_text": visible_text,
        "zoned_text": zones_result["text"],
        "zones": zones_result["zones"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Извлечение данных из .eml")
    parser.add_argument("file", help="Путь к .eml")
    args = parser.parse_args()

    result = parse_eml(args.file)
    print("Тема:", result["subject"])
    print("Отправитель:", result["from"])
    print("Дата:", result["date"])
    print("Есть HTML:", bool(result["html"]))
    print("Есть обычный текст:", bool(result["plain_text"]))
    print("Изображений:", len(result["images"]))
    print("Вложений:", len(result["attachments"]))
    if result["html_analysis"] is not None:
        print_report(result["html_analysis"])

    print("Текст для нейронки:")
    print(result["visible_text"])
    print("\nЗоны письма:")

    for name, bounds in result["zones"].items():
        print(f"{name}: {bounds}")

        if bounds is not None:
            start, end = bounds
            print(result["zoned_text"][start:end])

    print("\nАнализ изображений:")

    for item in result["image_results"]:
        print("Изображение:", item["image_number"])
        print("Файл:", item["filename"])

        if "error" in item:
            print("Ошибка:", item["error"])
            continue

        analysis = item["analysis"]

        print("QR-коды:", analysis["qr_codes"])
        print("OCR-текст:", analysis["ocr_text"])
        print("Результат ИИ:", analysis["ai_detection"])