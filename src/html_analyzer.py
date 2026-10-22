"""HTML-анализатор

Модуль извлекает из HTML признаки, которые нельзя получить из обычного текста:
ссылки и возможную подмену адреса;
скрытый текст (inline CSS, атрибут hidden, простые правила из <style>);
видимый текст письма;
структурные признаки: формы, поля пароля, iframe, script, meta refresh, изображения.
эти признаки сами по себе НЕ доказывают, что текст создан ИИ.
Они описывают структуру HTML и потенциально подозрительные элементы документа.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, Tag


# Стили, которые делают элемент невидимым для пользователя.
_HIDDEN_STYLE_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|;)\s*display\s*:\s*none\s*(?:!important)?\s*(?:;|$)", re.I), "display:none"),
    (
        re.compile(
            r"(?:^|;)\s*visibility\s*:\s*(?:hidden|collapse)\s*(?:!important)?\s*(?:;|$)",
            re.I,
        ),
        "visibility:hidden",
    ),
    (
        re.compile(r"(?:^|;)\s*opacity\s*:\s*0(?:\.0+)?\s*(?:!important)?\s*(?:;|$)", re.I),
        "opacity:0",
    ),
    (
        re.compile(
            r"(?:^|;)\s*font-size\s*:\s*0(?:px|pt|pc|em|rem|%|vh|vw)?\s*(?:!important)?\s*(?:;|$)",
            re.I,
        ),
        "font-size:0",
    ),
    (
        re.compile(r"(?:^|;)\s*color\s*:\s*transparent\s*(?:!important)?\s*(?:;|$)", re.I),
        "color:transparent",
    ),
)

# Очень простой разбор обычных CSS-правил: selector { declarations }.
# Для учебного проекта этого достаточно; сложные @media/@supports правила здесь не раскрываются.
_CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)

# Текст ссылки считаем URL-подобным только если он действительно похож на адрес.
_DOMAIN_TEXT_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?(?:[/?#].*)?$",
    re.I,
)


@dataclass
class LinkInfo:
    """Информация об одной HTML-ссылке."""

    text: str
    href: str
    visible_host: Optional[str]
    actual_host: Optional[str]
    suspicious: bool
    reasons: List[str]


@dataclass
class HiddenElementInfo:
    """Информация об элементе, который скрыт от пользователя."""

    tag: str
    element_id: Optional[str]
    classes: List[str]
    text: str
    reasons: List[str]


@dataclass
class HtmlAnalysisResult:
    """Полный результат анализа одного HTML-документа."""

    links_total: int
    suspicious_links_count: int
    links: List[LinkInfo]
    hidden_elements_count: int
    hidden_elements: List[HiddenElementInfo]
    hidden_texts: List[str]
    visible_text: str
    images_count: int
    forms_count: int
    password_fields_count: int
    iframes_count: int
    scripts_count: int
    meta_refresh_count: int

    def to_dict(self) -> Dict[str, object]:
        """Преобразовать результат в обычный словарь для JSON/дальнейшей обработки."""

        return asdict(self)


def _clean_text(text: str) -> str:
    """Схлопнуть повторяющиеся пробелы и переносы строк."""

    return " ".join(text.split())


def _hidden_style_reasons(style: str) -> List[str]:
    """Вернуть список причин, по которым CSS style скрывает элемент."""

    # Добавляем ; по краям, чтобы одинаково обрабатывать первый и последний declaration.
    normalized = ";" + style.strip().strip(";") + ";"
    reasons: List[str] = []
    for pattern, reason in _HIDDEN_STYLE_PATTERNS:
        if pattern.search(normalized):
            reasons.append(reason)
    return reasons


def _normalize_host(host: Optional[str]) -> Optional[str]:
    if not host:
        return None
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_from_url(value: str) -> Optional[str]:
    """Достать домен из URL или из строки вида example.com/path."""

    value = value.strip()
    if not value:
        return None

    candidate = value
    if "://" not in candidate:
        candidate = "//" + candidate

    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None

    return _normalize_host(parsed.hostname)


def _visible_host_from_link_text(text: str) -> Optional[str]:
    """Если подпись ссылки похожа на URL, вернуть показанный пользователю домен."""

    compact = text.strip().strip("<>()[]{}.,;\"'")
    if not _DOMAIN_TEXT_RE.fullmatch(compact):
        return None
    return _host_from_url(compact)


def _analyze_link(tag: Tag) -> LinkInfo:
    text = _clean_text(tag.get_text(" ", strip=True))
    href = str(tag.get("href", "")).strip()

    visible_host = _visible_host_from_link_text(text)
    actual_host: Optional[str] = None
    reasons: List[str] = []

    lower_href = href.lower()

    if lower_href.startswith(("http://", "https://", "//")):
        actual_host = _host_from_url(href)
    elif lower_href.startswith("javascript:"):
        reasons.append("ссылка использует javascript:")
    elif lower_href.startswith("data:"):
        reasons.append("ссылка использует data:")

    # Проверяем именно подмену URL: подпись сама должна быть похожа на адрес.
    if visible_host and actual_host and visible_host != actual_host:
        reasons.append(
            f"показан домен '{visible_host}', но переход ведёт на '{actual_host}'"
        )

    return LinkInfo(
        text=text,
        href=href,
        visible_host=visible_host,
        actual_host=actual_host,
        suspicious=bool(reasons),
        reasons=reasons,
    )


def _collect_hidden_tags(soup: BeautifulSoup) -> Dict[int, Tuple[Tag, Set[str]]]:
    """Найти элементы, скрытые HTML-атрибутами и CSS."""

    hidden: Dict[int, Tuple[Tag, Set[str]]] = {}

    def add(tag: Tag, reason: str) -> None:
        key = id(tag)
        if key not in hidden:
            hidden[key] = (tag, set())
        hidden[key][1].add(reason)

    # 1. Атрибут hidden и inline style="...".
    for tag in soup.find_all(True):
        if tag.has_attr("hidden"):
            add(tag, "HTML-атрибут hidden")

        style = str(tag.get("style", ""))
        for reason in _hidden_style_reasons(style):
            add(tag, reason)

    # 2. Простые CSS-правила из <style>.
    for style_tag in soup.find_all("style"):
        css = style_tag.string or style_tag.get_text(" ", strip=False)
        css = _CSS_COMMENT_RE.sub("", css)

        for selector_group, declarations in _CSS_RULE_RE.findall(css):
            style_reasons = _hidden_style_reasons(declarations)
            if not style_reasons:
                continue

            for selector in selector_group.split(","):
                selector = selector.strip()
                if not selector or selector.startswith("@"):
                    continue

                try:
                    matched = soup.select(selector)
                except Exception:
                    # Некорректный/неподдерживаемый CSS-селектор не должен ломать весь анализ.
                    continue

                for tag in matched:
                    for reason in style_reasons:
                        add(tag, f"CSS '{selector}': {reason}")

    return hidden


def _is_inside_hidden(tag: Tag, hidden_ids: Set[int]) -> bool:
    current: Optional[Tag] = tag
    while isinstance(current, Tag):
        if id(current) in hidden_ids:
            return True
        current = current.parent if isinstance(current.parent, Tag) else None
    return False


def _extract_visible_text(soup: BeautifulSoup, hidden_ids: Set[int]) -> str:
    """Извлечь текст, который должен быть видим пользователю."""

    ignored_parents = {"style", "script", "noscript", "template", "head", "title"}
    pieces: List[str] = []

    for node in soup.find_all(string=True):
        if not isinstance(node, NavigableString):
            continue
        if isinstance(node, (Comment, Doctype, Declaration)):
            continue

        parent = node.parent
        if not isinstance(parent, Tag):
            continue
        if parent.name in ignored_parents:
            continue
        if _is_inside_hidden(parent, hidden_ids):
            continue

        text = _clean_text(str(node))
        if text:
            pieces.append(text)

    return "\n".join(pieces)


def analyze_html(html: str) -> HtmlAnalysisResult:
    """Проанализировать HTML-код и вернуть найденные признаки.

    Args:
        html: Полный HTML-код письма или документа.

    Returns:
        HtmlAnalysisResult с ссылками, скрытым текстом, видимым текстом и
        структурными признаками HTML.
    """

    soup = BeautifulSoup(html, "html.parser")

    links = [_analyze_link(tag) for tag in soup.find_all("a")]
    suspicious_links_count = sum(link.suspicious for link in links)

    hidden_map = _collect_hidden_tags(soup)
    hidden_ids = set(hidden_map.keys())

    hidden_elements: List[HiddenElementInfo] = []
    hidden_texts: List[str] = []
    seen_hidden_texts: Set[str] = set()

    for tag, reasons in hidden_map.values():
        text = _clean_text(tag.get_text(" ", strip=True))
        hidden_elements.append(
            HiddenElementInfo(
                tag=tag.name,
                element_id=str(tag.get("id")) if tag.get("id") else None,
                classes=[str(c) for c in tag.get("class", [])],
                text=text,
                reasons=sorted(reasons),
            )
        )
        if text and text not in seen_hidden_texts:
            seen_hidden_texts.add(text)
            hidden_texts.append(text)

    meta_refresh_count = 0
    for meta in soup.find_all("meta"):
        http_equiv = str(meta.get("http-equiv", "")).strip().lower()
        if http_equiv == "refresh":
            meta_refresh_count += 1

    password_fields_count = sum(
        1
        for field in soup.find_all("input")
        if str(field.get("type", "")).strip().lower() == "password"
    )

    return HtmlAnalysisResult(
        links_total=len(links),
        suspicious_links_count=suspicious_links_count,
        links=links,
        hidden_elements_count=len(hidden_elements),
        hidden_elements=hidden_elements,
        hidden_texts=hidden_texts,
        visible_text=_extract_visible_text(soup, hidden_ids),
        images_count=len(soup.find_all("img")),
        forms_count=len(soup.find_all("form")),
        password_fields_count=password_fields_count,
        iframes_count=len(soup.find_all("iframe")),
        scripts_count=len(soup.find_all("script")),
        meta_refresh_count=meta_refresh_count,
    )


def analyze_html_file(path: str | Path) -> HtmlAnalysisResult:
    """Прочитать HTML-файл в UTF-8 и выполнить анализ."""

    file_path = Path(path)
    html = file_path.read_text(encoding="utf-8")
    return analyze_html(html)


def print_report(result: HtmlAnalysisResult) -> None:
    """Красиво вывести результат HTML-анализа в терминал."""

    print("\n========== HTML ANALYSIS ==========")
    print(f"Ссылок: {result.links_total}")
    print(f"Подозрительных ссылок: {result.suspicious_links_count}")

    for index, link in enumerate(result.links, start=1):
        status = "ПОДОЗРИТЕЛЬНАЯ" if link.suspicious else "OK"
        print(f"\nСсылка #{index} [{status}]")
        print(f"  Текст: {link.text or '(пустой)'}")
        print(f"  href:  {link.href or '(нет href)'}")
        for reason in link.reasons:
            print(f"  Причина: {reason}")

    print(f"\nСкрытых HTML-элементов: {result.hidden_elements_count}")
    for index, item in enumerate(result.hidden_elements, start=1):
        label = item.tag
        if item.element_id:
            label += f"#{item.element_id}"
        if item.classes:
            label += "." + ".".join(item.classes)
        print(f"  #{index} <{label}>: {', '.join(item.reasons)}")
        if item.text:
            print(f"      Скрытый текст: {item.text}")

    print("\nСтруктурные признаки:")
    print(f"  Изображения <img>: {result.images_count}")
    print(f"  Формы <form>: {result.forms_count}")
    print(f"  Поля пароля: {result.password_fields_count}")
    print(f"  iframe: {result.iframes_count}")
    print(f"  script: {result.scripts_count}")
    print(f"  meta refresh: {result.meta_refresh_count}")

    print("\n========== VISIBLE TEXT ==========")
    print(result.visible_text or "(видимый текст отсутствует)")
    print("==================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ HTML письма/документа: ссылки, скрытый текст и структура"
    )
    parser.add_argument("html_file", type=str, help="Путь к .html файлу")
    parser.add_argument(
        "--json",
        dest="json_path",
        type=str,
        default=None,
        help="Необязательно: сохранить полный отчёт в JSON-файл",
    )
    args = parser.parse_args()

    try:
        result = analyze_html_file(args.html_file)
    except FileNotFoundError:
        print(f"Ошибка: файл '{args.html_file}' не найден")
        return
    except UnicodeDecodeError:
        print("Ошибка: HTML-файл не удалось прочитать как UTF-8")
        return

    print_report(result)

    if args.json_path:
        output_path = Path(args.json_path)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON-отчёт сохранён: {output_path}")


if __name__ == "__main__":
    main()
