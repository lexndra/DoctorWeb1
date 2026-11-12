// Простой интерфейс для отправки файла в Python-сервер.
const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const button = document.getElementById('analyzeButton');
const message = document.getElementById('message');
const result = document.getElementById('result');
const zones = document.getElementById('zones');
const downloadButton = document.getElementById('downloadButton');

let currentReport = null;

function showError(text) {
    message.textContent = text;
    message.hidden = false;
}

function percent(value) {
    const number = Number(value);
    return (Math.max(0, Math.min(1, Number.isFinite(number) ? number : 0)) * 100).toFixed(1) + '%';
}

// Подсвечиваем фрагменты, которые отметила модель.
function showText(zone, element) {
    const text = String(zone.text || '').slice(0, 5000);
    const offset = Number(zone.start_char || 0);
    const spans = Array.isArray(zone.ai_spans) ? zone.ai_spans : [];
    const sorted = spans.map(item => ({
        start: Number(item.start_char) - offset,
        end: Number(item.end_char) - offset
    })).sort((a, b) => a.start - b.start);

    let position = 0;
    for (const span of sorted) {
        const start = Math.max(0, Math.min(text.length, span.start));
        const end = Math.max(0, Math.min(text.length, span.end));
        if (!Number.isFinite(start) || !Number.isFinite(end) || start < position || end <= start) continue;
        element.append(document.createTextNode(text.slice(position, start)));
        const highlight = document.createElement('mark');
        highlight.textContent = text.slice(start, end);
        element.append(highlight);
        position = end;
    }
    element.append(document.createTextNode(text.slice(position)));
    if (String(zone.text || '').length > text.length) {
        element.append(document.createTextNode('… (текст сокращён)'));
    }
}

function showResult(report) {
    currentReport = report;
    document.getElementById('fileName').textContent = report.file_id || '—';
    document.getElementById('verdict').textContent = report.verdict || 'Нет данных';
    document.getElementById('score').textContent = percent(report.overall_ai_index);
    document.getElementById('explanation').textContent = report.explanation || 'Пояснения нет.';

    zones.replaceChildren();
    const list = Array.isArray(report.zones) ? report.zones : [];
    if (!list.length) {
        zones.textContent = 'Части текста не найдены.';
    }
    for (const zone of list) {
        const block = document.createElement('div');
        block.className = 'zone';
        const heading = document.createElement('b');
        heading.textContent = (zone.name || 'Текст') + ' — ' + percent(zone.ai_probability);
        const paragraph = document.createElement('p');
        showText(zone, paragraph);
        block.append(heading, paragraph);
        zones.append(block);
    }
    result.hidden = false;
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) return showError('Сначала выберите файл.');
    if (file.size > 5 * 1024 * 1024) return showError('Максимальный размер файла — 5 МБ.');

    button.disabled = true;
    button.textContent = 'Проверяем...';
    message.hidden = true;
    result.hidden = true;

    try {
        const data = new FormData();
        data.append('file', file);
        const response = await fetch('/api/analyze', { method: 'POST', body: data });
        const report = await response.json();
        if (!response.ok) throw new Error(report.error || 'Ошибка на сервере');
        showResult(report);
    } catch (error) {
        showError(error.message || 'Не удалось выполнить анализ.');
    } finally {
        button.disabled = false;
        button.textContent = 'Начать анализ';
    }
});

// Отчёт можно сохранить в виде обычного JSON-файла.
downloadButton.addEventListener('click', () => {
    if (!currentReport) return;
    const blob = new Blob([JSON.stringify(currentReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'result.json';
    link.click();
    URL.revokeObjectURL(url);
});
// Переключение темы
const themeButton = document.getElementById('themeButton');

function updateThemeButton() {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';

    themeButton.textContent = dark
        ? '☀️ Светлая тема'
        : '🌙 Тёмная тема';
}

updateThemeButton();

themeButton.addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';

    const theme = dark ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    updateThemeButton();
});