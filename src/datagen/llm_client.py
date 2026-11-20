"""Клиент языковой модели с API, совместимым с OpenAI (п. 4.5.1, 4.2.2 ТЗ).

Модель разворачивается локально (например, Ollama: `ollama serve`, endpoint http://localhost:11434/v1),
поэтому данные не покидают контур. Настройки берутся из .env:
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TIMEOUT
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import requests

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "local")
        self.timeout = float(timeout or os.getenv("LLM_TIMEOUT", "180"))
        self.max_retries = max_retries

    def chat(self, messages: List[dict], temperature: float = 0.8, max_tokens: int = 600) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # сетевые ошибки, таймауты, 5xx
                last_err = e
                wait = 2 ** attempt
                log.warning("LLM запрос не удался (попытка %d/%d): %s; повтор через %ds", attempt, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM недоступна после {self.max_retries} попыток: {last_err}")

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5)
            return r.ok
        except Exception:
            return False
