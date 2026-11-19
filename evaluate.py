#!/usr/bin/env python3
"""Расчёт всех метрик ТЗ (п. 4.1.11) на тестовой выборке датасета одной командой.

    python evaluate.py                      # метод на признаках
    python evaluate.py --methods features,lm

Реализация: src/evaluation/run.py. Отчёт: results/metrics/report.md
"""
from src.evaluation.run import main

if __name__ == "__main__":
    main()
