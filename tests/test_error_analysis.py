"""Valida el encabezado exacto del CSV de análisis de errores y que las
categorías generadas pertenezcan siempre al conjunto controlado (A.6)."""

from __future__ import annotations
import csv

import numpy as np

from src.error_analysis import generate_error_analysis, CATEGORIES, select_errors


def _synthetic_case():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=40, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    texts = []
    for i in range(n):
        if i % 5 == 0:
            texts.append("not bad, actually pretty good #great")
        elif i % 5 == 1:
            texts.append("soooo tired of this 😀")
        elif i % 5 == 2:
            texts.append("yeah right, best service ever")
        else:
            texts.append("this is a normal review text")
    indices = np.arange(1000, 1000 + n)
    return y_true, y_pred, texts, indices


def test_csv_header_exacto(tmp_path):
    y_true, y_pred, texts, indices = _synthetic_case()
    csv_path, _md_path = generate_error_analysis(y_true, y_pred, texts, indices, out_dir=str(tmp_path))

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == ["index", "text", "true_label", "predicted_label", "category"]


def test_categorias_controladas(tmp_path):
    y_true, y_pred, texts, indices = _synthetic_case()
    csv_path, _md_path = generate_error_analysis(y_true, y_pred, texts, indices, out_dir=str(tmp_path))

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 20
    for row in rows:
        assert row["category"] in CATEGORIES
        assert row["true_label"] in {"negative", "positive"}
        assert row["predicted_label"] in {"negative", "positive"}


def test_select_errors_incluye_ambas_clases_si_existen():
    y_true = np.array([0, 0, 0, 1, 1, 1] * 10)
    y_pred = 1 - y_true  # todos son errores, ambas clases presentes
    positions = select_errors(y_true, y_pred, seed=42, min_errors=20)

    assert len(positions) >= 20
    assert set(y_true[positions]) == {0, 1}


def test_sin_errores_devuelve_vacio():
    y_true = np.array([0, 1, 0, 1])
    y_pred = y_true.copy()
    positions = select_errors(y_true, y_pred, seed=42, min_errors=20)
    assert len(positions) == 0
