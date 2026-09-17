"""Valida las respuestas 4xx de POST /api/v1/predict para los ejemplos
inválidos exactos del Anexo A.6, mockeando la carga del modelo (no
depende de un Tracking Server real)."""

from __future__ import annotations
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

INVALID_PAYLOADS = [
    {},
    {"text": None},
    {"text": ""},
    {"text": " "},
    {"text": []},
    {"text": ["ok", ""]},
    {"text": ["ok", 7]},
]


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_predict_payload_invalido_es_4xx(payload):
    response = client.post("/api/v1/predict", json=payload)
    assert 400 <= response.status_code < 500


def test_predict_lista_mas_de_32_elementos_es_4xx():
    response = client.post("/api/v1/predict", json={"text": ["ok"] * 33})
    assert 400 <= response.status_code < 500


def test_predict_texto_mas_de_1000_caracteres_es_4xx():
    response = client.post("/api/v1/predict", json={"text": "a" * 1001})
    assert 400 <= response.status_code < 500


def test_predict_valido_devuelve_predictions_en_orden():
    fake_model = type("FakeModel", (), {"predict": staticmethod(lambda texts: ["positive"] * len(texts))})()

    with patch("api.main._load_champion_model", return_value=(fake_model, "FAKE_RUN_ID")):
        response = client.post("/api/v1/predict", json={"text": ["i loved it", "worst day ever"]})

    assert response.status_code == 200
    body = response.json()
    assert body["model_run_id"] == "FAKE_RUN_ID"
    assert body["predictions"] == ["positive", "positive"]


def test_predict_string_unico_devuelve_lista_de_un_elemento():
    fake_model = type("FakeModel", (), {"predict": staticmethod(lambda texts: ["negative"] * len(texts))})()

    with patch("api.main._load_champion_model", return_value=(fake_model, "FAKE_RUN_ID")):
        response = client.post("/api/v1/predict", json={"text": "worst day ever"})

    assert response.status_code == 200
    assert response.json()["predictions"] == ["negative"]
