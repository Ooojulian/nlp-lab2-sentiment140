"""
Clasificador (Sección 3, Anexo A.3/A.4): Logistic Regression, Linear SVM,
SGDClassifier. T0 (most_frequent) se maneja aparte en baselines.py, no
entrena un estimador de sklearn.
"""

from __future__ import annotations

import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC


def build_classifier(config: dict):
    """Dado un classifier_config (dict, ver config_schema.py), devuelve un
    estimador de sklearn sin ajustar. parameters={} usa los hiperparámetros
    por defecto de la biblioteca, tal como exige la guía para B0/comparaciones
    obligatorias salvo que se declaren explícitamente."""
    ctype = config["type"]
    params = config.get("parameters") or {}

    if ctype == "logistic_regression":
        return LogisticRegression(**params)
    if ctype == "linear_svm":
        return LinearSVC(**params)
    if ctype == "sgd":
        return SGDClassifier(**params)

    raise ValueError(f"classifier.type no soportado por build_classifier: {ctype}")


def sklearn_version() -> str:
    return sklearn.__version__
