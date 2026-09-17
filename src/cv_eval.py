"""
Evaluación 3-fold CV compartida por comparisons.py y ablation.py: mismo
patrón que baselines.run_b0, generalizado a preprocesamiento +
representación + clasificador configurables.
"""

from __future__ import annotations
import pandas as pd
from sklearn.metrics import f1_score

from .data import LABEL_COL, TEXT_COL
from .preprocessing import build_preprocessor
from .representations import build_representation
from .classifiers import build_classifier


def macro_f1_per_fold(df: pd.DataFrame, predict_fn) -> list[float]:
    scores = []
    for fold in sorted(df["fold"].unique()):
        train_part = df[df["fold"] != fold]
        val_part = df[df["fold"] == fold]
        y_pred = predict_fn(train_part, val_part)
        scores.append(f1_score(val_part[LABEL_COL], y_pred, average="macro"))
    return scores


def run_configuration(df: pd.DataFrame, configuration: dict) -> list[float]:
    """Evalúa una configuración (preprocessing+representation+classifier)
    con 3-fold CV sobre df (salida de data.load_sample_with_text)."""
    preprocess = build_preprocessor(configuration["preprocessing"])

    def predict_fn(train_part, val_part):
        train_texts = preprocess(train_part[TEXT_COL].tolist())
        val_texts = preprocess(val_part[TEXT_COL].tolist())

        representation = build_representation(configuration["representation"])
        X_train = representation.fit_transform(train_texts)
        X_val = representation.transform(val_texts)

        clf = build_classifier(configuration["classifier"])
        clf.fit(X_train, train_part[LABEL_COL])
        return clf.predict(X_val)

    return macro_f1_per_fold(df, predict_fn)
