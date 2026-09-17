"""
T0 y B0 (Sección 2, A.3, A.4) como runs experimentales.

Ambos se ejecutan sobre los MISMOS 3 folds de protocol/partitions.csv
generado por protocol.py — nunca vuelven a muestrear ni a rehacer folds.

Uso:
    python -m src.baselines
"""

from __future__ import annotations
import os
import re
import sklearn

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .mlflow_utils import init_experiment
from .mlflow_logging import log_experiment_run
from .config_schema import t0_config, b0_config
from .data import LABEL_COL, TEXT_COL, load_sample_with_text

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def b0_preprocess(text: str) -> str:
    """Exactamente lo que exige B0: minúsculas, URL->'url', mención->'user',
    normalizar espacios. Nada más (Sección 2)."""
    t = text.lower()
    t = URL_RE.sub("url", t)
    t = MENTION_RE.sub("user", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def majority_class(y_train: np.ndarray) -> int:
    """T0: clase más frecuente; empate -> negative (label 0)."""
    counts = np.bincount(y_train, minlength=2)
    if counts[0] >= counts[1]:
        return 0  # negative
    return 1  # positive


def macro_f1_per_fold(df: pd.DataFrame, predict_fn) -> list[float]:
    scores = []
    for fold in sorted(df["fold"].unique()):
        train_part = df[df["fold"] != fold]
        val_part = df[df["fold"] == fold]
        y_pred = predict_fn(train_part, val_part)
        scores.append(f1_score(val_part[LABEL_COL], y_pred, average="macro"))
    return scores


def run_t0(df: pd.DataFrame) -> list[float]:
    def predict_fn(train_part, val_part):
        pred = majority_class(train_part[LABEL_COL].to_numpy())
        return np.full(len(val_part), pred)
    return macro_f1_per_fold(df, predict_fn)


def run_b0(df: pd.DataFrame) -> list[float]:
    def predict_fn(train_part, val_part):
        X_train_text = train_part[TEXT_COL].map(b0_preprocess)
        X_val_text = val_part[TEXT_COL].map(b0_preprocess)

        vec = CountVectorizer(ngram_range=(1, 1))  # BoW unigramas, defaults de la librería
        X_train = vec.fit_transform(X_train_text)
        X_val = vec.transform(X_val_text)

        # Hiperparámetros por defecto de la librería. Si aparece
        # ConvergenceWarning, NO SE TOCA (instrucción explícita de la guía).
        clf = LogisticRegression()
        clf.fit(X_train, train_part[LABEL_COL])
        return clf.predict(X_val)
    return macro_f1_per_fold(df, predict_fn)


if __name__ == "__main__":
    PROTOCOL_RUN_ID = os.environ["LAB_PROTOCOL_RUN_ID"]     # del output de protocol.py
    MEMBER_ID = os.environ.get("LAB_MEMBER_ID", "E01")
    PARTITIONS_PATH = os.environ.get("LAB_PARTITIONS_PATH", "protocol_artifacts/partitions.csv")

    init_experiment(os.environ.get("MLFLOW_TRACKING_URI"))
    df = load_sample_with_text(PARTITIONS_PATH)

    t0_scores = run_t0(df)
    log_experiment_run(
        lab_experiment_id="T0",
        lab_stage="reference",
        lab_protocol_run_id=PROTOCOL_RUN_ID,
        lab_member_id=MEMBER_ID,
        lab_configuration_id="CFG_T0",
        fold_scores=t0_scores,
        configuration=t0_config(library="scikit-learn", library_version=sklearn.__version__),
        out_dir="t0_artifacts",
    )

    b0_scores = run_b0(df)
    log_experiment_run(
        lab_experiment_id="B0",
        lab_stage="baseline",
        lab_protocol_run_id=PROTOCOL_RUN_ID,
        lab_member_id=MEMBER_ID,
        lab_configuration_id="CFG_B0",
        fold_scores=b0_scores,
        configuration=b0_config(sklearn_version=sklearn.__version__),
        out_dir="b0_artifacts",
    )
