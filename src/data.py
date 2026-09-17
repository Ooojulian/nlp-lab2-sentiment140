"""
Acceso al dataset y a la muestra/folds del protocolo (Sección 2, A.2).

Centraliza lo que antes vivía en protocol.py y se necesitaba también desde
baselines.py/comparisons.py/ablation.py/final_model.py, para no duplicar
DATASET_ID, TEXT_COL/LABEL_COL ni la lógica de carga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset

DATASET_ID = "adilbekovich/Sentiment140Twitter"
DATASET_REVISION = "b6037e127257d95b9b23d31f78b264b9ebe697fd"

# AJUSTA esto tras correr inspect_dataset() si los nombres reales difieren.
TEXT_COL = "text"
LABEL_COL = "label"


def load_dataset_splits():
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    assert len(ds["train"]) == 1_360_000, f"train esperado=1.360.000, obtenido={len(ds['train'])}"
    assert len(ds["test"]) == 240_000, f"test esperado=240.000, obtenido={len(ds['test'])}"
    return ds


def load_train():
    return load_dataset_splits()["train"]


def load_test():
    return load_dataset_splits()["test"]


def inspect_dataset():
    """Corre esto primero, a mano, para confirmar nombres de columnas."""
    train = load_train()
    print("Columnas:", train.column_names)
    print("Ejemplo:", train[0])
    return train


def load_sample_with_text(partitions_path: str) -> pd.DataFrame:
    """Une partitions.csv (index, fold) con el texto/label real desde train,
    usando 'index' como la posición original guardada por protocol.py."""
    partitions = pd.read_csv(partitions_path)
    train = load_train().to_pandas()
    train["index"] = np.arange(len(train))

    merged = partitions.merge(train[["index", TEXT_COL, LABEL_COL]], on="index", how="left")
    assert merged[TEXT_COL].isna().sum() == 0, "Índices de partitions.csv no encontrados en train"
    return merged
