"""
Run de protocolo (Sección 2 y A.2). Se ejecuta UNA sola vez para todo el
equipo. Genera protocol/partitions.csv y protocol/members.csv, y los
registra en un único run con lab_run_type=protocol.

Uso:
    python -m src.protocol

Antes de correrlo en serio:
1. Ejecuta `inspect_dataset()` y VERIFICA los nombres reales de las
   columnas de texto/etiqueta. adilbekovich/Sentiment140Twitter puede no
   llamarlas "text"/"label" — no lo estoy asumiendo a ciegas, ajusta
   TEXT_COL/LABEL_COL en src/data.py según lo que veas impreso.
2. Configura MLFLOW_TRACKING_URI apuntando a tu Tracking Server real
   (no queda válido con un store local para la entrega final).
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import mlflow

from .mlflow_utils import init_experiment, EXPERIMENT_NAME
from .data import (
    DATASET_ID,
    DATASET_REVISION,
    LABEL_COL,
    TEXT_COL,
    inspect_dataset,
    load_train,
)

SAMPLE_SIZE = 200_000
RANDOM_SEED = 42
CV_FOLDS = 3

OUT_DIR = "protocol_artifacts"  # carpeta local antes de subir a MLflow como artefactos


def build_sample(train) -> pd.DataFrame:
    """Muestra estratificada de SAMPLE_SIZE registros con semilla 42,
    conservando el índice original de train ANTES de muestrear (A.2)."""
    df = train.to_pandas()
    df["index"] = np.arange(len(df))  # posición original en el split train, empezando en 0

    labels = df[LABEL_COL].to_numpy()
    n = len(df)
    frac = SAMPLE_SIZE / n

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []
    for cls in np.unique(labels):
        idx_cls = df.index[df[LABEL_COL] == cls].to_numpy()
        k = int(round(frac * len(idx_cls)))
        chosen = rng.choice(idx_cls, size=k, replace=False)
        sampled_parts.append(chosen)

    sampled_positions = np.concatenate(sampled_parts)
    sample_df = df.loc[sampled_positions, ["index", LABEL_COL]].reset_index(drop=True)
    return sample_df


def build_folds(sample_df: pd.DataFrame) -> pd.DataFrame:
    """3 folds estratificados, shuffle=True, seed=42, sobre la muestra."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_assignment = np.empty(len(sample_df), dtype=int)

    X = sample_df["index"].to_numpy()
    y = sample_df[LABEL_COL].to_numpy()

    # fold=k significa: ese registro actúa como VALIDACIÓN en el fold k.
    for fold_id, (_, val_idx) in enumerate(skf.split(X, y)):
        fold_assignment[val_idx] = fold_id

    out = sample_df[["index"]].copy()
    out["fold"] = fold_assignment
    out = out.sort_values("index").reset_index(drop=True)

    assert len(out) == len(out["index"].unique()), "Hay índices repetidos en partitions.csv"
    return out


def write_partitions_csv(partitions: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "partitions.csv")
    partitions.to_csv(path, columns=["index", "fold"], index=False)
    return path


def write_members_csv(members: list[dict], out_dir: str) -> str:
    """members: [{"member_id": "E01", "notebook_arn": "arn:..."}]"""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(members).sort_values("member_id").reset_index(drop=True)
    path = os.path.join(out_dir, "members.csv")
    df.to_csv(path, columns=["member_id", "notebook_arn"], index=False)
    return path


def log_protocol_run(members: list[dict], tracking_uri: str | None = None) -> str:
    """Crea el ÚNICO run con lab_run_type=protocol. Devuelve el run_id
    (guárdalo: lo necesitas como lab_protocol_run_id en todos los runs
    posteriores)."""
    init_experiment(tracking_uri)

    train = load_train()
    sample = build_sample(train)
    partitions = build_folds(sample)

    local_dir = OUT_DIR
    partitions_path = write_partitions_csv(partitions, local_dir)
    members_path = write_members_csv(members, local_dir)

    with mlflow.start_run(run_name="protocol") as run:
        mlflow.set_tag("lab_run_type", "protocol")

        mlflow.log_param("dataset_id", DATASET_ID)
        mlflow.log_param("dataset_revision", DATASET_REVISION)
        mlflow.log_param("sampling_strategy", "stratified")
        mlflow.log_param("sample_size", len(partitions))
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("cv_strategy", "StratifiedKFold")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("cv_shuffle", True)

        mlflow.log_artifact(partitions_path, artifact_path="protocol")
        mlflow.log_artifact(members_path, artifact_path="protocol")

        print(f"Protocol run_id = {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    # NO EJECUTAR AÚN: faltan member_id/ARN del resto del equipo. Este run
    # solo se corre una vez para TODOS los integrantes (no se puede repetir).
    members = [
        {
            "member_id": "1019982682",
            "notebook_arn": "arn:aws:sagemaker:us-west-2:025118392450:notebook-instance/nlp-lab2-sentiment140-julian",
        },
        {
            "member_id": "1010126599",
            "notebook_arn": "arn:aws:sagemaker:us-east-1:146249747419:notebook-instance/T2PLN",
        },
        # TODO: agregar aquí member_id/notebook_arn del integrante 3
        # antes de correr `python -m src.protocol` en serio.
    ]
    log_protocol_run(members, tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"))
