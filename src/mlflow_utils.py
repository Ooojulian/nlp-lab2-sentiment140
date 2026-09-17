"""
Utilidades comunes de MLflow para el laboratorio.

IMPORTANTE sobre procedencia (A.2): /opt/ml/metadata/resource-metadata.json
SOLO existe corriendo dentro de una SageMaker Notebook Instance real. Esta
función NO fabrica un valor de reemplazo si el archivo no existe: lo hace
a propósito, porque un run con procedencia inventada es un run inválido
según el contrato (el evaluador contrasta ResourceArn contra la asignación
oficial del curso) y además sería una violación de integridad académica.
Corre esto solo desde tu Notebook Instance asignada.
"""

from __future__ import annotations
import json
import os
import shutil

import mlflow

EXPERIMENT_NAME = "nlp-lab2-sentiment140"
SAGEMAKER_METADATA_PATH = "/opt/ml/metadata/resource-metadata.json"


def init_experiment(tracking_uri: str | None = None) -> str:
    """Configura el Tracking URI (si se pasa) y asegura que el Experiment
    exacto exista. Devuelve el experiment_id nativo de MLflow."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"No se pudo crear/obtener el experiment '{EXPERIMENT_NAME}'.")
    return exp.experiment_id


def read_sagemaker_provenance(dest_dir: str) -> tuple[str, str]:
    """Copia SIN editar /opt/ml/metadata/resource-metadata.json a
    <dest_dir>/provenance/sagemaker-resource-metadata.json.

    Devuelve (ruta_local, resource_arn) para que puedas usar resource_arn
    como valor exacto del tag notebook_arn.

    Lanza RuntimeError si no corres dentro de una Notebook Instance real:
    NO se genera un valor ficticio.
    """
    if not os.path.exists(SAGEMAKER_METADATA_PATH):
        raise RuntimeError(
            f"No se encontró {SAGEMAKER_METADATA_PATH}. Este run debe ejecutarse "
            "dentro de la SageMaker Notebook Instance asignada por el curso; "
            "no se acepta un valor simulado (A.2 exige correspondencia exacta "
            "con la asignación oficial)."
        )

    prov_dir = os.path.join(dest_dir, "provenance")
    os.makedirs(prov_dir, exist_ok=True)
    local_path = os.path.join(prov_dir, "sagemaker-resource-metadata.json")
    shutil.copyfile(SAGEMAKER_METADATA_PATH, local_path)

    with open(local_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    resource_arn = metadata.get("ResourceArn")
    if not resource_arn:
        raise RuntimeError("resource-metadata.json no contiene 'ResourceArn'.")

    return local_path, resource_arn
