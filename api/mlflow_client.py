"""
Funciones que consultan MLflow para los endpoints /audit/* y /api/v1/predict
(Anexo A.1, A.2, A.5, A.6).

Todas las funciones que hablan con el Tracking Server lanzan
MlflowUnavailableError si MLflow no responde (conexión/timeout), que
main.py traduce a HTTP 503 {"detail": "mlflow_unavailable"} — distinto de
los 404/409 de lógica de negocio (alias inexistente, protocolo duplicado).
"""

from __future__ import annotations
import csv
import io
import json
from dataclasses import dataclass, field

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "nlp-lab2-sentiment140"
MODEL_NAME = "sentiment140"
MODEL_ALIAS = "champion"

REQUIRED_EXPERIMENT_TAGS = {
    "lab_run_type", "lab_protocol_run_id", "lab_experiment_id",
    "lab_stage", "lab_member_id", "lab_configuration_id", "notebook_arn",
}
REQUIRED_EXPERIMENT_METRICS = {
    "macro_f1_fold_0", "macro_f1_fold_1", "macro_f1_fold_2",
    "macro_f1_mean", "macro_f1_std",
}
REQUIRED_EXPERIMENT_ARTIFACTS = {"run/configuration.json", "provenance/sagemaker-resource-metadata.json"}

REQUIRED_FINAL_TAGS = {
    "lab_run_type", "lab_protocol_run_id", "lab_selected_experiment_run_id",
    "lab_configuration_id", "lab_member_id", "notebook_arn",
}
REQUIRED_FINAL_PARAMS = {"training_size"}
REQUIRED_FINAL_METRICS = {"test_macro_f1"}
REQUIRED_FINAL_ARTIFACTS = {
    "run/configuration.json", "provenance/sagemaker-resource-metadata.json",
    "reports/error_analysis.csv", "reports/error_analysis.md",
}


class MlflowUnavailableError(Exception):
    pass


def _client() -> MlflowClient:
    return MlflowClient()


def _wrap_mlflow_errors(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except MlflowException as exc:
        raise MlflowUnavailableError(str(exc)) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise MlflowUnavailableError(str(exc)) from exc


def get_experiment_id() -> str | None:
    exp = _wrap_mlflow_errors(mlflow.get_experiment_by_name, EXPERIMENT_NAME)
    return exp.experiment_id if exp else None


def list_presented_runs():
    """Todos los runs con lab_run_type en {protocol, experiment, final}
    (A.1: los exploratorios sin ese tag se ignoran)."""
    experiment_id = get_experiment_id()
    if experiment_id is None:
        return []

    runs = _wrap_mlflow_errors(
        _client().search_runs,
        experiment_ids=[experiment_id],
        filter_string="",
        max_results=50_000,
    )
    return [r for r in runs if r.data.tags.get("lab_run_type") in {"protocol", "experiment", "final"}]


def list_artifacts_recursive(run_id: str, path: str = "") -> list[str]:
    client = _client()
    entries = _wrap_mlflow_errors(client.list_artifacts, run_id, path)
    paths = []
    for entry in entries:
        if entry.is_dir:
            paths.extend(list_artifacts_recursive(run_id, entry.path))
        else:
            paths.append(entry.path)
    return sorted(paths)


def read_run_configuration(run_id: str, artifacts: list[str]) -> dict | None:
    if "run/configuration.json" not in artifacts:
        return None
    try:
        local_path = _wrap_mlflow_errors(
            mlflow.artifacts.download_artifacts, run_id=run_id, artifact_path="run/configuration.json"
        )
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def read_members_csv(protocol_run: object) -> list[dict]:
    local_path = _wrap_mlflow_errors(
        mlflow.artifacts.download_artifacts, run_id=protocol_run.info.run_id, artifact_path="protocol/members.csv"
    )
    with open(local_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_valid_experiment_run(run) -> bool:
    if run.info.status != "FINISHED":
        return False
    tags = set(run.data.tags.keys())
    if not REQUIRED_EXPERIMENT_TAGS.issubset(tags):
        return False
    if not REQUIRED_EXPERIMENT_METRICS.issubset(run.data.metrics.keys()):
        return False
    artifacts = set(list_artifacts_recursive(run.info.run_id))
    return REQUIRED_EXPERIMENT_ARTIFACTS.issubset(artifacts)


def is_valid_final_run(run) -> bool:
    if run.info.status != "FINISHED":
        return False
    tags = set(run.data.tags.keys())
    if not REQUIRED_FINAL_TAGS.issubset(tags):
        return False
    if not REQUIRED_FINAL_PARAMS.issubset(run.data.params.keys()):
        return False
    if not REQUIRED_FINAL_METRICS.issubset(run.data.metrics.keys()):
        return False
    artifacts = set(list_artifacts_recursive(run.info.run_id))
    return REQUIRED_FINAL_ARTIFACTS.issubset(artifacts)


def get_unique_protocol_run():
    runs = [r for r in list_presented_runs() if r.data.tags.get("lab_run_type") == "protocol"]
    if len(runs) != 1:
        return None
    return runs[0]


def resolve_champion_model_version():
    """Devuelve el ModelVersion del alias champion, o None si no existe."""
    try:
        return _wrap_mlflow_errors(_client().get_model_version_by_alias, MODEL_NAME, MODEL_ALIAS)
    except MlflowUnavailableError:
        raise
    except MlflowException:
        return None
