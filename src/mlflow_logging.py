"""
Registro común de runs experimentales y finales en MLflow (A.2).

Centraliza lo que antes vivía solo en baselines.py, para que
comparisons.py, ablation.py y final_model.py no dupliquen la construcción
de configuration.json ni el set de tags/metrics/artifacts obligatorios.
"""

from __future__ import annotations
import json
import os

import numpy as np
import mlflow

from .mlflow_utils import read_sagemaker_provenance


def log_experiment_run(
    lab_experiment_id: str,
    lab_stage: str,
    lab_protocol_run_id: str,
    lab_member_id: str,
    lab_configuration_id: str,
    fold_scores: list[float],
    configuration: dict,
    out_dir: str,
    extra_tags: dict[str, str] | None = None,
    extra_params: dict[str, object] | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> str:
    """Registra un run experimental (incluye T0, B0, P_*, R_*, C_*, EXTRA,
    ABLATION). extra_tags/extra_params/extra_metrics permiten añadir los
    campos propios de ablación (lab_ablation_parent_run_id,
    ablation_reverted_decision, macro_f1_delta) sin duplicar esta función."""
    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores, ddof=0))  # poblacional, exige la guía

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "configuration.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(configuration, f, indent=2)

    _, notebook_arn = read_sagemaker_provenance(out_dir)
    provenance_path = os.path.join(out_dir, "provenance", "sagemaker-resource-metadata.json")

    with mlflow.start_run(run_name=lab_experiment_id) as run:
        mlflow.set_tag("lab_run_type", "experiment")
        mlflow.set_tag("lab_protocol_run_id", lab_protocol_run_id)
        mlflow.set_tag("lab_experiment_id", lab_experiment_id)
        mlflow.set_tag("lab_stage", lab_stage)
        mlflow.set_tag("lab_member_id", lab_member_id)
        mlflow.set_tag("lab_configuration_id", lab_configuration_id)
        mlflow.set_tag("notebook_arn", notebook_arn)
        for k, v in (extra_tags or {}).items():
            mlflow.set_tag(k, v)

        for k, v in (extra_params or {}).items():
            mlflow.log_param(k, v)

        for i, s in enumerate(fold_scores):
            mlflow.log_metric(f"macro_f1_fold_{i}", s)
        mlflow.log_metric("macro_f1_mean", mean)
        mlflow.log_metric("macro_f1_std", std)
        for k, v in (extra_metrics or {}).items():
            mlflow.log_metric(k, v)

        mlflow.log_artifact(config_path, artifact_path="run")
        mlflow.log_artifact(provenance_path, artifact_path="provenance")

        print(f"{lab_experiment_id} run_id={run.info.run_id} mean={mean:.4f} std={std:.4f}")
        return run.info.run_id


def load_run_configuration(run_id: str) -> dict:
    """Descarga y parsea run/configuration.json de un run ya registrado
    (usado por ablation.py y final_model.py para partir de la configuración
    efectiva de un run experimental existente, en vez de reconstruirla a mano)."""
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="run/configuration.json"
    )
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)
