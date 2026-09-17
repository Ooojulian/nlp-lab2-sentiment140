"""
API FastAPI del laboratorio (Sección 6, Anexo A.5/A.6).

Uso:
    export MLFLOW_TRACKING_URI=<url del tracking server>
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import logging

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import mlflow_client as mc
from .schemas import PredictRequest, PredictResponse, HealthResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="nlp-lab2-sentiment140 API")

_model_cache: dict[str, object] = {"pyfunc": None, "run_id": None}


def _load_champion_model():
    """Carga (o reutiliza) sentiment140@champion. Nunca cachea un fallo:
    si la carga falla, el próximo request vuelve a intentar (A.5: la API
    resuelve el modelo desde el alias, nunca desde una copia local)."""
    version = mc.resolve_champion_model_version()
    if version is None:
        _model_cache["pyfunc"] = None
        _model_cache["run_id"] = None
        return None, None

    if _model_cache["run_id"] != version.run_id or _model_cache["pyfunc"] is None:
        loaded = mlflow.pyfunc.load_model(f"models:/{mc.MODEL_NAME}@{mc.MODEL_ALIAS}")
        _model_cache["pyfunc"] = loaded
        _model_cache["run_id"] = version.run_id

    return _model_cache["pyfunc"], version.run_id


@app.exception_handler(mc.MlflowUnavailableError)
async def mlflow_unavailable_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": "mlflow_unavailable"})


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    texts = request.as_list()

    try:
        model, run_id = _load_champion_model()
    except mc.MlflowUnavailableError:
        raise
    if model is None:
        raise HTTPException(status_code=503, detail="model_unavailable")

    predictions = model.predict(texts)
    return PredictResponse(model_run_id=run_id, predictions=list(predictions))


@app.get("/audit/protocol")
def audit_protocol():
    protocol_run = mc.get_unique_protocol_run()
    if protocol_run is None:
        raise HTTPException(status_code=409, detail="protocol_not_unique")

    params = protocol_run.data.params
    return {
        "protocol_run_id": protocol_run.info.run_id,
        "dataset_id": params.get("dataset_id"),
        "dataset_revision": params.get("dataset_revision"),
        "sampling_strategy": params.get("sampling_strategy"),
        "sample_size": int(params.get("sample_size", 0)),
        "random_seed": int(params.get("random_seed", 0)),
        "cv_strategy": params.get("cv_strategy"),
        "cv_folds": int(params.get("cv_folds", 0)),
        "cv_shuffle": params.get("cv_shuffle") in ("True", "true", True),
        "partitions_artifact": "protocol/partitions.csv",
        "members_artifact": "protocol/members.csv",
    }


@app.get("/audit/runs")
def audit_runs():
    runs = mc.list_presented_runs()
    result = []
    for run in sorted(runs, key=lambda r: r.info.run_id):
        artifacts = mc.list_artifacts_recursive(run.info.run_id)
        run_type = run.data.tags.get("lab_run_type")
        configuration = None
        if run_type in ("experiment", "final"):
            configuration = mc.read_run_configuration(run.info.run_id, artifacts)

        result.append({
            "run_id": run.info.run_id,
            "status": run.info.status,
            "run_type": run_type,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "artifacts": artifacts,
            "configuration": configuration,
        })

    return {"runs": result}


@app.get("/audit/contributions")
def audit_contributions():
    runs = mc.list_presented_runs()
    experiment_runs = [r for r in runs if r.data.tags.get("lab_run_type") == "experiment"]

    protocol_run = mc.get_unique_protocol_run()
    known_members: dict[str, str] = {}
    if protocol_run is not None:
        try:
            for row in mc.read_members_csv(protocol_run):
                known_members[row["member_id"]] = row["notebook_arn"]
        except mc.MlflowUnavailableError:
            raise
        except Exception:
            known_members = {}

    invalid_run_ids: list[str] = []
    unattributed_run_ids: list[str] = []
    by_member: dict[tuple[str, str], dict] = {}

    for run in experiment_runs:
        run_id = run.info.run_id
        member_id = run.data.tags.get("lab_member_id")
        notebook_arn = run.data.tags.get("notebook_arn")
        valid = mc.is_valid_experiment_run(run)

        attributable = (
            member_id in known_members and known_members[member_id] == notebook_arn
        ) if known_members else bool(member_id and notebook_arn)

        if not valid:
            invalid_run_ids.append(run_id)
            continue

        if not attributable:
            unattributed_run_ids.append(run_id)
            invalid_run_ids.append(run_id)
            continue

        key = (member_id, notebook_arn)
        entry = by_member.setdefault(key, {
            "member_id": member_id,
            "notebook_arn": notebook_arn,
            "run_ids": [],
            "counted_run_ids": [],
            "configuration_ids": set(),
            "stages": set(),
        })
        entry["run_ids"].append(run_id)

        experiment_id = run.data.tags.get("lab_experiment_id")
        if experiment_id not in ("T0", "B0"):
            entry["counted_run_ids"].append(run_id)
            config_id = run.data.tags.get("lab_configuration_id")
            if config_id:
                entry["configuration_ids"].add(config_id)
            stage = run.data.tags.get("lab_stage")
            if stage:
                entry["stages"].add(stage)

    members = []
    for entry in by_member.values():
        members.append({
            "member_id": entry["member_id"],
            "notebook_arn": entry["notebook_arn"],
            "run_ids": sorted(entry["run_ids"]),
            "counted_run_ids": sorted(entry["counted_run_ids"]),
            "configuration_ids": sorted(entry["configuration_ids"]),
            "valid_configurations": len(entry["configuration_ids"]),
            "stages": sorted(entry["stages"]),
        })
    members.sort(key=lambda m: m["member_id"])

    return {
        "members": members,
        "invalid_run_ids": sorted(set(invalid_run_ids)),
        "unattributed_run_ids": sorted(set(unattributed_run_ids)),
    }


@app.get("/audit/model")
def audit_model():
    version = mc.resolve_champion_model_version()
    if version is None:
        raise HTTPException(status_code=404, detail="champion_not_found")

    run = mlflow.get_run(version.run_id)
    if not mc.is_valid_final_run(run):
        raise HTTPException(status_code=409, detail="champion_invalid")

    artifacts = mc.list_artifacts_recursive(run.info.run_id)
    configuration = mc.read_run_configuration(run.info.run_id, artifacts)
    if configuration is None:
        raise HTTPException(status_code=409, detail="champion_invalid")

    return {
        "model_name": mc.MODEL_NAME,
        "alias": mc.MODEL_ALIAS,
        "version": int(version.version),
        "run_id": run.info.run_id,
        "protocol_run_id": run.data.tags.get("lab_protocol_run_id"),
        "selected_experiment_run_id": run.data.tags.get("lab_selected_experiment_run_id"),
        "configuration_id": run.data.tags.get("lab_configuration_id"),
        "configuration": configuration,
        "training_size": int(run.data.params.get("training_size", 0)),
        "test_macro_f1": float(run.data.metrics.get("test_macro_f1", 0.0)),
    }


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        model, run_id = _load_champion_model()
    except mc.MlflowUnavailableError:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_run_id": None})

    if model is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_run_id": None})

    try:
        model.predict(["health check"])
    except Exception:
        logger.exception("Fallo de sanity-check de inferencia en /health")
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_run_id": None})

    return {"status": "ok", "model_run_id": run_id}
