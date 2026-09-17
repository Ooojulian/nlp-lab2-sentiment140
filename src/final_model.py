"""
Modelo final (Sección 5, Anexo A.2/A.5): reentrena la configuración
seleccionada con TODO train (1.360.000 registros), evalúa UNA sola vez
sobre TODO test (240.000 registros), registra el run final y publica el
modelo en el Model Registry como sentiment140@champion.

No ajusta la configuración a partir del resultado de test (Sección 5): el
run final SOLO reentrena y evalúa; la selección ya ocurrió en comparisons.py
/ablation.py usando validación cruzada.

Uso:
    python -m src.final_model --selected-experiment-run-id <run_id> \
        --configuration-id CFG_XXX --member-id E01
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import tempfile

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

from .mlflow_utils import init_experiment, read_sagemaker_provenance, EXPERIMENT_NAME
from .mlflow_logging import load_run_configuration
from .data import LABEL_COL, TEXT_COL, load_train, load_test
from .preprocessing import build_preprocessor
from .representations import build_representation
from .classifiers import build_classifier
from .error_analysis import generate_error_analysis
from .pyfunc_model import SentimentPipelineModel

MODEL_NAME = "sentiment140"
MODEL_ALIAS = "champion"
TRAINING_SIZE = 1_360_000


def _texts_labels(split, text_col: str, label_col: str):
    df = split.to_pandas()
    return df[text_col].tolist(), df[label_col].to_numpy(), np.arange(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-experiment-run-id", required=True, dest="selected_experiment_run_id")
    parser.add_argument("--configuration-id", required=True, dest="configuration_id")
    parser.add_argument("--member-id", required=True, dest="member_id")
    parser.add_argument("--protocol-run-id", dest="protocol_run_id",
                         default=os.environ.get("LAB_PROTOCOL_RUN_ID"))
    parser.add_argument("--out-dir", dest="out_dir", default="final_artifacts")
    args = parser.parse_args()

    if not args.protocol_run_id:
        raise SystemExit("Falta --protocol-run-id (o variable LAB_PROTOCOL_RUN_ID).")

    init_experiment(os.environ.get("MLFLOW_TRACKING_URI"))

    configuration = load_run_configuration(args.selected_experiment_run_id)

    print("Cargando train completo (1.360.000 registros)...")
    train_texts, train_labels, _ = _texts_labels(load_train(), TEXT_COL, LABEL_COL)
    print("Cargando test completo (240.000 registros)...")
    test_texts, test_labels, test_indices = _texts_labels(load_test(), TEXT_COL, LABEL_COL)

    preprocess = build_preprocessor(configuration["preprocessing"])
    train_processed = preprocess(train_texts)
    test_processed = preprocess(test_texts)

    representation = build_representation(configuration["representation"])
    X_train = representation.fit_transform(train_processed)
    X_test = representation.transform(test_processed)

    classifier = build_classifier(configuration["classifier"])
    classifier.fit(X_train, train_labels)
    y_pred = classifier.predict(X_test)

    test_macro_f1 = float(f1_score(test_labels, y_pred, average="macro"))
    print(f"test_macro_f1={test_macro_f1:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    config_path = os.path.join(args.out_dir, "configuration.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(configuration, f, indent=2)

    _, notebook_arn = read_sagemaker_provenance(args.out_dir)
    provenance_path = os.path.join(args.out_dir, "provenance", "sagemaker-resource-metadata.json")

    csv_path, md_path = generate_error_analysis(
        y_true=test_labels,
        y_pred=y_pred,
        texts=test_texts,
        indices=test_indices,
        out_dir="reports",
    )

    with tempfile.TemporaryDirectory() as tmp:
        rep_path = os.path.join(tmp, "representation.pkl")
        clf_path = os.path.join(tmp, "classifier.pkl")
        with open(rep_path, "wb") as f:
            pickle.dump(representation, f)
        with open(clf_path, "wb") as f:
            pickle.dump(classifier, f)

        with mlflow.start_run(run_name="final") as run:
            mlflow.set_tag("lab_run_type", "final")
            mlflow.set_tag("lab_protocol_run_id", args.protocol_run_id)
            mlflow.set_tag("lab_selected_experiment_run_id", args.selected_experiment_run_id)
            mlflow.set_tag("lab_configuration_id", args.configuration_id)
            mlflow.set_tag("lab_member_id", args.member_id)
            mlflow.set_tag("notebook_arn", notebook_arn)

            mlflow.log_param("training_size", TRAINING_SIZE)
            mlflow.log_metric("test_macro_f1", test_macro_f1)

            mlflow.log_artifact(config_path, artifact_path="run")
            mlflow.log_artifact(provenance_path, artifact_path="provenance")
            mlflow.log_artifact(csv_path, artifact_path="reports")
            mlflow.log_artifact(md_path, artifact_path="reports")

            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SentimentPipelineModel(),
                artifacts={
                    "configuration": config_path,
                    "representation": rep_path,
                    "classifier": clf_path,
                },
                code_path=["src"],
            )

            final_run_id = run.info.run_id
            print(f"final run_id={final_run_id}")

    client = MlflowClient()
    registered = mlflow.register_model(model_info.model_uri, MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, registered.version)
    print(f"{MODEL_NAME}@{MODEL_ALIAS} -> version {registered.version} (run_id={final_run_id})")


if __name__ == "__main__":
    main()
