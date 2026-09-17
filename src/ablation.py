"""
Análisis de ablación (Sección 4, Anexo A.4 bloque ABLATION).

Lee la configuración efectiva del run "padre" (pipeline candidato o una
ablación previa), revierte UNA decisión ablacionable a la forma de B0
(config_schema.revert_decision_to_b0), reevalúa con los mismos 3 folds, y
registra el run con lab_experiment_id=ABLATION, lab_stage=ablation, más los
campos exclusivos de ablación: lab_ablation_parent_run_id,
ablation_reverted_decision, macro_f1_delta = mean(parent) - mean(ablation).

Uso:
    python -m src.ablation --decision preprocessing.stopwords \
        --parent-run-id <run_id_candidato> \
        --configuration-id CFG_ABL_1 --member-id E01
"""

from __future__ import annotations
import argparse
import os

import mlflow

from .mlflow_utils import init_experiment
from .mlflow_logging import log_experiment_run, load_run_configuration
from .config_schema import revert_decision_to_b0
from .classifiers import sklearn_version
from .data import load_sample_with_text
from .cv_eval import run_configuration

VALID_DECISIONS = {
    "preprocessing.stopwords",
    "preprocessing.lemmatize",
    "preprocessing.elongation",
    "preprocessing.emoji",
    "representation",
    "classifier",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, choices=sorted(VALID_DECISIONS))
    parser.add_argument("--parent-run-id", required=True, dest="parent_run_id")
    parser.add_argument("--configuration-id", required=True, dest="configuration_id")
    parser.add_argument("--member-id", required=True, dest="member_id")
    parser.add_argument("--protocol-run-id", dest="protocol_run_id",
                         default=os.environ.get("LAB_PROTOCOL_RUN_ID"))
    parser.add_argument("--partitions-path", dest="partitions_path",
                         default=os.environ.get("LAB_PARTITIONS_PATH", "protocol_artifacts/partitions.csv"))
    parser.add_argument("--out-dir", dest="out_dir", default="ablation_artifacts")
    args = parser.parse_args()

    if not args.protocol_run_id:
        raise SystemExit("Falta --protocol-run-id (o variable LAB_PROTOCOL_RUN_ID).")

    init_experiment(os.environ.get("MLFLOW_TRACKING_URI"))

    parent_run = mlflow.get_run(args.parent_run_id)
    parent_mean = parent_run.data.metrics.get("macro_f1_mean")
    if parent_mean is None:
        raise SystemExit(f"El run padre {args.parent_run_id} no tiene macro_f1_mean registrado.")

    parent_config = load_run_configuration(args.parent_run_id)
    ablated_config = revert_decision_to_b0(parent_config, args.decision, sklearn_version())

    df = load_sample_with_text(args.partitions_path)
    fold_scores = run_configuration(df, ablated_config)
    ablation_mean = sum(fold_scores) / len(fold_scores)
    macro_f1_delta = parent_mean - ablation_mean

    log_experiment_run(
        lab_experiment_id="ABLATION",
        lab_stage="ablation",
        lab_protocol_run_id=args.protocol_run_id,
        lab_member_id=args.member_id,
        lab_configuration_id=args.configuration_id,
        fold_scores=fold_scores,
        configuration=ablated_config,
        out_dir=args.out_dir,
        extra_tags={"lab_ablation_parent_run_id": args.parent_run_id},
        extra_params={"ablation_reverted_decision": args.decision},
        extra_metrics={"macro_f1_delta": macro_f1_delta},
    )


if __name__ == "__main__":
    main()
