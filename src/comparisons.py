"""
Comparaciones obligatorias (Sección 3, Anexo A.4).

Construye la configuration.json de cada tag P_*/R_*/C_*/EXTRA a partir de
config_schema.b0_config() (etapa preprocessing) o de la configuración ya
seleccionada por el equipo en la etapa anterior (etapas representation y
classifier), la evalúa con los mismos 3 folds del protocolo, y la registra
como run experimental.

lab_configuration_id NO se genera automáticamente: lo decide el equipo
(A.1 — "Identificador definido por el equipo para una configuración
efectiva"). Dos runs con configuración JSON equivalente DEBEN compartir el
mismo lab_configuration_id; pásenlo siempre por --configuration-id y
coordinen entre integrantes para no duplicar identificadores por error.

Uso (ejemplos):
    python -m src.comparisons --tag P_STOPWORDS \
        --configuration-id CFG_001 --member-id E01

    python -m src.comparisons --tag R_TFIDF_UNI \
        --selected-preprocessing-config selected/preprocessing.json \
        --configuration-id CFG_004 --member-id E01

    python -m src.comparisons --tag C_LINEAR_SVM \
        --selected-preprocessing-config selected/preprocessing.json \
        --selected-representation-config selected/representation.json \
        --configuration-id CFG_007 --member-id E01

    python -m src.comparisons --tag EXTRA --lab-stage classifier \
        --selected-preprocessing-config selected/preprocessing.json \
        --selected-representation-config selected/representation.json \
        --classifier-type sgd --classifier-params '{"alpha": 0.001}' \
        --configuration-id CFG_EXTRA_1 --member-id E01
"""

from __future__ import annotations
import argparse
import json
import os

from .mlflow_utils import init_experiment
from .mlflow_logging import log_experiment_run
from .config_schema import (
    b0_config,
    preprocessing_config,
    representation_config,
    classifier_config,
    full_config,
)
from .classifiers import sklearn_version
from .data import load_sample_with_text
from .cv_eval import run_configuration

PREPROCESSING_TAGS = {
    "P_STOPWORDS": {"stopwords": "remove", "negators": []},
    "P_STOPWORDS_NEGATION": {
        "stopwords": "remove_preserve_negation",
        "negators": [
            "no", "not", "never", "n't", "nobody", "nothing",
            "neither", "nor", "none", "cannot", "without",
        ],
    },
    "P_LEMMA": {"lemmatize": True},
    "P_ELONGATION": {"elongation": "normalize", "elongation_spec": "reduce_repeated_chars_to_2"},
    "P_EMOJI": {"emoji": "text", "emoji_spec": "emoji.demojize"},
}

REPRESENTATION_TAGS = {
    "R_BOW": dict(rtype="bow", ngram_range=[1, 1]),
    "R_TFIDF_UNI": dict(rtype="tfidf", ngram_range=[1, 1]),
    "R_TFIDF_UNI_BI": dict(rtype="tfidf", ngram_range=[1, 2]),
    "R_SPACY": dict(rtype="spacy_embedding"),
}

CLASSIFIER_TAGS = {
    "C_LOGREG": "logistic_regression",
    "C_LINEAR_SVM": "linear_svm",
    "C_SGD": "sgd",
}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_preprocessing_tag_config(tag: str) -> dict:
    """Etapa preprocessing: igual a B0 salvo el campo que exige A.4 para
    ese tag exacto."""
    base = b0_config(sklearn_version())
    overrides = PREPROCESSING_TAGS[tag]
    pre = dict(base["preprocessing"])
    pre.update(overrides)
    return full_config(
        preprocessing=preprocessing_config(**pre),
        representation=base["representation"],
        classifier=base["classifier"],
    )


def build_representation_tag_config(tag: str, selected_preprocessing: dict, spacy_model: str | None) -> dict:
    """Etapa representation: preprocesamiento seleccionado + tipo de
    representación del tag + Logistic Regression (fija, A.4)."""
    from .representations import SPACY_EMBEDDING_MODEL, DOCUMENT_VECTOR_METHOD
    import spacy as spacy_lib

    spec = REPRESENTATION_TAGS[tag]
    if spec["rtype"] == "spacy_embedding":
        model_name = spacy_model or SPACY_EMBEDDING_MODEL
        rep = representation_config(
            rtype="spacy_embedding",
            library="spacy",
            library_version=spacy_lib.__version__,
            spacy_model=model_name,
            spacy_model_version=spacy_lib.load(model_name).meta.get("version", "unknown"),
            parameters={"document_vector_method": DOCUMENT_VECTOR_METHOD},
        )
    else:
        rep = representation_config(
            rtype=spec["rtype"],
            library="scikit-learn",
            library_version=sklearn_version(),
            ngram_range=spec["ngram_range"],
        )

    clf = classifier_config("logistic_regression", "scikit-learn", sklearn_version(), parameters={})
    return full_config(preprocessing=selected_preprocessing, representation=rep, classifier=clf)


def build_classifier_tag_config(tag: str, selected_preprocessing: dict, selected_representation: dict) -> dict:
    """Etapa classifier: preprocesamiento + representación seleccionados +
    tipo de clasificador del tag (A.4)."""
    ctype = CLASSIFIER_TAGS[tag]
    clf = classifier_config(ctype, "scikit-learn", sklearn_version(), parameters={})
    return full_config(
        preprocessing=selected_preprocessing,
        representation=selected_representation,
        classifier=clf,
    )


def build_extra_config(
    lab_stage: str,
    selected_preprocessing: dict | None,
    selected_representation: dict | None,
    classifier_type: str | None,
    classifier_params: dict | None,
) -> dict:
    """EXTRA: comparación adicional del equipo, no sustituye ninguna
    obligatoria (A.4). El equipo controla qué campo varía; aquí solo se
    exige que el classifier.type sea uno de los controlados por A.3."""
    clf_type = classifier_type or "logistic_regression"
    clf = classifier_config(clf_type, "scikit-learn", sklearn_version(), parameters=classifier_params or {})
    return full_config(
        preprocessing=selected_preprocessing or b0_config(sklearn_version())["preprocessing"],
        representation=selected_representation or b0_config(sklearn_version())["representation"],
        classifier=clf,
    )


def resolve_configuration(args: argparse.Namespace) -> tuple[dict, str]:
    """Devuelve (configuration, lab_stage) para el --tag pedido."""
    tag = args.tag

    if tag in PREPROCESSING_TAGS:
        return build_preprocessing_tag_config(tag), "preprocessing"

    if tag in REPRESENTATION_TAGS:
        selected_pre = (
            _load_json(args.selected_preprocessing_config)
            if args.selected_preprocessing_config
            else b0_config(sklearn_version())["preprocessing"]
        )
        return (
            build_representation_tag_config(tag, selected_pre, args.spacy_model),
            "representation",
        )

    if tag in CLASSIFIER_TAGS:
        selected_pre = (
            _load_json(args.selected_preprocessing_config)
            if args.selected_preprocessing_config
            else b0_config(sklearn_version())["preprocessing"]
        )
        selected_rep = (
            _load_json(args.selected_representation_config)
            if args.selected_representation_config
            else b0_config(sklearn_version())["representation"]
        )
        return build_classifier_tag_config(tag, selected_pre, selected_rep), "classifier"

    if tag == "EXTRA":
        if not args.lab_stage:
            raise SystemExit("--lab-stage es obligatorio para --tag EXTRA "
                              "(preprocessing|representation|classifier|ablation).")
        selected_pre = _load_json(args.selected_preprocessing_config) if args.selected_preprocessing_config else None
        selected_rep = _load_json(args.selected_representation_config) if args.selected_representation_config else None
        classifier_params = json.loads(args.classifier_params) if args.classifier_params else None
        return (
            build_extra_config(args.lab_stage, selected_pre, selected_rep, args.classifier_type, classifier_params),
            args.lab_stage,
        )

    raise SystemExit(f"--tag desconocido: {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True,
                         help="lab_experiment_id: P_*, R_*, C_* o EXTRA.")
    parser.add_argument("--configuration-id", required=True, dest="configuration_id")
    parser.add_argument("--member-id", required=True, dest="member_id")
    parser.add_argument("--protocol-run-id", dest="protocol_run_id",
                         default=os.environ.get("LAB_PROTOCOL_RUN_ID"))
    parser.add_argument("--partitions-path", dest="partitions_path",
                         default=os.environ.get("LAB_PARTITIONS_PATH", "protocol_artifacts/partitions.csv"))
    parser.add_argument("--selected-preprocessing-config", dest="selected_preprocessing_config",
                         help="Path a JSON con el preprocessing_config ya seleccionado por el equipo "
                              "(requerido para etapas representation/classifier; default B0).")
    parser.add_argument("--selected-representation-config", dest="selected_representation_config",
                         help="Path a JSON con el representation_config ya seleccionado por el equipo "
                              "(requerido para etapa classifier; default B0).")
    parser.add_argument("--spacy-model", dest="spacy_model",
                         help="Modelo spaCy con vectores para R_SPACY (default en_core_web_md).")
    parser.add_argument("--lab-stage", dest="lab_stage",
                         choices=["preprocessing", "representation", "classifier", "ablation"],
                         help="Obligatorio solo para --tag EXTRA.")
    parser.add_argument("--classifier-type", dest="classifier_type",
                         choices=["logistic_regression", "linear_svm", "sgd"],
                         help="Solo para --tag EXTRA.")
    parser.add_argument("--classifier-params", dest="classifier_params",
                         help="JSON de hiperparámetros explícitos. Solo para --tag EXTRA.")
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    args = parser.parse_args()

    if not args.protocol_run_id:
        raise SystemExit("Falta --protocol-run-id (o variable LAB_PROTOCOL_RUN_ID).")

    configuration, lab_stage = resolve_configuration(args)

    init_experiment(os.environ.get("MLFLOW_TRACKING_URI"))
    df = load_sample_with_text(args.partitions_path)
    fold_scores = run_configuration(df, configuration)

    out_dir = args.out_dir or f"{args.tag.lower()}_artifacts"
    log_experiment_run(
        lab_experiment_id=args.tag,
        lab_stage=lab_stage,
        lab_protocol_run_id=args.protocol_run_id,
        lab_member_id=args.member_id,
        lab_configuration_id=args.configuration_id,
        fold_scores=fold_scores,
        configuration=configuration,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
