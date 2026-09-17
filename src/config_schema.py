"""
Construcción canónica de run/configuration.json (Anexo A.3).

Regla de oro: NADIE en el equipo escribe este JSON a mano en su notebook.
Todos importan estas funciones. Así se garantiza que dos runs con la misma
configuración efectiva produzcan exactamente el mismo diccionario (salvo
orden de claves, que no afecta equivalencia según A.3), y por lo tanto
puedan compartir el mismo configuration_id sin errores humanos.

El orden de LISTAS sí afecta equivalencia (ngram_range, negators) -> se
construyen siempre en el mismo orden.
"""

from __future__ import annotations
from typing import Optional


def preprocessing_config(
    lowercase: bool = True,
    url: str = "token:url",
    mention: str = "token:user",
    whitespace: str = "normalize",
    stopwords: str = "keep",             # keep | remove | remove_preserve_negation
    negators: Optional[list] = None,
    lemmatize: bool = False,
    elongation: str = "keep",            # keep | normalize
    elongation_spec: Optional[str] = None,
    emoji: str = "keep",                 # keep | text
    emoji_spec: Optional[str] = None,
    resources: Optional[dict] = None,
    additional: Optional[dict] = None,
) -> dict:
    # Validaciones defensivas: el Anexo A.3 exige estos campos SIEMPRE presentes,
    # con null exacto donde no aplica. Un campo faltante invalida el run.
    if stopwords == "remove_preserve_negation" and not negators:
        raise ValueError("stopwords=remove_preserve_negation exige negators no vacío (A.3).")
    if stopwords != "remove_preserve_negation" and negators:
        raise ValueError("negators debe ser [] salvo stopwords=remove_preserve_negation (A.3).")
    if elongation == "normalize" and elongation_spec is None:
        raise ValueError("elongation=normalize exige elongation_spec no nulo (A.3).")
    if elongation == "keep" and elongation_spec is not None:
        raise ValueError("elongation=keep exige elongation_spec=null (A.3).")
    if emoji == "text" and emoji_spec is None:
        raise ValueError("emoji=text exige emoji_spec no nulo (A.3).")
    if emoji == "keep" and emoji_spec is not None:
        raise ValueError("emoji=keep exige emoji_spec=null (A.3).")

    return {
        "lowercase": lowercase,
        "url": url,
        "mention": mention,
        "whitespace": whitespace,
        "stopwords": stopwords,
        "negators": negators or [],
        "lemmatize": lemmatize,
        "elongation": elongation,
        "elongation_spec": elongation_spec,
        "emoji": emoji,
        "emoji_spec": emoji_spec,
        "resources": resources or {},
        "additional": additional or {},
    }


def representation_config(
    rtype: str,                          # bow | tfidf | spacy_embedding
    library: str,
    library_version: str,
    ngram_range: Optional[list] = None,
    spacy_model: Optional[str] = None,
    spacy_model_version: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> dict:
    parameters = dict(parameters or {})

    if rtype in ("bow", "tfidf"):
        if not ngram_range or len(ngram_range) != 2:
            raise ValueError("bow/tfidf exigen ngram_range = [min_n, max_n] (A.3).")
        spacy_model = None
        spacy_model_version = None
    elif rtype == "spacy_embedding":
        ngram_range = []
        if not parameters.get("document_vector_method"):
            raise ValueError(
                "representation.type=spacy_embedding exige "
                "parameters.document_vector_method (string no vacío) (A.3)."
            )
        if not spacy_model or not spacy_model_version:
            raise ValueError("spacy_embedding exige spacy_model y spacy_model_version (A.3).")
    else:
        raise ValueError(f"representation.type no controlado: {rtype}")

    return {
        "type": rtype,
        "ngram_range": ngram_range,
        "library": library,
        "library_version": library_version,
        "spacy_model": spacy_model,
        "spacy_model_version": spacy_model_version,
        "parameters": parameters,
    }


def classifier_config(
    ctype: str,                          # logistic_regression | linear_svm | sgd | most_frequent
    library: str,
    library_version: str,
    parameters: Optional[dict] = None,
) -> dict:
    valid = {"logistic_regression", "linear_svm", "sgd", "most_frequent"}
    if ctype not in valid:
        raise ValueError(f"classifier.type no controlado: {ctype}")
    return {
        "type": ctype,
        "library": library,
        "library_version": library_version,
        "parameters": parameters or {},
    }


def full_config(preprocessing: Optional[dict], representation: Optional[dict], classifier: dict) -> dict:
    """Ensambla el JSON de primer nivel exigido por A.3: exactamente
    preprocessing, representation, classifier (o null donde el Anexo lo permite: T0)."""
    return {
        "preprocessing": preprocessing,
        "representation": representation,
        "classifier": classifier,
    }


# ---------------------------------------------------------------------------
# Formas canónicas fijas por la guía (Sección 2 y A.3). NO se deben modificar
# salvo library/library_version, que reflejan lo realmente instalado.
# ---------------------------------------------------------------------------

def t0_config(library: str, library_version: str) -> dict:
    return full_config(
        preprocessing=None,
        representation=None,
        classifier=classifier_config("most_frequent", library, library_version, parameters={}),
    )


def b0_config(sklearn_version: str) -> dict:
    pre = preprocessing_config(
        lowercase=True,
        url="token:url",
        mention="token:user",
        whitespace="normalize",
        stopwords="keep",
        negators=[],
        lemmatize=False,
        elongation="keep",
        elongation_spec=None,
        emoji="keep",
        emoji_spec=None,
        resources={},
        additional={},
    )
    rep = representation_config(
        rtype="bow",
        library="scikit-learn",
        library_version=sklearn_version,
        ngram_range=[1, 1],
    )
    clf = classifier_config(
        ctype="logistic_regression",
        library="scikit-learn",
        library_version=sklearn_version,
        parameters={},  # B0 usa hiperparámetros por defecto -> no se expanden (A.3)
    )
    return full_config(pre, rep, clf)


def revert_decision_to_b0(config: dict, decision: str, sklearn_version: str) -> dict:
    """Implementa A.4: revertir UNA decisión ablacionable a la forma de B0,
    manteniendo todo lo demás del candidato sin cambio. Úsalo en la etapa
    de ablación (todavía no la necesitas para protocolo/T0/B0)."""
    import copy
    cfg = copy.deepcopy(config)

    if decision == "preprocessing.stopwords":
        cfg["preprocessing"]["stopwords"] = "keep"
        cfg["preprocessing"]["negators"] = []
    elif decision == "preprocessing.lemmatize":
        cfg["preprocessing"]["lemmatize"] = False
    elif decision == "preprocessing.elongation":
        cfg["preprocessing"]["elongation"] = "keep"
        cfg["preprocessing"]["elongation_spec"] = None
    elif decision == "preprocessing.emoji":
        cfg["preprocessing"]["emoji"] = "keep"
        cfg["preprocessing"]["emoji_spec"] = None
    elif decision == "representation":
        cfg["representation"] = representation_config(
            "bow", "scikit-learn", sklearn_version, ngram_range=[1, 1]
        )
    elif decision == "classifier":
        cfg["classifier"] = classifier_config(
            "logistic_regression", "scikit-learn", sklearn_version, parameters={}
        )
    else:
        raise ValueError(f"Decisión ablacionable no reconocida: {decision}")

    return cfg
