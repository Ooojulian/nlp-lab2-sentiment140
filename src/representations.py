"""
Representación del texto (Sección 3, Anexo A.3/A.4): BoW, TF-IDF, embeddings
de spaCy.

Decisión del equipo: R_SPACY usa el modelo en_core_web_md (tiene vectores
preentrenados de tokens) y document_vector_method="mean_token_vectors":
promedio de los vectores de los tokens con vector propio (si ninguno tiene
vector, vector cero). Es una decisión razonable pero no la única válida;
ajústenla si su equipo prefiere otro modelo/método.
"""

from __future__ import annotations
from typing import Protocol

import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SPACY_EMBEDDING_MODEL = "en_core_web_md"
DOCUMENT_VECTOR_METHOD = "mean_token_vectors"

_nlp_cache: dict[str, object] = {}


class Representation(Protocol):
    library: str
    library_version: str

    def fit_transform(self, texts: list[str]): ...
    def transform(self, texts: list[str]): ...


class SklearnVectorizerRepresentation:
    def __init__(self, vectorizer):
        self._vectorizer = vectorizer
        self.library = "scikit-learn"
        self.library_version = sklearn.__version__

    def fit_transform(self, texts: list[str]):
        return self._vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self._vectorizer.transform(texts)


class SpacyEmbeddingRepresentation:
    """Sin parámetros que ajustar: fit_transform y transform calculan el
    mismo tipo de vector, solo cambia el conjunto de textos."""

    def __init__(self, model_name: str = SPACY_EMBEDDING_MODEL, batch_size: int = 256):
        import spacy

        if model_name not in _nlp_cache:
            _nlp_cache[model_name] = spacy.load(model_name)
        self._nlp = _nlp_cache[model_name]
        self._batch_size = batch_size
        self.library = "spacy"
        self.library_version = spacy.__version__
        self.model_name = model_name
        self.model_version = self._nlp.meta.get("version", "unknown")

    def _embed(self, texts: list[str]) -> np.ndarray:
        dim = self._nlp.vocab.vectors_length
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, doc in enumerate(self._nlp.pipe(texts, batch_size=self._batch_size)):
            token_vectors = [tok.vector for tok in doc if tok.has_vector]
            if token_vectors:
                vectors[i] = np.mean(token_vectors, axis=0)
        return vectors

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)


def build_representation(config: dict) -> Representation:
    """Dado un representation_config (dict, ver config_schema.py), devuelve
    un objeto con fit_transform/transform."""
    rtype = config["type"]

    if rtype == "bow":
        ngram_range = tuple(config["ngram_range"])
        return SklearnVectorizerRepresentation(CountVectorizer(ngram_range=ngram_range))

    if rtype == "tfidf":
        ngram_range = tuple(config["ngram_range"])
        return SklearnVectorizerRepresentation(TfidfVectorizer(ngram_range=ngram_range))

    if rtype == "spacy_embedding":
        model_name = config.get("spacy_model") or SPACY_EMBEDDING_MODEL
        return SpacyEmbeddingRepresentation(model_name=model_name)

    raise ValueError(f"representation.type no controlado: {rtype}")
