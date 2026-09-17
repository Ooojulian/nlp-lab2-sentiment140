"""
Preprocesamiento configurable (Sección 3, Anexo A.3/A.4).

Decisiones del equipo (la guía las deja abiertas explícitamente):
- Negadores preservados con stopwords=remove_preserve_negation: no, not,
  never, n't, nobody, nothing, neither, nor, none, cannot, without. Lista
  razonable en inglés (Sentiment140 es inglés); ajústenla si su equipo
  decide otra.
- elongation=normalize reduce cualquier carácter repetido 3+ veces a 2
  ("soooo" -> "soo"), documentado como elongation_spec=
  "reduce_repeated_chars_to_2" (forma ilustrativa del Anexo A.3).
- emoji=text usa emoji.demojize (emoji_spec="emoji.demojize").
- Modelo spaCy para stopwords/lemma: en_core_web_sm (no necesita vectores
  para esto; R_SPACY usa un modelo distinto, ver representations.py).
"""

from __future__ import annotations
import re
from typing import Callable

import spacy

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")

SPACY_PIPE_MODEL = "en_core_web_sm"
ELONGATION_SPEC = "reduce_repeated_chars_to_2"
EMOJI_SPEC = "emoji.demojize"

# Decisión del equipo (ver docstring del módulo).
NEGATORS = [
    "no", "not", "never", "n't", "nobody", "nothing",
    "neither", "nor", "none", "cannot", "without",
]

_nlp_cache: dict[str, "spacy.language.Language"] = {}


def _get_spacy_pipeline(model_name: str = SPACY_PIPE_MODEL):
    if model_name not in _nlp_cache:
        _nlp_cache[model_name] = spacy.load(model_name)
    return _nlp_cache[model_name]


def _normalize_elongation(text: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", text)


def _demojize(text: str) -> str:
    import emoji as emoji_lib
    return emoji_lib.demojize(text, delimiters=(" :", ": "))


def build_preprocessor(config: dict | None) -> Callable[[list[str]], list[str]]:
    """Dado un preprocessing_config (dict, ver config_schema.py), devuelve
    una función texts -> texts procesados. config=None (caso T0) devuelve
    la identidad."""
    if config is None:
        return lambda texts: list(texts)

    lowercase = config["lowercase"]
    url_mode = config["url"]
    mention_mode = config["mention"]
    whitespace_mode = config["whitespace"]
    stopwords_mode = config["stopwords"]
    negators = set(config.get("negators") or [])
    lemmatize = config["lemmatize"]
    elongation_mode = config["elongation"]
    emoji_mode = config["emoji"]

    needs_spacy = stopwords_mode != "keep" or lemmatize
    nlp = _get_spacy_pipeline() if needs_spacy else None

    def _token_url_mention(t: str) -> str:
        if url_mode == "drop":
            t = URL_RE.sub("", t)
        elif url_mode.startswith("token:"):
            t = URL_RE.sub(url_mode.split(":", 1)[1], t)
        if mention_mode == "drop":
            t = MENTION_RE.sub("", t)
        elif mention_mode.startswith("token:"):
            t = MENTION_RE.sub(mention_mode.split(":", 1)[1], t)
        return t

    def _spacy_pass(texts: list[str]) -> list[str]:
        out = []
        for doc in nlp.pipe(texts, batch_size=256):
            kept = []
            for tok in doc:
                if stopwords_mode != "keep" and tok.is_stop:
                    if not (stopwords_mode == "remove_preserve_negation" and tok.lower_ in negators):
                        continue
                kept.append(tok.lemma_ if lemmatize else tok.text)
            out.append(" ".join(kept))
        return out

    def preprocess(texts: list[str]) -> list[str]:
        processed = list(texts)

        if lowercase:
            processed = [t.lower() for t in processed]

        processed = [_token_url_mention(t) for t in processed]

        if elongation_mode == "normalize":
            processed = [_normalize_elongation(t) for t in processed]

        if emoji_mode == "text":
            processed = [_demojize(t) for t in processed]

        if needs_spacy:
            processed = _spacy_pass(processed)

        if whitespace_mode == "normalize":
            processed = [WHITESPACE_RE.sub(" ", t).strip() for t in processed]

        return processed

    return preprocess
