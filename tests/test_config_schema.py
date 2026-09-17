"""Valida que b0_config/t0_config/revert_decision_to_b0 produzcan
exactamente las formas canónicas del Anexo A.3."""

from __future__ import annotations

from src.config_schema import b0_config, t0_config, revert_decision_to_b0

LIBRARY = "scikit-learn"
VERSION = "1.5.0"


def test_b0_config_forma_canonica():
    assert b0_config(VERSION) == {
        "preprocessing": {
            "lowercase": True,
            "url": "token:url",
            "mention": "token:user",
            "whitespace": "normalize",
            "stopwords": "keep",
            "negators": [],
            "lemmatize": False,
            "elongation": "keep",
            "elongation_spec": None,
            "emoji": "keep",
            "emoji_spec": None,
            "resources": {},
            "additional": {},
        },
        "representation": {
            "type": "bow",
            "ngram_range": [1, 1],
            "library": LIBRARY,
            "library_version": VERSION,
            "spacy_model": None,
            "spacy_model_version": None,
            "parameters": {},
        },
        "classifier": {
            "type": "logistic_regression",
            "library": LIBRARY,
            "library_version": VERSION,
            "parameters": {},
        },
    }


def test_t0_config_forma_canonica():
    assert t0_config(LIBRARY, VERSION) == {
        "preprocessing": None,
        "representation": None,
        "classifier": {
            "type": "most_frequent",
            "library": LIBRARY,
            "library_version": VERSION,
            "parameters": {},
        },
    }


def test_revert_stopwords():
    candidate = b0_config(VERSION)
    candidate["preprocessing"]["stopwords"] = "remove"
    reverted = revert_decision_to_b0(candidate, "preprocessing.stopwords", VERSION)
    assert reverted["preprocessing"]["stopwords"] == "keep"
    assert reverted["preprocessing"]["negators"] == []


def test_revert_lemmatize():
    candidate = b0_config(VERSION)
    candidate["preprocessing"]["lemmatize"] = True
    reverted = revert_decision_to_b0(candidate, "preprocessing.lemmatize", VERSION)
    assert reverted["preprocessing"]["lemmatize"] is False


def test_revert_elongation():
    candidate = b0_config(VERSION)
    candidate["preprocessing"]["elongation"] = "normalize"
    candidate["preprocessing"]["elongation_spec"] = "reduce_repeated_chars_to_2"
    reverted = revert_decision_to_b0(candidate, "preprocessing.elongation", VERSION)
    assert reverted["preprocessing"]["elongation"] == "keep"
    assert reverted["preprocessing"]["elongation_spec"] is None


def test_revert_emoji():
    candidate = b0_config(VERSION)
    candidate["preprocessing"]["emoji"] = "text"
    candidate["preprocessing"]["emoji_spec"] = "emoji.demojize"
    reverted = revert_decision_to_b0(candidate, "preprocessing.emoji", VERSION)
    assert reverted["preprocessing"]["emoji"] == "keep"
    assert reverted["preprocessing"]["emoji_spec"] is None


def test_revert_representation():
    candidate = b0_config(VERSION)
    candidate["representation"] = {
        "type": "tfidf", "ngram_range": [1, 2], "library": LIBRARY,
        "library_version": VERSION, "spacy_model": None,
        "spacy_model_version": None, "parameters": {},
    }
    reverted = revert_decision_to_b0(candidate, "representation", VERSION)
    assert reverted["representation"] == b0_config(VERSION)["representation"]


def test_revert_classifier():
    candidate = b0_config(VERSION)
    candidate["classifier"] = {
        "type": "linear_svm", "library": LIBRARY, "library_version": VERSION, "parameters": {},
    }
    reverted = revert_decision_to_b0(candidate, "classifier", VERSION)
    assert reverted["classifier"] == b0_config(VERSION)["classifier"]


def test_revert_decision_desconocida():
    import pytest
    with pytest.raises(ValueError):
        revert_decision_to_b0(b0_config(VERSION), "no_existe", VERSION)
