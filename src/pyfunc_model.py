"""
Wrapper mlflow.pyfunc para el modelo final (Sección 5, A.5): recibe texto
crudo y devuelve la etiqueta de sentimiento como string ("negative" /
"positive"), no la clase numérica 0/1.

Serializa dentro del pyfunc: la configuration.json (para reconstruir el
preprocesador en load_context, evitando picklear objetos spaCy pesados),
la representación YA AJUSTADA (vectorizador o embebedder) y el clasificador
entrenado. code_path=["src"] en mlflow.pyfunc.log_model asegura que estos
módulos estén disponibles al cargar el modelo en otro proceso/máquina.
"""

from __future__ import annotations
import json
import pickle

import pandas as pd
import mlflow.pyfunc

_LABEL_NAMES = {0: "negative", 1: "positive"}


class SentimentPipelineModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from .preprocessing import build_preprocessor

        with open(context.artifacts["configuration"], "r", encoding="utf-8") as f:
            self._configuration = json.load(f)

        with open(context.artifacts["representation"], "rb") as f:
            self._representation = pickle.load(f)

        with open(context.artifacts["classifier"], "rb") as f:
            self._classifier = pickle.load(f)

        self._preprocess = build_preprocessor(self._configuration["preprocessing"])

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = list(model_input)

        processed = self._preprocess(texts)
        X = self._representation.transform(processed)
        y_pred = self._classifier.predict(X)
        return [_LABEL_NAMES[int(label)] for label in y_pred]
