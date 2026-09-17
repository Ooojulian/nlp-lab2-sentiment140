"""Modelos Pydantic del contrato de la API (Anexo A.5/A.6)."""

from __future__ import annotations
from typing import Union

from pydantic import BaseModel, field_validator

MAX_BATCH_SIZE = 32
MAX_TEXT_LENGTH = 1000


class PredictRequest(BaseModel):
    text: Union[str, list[str]]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise ValueError("text debe ser un string o una lista de strings")

        if len(items) == 0:
            raise ValueError("text no puede ser una lista vacía")
        if len(items) > MAX_BATCH_SIZE:
            raise ValueError(f"text no puede tener más de {MAX_BATCH_SIZE} elementos")

        for item in items:
            if not isinstance(item, str):
                raise ValueError("todos los elementos de text deben ser strings")
            if item.strip() == "":
                raise ValueError("text no puede ser un string vacío o solo espacios")
            if len(item) > MAX_TEXT_LENGTH:
                raise ValueError(f"cada texto debe tener máximo {MAX_TEXT_LENGTH} caracteres")

        return value

    def as_list(self) -> list[str]:
        return [self.text] if isinstance(self.text, str) else self.text


class PredictResponse(BaseModel):
    model_run_id: str
    predictions: list[str]


class HealthResponse(BaseModel):
    status: str
    model_run_id: str | None
