"""
Análisis de errores del modelo final (Sección 5, Anexo A.6).

La categorización automática es una HEURÍSTICA de arranque basada en
reglas simples sobre el texto (palabras clave/regex). NO es un juicio
lingüístico confiable: el enunciado pide "clasifique... interprete", lo
que exige criterio humano. Antes de entregar, el equipo debe revisar
reports/error_analysis.csv fila por fila y corregir la categoría cuando la
heurística se equivoque, y completar a mano la interpretación en
reports/error_analysis.md (los placeholders [COMPLETAR: ...] no deben
llegar a la entrega).
"""

from __future__ import annotations
import csv
import os
import re

import numpy as np

CATEGORIES = [
    "negation", "intensification", "contrast", "mixed", "emoji",
    "elongation", "informal", "hashtag", "sarcasm", "other",
]

_NEGATION_RE = re.compile(r"\b(no|not|never|n't|nobody|nothing|neither|nor|none|cannot|without)\b", re.I)
_CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet)\b", re.I)
_INTENSIFIER_RE = re.compile(r"\b(very|so|really|extremely|totally|absolutely)\b", re.I)
_INFORMAL_RE = re.compile(r"\b(u|ur|lol|omg|lmao|gonna|wanna|gotta|thx)\b", re.I)
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_SARCASM_HINTS_RE = re.compile(r"\b(yeah right|as if|totally not|great\.\.\.|oh (great|joy))\b", re.I)


def _classify_error(text: str) -> str:
    """Heurística de arranque, ver docstring del módulo."""
    signals = []
    if _EMOJI_RE.search(text):
        signals.append("emoji")
    if _ELONGATION_RE.search(text):
        signals.append("elongation")
    if "#" in text:
        signals.append("hashtag")
    if _SARCASM_HINTS_RE.search(text):
        signals.append("sarcasm")
    if _NEGATION_RE.search(text):
        signals.append("negation")
    if _CONTRAST_RE.search(text):
        signals.append("contrast")
    if _INTENSIFIER_RE.search(text):
        signals.append("intensification")
    if _INFORMAL_RE.search(text) or (text and text == text.lower() and len(text.split()) > 3):
        signals.append("informal")

    if not signals:
        return "other"
    if len(signals) >= 2:
        return "mixed"
    return signals[0]


def _label_name(label: int) -> str:
    return "negative" if label == 0 else "positive"


def select_errors(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 42, min_errors: int = 20) -> np.ndarray:
    """Índices (posiciones dentro de y_true/y_pred) de al menos min_errors
    errores, incluyendo ambas clases reales cuando existan (A.6/Sección 5)."""
    rng = np.random.default_rng(seed)
    error_positions = np.flatnonzero(y_true != y_pred)
    if len(error_positions) == 0:
        return error_positions

    by_class = {
        cls: error_positions[y_true[error_positions] == cls]
        for cls in np.unique(y_true[error_positions])
    }

    chosen: list[int] = []
    for cls_positions in by_class.values():
        if len(cls_positions) > 0:
            chosen.append(int(rng.choice(cls_positions)))

    remaining = np.setdiff1d(error_positions, np.array(chosen, dtype=error_positions.dtype))
    n_more = max(min_errors - len(chosen), 0)
    n_more = min(n_more, len(remaining))
    if n_more > 0:
        extra = rng.choice(remaining, size=n_more, replace=False)
        chosen.extend(int(x) for x in extra)

    return np.array(sorted(set(chosen)), dtype=error_positions.dtype)


def generate_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts,
    indices,
    out_dir: str,
    seed: int = 42,
    min_errors: int = 20,
) -> tuple[str, str]:
    """Genera reports/error_analysis.csv y reports/error_analysis.md.
    indices: posición original en test (A.6) para cada fila de y_true/y_pred/texts.
    Devuelve (csv_path, md_path)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    texts = list(texts)
    indices = list(indices)

    error_positions = select_errors(y_true, y_pred, seed=seed, min_errors=min_errors)

    rows = []
    category_counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    for pos in error_positions:
        category = _classify_error(texts[pos])
        category_counts[category] += 1
        rows.append({
            "index": indices[pos],
            "text": texts[pos],
            "true_label": _label_name(int(y_true[pos])),
            "predicted_label": _label_name(int(y_pred[pos])),
            "category": category,
        })
    rows.sort(key=lambda r: r["index"])

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "error_analysis.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "text", "true_label", "predicted_label", "category"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(out_dir, "error_analysis.md")
    _write_markdown(md_path, category_counts, len(rows))

    return csv_path, md_path


def _write_markdown(md_path: str, category_counts: dict[str, int], total_errors: int) -> None:
    nonzero = {c: n for c, n in category_counts.items() if n > 0}
    ranked = sorted(nonzero.items(), key=lambda kv: kv[1], reverse=True)

    lines = [
        "# Análisis de errores — modelo final",
        "",
        f"Total de errores analizados: {total_errors} (semilla 42).",
        "",
        "## Frecuencia por categoría",
        "",
        "| category | count |",
        "|---|---|",
    ]
    for category in CATEGORIES:
        lines.append(f"| {category} | {category_counts.get(category, 0)} |")

    lines += ["", "## Interpretación", ""]
    if not ranked:
        lines.append("No se encontraron errores en la muestra evaluada.")
    elif len(ranked) == 1:
        (only_cat, _n) = ranked[0]
        lines.append(
            f"Todos los errores caen en la categoría **{only_cat}**. "
            "[COMPLETAR: interpretación basada en los casos reales de esta categoría]."
        )
    else:
        top_two = ranked[:2]
        for category, count in top_two:
            lines.append(
                f"- **{category}** ({count} casos): "
                "[COMPLETAR: interpretación basada en los casos reales de esta categoría]."
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
