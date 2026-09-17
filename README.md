# nlp-lab2-sentiment140

Universidad Sergio Arboleda — Procesamiento de Lenguaje Natural, Laboratorio II 2026 S02.
Análisis binario de sentimientos sobre `adilbekovich/Sentiment140Twitter` (revisión
`b6037e127257d95b9b23d31f78b264b9ebe697fd`), con MLflow como registro oficial de
experimentación y modelo, y una API FastAPI para inferencia/auditoría.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # stopwords/lemmatización (P_STOPWORDS*, P_LEMMA)
python -m spacy download en_core_web_md   # vectores preentrenados (R_SPACY)
```

Los modelos de spaCy NO están en `requirements.txt` (no son paquetes pip normales):
se descargan aparte con `spacy download`, tal como se indica arriba.

Variables de entorno usadas por los scripts:

| Variable | Uso |
|---|---|
| `MLFLOW_TRACKING_URI` | URL del Tracking Server. Sin ella, cae a `./mlruns` local — **no válido para la entrega final** (Sección 6/9). |
| `LAB_PROTOCOL_RUN_ID` | `run_id` del run de protocolo (salida de `protocol.py`). Requerido por `baselines.py`, `comparisons.py`, `ablation.py`, `final_model.py`. |
| `LAB_MEMBER_ID` | `member_id` del integrante que ejecuta el script (default `E01`). |
| `LAB_PARTITIONS_PATH` | Ruta local a `partitions.csv` (default `protocol_artifacts/partitions.csv`). |

## Flujo completo

### 1. Inspeccionar el dataset (una vez, cualquier integrante)

```python
from src.data import inspect_dataset
inspect_dataset()
```

Verifica los nombres reales de columnas. Si no son `text`/`label`, ajusta
`TEXT_COL`/`LABEL_COL` en `src/data.py`.

### 2. Run de protocolo (una sola vez para todo el equipo)

Edita la lista `members` al final de `src/protocol.py` con los `member_id`/ARN
reales de SageMaker de **todo el equipo** antes de correrlo en serio.

```bash
python -m src.protocol
```

Imprime `Protocol run_id = ...` — guárdalo como `LAB_PROTOCOL_RUN_ID`. **No se
vuelve a correr** una vez generado (A.1: "único run que fija el contexto común").

### 3. T0 y B0

```bash
export LAB_PROTOCOL_RUN_ID=<run_id del paso 2>
export LAB_MEMBER_ID=E01
python -m src.baselines
```

Debe ejecutarse **dentro de la SageMaker Notebook Instance real** de cada
integrante — `read_sagemaker_provenance` falla a propósito si no existe
`/opt/ml/metadata/resource-metadata.json` (A.2).

### 4. Comparaciones obligatorias

Etapa **preprocessing** (fija: BoW unigramas + LogReg), un tag por comparación:

```bash
python -m src.comparisons --tag P_STOPWORDS --configuration-id CFG_001 --member-id E01
python -m src.comparisons --tag P_STOPWORDS_NEGATION --configuration-id CFG_002 --member-id E01
python -m src.comparisons --tag P_LEMMA --configuration-id CFG_003 --member-id E02
python -m src.comparisons --tag P_ELONGATION --configuration-id CFG_004 --member-id E02
python -m src.comparisons --tag P_EMOJI --configuration-id CFG_005 --member-id E02
```

Después de comparar, el equipo selecciona el preprocesamiento para la etapa
siguiente (puede ser B0 o cualquiera de las anteriores) y lo guarda como JSON,
por ejemplo `selected/preprocessing.json` (el `preprocessing` de la
`configuration.json` del run elegido, tal cual quedó registrado en MLflow).

Etapa **representation** (fija: preprocesamiento seleccionado + LogReg):

```bash
python -m src.comparisons --tag R_BOW \
  --selected-preprocessing-config selected/preprocessing.json \
  --configuration-id CFG_006 --member-id E01

python -m src.comparisons --tag R_TFIDF_UNI \
  --selected-preprocessing-config selected/preprocessing.json \
  --configuration-id CFG_007 --member-id E01

python -m src.comparisons --tag R_TFIDF_UNI_BI \
  --selected-preprocessing-config selected/preprocessing.json \
  --configuration-id CFG_008 --member-id E01

python -m src.comparisons --tag R_SPACY \
  --selected-preprocessing-config selected/preprocessing.json \
  --configuration-id CFG_009 --member-id E01
```

Guarden la representación elegida como `selected/representation.json`.

Etapa **classifier** (fija: preprocesamiento + representación seleccionados):

```bash
python -m src.comparisons --tag C_LOGREG \
  --selected-preprocessing-config selected/preprocessing.json \
  --selected-representation-config selected/representation.json \
  --configuration-id CFG_010 --member-id E02

python -m src.comparisons --tag C_LINEAR_SVM \
  --selected-preprocessing-config selected/preprocessing.json \
  --selected-representation-config selected/representation.json \
  --configuration-id CFG_011 --member-id E02

python -m src.comparisons --tag C_SGD \
  --selected-preprocessing-config selected/preprocessing.json \
  --selected-representation-config selected/representation.json \
  --configuration-id CFG_012 --member-id E02
```

**Importante sobre `--configuration-id`**: NO se genera automáticamente. El
equipo lo decide (A.1) y dos runs con `configuration.json` equivalente DEBEN
compartir el mismo id. Coordinen entre integrantes antes de correr scripts para
no duplicar identificadores sin querer.

**Contribución individual** (Sección 3): cada integrante necesita al menos 3
configuraciones distintas contadas (T0/B0 no cuentan), distribuidas en al
menos 2 etapas. Si falta, usen `EXTRA`:

```bash
python -m src.comparisons --tag EXTRA --lab-stage classifier \
  --selected-preprocessing-config selected/preprocessing.json \
  --selected-representation-config selected/representation.json \
  --classifier-type sgd --classifier-params '{"alpha": 0.001}' \
  --configuration-id CFG_EXTRA_1 --member-id E03
```

### 5. Selección y ablación

Seleccionen el **pipeline candidato** con base en las comparaciones (CV, nunca
`test`). Reviertan una a la vez las decisiones que difieren de B0:

```bash
python -m src.ablation --decision preprocessing.stopwords \
  --parent-run-id <run_id del candidato> \
  --configuration-id CFG_ABL_1 --member-id E01

python -m src.ablation --decision representation \
  --parent-run-id <run_id del candidato> \
  --configuration-id CFG_ABL_2 --member-id E01
```

Decisiones ablacionables válidas (A.4): `preprocessing.stopwords`,
`preprocessing.lemmatize`, `preprocessing.elongation`, `preprocessing.emoji`,
`representation`, `classifier`. Si el candidato no difiere de B0 en ninguna,
no se exige ablación; si difiere en 1, evalúen 1; si difiere en 2+, evalúen
al menos 2 (Sección 4).

### 6. Modelo final

```bash
python -m src.final_model --selected-experiment-run-id <run_id_candidato_final> \
  --configuration-id CFG_XXX --member-id E01
```

Reentrena con los 1.360.000 registros de `train`, evalúa una sola vez sobre los
240.000 de `test`, genera `reports/error_analysis.csv`/`.md`, registra el run
final y publica `sentiment140@champion` en el Model Registry.

**Revisen manualmente** `reports/error_analysis.md`: la categorización y la
interpretación que genera el script son un punto de partida heurístico, no el
resultado final que se entrega (ver docstring de `src/error_analysis.py`).

### 7. API

```bash
export MLFLOW_TRACKING_URI=<url del tracking server>
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints: `POST /api/v1/predict`, `GET /audit/protocol`, `GET /audit/runs`,
`GET /audit/contributions`, `GET /audit/model`, `GET /health` (contrato exacto
en Anexo A.5/A.6 de la guía).

### 8. Sustentación

`notebooks/experiment_audit.ipynb` reconstruye protocolo, runs y contribuciones
directamente desde el Tracking Server.

## Decisiones del equipo

La guía deja explícitamente abiertas estas decisiones (Sección 1):

- **Negadores preservados** (`P_STOPWORDS_NEGATION`): `no, not, never, n't,
  nobody, nothing, neither, nor, none, cannot, without` — ver
  `src/preprocessing.py::NEGATORS`.
- **Alargamientos** (`elongation=normalize`): reduce cualquier carácter
  repetido 3+ veces a 2 (`"soooo"` → `"soo"`), `elongation_spec =
  "reduce_repeated_chars_to_2"`.
- **Emojis** (`emoji=text`): `emoji.demojize`, `emoji_spec = "emoji.demojize"`.
- **Modelo spaCy para stopwords/lemma**: `en_core_web_sm`.
- **Modelo spaCy para `R_SPACY`**: `en_core_web_md` (tiene vectores
  preentrenados de tokens).
- **`document_vector_method` de `R_SPACY`**: `"mean_token_vectors"` — promedio
  de `token.vector` de los tokens con vector propio; vector cero si ninguno
  tiene vector.

Ajusten cualquiera de estas decisiones si su equipo prefiere otra alternativa
razonable; lo importante es que quede reflejada consistentemente en
`configuration.json` (Anexo A.3).

## Herramientas de IA utilizadas

Claude Code (Anthropic) se usó para generar el scaffold completo del proyecto:
protocolo experimental, baselines T0/B0, comparaciones obligatorias, análisis
de ablación, reentrenamiento del modelo final, análisis de errores (heurística
de categorización) y la API FastAPI con sus endpoints de auditoría. El equipo
es responsable de verificar el código, ejecutar los experimentos reales dentro
de las SageMaker Notebook Instances asignadas, revisar/corregir el análisis de
errores generado automáticamente, y validar que la API cumple el contrato
técnico antes de la entrega.

No se compartieron credenciales, tokens ni información sensible de AWS Academy
con la herramienta.

## Pendiente por completar (equipo)

- Reemplazar `<MEMBER_ID>` / `<NOTEBOOK_ARN>` de ejemplo en `src/protocol.py`
  por los datos reales de **todos** los integrantes asignados por el curso.
- Ejecutar todo el flujo dentro de las SageMaker Notebook Instances reales.
- Levantar un MLflow Tracking Server accesible externamente (compatible con
  Model Registry y aliases).
- Decidir en equipo los `lab_configuration_id` para no duplicarlos.
- Completar a mano `reports/error_analysis.md` (los `[COMPLETAR: ...]` no
  deben llegar a la entrega) y revisar `reports/error_analysis.csv` fila por
  fila.
- Desplegar la API en una URL pública accesible durante la ventana de
  evaluación.
