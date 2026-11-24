# Predicción de aprobación de préstamos de vivienda

**Pipeline completo de datos y modelo predictivo:** EDA → SQLite → ETL → tabla curada → modelo de clasificación.

Prueba de Suficiencia · Administración de Datos · LEAD University
Profesor: Alejandro Zamora · Estudiante: Juan Pablo Cubero Torres

---

## 1. Dataset seleccionado

**Opción B: Home Loan Approval Dataset**
Fuente: https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval

| Archivo | Contenido | Uso en el proyecto |
|---|---|---|
| `data/loan_sanction_train.csv` | 614 solicitudes, 13 columnas, incluye la variable objetivo | **Se usa** en todo el pipeline |
| `data/loan_sanction_test.csv` | 367 solicitudes **sin** la variable objetivo | No se usa: sin la respuesta real no sirve para entrenar ni para evaluar |

Los dos archivos ya vienen en `data/`, así que no hace falta descargar nada. Si se quieren bajar de nuevo, se descarga el zip desde el link de Kaggle (se necesita una cuenta gratuita) y los CSV se copian a `data/`.

**Qué representa cada registro:** una solicitud de préstamo de vivienda. Incluye datos del solicitante (género, estado civil, dependientes, educación y si es independiente), su ingreso y el del coaplicante, el monto y el plazo del préstamo, el historial crediticio y la zona de la propiedad.

## 2. Problema que se resuelve

Una empresa financiera recibe solicitudes de préstamo de vivienda y tiene que decidir si las aprueba. Revisar cada caso a mano es lento y puede ser inconsistente. El modelo estima la probabilidad de que una solicitud sea aprobada. Con eso la empresa puede:

- resolver rápido los casos claros,
- mandar a revisión detallada los casos riesgosos,
- reducir el riesgo de aprobar préstamos que luego no se paguen.

**Pregunta del modelo:** ¿la solicitud de préstamo debería ser aprobada o rechazada?

## 3. Variable objetivo

| | |
|---|---|
| Columna original | `Loan_Status` (Y = aprobado, N = rechazado) |
| Columna en la capa curada | `aprobado` (1 = aprobado, 0 = rechazado) |
| Tipo de problema | Clasificación binaria |
| Balance de clases | 422 aprobados (68.7%) y 192 rechazados (31.3%), un desbalance moderado |

## 4. Arquitectura del pipeline

```
 data/loan_sanction_train.csv
            │
            │  02_etl_pipeline.ipynb — EXTRACCIÓN
            ▼
 ┌──────────────────────── database/prestamos.db (SQLite) ────────────────────────┐
 │  prestamos_raw          datos originales tal cual (con nulos)                   │
 │        │                                                                         │
 │        │  02_etl_pipeline.ipynb — TRANSFORMACIÓN (Python consultando con SQL)   │
 │        ▼                                                                         │
 │  prestamos_curados      datos limpios, sin nulos, listos para el modelo  — CARGA│
 │  metadatos_columnas     diccionario: qué columna es predictora / objetivo       │
 │  etl_log                bitácora: una fila por cada ejecución del ETL           │
 │  vw_entrenamiento       vista: filas de train                                   │
 │  vw_prueba              vista: filas de test                                    │
 │  vw_resumen_aprobacion  vista de negocio: % de aprobación por zona e historial  │
 └─────────────────────────────────────────────────────────────────────────────────┘
            │                                     │
            ▼                                     ▼
 03_model_training.ipynb                scripts/consumo_independiente.py
 (consume las vistas, entrena            (otro consumidor con SQLAlchemy:
  6 modelos, guarda el modelo)            consulta la BD y usa el modelo guardado)
            │
            ▼
 models/modelo_final.joblib
```

**Separación entre carga, almacenamiento y consumo:**

- **Carga:** solo el notebook 02 lee el CSV y escribe en la base.
- **Almacenamiento:** la base SQLite.
- **Consumo:** el notebook 03 y el script leen únicamente desde la base, nunca del CSV.

### Hallazgos del EDA que guiaron el ETL

| Hallazgo (notebook 01) | Decisión que se tomó |
|---|---|
| Nulos en 7 columnas: `Credit_History` 50, `Self_Employed` 32, `LoanAmount` 22, `Dependents` 15, `Loan_Amount_Term` 14, `Gender` 13, `Married` 3 | Imputar con estadísticas calculadas solo con train |
| Las solicitudes con `Credit_History` nulo se aprueban en 74%, distinto de historial 0 (8%) y de historial 1 (80%) | No imputarlas: se marcan como `historial_desconocido` |
| Ingresos y montos con mucha asimetría y valores extremos | Transformación `log(1 + x)`. Los extremos no se eliminan porque son clientes reales |
| `Credit_History` es la variable más relacionada con la aprobación (correlación de 0.56) | Se conserva y se refuerza con variables de capacidad de pago |
| Desbalance 69/31 | Split estratificado, `class_weight="balanced"` y métricas más allá de la accuracy |

### Pasos del ETL

1. **Extracción:** se lee el CSV y se inserta sin cambios en `prestamos_raw`. Las tablas se crean con `database/schema.sql`, que define tipos, llaves y restricciones. Después los datos se vuelven a consultar desde la base con `pd.read_sql`.
2. **Transformación:**
   - Se separan train y test (80/20, estratificado, semilla 42) **antes** de imputar. Así la moda y la mediana se calculan solo con train y no hay *data leakage*.
   - Se imputan nulos: moda para las categóricas, mediana para `LoanAmount` y moda para `Loan_Amount_Term`.
   - `Credit_History` nulo **no** se imputa. Se crea la bandera `historial_desconocido`, porque en el EDA esos clientes se aprueban en 74%, un comportamiento distinto al de historial 0 (8%) y al de historial 1 (80%).
   - Se convierten tipos: `Dependents` pasa de '3+' a 3.
   - Se codifican las categóricas: las binarias a 0/1 y `Property_Area` con one-hot, usando Rural como base.
   - Se aplica `log(1 + x)` a ingresos y monto, que tienen mucha asimetría.
   - Se crean variables derivadas: ingreso total, cuota aproximada y relación cuota/ingreso.
   - Se eliminan las columnas originales ya transformadas. `Loan_ID` queda solo como llave.
   - Se separan las 16 variables predictoras de la objetivo.
   - El escalado (`StandardScaler`) va dentro del pipeline del modelo y se ajusta solo con train.
3. **Carga:** el resultado se guarda en `prestamos_curados`, junto con el diccionario de columnas, la bitácora y las vistas.

### Controles de calidad

- **Sin data leakage:** el split se hace antes de imputar, y el escalado se ajusta solo con train dentro del pipeline del modelo.
- **La variable objetivo no se usa como predictora:** `metadatos_columnas` marca `aprobado` como objetivo, y el ETL y la verificación comprueban que no esté entre las 16 predictoras. `Loan_ID` tampoco entra al modelo.
- **Desbalance:** se maneja con split estratificado, `class_weight="balanced"` y selección por costo de errores en lugar de accuracy.
- **Reproducibilidad:** el ETL se puede correr las veces que se quiera. Recrea las tablas desde `schema.sql`, y la semilla 42 fija el split y los modelos.

## 5. Tecnologías utilizadas

| Uso | Herramienta |
|---|---|
| Lenguaje | Python 3.10 a 3.13 (probado en 3.13; **no** funciona con 3.14, porque las versiones fijadas de numpy y scipy no tienen instalador para esa versión) |
| Base de datos | SQLite (con `sqlite3` y SQLAlchemy desde Python) |
| Manipulación de datos | pandas, numpy |
| Modelos y métricas | scikit-learn |
| Visualización | matplotlib, seaborn |
| Guardado del modelo | joblib |
| Notebooks | Jupyter |
| Lectura del PDF (verificación) | pypdf |

Las versiones exactas están en `requirements.txt`. Están fijadas porque con otra versión de scikit-learn el Random Forest da resultados distintos.

## 6. Estructura del repositorio

```
proyecto-suficiencia-admin-datos/
├── data/
│   ├── loan_sanction_train.csv        dataset original (el que se usa)
│   └── loan_sanction_test.csv         sin variable objetivo (no se usa)
├── database/
│   ├── prestamos.db                   base SQLite ya generada
│   ├── schema.sql                     creación de tablas y vistas
│   └── consultas.sql                  consultas documentadas de validación y consumo
├── notebooks/
│   ├── 01_eda.ipynb                   análisis exploratorio
│   ├── 02_etl_pipeline.ipynb          ETL: CSV → SQLite → tabla curada
│   └── 03_model_training.ipynb        entrenamiento, evaluación y guardado del modelo
├── models/
│   └── modelo_final.joblib            modelo final (escalador + regresión logística)
├── scripts/
│   ├── consumo_independiente.py       consumo de la BD desde otro plano (SQLAlchemy)
│   └── verificar_proyecto.py          verificación automática de todo el proyecto
├── reports/
│   ├── reporte_final.pdf              documento ejecutivo (3 páginas)
│   └── figuras/                       gráficos generados por los notebooks
├── .gitignore                         excluye el entorno virtual y archivos temporales
├── .gitattributes                     marca .pdf, .db, .joblib y .png como binarios
├── README.md
└── requirements.txt
```

## 7. Instrucciones para ejecutar el proyecto

Todos los comandos se corren desde la carpeta raíz del proyecto (`proyecto-suficiencia-admin-datos`).

### 7.1 Instalar (una sola vez)

**Windows (PowerShell):**
```
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- `Set-ExecutionPolicy` permite activar el entorno **solo en esa ventana**. Sin esa línea, PowerShell bloquea `Activate.ps1`.
- Si se usa CMD en lugar de PowerShell, se activa con `.venv\Scripts\activate.bat`.
- Si `py -3.13` no encuentra Python 3.13, se instala con `winget install Python.Python.3.13`.
- Cada vez que se abre una terminal nueva hay que volver a activar el entorno. El prompt muestra `(.venv)` cuando está activo.

**macOS / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 7.2 Correr el pipeline completo (en este orden)

```
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_etl_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_model_training.ipynb
cd ..
python scripts/consumo_independiente.py
```

Qué genera cada paso:

| Paso | Genera |
|---|---|
| `01_eda.ipynb` | gráficos en `reports/figuras/` |
| `02_etl_pipeline.ipynb` | `database/prestamos.db` (se recrea completa) |
| `03_model_training.ipynb` | `models/modelo_final.joblib` y gráficos de evaluación |
| `consumo_independiente.py` | imprime en pantalla las consultas a la BD y las predicciones del modelo |

Si preferís ver el detalle, abrí Jupyter (`jupyter notebook`) y corré los notebooks en orden con *Run All*.

### 7.3 Reconstruir la base de datos

- **Completa, con datos:** correr `notebooks/02_etl_pipeline.ipynb`. Se puede correr las veces que se quiera: recrea las tablas y agrega una fila nueva a `etl_log`.
- **Solo la estructura, sin datos.** Ojo: este comando borra los datos de las tablas.
  ```
  python -c "import sqlite3; sqlite3.connect('database/prestamos.db').executescript(open('database/schema.sql', encoding='utf-8').read())"
  ```
  Después se corre el notebook 02 para volver a cargar los datos.
- **Consultas de revisión:** `database/consultas.sql` tiene 8 consultas comentadas: filas cargadas, nulos, balance, split train/test, predictoras, vistas y bitácora. Se pueden ejecutar en [DB Browser for SQLite](https://sqlitebrowser.org/), que es gratuito: *Open Database* → `database/prestamos.db` → pestaña *Execute SQL*. En macOS y Linux también sirve `sqlite3 database/prestamos.db < database/consultas.sql`.
- **Resultados esperados:** 614 filas en `prestamos_raw`; 491 en train y 123 en test, ambos con ~69% de aprobados; 0 nulos en `prestamos_curados`.

### 7.4 Reutilizar el modelo desde Python

```python
import joblib
import sqlite3
import pandas as pd

paquete = joblib.load("models/modelo_final.joblib")
modelo = paquete["modelo"]            # pipeline: StandardScaler + LogisticRegression
columnas = paquete["predictoras"]     # las 16 columnas que espera el modelo

con = sqlite3.connect("database/prestamos.db")
datos = pd.read_sql("SELECT * FROM vw_prueba", con)
prob_aprobacion = modelo.predict_proba(datos[columnas])[:, 1]
```

### 7.5 Verificar que todo está correcto

```
python scripts/verificar_proyecto.py
```

El script hace 46 chequeos. Revisa que:

- existan todos los entregables,
- los notebooks se hayan ejecutado sin errores,
- las tablas tengan las filas correctas y la capa curada no tenga nulos,
- no haya leakage en la imputación,
- el modelo cargue y sus métricas sean **exactamente** las que dice este README,
- el PDF tenga máximo 3 páginas.

Si todo está bien, termina con `RESULTADO: todo correcto`. Antes de correrlo tiene que existir `database/prestamos.db` con datos, o sea, haberse corrido el notebook 02.

## 8. Modelos comparados

Se probaron 6 modelos de clasificación con validación cruzada estratificada de 5 folds sobre train (491 solicitudes). En los modelos que lo permiten se usó `class_weight="balanced"` para manejar el desbalance.

En un préstamo, los dos errores no cuestan lo mismo:

- **Falso positivo (FP):** aprobar a quien debía rechazarse. El banco puede perder el capital.
- **Falso negativo (FN):** rechazar a quien sí calificaba. El banco deja de ganar intereses.

Como supuesto de trabajo, un FP cuesta 3 veces un FN. El modelo final se eligió por el **menor costo de errores en validación cruzada**, no por accuracy.

| Modelo (CV, 5 folds) | Accuracy | F1 | ROC-AUC | Especificidad* | FP | FN | Costo (3·FP + FN) |
|---|---|---|---|---|---|---|---|
| **Regresión logística balanceada** | 0.745 | 0.820 | 0.740 | **0.525** | 73 | 52 | **271** |
| Logística sin balancear | 0.794 | 0.866 | 0.740 | 0.408 | 91 | 10 | 283 |
| Gradient Boosting | 0.784 | 0.858 | 0.763 | 0.422 | 89 | 17 | 284 |
| Árbol de decisión (balanceado) | 0.756 | 0.834 | 0.674 | 0.454 | 84 | 36 | 288 |
| Random Forest (balanceado) | 0.729 | 0.809 | 0.753 | 0.493 | 78 | 55 | 289 |
| KNN (k = 15) | 0.758 | 0.846 | 0.699 | 0.292 | 109 | 10 | 337 |

\* Especificidad: porcentaje de las solicitudes que debían rechazarse que el modelo detecta.

Los modelos con mayor accuracy (logística sin balancear y Gradient Boosting) la consiguen aprobando casi todo. Detectan solo cerca del 41% de los rechazos, así que dejan pasar la mayoría de los préstamos riesgosos.

## 9. Resultados principales del modelo

**Modelo final: regresión logística balanceada.** Se eligió por tres razones:

- tiene el menor costo de errores y la mayor especificidad en validación cruzada,
- su ROC-AUC es comparable a la de los modelos más complejos,
- es interpretable: se puede explicar por qué se rechaza una solicitud.

Métricas en el set de prueba (123 solicitudes que el modelo nunca vio):

| Accuracy | Precision | Recall | F1-score | ROC-AUC | Especificidad | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.789 | 0.847 | 0.847 | 0.847 | 0.862 | 0.658 | 13 | 13 |

- **Matriz de confusión:** de 38 solicitudes que debían rechazarse, el modelo detecta 25. De 85 que debían aprobarse, aprueba 72.
- **Variable más importante:** el historial crediticio, tanto en la logística como en el Random Forest. Le siguen las variables de capacidad de pago y la zona semiurbana.
- **Falsos positivos:** los 13 tienen buen historial crediticio. Al banco le faltan datos para detectarlos: deudas actuales, estabilidad laboral, garantías.
- **Umbral de decisión:** al subirlo de 0.5 a 0.6, los FP bajan de 13 a 7, pero los FN suben de 13 a 27. El umbral debe definirlo el área de riesgo con sus costos reales.
- **Test vs CV:** en el test, el Random Forest tuvo un costo un poco menor (42 contra 52, unas 3 solicitudes de diferencia). Con solo 123 casos esa diferencia es ruido. La selección se basó en la validación cruzada porque es más estable y porque elegir mirando el test sería *data leakage*.

## 10. Limitaciones

- El dataset es pequeño (614 solicitudes), así que las métricas varían entre folds.
- El modelo depende mucho de `Credit_History`.
- Las variables de ingreso y monto están correlacionadas entre sí. Sus coeficientes individuales se interpretan en conjunto, no uno por uno.
- El costo 3:1 es un supuesto. Con los costos reales del banco, el modelo o el umbral elegido podrían cambiar.
- El modelo aprende las decisiones históricas del banco. Si esas decisiones tenían sesgos, el modelo los repite, así que conviene monitorearlo.
