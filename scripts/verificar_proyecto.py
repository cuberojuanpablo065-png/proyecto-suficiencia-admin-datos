# Verificación automática del proyecto.
# Revisa que todos los entregables existan y que lo que dicen el README y el PDF
# coincida con lo que realmente hay en la base de datos y en el modelo.
#
# Uso (desde la raíz del repo, después de correr los 3 notebooks):
#   python scripts/verificar_proyecto.py

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
fallas = []


def revisar(descripcion, condicion, detalle=""):
    if condicion:
        print(f"  [OK]    {descripcion}")
    else:
        print(f"  [FALLA] {descripcion} {detalle}")
        fallas.append(descripcion)


# valores que aparecen en el README y en el PDF (set de prueba, 123 solicitudes)
ESPERADO = {"accuracy": 0.789, "precision": 0.847, "recall": 0.847, "f1": 0.847,
            "roc_auc": 0.862, "especificidad": 0.658, "FP": 13, "FN": 13}

# ------------------------------------------------------------------
print("\n1. Estructura del repositorio")
archivos = [
    "README.md", "requirements.txt",
    "data/loan_sanction_train.csv",
    "notebooks/01_eda.ipynb", "notebooks/02_etl_pipeline.ipynb", "notebooks/03_model_training.ipynb",
    "database/prestamos.db", "database/schema.sql", "database/consultas.sql",
    "models/modelo_final.joblib",
    "scripts/consumo_independiente.py",
    "reports/reporte_final.pdf",
]
for a in archivos:
    revisar(f"existe {a}", (RAIZ / a).exists())

# ------------------------------------------------------------------
print("\n2. Notebooks ejecutados sin errores")
for nb in ["01_eda", "02_etl_pipeline", "03_model_training"]:
    contenido = json.loads((RAIZ / "notebooks" / f"{nb}.ipynb").read_text(encoding="utf-8"))
    celdas = [c for c in contenido["cells"] if c["cell_type"] == "code"]
    con_error = [c for c in celdas if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    sin_ejecutar = [c for c in celdas if c.get("execution_count") is None]
    revisar(f"{nb}.ipynb: {len(celdas)} celdas, todas ejecutadas y sin error",
            not con_error and not sin_ejecutar,
            f"({len(con_error)} con error, {len(sin_ejecutar)} sin ejecutar)")

# ------------------------------------------------------------------
print("\n3. Base de datos (carga y almacenamiento)")
con = sqlite3.connect(RAIZ / "database" / "prestamos.db")
objetos = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type IN ('table','view')", con)["name"])
for o in ["prestamos_raw", "prestamos_curados", "metadatos_columnas", "etl_log",
          "vw_entrenamiento", "vw_prueba", "vw_resumen_aprobacion"]:
    revisar(f"existe la tabla/vista {o}", o in objetos)

csv = pd.read_csv(RAIZ / "data" / "loan_sanction_train.csv")
raw = pd.read_sql("SELECT * FROM prestamos_raw", con)
curados = pd.read_sql("SELECT * FROM prestamos_curados", con)

revisar(f"prestamos_raw tiene las {len(csv)} filas del CSV", len(raw) == len(csv), f"(tiene {len(raw)})")
revisar("prestamos_raw conserva los nulos originales (datos sin tocar)",
        raw.isna().sum().sum() == csv.isna().sum().sum())
revisar("prestamos_curados tiene las mismas filas que raw", len(curados) == len(raw))
revisar("prestamos_curados no tiene nulos", curados.isna().sum().sum() == 0)
revisar("todas las llaves de la tabla curada existen en raw",
        curados["loan_id"].isin(raw["loan_id"]).all())

n_train = (curados["conjunto"] == "train").sum()
n_test = (curados["conjunto"] == "test").sum()
revisar("split train/test 491 / 123", (n_train, n_test) == (491, 123), f"({n_train} / {n_test})")
tasa_train = curados.loc[curados.conjunto == "train", "aprobado"].mean()
tasa_test = curados.loc[curados.conjunto == "test", "aprobado"].mean()
revisar("split estratificado (misma proporción de aprobados)", abs(tasa_train - tasa_test) < 0.02,
        f"({tasa_train:.3f} vs {tasa_test:.3f})")

roles = pd.read_sql("SELECT columna, rol FROM metadatos_columnas", con)
predictoras = roles.loc[roles.rol == "predictora", "columna"].tolist()
revisar("16 columnas predictoras y 1 objetivo", len(predictoras) == 16 and (roles.rol == "objetivo").sum() == 1)
revisar("la variable objetivo y el loan_id no están entre las predictoras",
        "aprobado" not in predictoras and "loan_id" not in predictoras)
revisar("etl_log registra al menos una ejecución", pd.read_sql("SELECT COUNT(*) n FROM etl_log", con)["n"][0] >= 1)

# ------------------------------------------------------------------
print("\n4. Sin data leakage en la imputación")
ids_train = curados.loc[curados.conjunto == "train", "loan_id"]
mediana_train = raw.loc[raw.loan_id.isin(ids_train), "loan_amount"].median()
nulos_monto = raw.loc[raw.loan_amount.isna(), "loan_id"]
imputados = curados.set_index("loan_id").reindex(nulos_monto)["log_monto_prestamo"]
revisar(f"LoanAmount nulo se rellenó con la mediana de TRAIN ({mediana_train})",
        imputados.notna().all() and np.allclose(imputados, np.log1p(mediana_train)))
nulos_hist = raw.loc[raw.credit_history.isna(), "loan_id"]
revisar("Credit_History nulo quedó marcado como historial_desconocido = 1",
        (curados.set_index("loan_id").reindex(nulos_hist)["historial_desconocido"] == 1).all())
con.close()

# ------------------------------------------------------------------
print("\n5. Modelo guardado")
paquete = joblib.load(RAIZ / "models" / "modelo_final.joblib")
modelo = paquete["modelo"]
revisar(f"el modelo se carga: {paquete['nombre']}", hasattr(modelo, "predict_proba"))
revisar("el modelo usa las mismas columnas que la base de datos", paquete["predictoras"] == predictoras)

prueba = curados[curados.conjunto == "test"]
X, y = prueba[predictoras], prueba["aprobado"]
pred = modelo.predict(X)
prob = modelo.predict_proba(X)[:, 1]
calculado = {
    "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
    "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
    "especificidad": recall_score(y, pred, pos_label=0),
    "FP": int(((pred == 1) & (y == 0)).sum()), "FN": int(((pred == 0) & (y == 1)).sum()),
}
for m, valor in ESPERADO.items():
    revisar(f"{m} = {valor} (igual que en README y PDF)", round(calculado[m], 3) == valor,
            f"(calculado: {round(calculado[m], 3)})")

# ------------------------------------------------------------------
print("\n6. Consumo desde otro plano")
r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "consumo_independiente.py")],
                   capture_output=True, text=True)
revisar("scripts/consumo_independiente.py corre sin errores", r.returncode == 0, r.stderr[-300:])

# ------------------------------------------------------------------
print("\n7. Documento PDF")
ruta_pdf = RAIZ / "reports" / "reporte_final.pdf"
try:
    from pypdf import PdfReader
    paginas = len(PdfReader(ruta_pdf).pages) if ruta_pdf.exists() else 0
    revisar(f"reporte_final.pdf tiene entre 1 y 3 páginas ({paginas})", 1 <= paginas <= 3)
except ImportError:
    print("  [AVISO] pypdf no está instalado: abrí el PDF y contá las páginas a mano (máximo 3)")

# ------------------------------------------------------------------
print("\n" + "=" * 60)
if fallas:
    print(f"RESULTADO: {len(fallas)} chequeo(s) fallaron:")
    for f in fallas:
        print("  -", f)
    sys.exit(1)
print("RESULTADO: todo correcto")
