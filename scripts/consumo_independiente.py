# Script independiente de consumo (Etapa 5).
# No usa el CSV ni los notebooks: solo se conecta a la base de datos con SQLAlchemy,
# extrae datos de la capa curada y usa el modelo guardado para predecir.
#
# Uso (desde la raíz del repo):
#   python scripts/consumo_independiente.py

from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

RAIZ = Path(__file__).resolve().parent.parent
motor = create_engine(f"sqlite:///{RAIZ / 'database' / 'prestamos.db'}")
paquete = joblib.load(RAIZ / "models" / "modelo_final.joblib")

with motor.connect() as con:
    # 1. consulta de negocio sobre los datos originales
    resumen = pd.read_sql(text("SELECT * FROM vw_resumen_aprobacion ORDER BY pct_aprobado DESC"), con)

    # 2. extraer el set de prueba desde la vista curada
    columnas = ", ".join(paquete["predictoras"])
    prueba = pd.read_sql(text(f"SELECT loan_id, {columnas}, aprobado FROM vw_prueba"), con)

print("== Tasa de aprobación histórica por zona e historial ==")
print(resumen.to_string(index=False))

modelo = paquete["modelo"]
prueba["prob_aprobacion"] = modelo.predict_proba(prueba[paquete["predictoras"]])[:, 1]
prueba["prediccion"] = (prueba["prob_aprobacion"] >= paquete["umbral"]).astype(int)

aciertos = (prueba["prediccion"] == prueba["aprobado"]).mean()
print(f"\n== Modelo: {paquete['nombre']} ==")
print(f"Solicitudes evaluadas: {len(prueba)} | accuracy: {aciertos:.3f}")

print("\n== 5 solicitudes con mayor riesgo (menor probabilidad de aprobación) ==")
print(prueba.nsmallest(5, "prob_aprobacion")[["loan_id", "historial_bueno", "prob_aprobacion", "aprobado"]]
      .round(3).to_string(index=False))
