-- Esquema de la base de datos del proyecto (SQLite)
-- Se ejecuta desde notebooks/02_etl_pipeline.ipynb, pero también se puede correr a mano:
--   sqlite3 database/prestamos.db < database/schema.sql

DROP VIEW IF EXISTS vw_entrenamiento;
DROP VIEW IF EXISTS vw_prueba;
DROP VIEW IF EXISTS vw_resumen_aprobacion;
-- primero la tabla curada porque tiene llave foránea hacia prestamos_raw
DROP TABLE IF EXISTS prestamos_curados;
DROP TABLE IF EXISTS prestamos_raw;
DROP TABLE IF EXISTS metadatos_columnas;

-- Capa 1: datos originales, tal cual vienen del CSV (incluye nulos)
CREATE TABLE prestamos_raw (
    loan_id             TEXT PRIMARY KEY,
    gender              TEXT,
    married             TEXT,
    dependents          TEXT,      -- viene como texto por el valor '3+'
    education           TEXT,
    self_employed       TEXT,
    applicant_income    INTEGER,
    coapplicant_income  REAL,
    loan_amount         REAL,      -- en miles
    loan_amount_term    REAL,      -- en meses
    credit_history      REAL,      -- 1, 0 o NULL
    property_area       TEXT,
    loan_status         TEXT NOT NULL CHECK (loan_status IN ('Y', 'N'))
);

-- Capa 2: datos limpios y transformados, listos para el modelo
CREATE TABLE prestamos_curados (
    loan_id                  TEXT PRIMARY KEY REFERENCES prestamos_raw(loan_id),
    conjunto                 TEXT NOT NULL CHECK (conjunto IN ('train', 'test')),
    es_hombre                INTEGER NOT NULL,
    casado                   INTEGER NOT NULL,
    dependientes             INTEGER NOT NULL,
    graduado                 INTEGER NOT NULL,
    independiente            INTEGER NOT NULL,
    zona_semiurbana          INTEGER NOT NULL,
    zona_urbana              INTEGER NOT NULL,
    historial_bueno          INTEGER NOT NULL,
    historial_desconocido    INTEGER NOT NULL,
    plazo_meses              REAL NOT NULL,
    log_ingreso_solicitante  REAL NOT NULL,
    log_ingreso_coaplicante  REAL NOT NULL,
    log_ingreso_total        REAL NOT NULL,
    log_monto_prestamo       REAL NOT NULL,
    cuota_aprox              REAL NOT NULL,
    relacion_cuota_ingreso   REAL NOT NULL,
    aprobado                 INTEGER NOT NULL CHECK (aprobado IN (0, 1))
);

-- Diccionario: qué columna es predictora, cuál es objetivo y cuál es llave
CREATE TABLE metadatos_columnas (
    columna      TEXT PRIMARY KEY,
    rol          TEXT NOT NULL CHECK (rol IN ('llave', 'control', 'predictora', 'objetivo')),
    descripcion  TEXT
);

-- Bitácora de ejecuciones del ETL (no se borra: acumula el historial de corridas)
CREATE TABLE IF NOT EXISTS etl_log (
    id_ejecucion      INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha             TEXT NOT NULL,
    filas_leidas      INTEGER,
    filas_raw         INTEGER,
    filas_curadas     INTEGER,
    filas_train       INTEGER,
    filas_test        INTEGER,
    nulos_restantes   INTEGER
);

-- Vistas de consumo para el modelo
CREATE VIEW vw_entrenamiento AS
SELECT * FROM prestamos_curados WHERE conjunto = 'train';

CREATE VIEW vw_prueba AS
SELECT * FROM prestamos_curados WHERE conjunto = 'test';

-- Vista de negocio: tasa de aprobación por zona e historial
CREATE VIEW vw_resumen_aprobacion AS
SELECT property_area,
       CASE WHEN credit_history IS NULL THEN 'sin historial'
            WHEN credit_history = 1 THEN 'cumple'
            ELSE 'no cumple' END AS historial,
       COUNT(*) AS solicitudes,
       ROUND(AVG(CASE WHEN loan_status = 'Y' THEN 1.0 ELSE 0 END) * 100, 1) AS pct_aprobado
FROM prestamos_raw
GROUP BY property_area, historial;
