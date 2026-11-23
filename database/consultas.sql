-- Consultas documentadas sobre prestamos.db
-- Correr con: sqlite3 database/prestamos.db < database/consultas.sql

-- 1. Validar que la carga raw tiene todas las filas del CSV (esperado: 614)
SELECT COUNT(*) AS filas_raw FROM prestamos_raw;

-- 2. Nulos por columna en la capa raw
SELECT SUM(gender IS NULL)           AS nulos_gender,
       SUM(married IS NULL)          AS nulos_married,
       SUM(dependents IS NULL)       AS nulos_dependents,
       SUM(self_employed IS NULL)    AS nulos_self_employed,
       SUM(loan_amount IS NULL)      AS nulos_loan_amount,
       SUM(loan_amount_term IS NULL) AS nulos_loan_amount_term,
       SUM(credit_history IS NULL)   AS nulos_credit_history
FROM prestamos_raw;

-- 3. Distribución de la variable objetivo
SELECT loan_status, COUNT(*) AS solicitudes,
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM prestamos_raw), 1) AS porcentaje
FROM prestamos_raw GROUP BY loan_status;

-- 4. Tamaño y balance de train / test en la capa curada
SELECT conjunto, COUNT(*) AS filas, ROUND(AVG(aprobado) * 100, 1) AS pct_aprobado
FROM prestamos_curados GROUP BY conjunto;

-- 5. Columnas predictoras (lo que consume el modelo)
SELECT columna, descripcion FROM metadatos_columnas WHERE rol = 'predictora';

-- 6. Dataset de entrenamiento que alimenta al modelo
SELECT * FROM vw_entrenamiento LIMIT 5;

-- 7. Vista de negocio
SELECT * FROM vw_resumen_aprobacion ORDER BY pct_aprobado DESC;

-- 8. Bitácora del ETL
SELECT * FROM etl_log;
