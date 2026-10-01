-- =====================================================================
-- Hourly ICU dataset for early sepsis prediction (MIMIC-IV, PostgreSQL)
--
-- Assumes the standard mimic-code PostgreSQL build:
--   schemas mimiciv_hosp, mimiciv_icu, and mimiciv_derived (sepsis3 table).
-- One row per ICU stay per hour. Output columns match sepsis/config.py.
--
-- Label convention (PhysioNet/CinC 2019): SepsisLabel = 1 from 6 hours
-- BEFORE sepsis onset onward. Onset = earlier of (suspected infection
-- time, SOFA>=2 time) from mimiciv_derived.sepsis3.
-- =====================================================================
WITH onset AS (
    SELECT stay_id,
           LEAST(suspected_infection_time, sofa_time) AS onset_time
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 IS TRUE
),
cohort AS (
    SELECT ie.stay_id, ie.subject_id, ie.hadm_id, ie.intime, ie.outtime,
           pat.anchor_age + (EXTRACT(YEAR FROM ie.intime) - pat.anchor_year) AS age,
           CASE WHEN pat.gender = 'M' THEN 1 ELSE 0 END AS gender,
           o.onset_time
    FROM mimiciv_icu.icustays ie
    JOIN mimiciv_hosp.patients pat ON pat.subject_id = ie.subject_id
    LEFT JOIN onset o ON o.stay_id = ie.stay_id
    WHERE ie.los >= 0.5                                   -- stay >= 12 h
      AND pat.anchor_age + (EXTRACT(YEAR FROM ie.intime) - pat.anchor_year) BETWEEN 18 AND 89
      -- drop stays where sepsis was already present at ICU entry (not a prediction problem)
      AND (o.onset_time IS NULL OR o.onset_time > ie.intime + INTERVAL '6 hours')
    ORDER BY ie.stay_id
    /*LIMIT_STAYS*/
),
grid AS (   -- one row per ICU hour, capped at 7 days
    SELECT c.stay_id, gs AS hour_ts,
           ROW_NUMBER() OVER (PARTITION BY c.stay_id ORDER BY gs) AS iculos
    FROM cohort c
    CROSS JOIN LATERAL generate_series(
        date_trunc('hour', c.intime),
        LEAST(date_trunc('hour', c.outtime), date_trunc('hour', c.intime) + INTERVAL '167 hours'),
        INTERVAL '1 hour') AS gs
),
vitals AS (  -- hourly mean of charted vitals (itemids: MIMIC-IV chartevents)
    SELECT ce.stay_id, date_trunc('hour', ce.charttime) AS hour_ts,
        AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END)                    AS "HR",
        AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END)                    AS "O2Sat",
        AVG(CASE WHEN ce.itemid = 223762 THEN ce.valuenum
                 WHEN ce.itemid = 223761 THEN (ce.valuenum - 32) / 1.8 END)       AS "Temp",
        AVG(CASE WHEN ce.itemid IN (220050, 220179) THEN ce.valuenum END)         AS "SBP",
        AVG(CASE WHEN ce.itemid IN (220052, 220181) THEN ce.valuenum END)         AS "MAP",
        AVG(CASE WHEN ce.itemid IN (220051, 220180) THEN ce.valuenum END)         AS "DBP",
        AVG(CASE WHEN ce.itemid = 220210 THEN ce.valuenum END)                    AS "Resp"
    FROM mimiciv_icu.chartevents ce
    WHERE ce.stay_id IN (SELECT stay_id FROM cohort)
      AND ce.itemid IN (220045, 220277, 223762, 223761, 220050, 220179,
                        220052, 220181, 220051, 220180, 220210)
      AND ce.valuenum IS NOT NULL
    GROUP BY ce.stay_id, date_trunc('hour', ce.charttime)
),
labs AS (    -- hourly mean of lab results (itemids: MIMIC-IV labevents)
    SELECT c.stay_id, date_trunc('hour', le.charttime) AS hour_ts,
        AVG(CASE WHEN le.itemid = 50813 THEN le.valuenum END)             AS "Lactate",
        AVG(CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum END)   AS "WBC",
        AVG(CASE WHEN le.itemid = 50912 THEN le.valuenum END)             AS "Creatinine",
        AVG(CASE WHEN le.itemid = 51265 THEN le.valuenum END)             AS "Platelets"
    FROM mimiciv_hosp.labevents le
    JOIN cohort c
      ON c.subject_id = le.subject_id
     AND le.charttime BETWEEN c.intime AND c.outtime
    WHERE le.itemid IN (50813, 51300, 51301, 50912, 51265)
      AND le.valuenum IS NOT NULL
    GROUP BY c.stay_id, date_trunc('hour', le.charttime)
)
SELECT g.stay_id, g.iculos,
       v."HR", v."O2Sat", v."Temp", v."SBP", v."MAP", v."DBP", v."Resp",
       l."Lactate", l."WBC", l."Creatinine", l."Platelets",
       c.age AS "Age", c.gender AS "Gender",
       CASE WHEN c.onset_time IS NOT NULL
             AND g.hour_ts >= date_trunc('hour', c.onset_time) - INTERVAL '6 hours'
            THEN 1 ELSE 0 END AS "SepsisLabel"
FROM grid g
JOIN cohort c USING (stay_id)
LEFT JOIN vitals v ON v.stay_id = g.stay_id AND v.hour_ts = g.hour_ts
LEFT JOIN labs   l ON l.stay_id = g.stay_id AND l.hour_ts = g.hour_ts
ORDER BY g.stay_id, g.iculos;
