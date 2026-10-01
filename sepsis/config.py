"""Single source of truth for column names, windows and physiological ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"

PID = "patient_id"      # one ICU stay
TIME = "ICULOS"         # hours since ICU admission (1, 2, 3, ...)
LABEL = "SepsisLabel"   # 1 from 6h before sepsis onset (PhysioNet 2019 convention)

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = ["Lactate", "WBC", "Creatinine", "Platelets"]
DYNAMIC = VITALS + LABS
STATIC = ["Age", "Gender"]          # Gender: 1 = male, 0 = female
WINDOWS = (3, 6, 12)                # sliding windows in hours

# Values outside these ranges are treated as sensor/entry errors -> NaN.
RANGES = {
    "HR": (20, 300), "O2Sat": (50, 100), "Temp": (30, 45), "SBP": (40, 300),
    "MAP": (20, 200), "DBP": (20, 200), "Resp": (4, 80), "Lactate": (0.1, 30),
    "WBC": (0.1, 200), "Creatinine": (0.1, 20), "Platelets": (5, 1500),
}

NON_VITAL_BASES = {"Age", "Gender", "ICULOS"}   # never reported as an "alert driver"
