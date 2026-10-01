"""Model loading + scoring. Reuses the exact training feature code (no train/serve skew)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import ARTIFACT_DIR, DYNAMIC, PID, TIME
from ..explain import alert_message, make_explainer, shap_matrix, top_drivers
from ..features import build_features


class SepsisService:
    def __init__(self, artifact_dir=None):
        d = Path(artifact_dir or os.getenv("MODEL_DIR", ARTIFACT_DIR))
        self.meta = json.loads((d / "meta.json").read_text())
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(d / "xgb_model.json"))
        self.explainer = make_explainer(self.model)
        self.names = self.meta["feature_names"]
        self.threshold = float(self.meta["threshold"])
        self.medians = self.meta["medians"]
        self.version = self.meta["model_version"]

    def _last_row_features(self, p) -> pd.DataFrame:
        df = pd.DataFrame([o.model_dump() for o in p.observations]).rename(columns={"hour": TIME})
        df = df.set_index(TIME).sort_index()
        df = df.reindex(range(int(df.index.min()), int(df.index.max()) + 1))   # hourly grid, gaps -> NaN
        df.index.name = TIME
        df = df.reset_index()
        for c in DYNAMIC:
            if c not in df:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)   # all-None column -> float NaN
        df[PID], df["Age"], df["Gender"] = p.patient_id, p.age, float(p.gender)
        return build_features(df).iloc[[-1]].reindex(columns=self.names)

    def score(self, patients, k: int = 3):
        X = pd.concat([self._last_row_features(p) for p in patients], ignore_index=True)
        probs = self.model.predict_proba(X)[:, 1]
        sv = shap_matrix(self.explainer, X)
        out = []
        for i, p in enumerate(patients):
            risk = float(probs[i])
            drivers = top_drivers(sv[i], self.names, X.iloc[i], self.medians, k) if risk >= self.threshold * 0.5 else []
            level = "HIGH" if risk >= self.threshold else "MEDIUM" if risk >= 0.5 * self.threshold else "LOW"
            out.append({
                "patient_id": p.patient_id, "hour": max(o.hour for o in p.observations),
                "risk_score": round(risk, 4), "threshold": round(self.threshold, 4),
                "alert": risk >= self.threshold, "risk_level": level,
                "alert_message": alert_message(drivers) if drivers else "No elevated risk",
                "top_drivers": drivers, "model_version": self.version})
        return out
