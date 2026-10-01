"""SHAP explanations -> human-readable 'top 3 drivers' for every alert."""
import re

import numpy as np

from .config import NON_VITAL_BASES

LABELS = {
    "HR": "heart rate", "O2Sat": "SpO2", "Temp": "temperature", "SBP": "systolic BP",
    "MAP": "mean arterial pressure", "DBP": "diastolic BP", "Resp": "respiratory rate",
    "Lactate": "lactate", "WBC": "WBC count", "Creatinine": "creatinine",
    "Platelets": "platelet count", "shock_index": "shock index (HR/SBP)",
    "Age": "age", "ICULOS": "ICU length of stay", "Gender": "gender",
}
_PATTERN = re.compile(r"^(?P<base>[A-Za-z0-9_]+?)(?:_(?P<kind>mean|std|slope)(?P<w>\d+)|_(?P<age>age))?$")


def parse_feature(name: str):
    """'MAP_slope6' -> ('MAP', 'slope', 6); 'Lactate' -> ('Lactate', 'level', None)."""
    m = _PATTERN.match(name)
    base, kind, w, age = m.group("base"), m.group("kind"), m.group("w"), m.group("age")
    if age:
        return base, "staleness", None
    return base, (kind or "level"), (int(w) if w else None)


def describe(name: str, value: float, median: float | None) -> str:
    base, kind, w = parse_feature(name)
    label = LABELS.get(base, base)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"Missing {label}"
    if kind == "slope":
        return f"{'Rapid rise' if value > 0 else 'Rapid drop'} in {label} ({value:+.2f}/h over {w}h)"
    if kind == "std":
        return f"Unstable {label} (SD {value:.2f} over {w}h)"
    if kind == "staleness":
        return f"Stale {label} measurement ({value:.0f}h since last)"
    tag = f" ({w}h avg)" if kind == "mean" else ""
    if median is None:
        return f"Abnormal {label}{tag} ({value:.2f})"
    return f"{'Elevated' if value > median else 'Low'} {label}{tag} ({value:.2f})"


def make_explainer(model):
    import shap
    return shap.TreeExplainer(model)


def shap_matrix(explainer, X) -> np.ndarray:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    return np.asarray(sv)


def top_drivers(shap_row, feature_names, values, medians, k: int = 3):
    """Aggregate SHAP by clinical variable (lactate level + lactate 6h mean count as one
    'lactate' driver), keep the k largest RISK-INCREASING ones, and describe each using
    its single most influential feature."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for name, s in zip(feature_names, shap_row):
        base = parse_feature(name)[0]
        if base in NON_VITAL_BASES:
            continue
        groups.setdefault(base, []).append((name, float(s)))
    ranked = sorted(groups.items(), key=lambda kv: sum(s for _, s in kv[1]), reverse=True)
    out = []
    for base, items in ranked[:k]:
        total = sum(s for _, s in items)
        if total <= 0:
            break
        top_feat = max(items, key=lambda t: t[1])[0]
        val = values[top_feat]
        val = None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        out.append({"variable": LABELS.get(base, base), "feature": top_feat,
                    "value": val, "shap": round(total, 4),
                    "description": describe(top_feat, val, medians.get(top_feat))})
    return out


def alert_message(drivers) -> str:
    return " + ".join(d["description"].split(" (")[0] for d in drivers) or "No dominant risk driver"
