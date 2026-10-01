"""Row-level and patient-level evaluation, including early-warning lead time."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import LABEL, PID, TIME


def choose_threshold(y, p, beta: float = 2.0) -> float:
    """Threshold maximising F-beta on validation data (beta=2 favours recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f))])


def row_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "threshold": float(thr),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
    }


def patient_metrics(df: pd.DataFrame, p, thr: float) -> dict:
    """Clinically meaningful view.
    Sepsis onset = first labelled hour + 6 (labels start 6h before onset).
    A septic patient is 'caught' if the first alert fires at or before onset."""
    d = df[[PID, TIME, LABEL]].copy()
    d["alert"] = np.asarray(p) >= thr
    lead, caught, n_septic = [], 0, 0
    fa_patients, n_clean, clean_hours, clean_alert_hours = 0, 0, 0, 0
    for _, g in d.groupby(PID, sort=False):
        first_alert = g.loc[g["alert"], TIME].min() if g["alert"].any() else np.nan
        if g[LABEL].max() == 1:
            n_septic += 1
            onset = g.loc[g[LABEL] == 1, TIME].min() + 6
            if not np.isnan(first_alert) and first_alert <= onset:
                caught += 1
                lead.append(onset - first_alert)
        else:
            n_clean += 1
            fa_patients += int(g["alert"].any())
            clean_hours += len(g)
            clean_alert_hours += int(g["alert"].sum())
    return {
        "n_septic_patients": n_septic,
        "patient_sensitivity": caught / max(n_septic, 1),
        "median_lead_time_hours": float(np.median(lead)) if lead else None,
        "pct_non_septic_patients_with_any_alert": fa_patients / max(n_clean, 1),
        "alert_hours_per_100_non_septic_patient_hours": 100 * clean_alert_hours / max(clean_hours, 1),
    }
