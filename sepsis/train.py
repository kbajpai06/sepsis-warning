"""End-to-end training: load -> features -> XGBoost / LightGBM / LSTM -> evaluate -> SHAP -> MLflow -> artifacts.

    python -m sepsis.train --source synthetic
    python -m sepsis.train --source challenge2019 --data-dir data/physionet2019
    python -m sepsis.train --source mimic --db-url postgresql+psycopg2://user:pw@localhost/mimiciv
"""
import argparse
import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import __version__  # noqa: E402
from .config import ARTIFACT_DIR, LABEL, PID, REPORT_DIR  # noqa: E402
from .evaluate import choose_threshold, patient_metrics, row_metrics  # noqa: E402
from .explain import alert_message, make_explainer, shap_matrix, top_drivers  # noqa: E402
from .features import (add_temporal_features, feature_columns, patient_split,  # noqa: E402
                       preprocess, truncate)
from .models.gbm import train_lgbm, train_xgb, trim_to_best  # noqa: E402


def load_data(source, data_dir=None, db_url=None, n_patients=1500, seed=42, limit_stays=None):
    if source == "synthetic":
        from .data.synthetic import make_synthetic
        return make_synthetic(n_patients=n_patients, seed=seed)
    if source == "challenge2019":
        from .data.physionet2019 import load_challenge2019
        return load_challenge2019(data_dir)
    if source == "mimic":
        from .data.mimic import load_mimic
        return load_mimic(db_url, limit_stays=limit_stays)
    raise ValueError(f"unknown source {source}")


def _plot_curves(y, probs: dict, fig_dir: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(rc, pr, label=name)
        ax[1].plot(fpr, tpr, label=name)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test, hourly rows)")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.6)
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_roc_curves.png", dpi=150)
    plt.close(fig)


def run(source="synthetic", data_dir=None, db_url=None, n_patients=1500, max_hours=72,
        skip_lstm=False, cell="lstm", epochs=25, use_mlflow=True, artifact_dir=None,
        report_dir=None, seed=42, mlflow_uri="sqlite:///mlflow.db", limit_stays=None,
        experiment="icu-sepsis-early-warning") -> dict:
    artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
    report_dir = Path(report_dir or REPORT_DIR)
    fig_dir = report_dir / "figures"
    for d in (artifact_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    ctx = nullcontext()
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)
        ctx = mlflow.start_run(run_name=f"{source}-{datetime.now():%Y%m%d-%H%M}")

    with ctx:
        # ---------------- data & features ----------------
        df = truncate(load_data(source, data_dir, db_url, n_patients, seed, limit_stays), max_hours)
        pre = preprocess(df)
        feats = add_temporal_features(pre)
        cols = feature_columns(feats)
        tr_ids, va_ids, te_ids = patient_split(df, seed=seed)

        def sub(d, ids):
            return d[d[PID].isin(ids)].reset_index(drop=True)

        F_tr, F_va, F_te = sub(feats, tr_ids), sub(feats, va_ids), sub(feats, te_ids)
        X_tr, X_va, X_te = F_tr[cols], F_va[cols], F_te[cols]
        y_tr, y_va, y_te = F_tr[LABEL].to_numpy(), F_va[LABEL].to_numpy(), F_te[LABEL].to_numpy()
        data_summary = {
            "patients": int(df[PID].nunique()), "rows": int(len(df)),
            "septic_patient_pct": float(df.groupby(PID)[LABEL].max().mean()),
            "positive_row_pct": float(df[LABEL].mean()), "n_features": len(cols),
            "split_patients": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
        }
        print("DATA:", json.dumps(data_summary))

        # ---------------- models ----------------
        probs_va, probs_te, models = {}, {}, {}
        models["xgboost"] = train_xgb(X_tr, y_tr, X_va, y_va, seed)
        models["lightgbm"] = train_lgbm(X_tr, y_tr, X_va, y_va, seed)
        for name, m in models.items():
            probs_va[name] = m.predict_proba(X_va)[:, 1]
            probs_te[name] = m.predict_proba(X_te)[:, 1]

        history = None
        if not skip_lstm:
            from .models.lstm import compute_stats, predict_rows, to_sequences, train_rnn
            pre_tr, pre_va, pre_te = sub(pre, tr_ids), sub(pre, va_ids), sub(pre, te_ids)
            stats = compute_stats(pre_tr)
            seq_tr, seq_va = to_sequences(pre_tr, stats)[:3], to_sequences(pre_va, stats)[:3]
            rnn, history = train_rnn(seq_tr, seq_va, cell=cell, epochs=epochs, seed=seed)
            name = cell
            probs_va[name] = predict_rows(rnn, pre_va, stats)
            probs_te[name] = predict_rows(rnn, pre_te, stats)
            models[name] = rnn

        # ---------------- evaluation (thresholds picked on VALIDATION only) ----------------
        results = {}
        for name in probs_te:
            thr = choose_threshold(y_va, probs_va[name])
            results[name] = {"row": row_metrics(y_te, probs_te[name], thr),
                             "patient": patient_metrics(F_te, probs_te[name], thr)}
        comp = pd.DataFrame({n: {**r["row"], **{f"pt_{k}": v for k, v in r["patient"].items()}}
                             for n, r in results.items()}).T
        comp.to_csv(report_dir / "model_comparison.csv")
        print(comp[["auroc", "auprc", "sensitivity", "specificity", "precision",
                    "pt_patient_sensitivity", "pt_median_lead_time_hours"]].round(3).to_string())
        _plot_curves(y_te, probs_te, fig_dir)

        # ---------------- SHAP on the serving model (XGBoost) ----------------
        xgb_params = models["xgboost"].get_params()
        xgb_model = trim_to_best(models["xgboost"])        # served model: best-iteration trees only
        assert np.allclose(xgb_model.predict_proba(X_te)[:, 1], probs_te["xgboost"], atol=1e-5)
        explainer = make_explainer(xgb_model)
        sample = X_te.sample(min(2000, len(X_te)), random_state=seed)
        sv = shap_matrix(explainer, sample)
        import shap
        shap.summary_plot(sv, sample, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        medians = {k: (None if pd.isna(v) else float(v)) for k, v in X_tr.median().items()}
        septic_idx = np.where(y_te == 1)[0]
        best_i = septic_idx[np.argmax(probs_te["xgboost"][septic_idx])]
        row = X_te.iloc[[best_i]]
        drivers = top_drivers(shap_matrix(explainer, row)[0], cols, row.iloc[0], medians, k=3)
        example = {"patient_id": str(F_te.loc[best_i, PID]), "hour": int(F_te.loc[best_i, "ICULOS"]),
                   "risk_score": float(probs_te["xgboost"][best_i]),
                   "alert_message": alert_message(drivers), "top_drivers": drivers}
        (report_dir / "example_alert.json").write_text(json.dumps(example, indent=2))

        # ---------------- artifacts ----------------
        xgb_model.save_model(str(artifact_dir / "xgb_model.json"))
        models["lightgbm"].booster_.save_model(str(artifact_dir / "lgbm_model.txt"))
        if not skip_lstm:
            import torch
            torch.save({"state_dict": models[cell].state_dict(), "stats": stats, "cell": cell},
                       artifact_dir / f"{cell}_model.pt")
        meta = {"model_version": f"{__version__}-{datetime.now(timezone.utc):%Y%m%d%H%M}",
                "data_source": source, "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_names": cols, "threshold": results["xgboost"]["row"]["threshold"],
                "medians": medians, "metrics": results, "data": data_summary}
        (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        (report_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

        # ---------------- MLflow ----------------
        if use_mlflow:
            import mlflow
            mlflow.log_params({"source": source, "max_hours": max_hours, "seed": seed,
                               "n_features": len(cols), "windows": "3,6,12", "rnn_cell": cell,
                               **{f"xgb_{k}": v for k, v in xgb_params.items()
                                  if k in ("max_depth", "learning_rate", "n_estimators", "subsample")}})
            mlflow.log_params({f"data_{k}": v for k, v in data_summary.items() if not isinstance(v, dict)})
            for name, r in results.items():
                for grp in ("row", "patient"):
                    for k, v in r[grp].items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"{name}_{grp}_{k}", float(v))
            if history:
                for h in history:
                    mlflow.log_metric("rnn_val_auprc", h["val_auprc"], step=h["epoch"])
                    mlflow.log_metric("rnn_train_loss", h["train_loss"], step=h["epoch"])
            mlflow.log_artifacts(str(report_dir), artifact_path="reports")
            mlflow.log_artifact(str(artifact_dir / "meta.json"), artifact_path="model")
            mlflow.log_artifact(str(artifact_dir / "xgb_model.json"), artifact_path="model")
    return {"results": results, "meta": meta, "example": example}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "challenge2019", "mimic"])
    ap.add_argument("--data-dir"); ap.add_argument("--db-url")
    ap.add_argument("--n-patients", type=int, default=1500)
    ap.add_argument("--max-hours", type=int, default=72)
    ap.add_argument("--cell", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--limit-stays", type=int)
    a = ap.parse_args()
    run(source=a.source, data_dir=a.data_dir, db_url=a.db_url, n_patients=a.n_patients,
        max_hours=a.max_hours, skip_lstm=a.skip_lstm, cell=a.cell, epochs=a.epochs,
        use_mlflow=not a.no_mlflow, limit_stays=a.limit_stays)


if __name__ == "__main__":
    main()
