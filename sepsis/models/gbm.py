"""Gradient-boosted tree baselines: XGBoost and LightGBM."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb


def _pos_weight(y) -> float:
    y = np.asarray(y)
    return float(np.sqrt((y == 0).sum() / max((y == 1).sum(), 1)))   # damped class weight


def train_xgb(X_tr, y_tr, X_va, y_va, seed: int = 42, params: dict | None = None):
    p = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
             scale_pos_weight=_pos_weight(y_tr), eval_metric="aucpr",
             early_stopping_rounds=50, tree_method="hist", n_jobs=-1, random_state=seed)
    p.update(params or {})
    model = xgb.XGBClassifier(**p)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def train_lgbm(X_tr, y_tr, X_va, y_va, seed: int = 42, params: dict | None = None):
    p = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
             scale_pos_weight=_pos_weight(y_tr), random_state=seed, n_jobs=-1, verbose=-1)
    p.update(params or {})
    model = lgb.LGBMClassifier(**p)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def trim_to_best(model: xgb.XGBClassifier) -> xgb.XGBClassifier:
    """Early stopping leaves extra trees in the booster. predict_proba ignores them but
    SHAP / pred_contribs would not, so explanations would not add up to the served score.
    Keep only trees up to the best iteration."""
    best = getattr(model, "best_iteration", None)
    booster = model.get_booster()
    if best is not None:
        booster = booster[: best + 1]
    clf = xgb.XGBClassifier()
    clf.load_model(bytearray(booster.save_raw("json")))
    return clf
