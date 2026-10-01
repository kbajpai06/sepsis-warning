import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sepsis import train
from sepsis.api.main import create_app
from sepsis.config import DYNAMIC, LABEL, PID, TIME
from sepsis.data.synthetic import make_synthetic
from sepsis.explain import make_explainer, shap_matrix


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    out = train.run(source="synthetic", n_patients=500, skip_lstm=True, use_mlflow=False,
                    artifact_dir=root / "art", report_dir=root / "rep")
    return root, out


def test_training_beats_chance_and_saves_artifacts(trained):
    root, out = trained
    assert out["results"]["xgboost"]["row"]["auroc"] > 0.8
    for f in ("xgb_model.json", "meta.json"):
        assert (root / "art" / f).exists()
    assert (root / "rep" / "figures" / "shap_summary.png").exists()
    assert len(out["example"]["top_drivers"]) <= 3


def test_shap_matches_native_treeshap(trained):
    import xgboost as xgb
    root, _ = trained
    m = xgb.XGBClassifier()
    m.load_model(str(root / "art" / "xgb_model.json"))
    names = json.loads((root / "art" / "meta.json").read_text())["feature_names"]
    X = np.random.default_rng(0).normal(size=(5, len(names)))
    import pandas as pd
    X = pd.DataFrame(X, columns=names)
    ours = shap_matrix(make_explainer(m), X)
    native = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    assert np.allclose(ours, native, atol=1e-3)


def _request(hours=None):
    df = make_synthetic(300, seed=11)
    pid = df[df[LABEL] == 1][PID].iloc[0]
    p = df[df[PID] == pid]
    if hours:
        p = p[p[TIME] <= hours]
    obs = [{"hour": int(r[TIME]), **{c: (None if r[c] != r[c] else float(r[c])) for c in DYNAMIC}}
           for _, r in p.iterrows()]
    return {"patient_id": "t1", "age": 60, "gender": 1, "observations": obs}


def test_api_endpoints(trained):
    root, _ = trained
    with TestClient(create_app(root / "art")) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "test_metrics" in c.get("/model-info").json()
        r = c.post("/predict", json=_request())
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["risk_score"] <= 1 and body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert len(body["top_drivers"]) <= 3
        b = c.post("/predict/batch", json={"patients": [_request(30), _request(50)]})
        assert len(b.json()) == 2
        assert c.get("/metrics").json()["requests"] >= 2


def test_api_validation_and_gaps(trained):
    root, _ = trained
    with TestClient(create_app(root / "art")) as c:
        assert c.post("/predict", json={"patient_id": "x", "age": 50, "gender": 1, "observations": []}).status_code == 422
        dup = {"patient_id": "x", "age": 50, "gender": 1, "observations": [{"hour": 1}, {"hour": 1}]}
        assert c.post("/predict", json=dup).status_code == 422
        gappy = {"patient_id": "x", "age": 50, "gender": 1,
                 "observations": [{"hour": 1, "HR": 80}, {"hour": 5, "HR": 90}]}
        assert c.post("/predict", json=gappy).status_code == 200      # missing hours are filled


def test_api_503_without_model(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/health").status_code == 503
