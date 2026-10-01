"""FastAPI real-time alerting service.   uvicorn sepsis.api.main:app --port 8000"""
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request

from .schemas import BatchRequest, PatientHistory, Prediction
from .service import SepsisService

log = logging.getLogger("sepsis.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_app(artifact_dir=None) -> FastAPI:
    stats = {"requests": 0, "patients_scored": 0, "alerts": 0, "latency_ms_total": 0.0}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.service = SepsisService(artifact_dir)
            log.info("model loaded: %s", app.state.service.version)
        except Exception as e:                      # keep process up; /health reports 503
            app.state.service = None
            log.error("model failed to load: %s", e)
        yield

    app = FastAPI(title="ICU Early Sepsis Warning API", version="1.0.0", lifespan=lifespan,
                  description="Research prototype. Not a medical device.")

    def svc(request: Request) -> SepsisService:
        s = request.app.state.service
        if s is None:
            raise HTTPException(503, "Model not loaded")
        return s

    @app.get("/health")
    def health(request: Request):
        s = request.app.state.service
        if s is None:
            raise HTTPException(503, "Model not loaded")
        return {"status": "ok", "model_version": s.version}

    @app.get("/model-info")
    def model_info(request: Request):
        s = svc(request)
        return {"model_version": s.version, "threshold": s.threshold, "n_features": len(s.names),
                "data_source": s.meta["data_source"], "trained_at": s.meta["trained_at"],
                "test_metrics": s.meta["metrics"]["xgboost"]}

    def _run(request, patients):
        t0 = time.perf_counter()
        preds = svc(request).score(patients)
        ms = (time.perf_counter() - t0) * 1000
        stats["requests"] += 1
        stats["patients_scored"] += len(preds)
        stats["alerts"] += sum(p["alert"] for p in preds)
        stats["latency_ms_total"] += ms
        for p in preds:
            log.info("patient=%s hour=%s risk=%.3f alert=%s latency_ms=%.1f",
                     p["patient_id"], p["hour"], p["risk_score"], p["alert"], ms)
        return preds

    @app.post("/predict", response_model=Prediction)
    def predict(patient: PatientHistory, request: Request):
        return _run(request, [patient])[0]

    @app.post("/predict/batch", response_model=List[Prediction])
    def predict_batch(body: BatchRequest, request: Request):
        return _run(request, body.patients)

    @app.get("/metrics")
    def metrics():
        n = max(stats["requests"], 1)
        return {**stats, "avg_latency_ms": stats["latency_ms_total"] / n}

    return app


app = create_app()
