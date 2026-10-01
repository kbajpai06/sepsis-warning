FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/artifacts
WORKDIR /app

# libgomp is required by xgboost; curl for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY sepsis/ ./sepsis/
# Model artifacts (xgb_model.json + meta.json) are mounted at runtime, not baked into the image:
#   docker run -v ./artifacts:/app/artifacts:ro ...

RUN useradd -m appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "sepsis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
