FROM python:3.12-slim

# onnxruntime (CPU) by default; build with --build-arg ORT_PACKAGE=onnxruntime-gpu for CUDA.
ARG ORT_PACKAGE=onnxruntime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    MODELS_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "${ORT_PACKAGE}"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
