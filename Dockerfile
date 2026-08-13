# =============================================================================
# Sentinel-RAG — multi-stage, multi-target build
#
#   docker build --target backend -t sentinel-rag:backend .
#   docker build --target ui      -t sentinel-rag:ui      .
#   docker build --target ingest  -t sentinel-rag:ingest  .
#
# Three design decisions worth knowing before editing:
#
# 1. torch comes from the CPU wheel index. The default PyPI wheel is the CUDA
#    build (~2.5 GB of GPU libraries this workload never touches).
#
# 2. Both models are downloaded at BUILD time, not first request. The embedding
#    model is ~420 MB; downloading it on the first query blows past the UI's
#    60 s timeout and makes a cold container look broken.
#
# 3. The UI is a separate target. It is an HTTP client with no ML dependencies,
#    so it has no reason to carry torch.
# =============================================================================

ARG PYTHON_VERSION=3.12


# ---------------------------------------------------------------- base -------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Patch OS CVEs; libgomp1 is required by torch/onnxruntime at runtime.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user — nothing here needs root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app


# ------------------------------------------------------------- pydeps --------
# Heavy dependency install, isolated so it is cached independently of app code.
FROM base AS pydeps

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first, so sentence-transformers resolves against it instead of
# pulling the CUDA build from PyPI.
RUN pip install --no-cache-dir torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt


# ------------------------------------------------------------- models --------
# Bake model weights into the image so no download happens on the request path.
FROM pydeps AS models

ENV HF_HOME=/opt/models/hf \
    FLASHRANK_CACHE_DIR=/opt/models/flashrank

RUN mkdir -p "$HF_HOME" "$FLASHRANK_CACHE_DIR" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')" \
    && python -c "from flashrank import Ranker; Ranker(cache_dir='/opt/models/flashrank')"


# ------------------------------------------------------------ backend --------
FROM base AS backend

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models/hf \
    FLASHRANK_CACHE_DIR=/opt/models/flashrank \
    HF_HUB_OFFLINE=1 \
    PORT=8000

COPY --from=pydeps /opt/venv /opt/venv
COPY --from=models /opt/models /opt/models
COPY --chown=appuser:appuser app/ ./app/

RUN chown -R appuser:appuser /opt/models
USER appuser

EXPOSE 8000

# No curl in a slim image — use the interpreter that is already here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/',timeout=4).status==200 else 1)"

# Single worker on purpose: each worker loads its own ~420 MB embedding model
# and its own NeMo rails. Scale with replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ----------------------------------------------------------------- ui --------
FROM base AS ui

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

COPY --chown=appuser:appuser ui/ ./ui/
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]


# ------------------------------------------------------------- ingest --------
# One-shot batch job: parse DATA/, embed locally, upsert to Qdrant. Adds the
# Office parsers the serving image deliberately omits.
FROM backend AS ingest

USER root
COPY requirements-ingest.txt .
RUN pip install --no-cache-dir -r requirements-ingest.txt
USER appuser

HEALTHCHECK NONE

ENTRYPOINT ["python", "-m", "app.ingestion.processor"]
CMD ["DATA"]
