# 12 — Docker Deployment

> **One-line summary:** three purpose-built images (backend, UI, ingest) from one
> multi-stage Dockerfile, with model weights baked in at build time so the first
> request never pays a download.

---

## What Gets Built

| Target | Contains | Approx. size | Purpose |
|---|---|---|---|
| `backend` | FastAPI + LangGraph + torch (CPU) + baked models | ~2.0–2.5 GB | Serves `/query` |
| `ui` | Streamlit + requests | ~250 MB | Chat client — no ML stack |
| `ingest` | `backend` + Office parsers | ~2.6 GB | One-shot batch ingestion |

There is **no Qdrant container** — the vector store is a managed cloud service.

---

## Prerequisites (your machine)

1. **Docker Desktop running.** The CLI can be installed while the engine is
   stopped; if `docker build` reports
   `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`,
   the engine is not started. Launch Docker Desktop and wait for "Engine running".
2. **A populated `.env`** in the repo root (see the README). It is *never* copied
   into an image — `.dockerignore` excludes it and Compose injects it at runtime.
3. **≥ 8 GB free disk.** The build downloads torch and two models.

---

## Quick Start

```powershell
# Build and start backend + UI
docker compose up --build

# UI    → http://localhost:8501
# API   → http://localhost:8000
```

First build takes roughly **10–20 minutes** (torch + a ~420 MB embedding model +
a FlashRank ONNX model). Rebuilds after a code change take seconds, because
dependencies and models live in earlier cached layers.

```powershell
docker compose logs -f backend      # follow logs
docker compose ps                   # health status
docker compose down                 # stop
docker compose down -v --rmi local  # stop and reclaim disk
```

---

## Ingestion

Ingestion is a batch job, not a service, so it is behind a Compose profile and
never starts with `up`:

```powershell
docker compose --profile ingest run --rm ingest DATA --wipe
```

`DATA/` is **bind-mounted read-only** rather than copied into the image — the
corpus is large and changes independently of the code. `processed_data/` is
mounted read-write so chunk metadata lands back on the host.

> Omit `--wipe` to append. The processor verifies the existing collection's
> vector dimension against the current embedding model and aborts on mismatch
> rather than silently mixing embedding spaces.

---

## Three Decisions Behind This Setup

### 1. CPU-only torch

`pip install torch` from PyPI fetches the **CUDA** build — roughly 2.5 GB of GPU
libraries this workload never executes. The Dockerfile installs torch first from
`https://download.pytorch.org/whl/cpu`, so `sentence-transformers` resolves
against the CPU wheel instead of pulling the default.

### 2. Models baked in at build time

```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')" \
 && python -c "from flashrank import Ranker; Ranker(cache_dir='/opt/models/flashrank')"
```

Without this, the **first** query triggers a ~420 MB download inside the request
path. The Streamlit client times out at 60 s, so a perfectly healthy cold
container looks broken. `HF_HUB_OFFLINE=1` in the backend then guarantees no
runtime call to Hugging Face at all.

This is also why `FLASHRANK_CACHE_DIR` is configurable in `app/config.py`: the
container points it at a baked image path, while locally it defaults to the OS
temp directory.

### 3. The UI is a separate image

The UI POSTs to `/query` and renders the response — no retrieval, no embeddings,
no LLM calls. Building it from `requirements-ui.txt` keeps it around 250 MB
instead of shipping torch to a container that only draws a chat window.

---

## Networking: the one thing that always breaks

Inside a container, `localhost` is **the container itself**. A `.env` containing
`BACKEND_URL=http://localhost:8000` makes the UI call *itself* and fail.

Compose therefore overrides it by service name:

```yaml
ui:
  environment:
    BACKEND_URL: "http://backend:8000"
```

This works because `environment:` takes precedence over `env_file:`, and the
app's `load_dotenv()` does not override real environment variables.

The UI also waits on `condition: service_healthy`, not merely "started" — NeMo
Guardrails and LangGraph import for ~30–60 s before the socket binds, and
without the health gate the first message can race a still-importing backend.

---

## Requirements Files

| File | Used by | Contents |
|---|---|---|
| `requirements.txt` | local dev | Everything, including evals |
| `requirements-prod.txt` | `backend` target | Exactly what `app/` imports on the request path |
| `requirements-ui.txt` | `ui` target | Streamlit client only |
| `requirements-ingest.txt` | `ingest` target | Office parsers layered on prod |

`requirements-prod.txt` is derived from an AST scan of every module under `app/`,
not maintained by hand. Two packages it previously omitted — `sentence-transformers`
and `groq` — are now mandatory: without them the container starts cleanly and
then fails on the first query, which is the worst possible failure mode.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `open //./pipe/dockerDesktopLinuxEngine` | Docker Desktop not running | Start it, wait for "Engine running" |
| `COPY ui/: not found` | `ui/` excluded in `.dockerignore` | It must **not** be — the `ui` target copies it |
| UI shows "Backend Offline" | `BACKEND_URL` points at `localhost` | Compose sets `http://backend:8000`; check with `docker compose exec ui env \| grep BACKEND` |
| Backend unhealthy for ~60 s after start | Normal — imports precede socket bind | `start_period: 90s` already accounts for it |
| Build downloads ~2.5 GB of nvidia wheels | torch resolved from PyPI, not the CPU index | Ensure the CPU-index `pip install torch` line runs *before* `requirements-prod.txt` |
| `ModuleNotFoundError: sentence_transformers` | Stale `requirements-prod.txt` | Rebuild without cache: `docker compose build --no-cache backend` |
| First query takes 60 s+ and times out | Models not baked in | Confirm the `models` stage ran; check `docker compose exec backend ls /opt/models` |

---

## Production Notes

- **Scale with replicas, not workers.** Each uvicorn worker loads its own
  ~420 MB embedding model and its own NeMo rails, so `--workers 4` multiplies
  memory ~4x. The `CMD` pins `--workers 1` deliberately.
- **Secrets.** `env_file` is right for local use. On a managed platform use its
  secret manager (Cloud Run secrets, ECS Secrets Manager) rather than shipping
  a `.env`.
- **Cloud Run.** It injects `$PORT`; either honour it in the `CMD` or configure
  the service to target 8000. `.gcloudignore` mirrors `.dockerignore`.
- **Non-root.** All targets run as uid 10001 (`appuser`). Model directories are
  chowned at build so no runtime write to a root-owned path is attempted.
- **Health checks** use the interpreter, not `curl` — the slim base image has no
  curl, and adding it purely for a health probe grows the attack surface.

---

## See Also

- `Dockerfile` · `docker-compose.yml` · `.dockerignore`
- [05 — Environment Variables](05_ENVIRONMENT_VARIABLES.md)
- [06 — Known Gotchas](06_KNOWN_GOTCHAS.md)
