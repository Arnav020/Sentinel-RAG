# 12 — Docker Deployment

> **One-line summary:** one image serves the API and the web client, with model
> weights baked in at build time so the first request never pays a download.
> A second target exists only for the batch ingestion job.

---

## What Gets Built

| Target | Contains | Approx. size | Purpose |
|---|---|---|---|
| `backend` | FastAPI + LangGraph + torch (CPU) + baked models + web client | **1.36 GB** (measured) | Serves the API **and** the UI |
| `ingest` | `backend` + Office parsers | ~1.8 GB | One-shot batch ingestion |

There is **no UI container**: the web client is static HTML/CSS/JS served by the
same FastAPI process, so there is no Node toolchain, no build step, and no
cross-origin configuration.

There is **no Qdrant container** either — the vector store is a managed cloud service.

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
# Build and start
docker compose up --build

# UI + API → http://localhost:8000
```

First build takes roughly **10–20 minutes** (torch + a ~420 MB embedding model +
a FlashRank ONNX model). Rebuilds after a code change take seconds, because
dependencies and models live in earlier cached layers.

Verified on a clean build: image **1.36 GB**, container reports `(healthy)`, and
the **first** query in a fresh container returns in **3.9 s** — the models are
baked in, so nothing is downloaded on the request path.

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

## Decisions Behind This Setup

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
path, so a perfectly healthy cold container looks broken while it downloads. `HF_HUB_OFFLINE=1` in the backend then guarantees no
runtime call to Hugging Face at all.

This is also why `FLASHRANK_CACHE_DIR` is configurable in `app/config.py`: the
container points it at a baked image path, while locally it defaults to the OS
temp directory.

### 3. The UI ships inside the backend image

The client is plain HTML/CSS/JS with no dependencies, so serving it from the
same FastAPI process costs a few hundred kilobytes and removes an entire
container, a Node toolchain, and a cross-origin hop. It is mounted **after**
every API route, because FastAPI resolves routes in declaration order and the
static mount is a catch-all.

---

## Networking

The client calls `/query` on its own origin, so there is no `BACKEND_URL` to
configure and no container-to-container hop to get wrong. (The previous
Streamlit UI needed `BACKEND_URL`, and pointing it at `localhost` inside a
container — where `localhost` is the container itself — was the single most
common failure.)

The health gate still matters: NeMo Guardrails and LangGraph import for
~30–60 s before the socket binds, which is why `start_period` is 90 s.

---

## Requirements Files

| File | Used by | Contents |
|---|---|---|
| `requirements.txt` | local dev | Everything, including evals |
| `requirements-prod.txt` | `backend` target | Exactly what `app/` imports on the request path |
| `requirements-ingest.txt` | `ingest` target | Office parsers layered on prod |

`requirements-prod.txt` and `requirements-ingest.txt` are **generated, not
hand-written**: the full transitive closure is resolved from the working
development environment and pinned at the versions actually installed there
(149 and 37 packages respectively), with markers evaluated for linux/cpython
3.12 rather than the Windows host.

Pinning the whole closure matters because this project's dependency conflicts
were settled by selectively upgrading and downgrading packages. A top-level-only
pin lets pip re-resolve inside the image and land back on a broken combination —
hand-maintained pins had already drifted (`langsmith` was recorded as `0.8.3`
while the working environment ran `0.10.18`).

Regenerate after changing dependencies locally rather than editing by hand.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `open //./pipe/dockerDesktopLinuxEngine` | Docker Desktop not running | Start it, wait for "Engine running" |
| `invalid env file: variable 'X ' contains whitespaces` | `.env` written as `KEY = value` | Docker's parser reads the name as `'KEY '`. Rewrite as `KEY=value` — python-dotenv accepts both, Docker only the latter |
| `COPY web/: not found` | `web/` excluded in `.dockerignore` | It must **not** be — the backend target copies it |
| Blank page at `:8000`, API works | `web/` missing from the image | `web/` must not be in `.dockerignore`; check `docker compose exec backend ls web` |
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
