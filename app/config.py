"""
Central configuration.

Two rules this module enforces, both of which were previously violated:

1. Every tunable that affects retrieval quality lives here, not at a call site.
   `limit`, `top_n` and the relevance threshold used to be hardcoded literals in
   `retriever.py`, which made them impossible to sweep in an experiment without
   editing code. An evaluation you cannot re-run at a different setting is not
   an evaluation.

2. Missing secrets fail loudly at import, not silently at the first request.
   `LANGSMITH_API_KEY` used to be written into the environment as `""`, which
   produces a stream of 401s rather than a clear error.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a number, got {raw!r}") from e


class Settings:
    # --- LOCAL MODEL CACHE ---------------------------------------------------
    # A container bakes the model into an image path at build time. Locally this
    # must NOT be the OS temp dir: temp cleanup deletes the model files but
    # leaves the directories behind, after which FlashRank sees the directory,
    # skips the download, and then fails to load the missing .onnx on every
    # single request. Observed in practice. A stable user cache dir avoids it.
    FLASHRANK_CACHE_DIR = os.getenv(
        "FLASHRANK_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "flashrank"),
    )

    # --- VECTOR DB (QDRANT) --------------------------------------------------
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kubernetes_docs")
    QDRANT_TIMEOUT = _int("QDRANT_TIMEOUT", 60)

    # --- EMBEDDINGS ----------------------------------------------------------
    # Dimension is NOT configured here on purpose: it is read from the loaded
    # model at runtime (see app/services/retrieval/embedding.py). A hardcoded
    # constant meant the collection-dimension safety check compared the wrong
    # value against itself and passed while corrupting the index.
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 64)

    # --- CHUNKING ------------------------------------------------------------
    # Character budgets. Overlap is a fraction of CHUNK_SIZE and exists because
    # a heading severed from its body is unrecoverable without it.
    CHUNK_SIZE = _int("CHUNK_SIZE", 1200)
    CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)
    CHUNK_MIN_SIZE = _int("CHUNK_MIN_SIZE", 120)

    # --- RETRIEVAL -----------------------------------------------------------
    # Candidate depth, chosen by measurement rather than convention. Vector
    # recall for the gold chunk is 0.81 @20, 0.91 @30, 0.98 @50 and 1.00 @100,
    # so a shallow pool - not the reranker - was the binding constraint:
    #     depth   hit@1   hit@5   rerank_ms
    #        20   0.663   0.855          31
    #        60   ~0.73   ~0.95          ~100
    #       100   0.759   0.976         194
    # 60 takes most of the accuracy for a third of the reranking cost. Going
    # deeper also gives the cross-encoder more chances to promote a topically
    # adjacent passage on a question the corpus cannot answer, which is the
    # trade recorded against RERANK_THRESHOLD below.
    RETRIEVAL_CANDIDATES = _int("RETRIEVAL_CANDIDATES", 60)
    RETRIEVAL_TOP_N = _int("RETRIEVAL_TOP_N", 5)  # kept after reranking
    # The cross-encoder score below which a chunk is treated as "not relevant".
    #
    # Derived, not guessed - see tools/calibrate_threshold.py. Measured on this
    # corpus (n=60 answerable, n=28 out-of-corpus):
    #     answerable      min 0.9393   p05 0.9956   p50 0.9997
    #     out-of-corpus   max 0.1523   p95 0.1510
    # Every threshold in [0.20, 0.90] scored 0% false-answer and 0%
    # false-abstention, so this sits mid-band: ~2.6x clear of the highest
    # out-of-corpus score and ~2.3x below the lowest answerable one, which
    # leaves margin for drift in either direction rather than tuning to the
    # sample. Re-derive whenever the corpus, chunker or reranker changes.
    #
    # Scope note, and the honest limitation of this design: the threshold
    # separates OFF-domain from in-domain almost perfectly (all 10 off-domain
    # probes score ~0.000). It does NOT reliably separate "in-domain and
    # covered" from "in-domain but uncovered", because a cross-encoder scores
    # topical relatedness rather than answer presence. At depth 60, 2 of 18
    # in-domain-uncovered questions still surface a passage above threshold -
    # e.g. "Kubernetes SIG groups" matches "expose groups of Pods".
    #
    # A threshold of 0.90 would exclude one of those two, but answerable
    # questions bottom out at 0.943 and those questions were GENERATED from the
    # passages they match, so their scores are optimistic. Tuning to that margin
    # would overfit to the eval set and start refusing real, less
    # perfectly-phrased questions. The second line of defence is the generator,
    # which is instructed to decline when the passages do not answer the
    # question; the `uncovered_retrieval_positive` slice measures exactly that.
    RERANK_THRESHOLD = _float("RERANK_THRESHOLD", 0.35)
    # A chunk this much below the top hit is dropped even if it clears the
    # absolute threshold: prevents one strong hit dragging in weak neighbours.
    RERANK_RELATIVE_FLOOR = _float("RERANK_RELATIVE_FLOOR", 0.02)
    # Used ONLY when the cross-encoder is unavailable and scores fall back to
    # cosine similarity. Cosine and cross-encoder scores are different scales -
    # applying RERANK_THRESHOLD to a cosine score silently changes the system's
    # abstention behaviour, which is how a broken reranker previously produced
    # confident answers to out-of-corpus questions. Deliberately conservative.
    VECTOR_FALLBACK_THRESHOLD = _float("VECTOR_FALLBACK_THRESHOLD", 0.55)

    # --- GROQ ----------------------------------------------------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Model roster. The Llama family this project was built on (llama-3.3-70b-
    # versatile, llama-3.1-8b-instant) was withdrawn from Groq and now returns
    # 404, so every role below was re-selected from what the account can
    # actually serve. Each model draws on its own rate-limit bucket.
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")
    PLANNER_MODEL = os.getenv("PLANNER_MODEL", "openai/gpt-oss-20b")
    # Injection detection, stage 1 - dedicated classifier, ~50 tokens.
    PROMPT_GUARD_MODEL = os.getenv("PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")
    # Injection detection, stage 2 - adjudicates intent for stage-1 flags only.
    # gpt-oss-safeguard is purpose-built for policy classification.
    INJECTION_CONFIRM_MODEL = os.getenv("INJECTION_CONFIRM_MODEL", "openai/gpt-oss-safeguard-20b")
    TOPIC_FILTER_MODEL = os.getenv("TOPIC_FILTER_MODEL", "openai/gpt-oss-safeguard-20b")

    # gpt-oss models emit reasoning tokens before any content. Too small a
    # max_tokens truncates the answer to "" and the verdict is lost silently.
    REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
    GENERATION_MAX_TOKENS = _int("GENERATION_MAX_TOKENS", 1600)
    CLASSIFIER_MAX_TOKENS = _int("CLASSIFIER_MAX_TOKENS", 900)

    # --- LLM GATEWAY (PORTKEY) ----------------------------------------------
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG", "")
    GROQ_SLUG = os.getenv("PORTKEY_GROQ_SLUG", "rag1")
    GROQ_SLUG_2 = os.getenv("PORTKEY_GROQ_SLUG_2", "rag2")
    # Portkey is OFF by default, and that is a deliberate downgrade from the
    # previous design. Two reasons:
    #   1. A saved Portkey Config (referenced by PORTKEY_CONFIG_SLUG) overrides
    #      the model this code asks for. The saved config still targets
    #      llama-3.3-70b-versatile / llama-3.1-8b-instant, which Groq has
    #      withdrawn, so every gateway call 404s. That config lives in the
    #      Portkey dashboard and cannot be fixed from this repository.
    #   2. Its response cache makes evaluation non-independent: a repeat run
    #      scores cached generations rather than the system.
    # Set USE_GATEWAY=true once the saved config targets live models.
    USE_GATEWAY = _flag("USE_GATEWAY", False)
    # Evaluation must never be scored against cached generations - a repeat run
    # would grade the cache rather than the system.
    GATEWAY_CACHE_ENABLED = _flag("GATEWAY_CACHE_ENABLED", True)

    # --- CONVERSATION MEMORY -------------------------------------------------
    # Bounded: history used to grow without limit, inflating every prompt and
    # eventually exceeding the model's context window.
    HISTORY_MAX_TURNS = _int("HISTORY_MAX_TURNS", 8)
    HISTORY_MAX_CHARS = _int("HISTORY_MAX_CHARS", 6000)
    CONTEXT_MAX_CHARS = _int("CONTEXT_MAX_CHARS", 12000)

    # --- API SECURITY --------------------------------------------------------
    # Comma-separated. When empty, auth is disabled - correct for local dev and
    # CI, and reported on /health so a deployment cannot be unprotected without
    # it being visible.
    API_KEYS = tuple(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip())
    AUTH_REQUIRED = _flag("AUTH_REQUIRED", bool(API_KEYS))
    RATE_LIMIT_PER_MINUTE = _int("RATE_LIMIT_PER_MINUTE", 20)
    RATE_LIMIT_ENABLED = _flag("RATE_LIMIT_ENABLED", True)

    # --- OBSERVABILITY -------------------------------------------------------
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_ENABLED = _flag("LOGFIRE_ENABLED", bool(LOGFIRE_TOKEN))
    LANGSMITH_TRACING = _flag("LANGSMITH_TRACING", False)
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "sentinel-rag")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # --- EVALUATION ----------------------------------------------------------
    # Judge must be a different model family from GENERATION_MODEL, or the
    # system grades its own output. validate() enforces that.
    # Qwen, so the judge is a different family from the gpt-oss generator -
    # `validate()` refuses to start otherwise. 3.6 rather than 3.8 purely for
    # quota: each model has its own 200k/day token budget, and 3.8's was spent.
    # Both were verified to produce valid structured output for RAGAS, and both
    # scored faithfulness within 0.03 of each other on the same samples.
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.6-27b")
    # Falls back to the serving key, but says so. The previous code fell back
    # silently while its comments claimed the judge had "its own rate-limit
    # budget" - and the repo's own notes record that Groq quotas are per
    # organisation anyway, so a second key buys no isolation. Isolation comes
    # from using a different MODEL, which the judge-family check above enforces.
    JUDGE_API_KEY = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    JUDGE_KEY_IS_SHARED = not os.getenv("JUDGE_GROQ")

    @classmethod
    def missing_required(cls) -> list[str]:
        """
        Names of secrets needed to serve traffic that are not configured.

        `QDRANT_API_KEY` is deliberately conditional. A managed Qdrant Cloud
        endpoint needs one; a local or self-hosted instance does not, and
        demanding it there is not a safety check - it just blocks a legitimate
        setup. Two cases are treated as fine:

          * the endpoint is localhost / an in-cluster host, or
          * the variable is present but empty, which is how a caller says
            "this deployment has no auth" on purpose.

        Only an entirely unset key against a remote endpoint is an error.
        `os.getenv` returns None when unset and "" when set-but-empty, so those
        two intentions stay distinguishable.
        """
        missing = []
        if not cls.QDRANT_URL:
            missing.append("QDRANT_CLUSTER_ENDPOINT")
        elif cls.QDRANT_API_KEY is None and not cls.qdrant_is_local():
            missing.append("QDRANT_API_KEY")
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        return missing

    @classmethod
    def qdrant_is_local(cls) -> bool:
        """True when the vector store is reachable without credentials."""
        url = (cls.QDRANT_URL or "").lower()
        return any(
            host in url
            for host in ("localhost", "127.0.0.1", "0.0.0.0", "://qdrant", "host.docker.internal")
        )

    @classmethod
    def validate(cls) -> None:
        """Raise on any configuration that would produce a broken system."""
        missing = cls.missing_required()
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not 0.0 <= cls.RERANK_THRESHOLD <= 1.0:
            raise ValueError("RERANK_THRESHOLD must be between 0 and 1.")
        if cls.RETRIEVAL_TOP_N > cls.RETRIEVAL_CANDIDATES:
            raise ValueError("RETRIEVAL_TOP_N cannot exceed RETRIEVAL_CANDIDATES.")
        if _family(cls.JUDGE_MODEL) == _family(cls.GENERATION_MODEL):
            raise ValueError(
                f"JUDGE_MODEL ({cls.JUDGE_MODEL}) and GENERATION_MODEL "
                f"({cls.GENERATION_MODEL}) are from the same family. A model "
                "grading its own lineage is not an independent evaluation."
            )


def _family(model: str) -> str:
    """Vendor prefix of a model id, used for the judge-independence check."""
    return model.split("/")[0] if "/" in model else model.split("-")[0]


settings = Settings()

# LangChain reads tracing config from the environment. Only set it when tracing
# is actually enabled and keyed, so an unset key cannot produce silent 401s.
if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
