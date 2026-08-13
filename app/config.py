import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- LOCAL MODEL CACHE ---
    # Where FlashRank stores its ONNX reranker. Configurable because the value
    # differs by environment: a container bakes the model into an image path at
    # build time (so first query doesn't pay a download), while locally the OS
    # temp dir is correct — the previously hardcoded "/tmp/flashrank" is not a
    # valid path on Windows and silently fell back to re-downloading every run.
    FLASHRANK_CACHE_DIR = os.getenv(
        "FLASHRANK_CACHE_DIR", os.path.join(tempfile.gettempdir(), "flashrank")
    )

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- GUARDRAILS ---
    # Model tiering matters here: every request passes the gate, so running it on
    # the 70B generation model burned the whole daily token budget on gate checks
    # (and still misclassified paraphrased jailbreaks). Each model below draws on
    # its own rate-limit budget, keeping the 70B quota for actual answers.
    #
    # Injection detection, stage 1 — dedicated classifier, small, fast, own budget.
    PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
    # Injection detection, stage 2 — adjudicates intent for stage-1 flags only.
    # Measured on a held-out set (no overlap with the prompt's examples):
    #   llama-3.1-8b-instant  accuracy 0.400
    #   openai/gpt-oss-20b    accuracy 1.000  <-- chosen
    #   openai/gpt-oss-120b   accuracy 0.933
    # Slower (~2.8s) but only runs on flagged traffic, and on its own budget.
    INJECTION_CONFIRM_MODEL = "openai/gpt-oss-20b"
    # Topical scope classifier. 8B was unreliable at NeMo's few-shot intent
    # matching but scores 16/16 on one direct scoped question — the prompting
    # approach was the problem, not the model. Draws on its own rate-limit
    # budget, so gate checks never compete with 70B answer generation.
    TOPIC_FILTER_MODEL = "llama-3.1-8b-instant"

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "rag1"     # primary: @rag1/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "rag2"  # fallback: @rag2/llama-3.1-8b-instant
    # Slug (pc-...) of a saved Portkey Config, required when the workspace has
    # "block_inline_config" enabled and rejects the fallback/cache/retry config
    # sent via the x-portkey-config header. Leave unset to send it inline.
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG", "")

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
