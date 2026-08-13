"""
Layer 2 of the guardrail stack — topical scope enforcement.

Replaces NeMo's few-shot dialog-rail intent classification, which needed ~2,300
tokens across 3 LLM calls per message and generalised poorly beyond its listed
canonical examples ("what is the capital of france" was listed and caught;
close variants were not).

This asks one small model one direct, scoped question with a ~150-token prompt,
which is both ~15x cheaper and more robust to paraphrase. It runs on its own
rate-limit budget so gate checks never compete with answer generation.
"""

import logfire
from groq import Groq

from app.config import settings

_client: Groq | None = None

_SYSTEM_PROMPT = """You are a topic classifier for an Enterprise IT assistant.

IN_SCOPE — anything an enterprise platform/infrastructure engineer would ask, including:
- Kubernetes: pods, Jobs, CronJobs, work queues, parallelism, autoscaling (HPA/VPA),
  operators, scheduling, monitoring, kubectl, YAML manifests, the Metrics Server
- Supporting infrastructure used by those workloads: Redis, message/work queues,
  containers, Docker, storage, logging
- Databricks and cloud job management: CLI, SDK, REST API, job orchestration
- Intel hardware: CPUs, FPGAs, NICs, SRIOV
- Enterprise networking: SDN, VLANs, BGP, routing
- Any command-line, API, configuration or troubleshooting question about the above
- Greetings, farewells, questions about your capabilities, and follow-ups that
  refer to earlier conversation

OFF_TOPIC — only clearly unrelated requests: jokes, poems, creative writing,
general trivia, sports, cooking, movies, travel, school homework, personal or
medical advice, politics.

Important: when a message is technical, infrastructure-related, or ambiguous,
answer IN_SCOPE. Wrongly blocking a real engineering question is far worse than
occasionally answering a borderline one.

Reply with exactly one word: IN_SCOPE or OFF_TOPIC."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def is_off_topic(message: str) -> bool:
    """
    True if the message falls outside the assistant's supported domains.

    Fails OPEN (False) on classifier error: an outage in the topical gate should
    degrade scope enforcement, not take the service down. Off-topic answers are
    a UX problem, not a safety one — injections are handled separately by
    app/guardrails/prompt_guard.py.
    """
    try:
        response = _get_client().chat.completions.create(
            model=settings.TOPIC_FILTER_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=8,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
    except Exception as e:
        logfire.error(f"⚠️ Topic filter unavailable, failing open: {e}")
        return False

    off_topic = "OFF_TOPIC" in verdict
    if off_topic:
        logfire.info(f"🛡️ Off-topic detected | query='{message[:80]}'")
    return off_topic
