import logfire
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action

from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, BLOCKING_FLOWS
from app.guardrails.prompt_guard import detect_injection
from app.guardrails.topic_filter import is_off_topic

_rails: LLMRails | None = None


# ── Colang actions ────────────────────────────────────────────────────────────
# NeMo owns policy (what to check, in what order, what to say when blocking);
# these actions own detection. Keeping them separate is what lets the gate use
# purpose-built classifiers instead of few-shot prompting a chat model.

@action(name="detect_injection_action", is_system_action=True)
async def detect_injection_action(context: dict | None = None) -> bool:
    message = (context or {}).get("user_message", "")
    is_injection, _score = detect_injection(message)
    return is_injection


@action(name="detect_off_topic_action", is_system_action=True)
async def detect_off_topic_action(context: dict | None = None) -> bool:
    message = (context or {}).get("user_message", "")
    return is_off_topic(message)


def initialize_rails() -> None:
    """Build the NeMo LLMRails singleton at app startup."""
    global _rails

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    # No `llm=` is passed: this config runs input rails only, and every decision
    # comes from a registered action, so NeMo never needs a generation model of
    # its own. That is what removes the ~2,300 tokens/message the old dialog-rail
    # design spent on intent classification.
    _rails = LLMRails(config)
    _rails.register_action(detect_injection_action, "detect_injection_action")
    _rails.register_action(detect_off_topic_action, "detect_off_topic_action")

    logfire.info("🛡️ NeMo Guardrails initialised (input rails: injection + topic).")


def _extract_response_text(result) -> str:
    """GenerationResponse.response is a list of message dicts; plain generate() returns a dict."""
    response = getattr(result, "response", result)
    if isinstance(response, list):
        return " ".join(
            msg.get("content", "") for msg in response if isinstance(msg, dict)
        ).strip()
    if isinstance(response, dict):
        return response.get("content", "")
    return str(response)


def _blocking_rail_activated(result) -> bool:
    """
    Read NeMo's structured activation log to decide whether a rail blocked the
    message, rather than substring-matching the reply against hand-copied
    refusal phrases (which broke silently whenever wording drifted).
    """
    log = getattr(result, "log", None)
    activated = getattr(log, "activated_rails", None) if log else None
    if not activated:
        return False

    for rail in activated:
        name = (getattr(rail, "name", "") or "").lower()
        if name in BLOCKING_FLOWS:
            # `stop` is set when the flow halted the request; treat a named
            # blocking flow as a block even if the attribute is absent.
            if getattr(rail, "stop", True):
                return True
    return False


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the guardrail gate.

    Layer 1 — Prompt Guard 2 injection classifier   (via NeMo input rail)
    Layer 2 — scoped topical classifier             (via NeMo input rail)
    Layer 3 — hardened system prompts in app/agents/nodes/responder.py, which
              catch anything that slips past 1 and 2.

    Returns:
        (True,  response) — a rail fired; return this immediately, skip the RAG pipeline.
        (False, None)     — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(
                messages=[{"role": "user", "content": message}],
                # Input rails only: generation belongs to LangGraph, and this
                # keeps the gate to exactly the two classifier calls above.
                options={"rails": ["input"], "log": {"activated_rails": True}},
            )
        except Exception as e:
            # Fail open: a gate outage should degrade protection, not the service.
            logfire.error(f"⚠️ Guardrails check failed, failing open: {e}")
            return False, None

        if _blocking_rail_activated(result):
            content = _extract_response_text(result)
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
