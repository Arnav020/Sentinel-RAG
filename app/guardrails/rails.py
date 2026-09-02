"""
The guardrail gate.

Layer 1 - Prompt Guard 2 injection classifier      (NeMo input rail)
Layer 2 - intent adjudicator for stage-1 flags     (NeMo input rail)
Layer 3 - topical scope classifier                 (NeMo input rail)
Layer 4 - hardened generation prompt               (responder.py)

The gate now sees recent conversation history, not just the current message. An
injection split across turns - one message establishing a persona, a later
innocuous one triggering it - previously passed straight through, because the
graph replayed history into the prompt but the gate never inspected it.
"""

from __future__ import annotations

import logfire
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action

from app.guardrails.colang_rules import BLOCKING_FLOWS, COLANG_CONTENT, YAML_CONTENT
from app.guardrails.prompt_guard import detect_injection
from app.guardrails.topic_filter import is_off_topic

_rails: LLMRails | None = None

# Set per call so the Colang actions, which only receive NeMo's own context, can
# see the conversation the gate was asked about.
_pending_history: str = ""


@action(name="detect_injection_action", is_system_action=True)
async def detect_injection_action(context: dict | None = None) -> bool:
    message = (context or {}).get("user_message", "")
    probe = f"{_pending_history}\n{message}".strip() if _pending_history else message
    is_injection, _score = detect_injection(probe)
    return is_injection


@action(name="detect_off_topic_action", is_system_action=True)
async def detect_off_topic_action(context: dict | None = None) -> bool:
    message = (context or {}).get("user_message", "")
    return is_off_topic(message)


def initialize_rails() -> None:
    """Build the NeMo LLMRails singleton. Called once at application startup."""
    global _rails
    config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)
    rails = LLMRails(config)
    rails.register_action(detect_injection_action, "detect_injection_action")
    rails.register_action(detect_off_topic_action, "detect_off_topic_action")
    _rails = rails
    logfire.info("Guardrails initialised (input rails: injection + topic).")


def rails_ready() -> bool:
    return _rails is not None


def _extract_response_text(result) -> str:
    response = getattr(result, "response", result)
    if isinstance(response, list):
        return " ".join(m.get("content", "") for m in response if isinstance(m, dict)).strip()
    if isinstance(response, dict):
        return response.get("content", "")
    return str(response)


def _blocking_rail_activated(result) -> bool:
    """
    Read NeMo's structured activation log rather than substring-matching the
    reply against hand-copied refusal phrases, which broke silently whenever the
    wording drifted.
    """
    log = getattr(result, "log", None)
    activated = getattr(log, "activated_rails", None) if log else None
    if not activated:
        return False
    for rail in activated:
        name = (getattr(rail, "name", "") or "").lower()
        if name in BLOCKING_FLOWS and getattr(rail, "stop", True):
            return True
    return False


def guard(message: str, history: str = "") -> tuple[bool, str | None]:
    """
    Run a message through the gate.

    Returns (True, refusal) if a rail fired - return it and skip the pipeline -
    or (False, None) if the message is clean.

    Fails OPEN: a gate outage degrades protection rather than taking the service
    down, and layer 4 still applies. Every fail-open path logs at error level so
    degraded protection is visible rather than silent.
    """
    global _pending_history

    if _rails is None:
        logfire.warning("Guardrails not initialised - skipping gate.")
        return False, None

    with logfire.span("Guardrails"):
        _pending_history = history or ""
        try:
            result = _rails.generate(
                messages=[{"role": "user", "content": message}],
                options={"rails": ["input"], "log": {"activated_rails": True}},
            )
        except Exception as e:
            logfire.error(f"Guardrails check failed, failing open: {e}")
            return False, None
        finally:
            _pending_history = ""

        if _blocking_rail_activated(result):
            logfire.info(f"Guardrails fired | {message[:80]!r}")
            return True, _extract_response_text(result)

        return False, None


def reset_for_tests() -> None:
    global _rails
    _rails = None
