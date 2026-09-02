"""
LLM access.

One entry point for every model call in the system, because the previous split
- LangChain `ChatOpenAI` in the planner, the native Portkey client in the
responder, a bare `groq.Groq` in each guardrail - meant no single place knew
which model actually answered, whether a response came from cache, or what it
cost.

Three things this fixes:

  * **Portkey is optional.** With no gateway key the app calls Groq directly, so
    local development, tests and CI work without a gateway account.
  * **The answering model is reported.** A gateway fallback silently substitutes
    a weaker model; an evaluation that cannot see that attributes the weaker
    model's output to the stronger one.
  * **Cache can be bypassed.** Evaluation must never grade a cached generation,
    or a repeat run measures the cache rather than the system.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import logfire

from app.config import settings


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    cache_hit: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """A model call failed after the client's own retries."""


_groq_client = None
_portkey_client = None


def _groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=settings.GROQ_API_KEY, max_retries=3)
    return _groq_client


def _gateway_config(model: str, bypass_cache: bool) -> Any:
    """
    Build a per-call gateway config.

    A saved config slug overrides `override_params`, so referencing one means
    the gateway - not this code - chooses the model. That is why the primary
    target is built from the model the caller actually asked for: routing a
    planner call to the generation model, as the previous hardcoded target list
    did, silently changes both cost and behaviour.
    """
    if settings.PORTKEY_CONFIG_SLUG:
        return settings.PORTKEY_CONFIG_SLUG
    config: dict[str, Any] = {
        "strategy": {"mode": "fallback"},
        "retry": {"attempts": 2, "on_status_codes": [429, 503]},
        "targets": [
            {"override_params": {"model": f"@{settings.GROQ_SLUG}/{model}"}},
            {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{model}"}},
        ],
    }
    if settings.GATEWAY_CACHE_ENABLED and not bypass_cache:
        config["cache"] = {"mode": "simple"}
    return config


def _portkey(model: str, bypass_cache: bool):
    """A client per (model, cache) pair - the config differs, so the client must."""
    global _portkey_client
    key = (model, bypass_cache)
    if _portkey_client is None or _portkey_client[0] != key:
        from portkey_ai import Portkey

        client = Portkey(
            api_key=settings.PORTKEY_API_KEY,
            config=_gateway_config(model, bypass_cache),
        )
        _portkey_client = (key, client)
    return _portkey_client[1]


def _cache_status(response) -> bool:
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", None) or {}
            status = headers.get("x-portkey-cache-status", "")
            if status:
                return status.upper() == "HIT"
    return False


def _is_reasoning_model(model: str) -> bool:
    """gpt-oss models emit reasoning tokens before content and accept effort."""
    return "gpt-oss" in model


def complete(
    messages: Iterable[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    feature: str = "rag",
    bypass_cache: bool = False,
) -> LLMResponse:
    """
    Run a chat completion. Raises LLMError on failure - callers decide the
    degradation policy rather than having an empty string handed to them.
    """
    model = model or settings.GENERATION_MODEL
    max_tokens = max_tokens or settings.GENERATION_MAX_TOKENS
    messages = list(messages)

    kwargs: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if _is_reasoning_model(model):
        # Without headroom the reasoning preamble consumes the whole budget and
        # `content` comes back empty - a silent failure, not an error.
        kwargs["reasoning_effort"] = settings.REASONING_EFFORT

    try:
        if settings.USE_GATEWAY and settings.PORTKEY_API_KEY:
            client = _portkey(model, bypass_cache)
            response = client.chat.completions.create(**kwargs)
            cache_hit = _cache_status(response)
            answered_by = getattr(response, "model", model) or model
        else:
            response = _groq().chat.completions.create(model=model, **kwargs)
            cache_hit = False
            answered_by = getattr(response, "model", model) or model

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        usage = getattr(response, "usage", None)

        if not content:
            raise LLMError(
                f"{model} returned empty content "
                f"(finish_reason={getattr(choice, 'finish_reason', '?')}). "
                "For a reasoning model this usually means max_tokens was too small."
            )

        return LLMResponse(
            content=content,
            model=str(answered_by),
            cache_hit=cache_hit,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    except LLMError:
        raise
    except Exception as e:
        logfire.error(f"LLM call failed ({model}, feature={feature}): {e}")
        raise LLMError(f"{type(e).__name__}: {e}") from e


def classify(
    system_prompt: str,
    user_message: str,
    *,
    model: str,
    max_tokens: int | None = None,
) -> str:
    """
    Single-label classification helper used by the guardrails.

    Returns the raw upper-cased verdict; an empty string means the call failed
    and the caller applies its own fail-open / fail-closed policy.
    """
    try:
        r = complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens or settings.CLASSIFIER_MAX_TOKENS,
            feature="guardrail",
            bypass_cache=False,
        )
        return r.content.strip().upper()
    except LLMError as e:
        logfire.error(f"Classifier {model} unavailable: {e}")
        return ""


def reset_clients() -> None:
    """Drop cached clients so tests can change configuration."""
    global _groq_client, _portkey_client
    _groq_client, _portkey_client = None, None
