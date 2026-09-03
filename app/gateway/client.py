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

import threading
import time
from collections import deque
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

        result = LLMResponse(
            content=content,
            model=str(answered_by),
            cache_hit=cache_hit,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        record_usage(model, result.total_tokens)
        return result

    except LLMError:
        raise
    except Exception as e:
        note_if_daily_quota(model, str(e))
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


# --------------------------------------------------------------------------
# Token accounting
#
# Every provider limit that actually bites is a *token* limit, not a request
# limit, and neither the response body nor the rate-limit headers report how
# much of the per-minute budget is left after the call. Batch callers were left
# pacing on a flat sleep, which stays under the limit only by accident: a
# behaviour run at 1.5s per item drew 236 rate-limit errors and lost 73 of 145
# items to failed generation, because a reasoning model's hidden reasoning
# tokens cost several times the visible answer.
#
# Recording what each call actually cost lets a caller pace on measurement.
# Bounded, in-memory, and never consulted on the serving path, so production is
# unaffected.
# --------------------------------------------------------------------------

_USAGE_WINDOW_SECONDS = 300
_usage_log: deque[tuple[float, str, int]] = deque(maxlen=4096)
_usage_lock = threading.Lock()


def record_usage(model: str, total_tokens: int) -> None:
    """Note that `model` just spent `total_tokens`."""
    if total_tokens <= 0:
        return
    now = time.time()
    with _usage_lock:
        _usage_log.append((now, model, total_tokens))
        cutoff = now - _USAGE_WINDOW_SECONDS
        while _usage_log and _usage_log[0][0] < cutoff:
            _usage_log.popleft()


def tokens_used_since(model: str, seconds: float) -> int:
    """Tokens `model` has spent in the last `seconds`. 0 if it has spent none."""
    cutoff = time.time() - seconds
    with _usage_lock:
        return sum(t for ts, m, t in _usage_log if m == model and ts >= cutoff)


# --------------------------------------------------------------------------
# Daily-quota observation
#
# Groq enforces two limits with the same 429: tokens per minute, which a pause
# clears, and tokens per DAY, which a pause does not. They are worth telling
# apart, because the second one means every remaining call in a batch will fail
# too. A behaviour run learned this expensively - it kept going for 17 minutes
# after the budget was gone, failed 73 of 145 items, and still printed a
# summary table of numbers that looked like measurements.
#
# Observation only: `complete` re-raises exactly as before, so production
# degradation policy is unchanged. Batch callers poll this to stop early.
# --------------------------------------------------------------------------

_daily_quota_hits: dict[str, float] = {}


def note_if_daily_quota(model: str, error_text: str) -> None:
    """Record that `model` reported a per-DAY token limit, not a per-minute one."""
    lowered = error_text.lower()
    if "tokens per day" in lowered or "(tpd)" in lowered:
        _daily_quota_hits[model] = time.time()
        logfire.error(f"{model} has exhausted its daily token budget.")


def daily_quota_exhausted(within_seconds: float = 600) -> set[str]:
    """Models that reported a per-day token limit within the last `within_seconds`."""
    cutoff = time.time() - within_seconds
    return {m for m, ts in _daily_quota_hits.items() if ts >= cutoff}


def clear_daily_quota_notes() -> None:
    """Forget recorded exhaustion - for tests and for retrying after a reset."""
    _daily_quota_hits.clear()
