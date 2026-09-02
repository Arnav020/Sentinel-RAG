"""
Layer 3: topical scope enforcement.

The scope description is built from the corpus at call time rather than being
hardcoded. The previous prompt listed Databricks, Intel hardware and enterprise
networking - none of which the knowledge base contained - so it admitted
questions the system could not ground and then answered them anyway.

This filter is a fast, friendly rejection for obviously unrelated requests. It is
NOT the mechanism that keeps answers grounded: that is the retrieval relevance
threshold, which measures coverage directly instead of predicting it. Because a
second, stronger control exists downstream, this one fails OPEN - an outage here
costs scope politeness, not grounding.
"""

from __future__ import annotations

import logfire

from app.config import settings
from app.gateway import classify
from app.services.scope import get_scope

_TEMPLATE = """You are a topic classifier for a {subject} documentation assistant.

The assistant's knowledge base covers:
{scope}

Reply IN_SCOPE for:
- Any question about {subject} concepts, configuration, commands, YAML, APIs,
  troubleshooting or operations, including areas of {subject} not listed above.
- Greetings, farewells, thanks, and questions about the assistant's capabilities.
- Follow-up messages that only make sense with earlier conversation.

Reply OFF_TOPIC only for clearly unrelated requests: jokes, poems, creative
writing, general trivia, sports, cooking, films, travel, homework, personal or
medical advice, politics, or programming questions with no {subject} connection.

When a message is technical, infrastructure-related, or ambiguous, answer
IN_SCOPE. Wrongly blocking a real engineering question is far worse than
occasionally letting a borderline one through, because the retrieval stage
declines to answer anything the documentation does not cover.

Reply with exactly one word: IN_SCOPE or OFF_TOPIC."""


def is_off_topic(message: str) -> bool:
    scope = get_scope()
    system = _TEMPLATE.format(subject=scope.subject, scope=scope.bullet_list())

    verdict = classify(system, message, model=settings.TOPIC_FILTER_MODEL)
    if not verdict:
        logfire.error("Topic filter unavailable, failing open.")
        return False

    off_topic = "OFF_TOPIC" in verdict
    if off_topic:
        logfire.info(f"Off-topic detected | {message[:80]!r}")
    return off_topic
