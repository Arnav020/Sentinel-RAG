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
- ANY follow-up that continues the conversation, however short or bare -
  "elaborate", "explain in more detail", "why?", "go on", "what about memory?".
  Judge these by what the conversation is about, not by the words in the message
  itself. A follow-up in a {subject} conversation is IN_SCOPE.

Reply OFF_TOPIC only for clearly unrelated requests: jokes, poems, creative
writing, general trivia, sports, cooking, films, travel, homework, personal or
medical advice, politics, or programming questions with no {subject} connection.

When a message is technical, infrastructure-related, or ambiguous, answer
IN_SCOPE. Wrongly blocking a real engineering question is far worse than
occasionally letting a borderline one through, because the retrieval stage
declines to answer anything the documentation does not cover.

Reply with exactly one word: IN_SCOPE or OFF_TOPIC."""


def is_off_topic(message: str, history: str = "") -> bool:
    """
    True when the message is outside the assistant's subject.

    History matters and its absence was a real bug. The gate runs before the
    planner, on the raw message, so a bare follow-up - "explain in elaborate",
    "why?", "go on" - carries no subject of its own and was classified
    OFF_TOPIC and blocked, even mid-conversation about Kubernetes. Judged
    alongside the preceding turns it is obviously in scope.
    """
    scope = get_scope()
    system = _TEMPLATE.format(subject=scope.subject, scope=scope.bullet_list())

    payload = (
        f"CONVERSATION SO FAR:\n{history}\n\nLATEST MESSAGE:\n{message}"
        if history.strip()
        else message
    )
    verdict = classify(system, payload, model=settings.TOPIC_FILTER_MODEL)
    if not verdict:
        logfire.error("Topic filter unavailable, failing open.")
        return False

    off_topic = "OFF_TOPIC" in verdict
    if off_topic:
        logfire.info(f"Off-topic detected | {message[:80]!r}")
    return off_topic
