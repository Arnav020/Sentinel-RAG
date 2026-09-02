"""
Planner: decide whether the turn needs retrieval, and if so rewrite the query.

Two corrections over the previous version:

  * The scope description is derived from the corpus rather than hardcoded. The
    old prompt advertised Databricks, Intel hardware and enterprise networking,
    none of which the knowledge base covers, so it routed questions to a store
    that could not answer them.
  * Conversation history is bounded (see `format_history`) instead of being
    replayed in full on every turn.
"""

from __future__ import annotations

import logfire

from app.agents.state import CONVERSATIONAL, AgentState, format_history, latest_user_message
from app.config import settings
from app.gateway import LLMError, complete
from app.services.scope import get_scope

_PROMPT = """You rewrite user messages into search queries for a {subject} documentation assistant.

The knowledge base covers:
{scope}

Decide between two outputs:

1. Reply with exactly CONVERSATIONAL if the latest message is a greeting, a
   thanks or farewell, a question about your own capabilities, or something
   answerable purely from the conversation history (for example "what did I
   just ask?").

2. Otherwise reply with a search query: a short noun phrase of 3-12 words in
   the vocabulary the documentation itself would use. Resolve pronouns and
   follow-ups against the history, so "how do I scale it?" after a question
   about Deployments becomes "scale a Deployment replicas".

Reply with the query alone. No explanation, no quotes, no URL, no file path,
and never a full sentence.

Examples:
  "hi there"                                 -> CONVERSATIONAL
  "what did I just ask?"                     -> CONVERSATIONAL
  "How do I monitor a Job?"                  -> Kubernetes Job status monitoring
  "how does HPA scale my pods"               -> HorizontalPodAutoscaler scaling behaviour
  "what about memory?"  (after CPU limits)   -> container memory limits and requests
  "tell me about the weather"                -> weather forecast

Note the last example: rewrite the query even when it is clearly outside the
knowledge base. Deciding what the documentation covers is not your job - the
retrieval stage measures that directly and declines when nothing matches."""


def planner_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    user_message = latest_user_message(messages)
    history = format_history(messages)
    scope = get_scope()

    system = _PROMPT.format(subject=scope.subject, scope=scope.bullet_list())
    user = f"CONVERSATION HISTORY:\n{history or '(none)'}\n\nLATEST MESSAGE:\n{user_message}"

    with logfire.span("Planner"):
        try:
            result = complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=settings.PLANNER_MODEL,
                temperature=0.0,
                max_tokens=settings.CLASSIFIER_MAX_TOKENS,
                feature="planner",
            )
            decision = result.content.strip()
        except LLMError as e:
            # Degrade rather than fail the turn: the raw message is a usable, if
            # unrefined, search query. Logged at error so the degradation is
            # visible rather than silently changing answer quality.
            logfire.error(f"Planner unavailable, using the raw message as the query: {e}")
            decision = user_message

        # Normalise: models do not reliably emit the bare token.
        normalised = decision.strip().strip(".,!?\"'`").upper()
        is_conversational = normalised == "CONVERSATIONAL"

        # A multi-line or over-long reply means the model explained itself
        # instead of answering; the first line is the usable part.
        if not is_conversational:
            decision = decision.splitlines()[0].strip().strip("\"'`")
            if len(decision) > 200 or not decision:
                decision = user_message

        logfire.info(f"Planner decision: {'CONVERSATIONAL' if is_conversational else decision}")

    if is_conversational:
        return {
            "current_query": CONVERSATIONAL,
            "status": "Answering from the conversation.",
            "plan": ["Intent: conversational", "Retrieval: skipped"],
        }

    return {
        "current_query": decision,
        "status": f"Searching the knowledge base for: {decision}",
        "plan": ["Intent: documentation lookup", f"Search query: {decision}"],
    }
