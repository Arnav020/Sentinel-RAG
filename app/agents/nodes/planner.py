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

_PROMPT = """You rewrite user messages into self-contained questions for a {subject} documentation assistant.

The knowledge base covers:
{scope}

Decide between two outputs:

1. Reply with exactly CONVERSATIONAL if the latest message is a greeting, a
   thanks or farewell, a question about your own capabilities, or something
   answerable purely from the conversation history (for example "what did I
   just ask?").

2. Otherwise reply with a SELF-CONTAINED QUESTION - a complete question that
   makes sense on its own, without the conversation.

   Preserve exactly what was asked. Do not compress the question into keywords
   and do not swap in a nearby technical term: "how does an HPA decide how many
   replicas to run" is a different question from "HorizontalPodAutoscaler
   scaling behaviour", and answering the second does not answer the first.

   Your only real job is to resolve references. Pronouns, "it", "that", and
   bare follow-ups like "elaborate" or "why?" must be replaced with the subject
   from the conversation history, so the question stands alone.

Then, on a second line, give the exact identifiers a documentation page would
use for this - field names, API kinds, camelCase spellings, allowed values.
Dense retrieval is sensitive to spelling: "restart policies" and
"RestartPolicy" embed differently, and the passage that answers a question
often uses the identifier rather than the prose form. Both queries are searched.

Reply in exactly this form, and nothing else:

QUESTION: <the self-contained question>
KEYWORDS: <identifiers and exact terms, or the key nouns if there are none>

Examples:
  "hi there"                                -> CONVERSATIONAL
  "what did I just ask?"                    -> CONVERSATIONAL
  "hi there"
      -> CONVERSATIONAL

  "what restart policies can a Job use?"
      -> QUESTION: What restart policies can a Kubernetes Job use?
         KEYWORDS: Job pod template restartPolicy Never OnFailure allowed

  "how does HPA scale my pods"
      -> QUESTION: How does a HorizontalPodAutoscaler decide how many replicas to run?
         KEYWORDS: HorizontalPodAutoscaler desiredReplicas currentMetricValue algorithm

  "what about memory?"  (after CPU limits)
      -> QUESTION: How do I set memory limits and requests on a container?
         KEYWORDS: resources limits requests memory container

  "could you elaborate?"  (after Pods)
      -> QUESTION: Can you explain Kubernetes Pods in more detail?
         KEYWORDS: Pod containers shared namespaces volumes pod template

  "tell me about the weather"
      -> QUESTION: What is the weather forecast?
         KEYWORDS: weather forecast

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

        question, keywords = _parse(decision, user_message)
        logfire.info(f"Planner decision: {'CONVERSATIONAL' if is_conversational else question}")

    if is_conversational:
        return {
            "current_query": CONVERSATIONAL,
            "search_terms": "",
            "status": "Answering from the conversation.",
            "plan": ["Intent: conversational", "Retrieval: skipped"],
        }

    plan = ["Intent: documentation lookup", f"Search query: {question}"]
    if keywords:
        plan.append(f"Keyword query: {keywords}")

    return {
        "current_query": question,
        "search_terms": keywords,
        "status": f"Searching the knowledge base for: {question}",
        "plan": plan,
    }


def _parse(raw: str, fallback: str) -> tuple[str, str]:
    """
    Pull the question and keyword line out of the planner's reply.

    Deliberately forgiving: a malformed reply degrades to "use the first usable
    line as the question, no keywords", which is exactly the behaviour before
    the keyword query existed. A parse failure must never cost an answer.
    """
    question, keywords = "", ""
    for line in raw.splitlines():
        line = line.strip().strip("`")
        low = line.lower()
        if low.startswith("question:"):
            question = line.split(":", 1)[1].strip().strip("\"'")
        elif low.startswith("keywords:"):
            keywords = line.split(":", 1)[1].strip().strip("\"'")

    if not question:
        # No labels - take the first substantial line, as before.
        for line in raw.splitlines():
            line = line.strip().strip("\"'`")
            if len(line) > 8:
                question = line
                break

    if not question or len(question) > 300:
        question = fallback
    return question, _clean_keywords(keywords)


# Special tokens a model can emit verbatim when it fails to stop cleanly, plus
# the run-on filler that follows them.
_STOP_MARKERS = ("<|endoftext|>", "<|", "```", "##")


def _clean_keywords(raw: str) -> str:
    """
    Trim the keyword line down to something worth searching, or drop it.

    Small models do not always stop after the keyword line. One observed reply
    was "Pod containers shared namespaces volumes pod template<|endoftext|>## 1
    The - - - ... ... .. ...", which is a valid keyword list with several
    hundred characters of degenerate continuation welded on. Searching that
    wastes an embedding call and shows the user a broken trace, so the
    continuation is cut at the first stop marker and anything still looking
    degenerate is discarded entirely.

    Dropping is safe: no keyword query simply means the search falls back to the
    question alone, which is where it started.
    """
    text = raw
    for marker in _STOP_MARKERS:
        cut = text.find(marker)
        if cut != -1:
            text = text[:cut]

    words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if not words:
        return ""

    # Degenerate output is dominated by punctuation runs and repeats.
    if len({w.lower() for w in words}) < len(words) * 0.6:
        return ""

    cleaned = " ".join(words[:20])
    return cleaned if 2 <= len(cleaned) <= 200 else ""
