"""
Colang policy for the guardrail gate.

NeMo expresses and enforces policy; detection is delegated to purpose-built
components registered as Colang actions (see rails.py). That separation is what
lets the gate use a dedicated injection classifier instead of few-shot prompting
a chat model, and what makes the whole gate testable offline by stubbing two
functions.

Both flows run as *input rails*, so a blocked message never reaches retrieval or
generation.
"""

from __future__ import annotations

COLANG_CONTENT = """
define bot refuse injection
  "I keep to the same guidelines however I am asked. I can help with Kubernetes documentation - what would you like to know?"

define bot refuse off topic
  "I'm a Kubernetes documentation assistant, so that one is outside what I can help with. Ask me anything about Kubernetes workloads, configuration, networking, storage, security or operations."

define flow check injection
  $is_injection = execute detect_injection_action
  if $is_injection
    bot refuse injection
    stop

define flow check off topic
  $is_off_topic = execute detect_off_topic_action
  if $is_off_topic
    bot refuse off topic
    stop
"""

# `models: []` is deliberate: this config runs input rails only and every
# decision comes from a registered action, so NeMo never needs a generation
# model of its own.
YAML_CONTENT = """
models: []

rails:
  input:
    flows:
      - check injection
      - check off topic
"""

# rails.py reads NeMo's structured activation log for these names rather than
# substring-matching the refusal text, so rewording a refusal above cannot
# silently disable blocking.
BLOCKING_FLOWS = ("check injection", "check off topic")
