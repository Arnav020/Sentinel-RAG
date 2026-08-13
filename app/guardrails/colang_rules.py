# Colang policy for the production guardrail system.
#
# Design note — why this is input rails + custom actions rather than the
# few-shot dialog rails the tutorial notebooks used:
#
# The original design classified intent by few-shot prompting a general chat
# model through NeMo's dialog rails. That cost ~2,300 tokens across 3 LLM calls
# on *every* message (which exhausted the daily token budget of the generation
# model and added ~3-25s of latency to every request), and it still missed
# paraphrased jailbreaks because it pattern-matched canonical examples instead
# of generalising.
#
# Here NeMo does what it is genuinely good at — expressing and enforcing policy
# — while the actual detection is delegated to purpose-built components
# registered as Colang actions (see app/guardrails/rails.py):
#   * detect_injection_action — Llama Prompt Guard 2, a dedicated injection classifier
#   * detect_off_topic_action — one compact scoped classification call
# Both run as *input rails*, so a blocked message never reaches generation.

COLANG_CONTENT = """
define bot refuse injection
  "I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?"

define bot refuse off topic
  "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!"

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

YAML_CONTENT = """
models: []

rails:
  input:
    flows:
      - check injection
      - check off topic

instructions:
  - type: general
    content: |
      You are an Enterprise IT Assistant specialising in:
      - Kubernetes (deployment, scaling, operators, networking)
      - Intel hardware (CPUs, FPGAs, NICs, SRIOV)
      - Enterprise networking (SDN, VLANs, BGP, routing)
      Only answer questions about these topics. Be professional and concise.
"""

# The two flows above are the only ones that block. rails.py checks NeMo's
# structured `activated_rails` log for these names rather than substring-matching
# the response text, so editing the refusal wording above cannot silently
# disable blocking.
BLOCKING_FLOWS = ("check injection", "check off topic")
