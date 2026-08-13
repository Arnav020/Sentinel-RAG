from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. Respond with 'CONVERSATIONAL' ONLY if the latest message is a greeting
       (hi, hello, thanks, bye) or can be answered using ONLY the conversation
       history above (e.g. "what did I just ask?", "what is my name").
    2. For ANY question seeking factual or technical information, output a
       refined search query. The knowledge base covers Kubernetes (Jobs,
       CronJobs, work queues, autoscaling, monitoring), Databricks job
       management (CLI, SDK, REST API), Intel hardware, and enterprise
       networking.

    When in doubt, prefer the search query — answering from documentation is
    always safer than answering from memory.

    The search query is embedded and matched against documentation chunks, so it
    must be a short natural-language phrase of keywords (3-12 words).
    Do NOT output a URL, a file path, an explanation, or a full sentence.

    Examples:
      "hi there"                                        -> CONVERSATIONAL
      "what did I just ask?"                            -> CONVERSATIONAL
      "How do I monitor a Kubernetes Job?"              -> Kubernetes Job status monitoring
      "how does HPA scale my pods"                      -> Kubernetes horizontal pod autoscaling
      "Which Databricks CLI command lists jobs?"        -> Databricks CLI job management commands
      "When should I use the REST API over the SDK?"    -> Databricks REST API versus SDK usage

    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    
    with logfire.span("🧠 Planner Decision"):
        try:
            decision = llm.invoke(prompt).content.strip()
        except Exception as e:
            # Degrade instead of crashing the whole request: treat the raw
            # message as a search query rather than failing the turn outright.
            logfire.error(f"Planner LLM call failed, defaulting to raw query: {e}")
            decision = user_message
        logfire.info(f"Intent identified: {decision}")

    # Normalize before comparing — LLMs don't always emit the bare token
    # exactly ("CONVERSATIONAL.", quoted, different case, etc.).
    is_conversational = decision.strip().strip(".,!?\"'").upper() == "CONVERSATIONAL"

    if is_conversational:
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
