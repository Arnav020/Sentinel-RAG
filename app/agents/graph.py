"""
The LangGraph state machine.

planner -> (retriever -> responder | responder) -> END

The retriever is skipped only for turns the planner marks conversational.
Every other turn goes through retrieval, and retrieval decides for itself
whether anything relevant exists - the responder is never asked to guess.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import CONVERSATIONAL, AgentState


def route_planner(state: AgentState) -> str:
    return "responder" if state.get("current_query") == CONVERSATIONAL else "retriever"


def build_graph(checkpointer=None):
    """
    Compile the graph.

    Exposed as a function so tests can build an isolated graph with their own
    checkpointer instead of sharing the module-level singleton's memory.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)

    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner", route_planner, {"retriever": "retriever", "responder": "responder"}
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


rag_agent = build_graph()
