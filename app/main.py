# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import sys

if sys.platform == "win32":
    # Windows consoles default to the cp1252 codepage, which can't encode the
    # emoji used throughout this codebase's log messages — without this,
    # every logfire span print raises UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard, rails_ready

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


@app.on_event("startup")
def startup_event():
    initialize_rails()

class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"
    
    
@app.get("/health")
def health():
    """
    Liveness + readiness for container health checks and the web client's
    status indicator. Reports whether the guardrail gate actually initialised,
    so a backend serving traffic with the gate down is visibly degraded rather
    than silently unprotected.
    """
    return {
        "status": "ok",
        "guardrails": rails_ready(),
    }


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }


# ─────────────────────────────────────────────────────────────────────────────
# Web client
#
# Mounted LAST on purpose: FastAPI resolves routes in declaration order, so
# every API route above still wins over the catch-all static mount.
#
# Serving the UI from this same process removes a whole container, a Node
# toolchain and a cross-origin hop — the client is plain HTML/CSS/JS with no
# build step, so there is nothing to compile or keep in sync.
# ─────────────────────────────────────────────────────────────────────────────
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:
    # Never fail startup over a missing UI — the API is the product surface the
    # eval suite and any programmatic client depend on.
    logfire.warning(f"⚠️ Web client directory not found at {_WEB_DIR} — API-only mode.")
