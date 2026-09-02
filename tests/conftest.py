"""
Shared test fixtures.

Design rule for this suite: the unit tier must run with no network, no secrets
and no model downloads, because a test suite that needs credentials is a test
suite that will not run in CI - which is how this project ended up with none.

Anything that genuinely needs Qdrant or a model is marked `integration` and
skipped unless the environment provides it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set before any app import so config picks up test-safe values.
os.environ.setdefault("QDRANT_CLUSTER_ENDPOINT", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_ENABLED", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("USE_GATEWAY", "false")

import logfire

logfire.configure(send_to_logfire=False, service_name="tests")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: requires a live Qdrant and/or model downloads")
    config.addinivalue_line("markers", "llm: requires a live LLM API key")


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_html(tmp_path: Path) -> Path:
    """
    An HTML document shaped like the real corpus: headings, prose, a code block
    and a table, with NO blank lines between elements.

    That last property is the whole point. The previous chunker split on blank
    lines, and real HTML extraction never produces one, which is why four of six
    documents were chunked as a single undifferentiated slab.
    """
    doc = (
        "<!--\ntitle: Test Doc\nsource_url: https://example.test/doc\n"
        "doc_topic: workloads\n-->\n"
        "<h1>Test Doc</h1>\n"
        "<p>A Job creates one or more Pods and retries until a specified number complete.</p>\n"
        "<h2>Restart policies</h2>\n"
        "<p>Jobs have strict requirements for Pod restart policies and only two are valid.</p>\n"
        "<pre>apiVersion: batch/v1\nkind: Job\nspec:\n  template:\n    spec:\n"
        "      restartPolicy: OnFailure</pre>\n"
        "<h2>Parallelism</h2>\n"
        "<p>The parallelism field controls how many Pods run at the same time.</p>\n"
        "<table><tr><th>Field</th><th>Meaning</th></tr>"
        "<tr><td>completions</td><td>Successful Pods required</td></tr>"
        "<tr><td>parallelism</td><td>Maximum concurrent Pods</td></tr></table>\n"
        "<ul><li>Use OnFailure to retry in place</li><li>Use Never to create a new Pod</li></ul>\n"
    )
    path = tmp_path / "sample.html"
    path.write_text(doc, encoding="utf-8")
    return path


@pytest.fixture
def sample_text(tmp_path: Path) -> Path:
    doc = (
        "OVERVIEW OF STEPS\n\n"
        "1. Start a storage service to hold the work queue.\n"
        "2. Create a queue and fill it with messages.\n\n"
        "STARTING REDIS\n\n"
        "For this example, you will start a single instance of Redis.\n\n"
        "    kubectl apply -f https://k8s.io/examples/redis-pod.yaml\n"
        "    kubectl apply -f https://k8s.io/examples/redis-service.yaml\n"
    )
    path = tmp_path / "sample.txt"
    path.write_text(doc, encoding="utf-8")
    return path


@pytest.fixture
def stub_llm(monkeypatch):
    """
    Replace every LLM call with a deterministic function.

    Returns a recorder so a test can assert on what the system asked for as well
    as what it did with the reply.
    """
    from app.gateway import client as gw

    calls: list[dict] = []

    def make(reply: str = "stub reply"):
        def fake_complete(
            messages,
            *,
            model=None,
            temperature=0.1,
            max_tokens=None,
            feature="rag",
            bypass_cache=False,
        ):
            calls.append({"messages": list(messages), "model": model, "feature": feature})
            value = reply(feature) if callable(reply) else reply
            return gw.LLMResponse(content=value, model=model or "stub-model")

        monkeypatch.setattr(gw, "complete", fake_complete)
        # Re-export sites bind the name at import time.
        for mod in (
            "app.agents.nodes.planner",
            "app.agents.nodes.responder",
            "app.guardrails.prompt_guard",
        ):
            if mod in sys.modules:
                monkeypatch.setattr(sys.modules[mod], "complete", fake_complete, raising=False)
        return calls

    return make
