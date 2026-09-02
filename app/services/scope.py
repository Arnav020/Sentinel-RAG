"""
The assistant's scope, derived from the corpus.

Previously the advertised scope lived in four hand-maintained prose lists - the
planner prompt, the topic-filter prompt, the responder prompt and the Colang
instructions - and all four had drifted away from what was actually indexed.
They claimed Intel hardware and enterprise networking coverage the corpus did
not contain, so the topic filter admitted questions the knowledge base could not
answer and the system answered them from whatever was nearest in vector space.

Scope is now a single derived value: whatever `doc_topic`s exist in the
collection. Add documents and the scope widens automatically; remove them and it
narrows. The prompts, the /health endpoint and the web client all read from here,
so they cannot disagree with each other or with the index.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import logfire

from app.services.retrieval.qdrant_service import (
    RetrievalError,
    collection_stats,
    distinct_payload_values,
)

# Human-readable descriptions for the topic keys produced by the corpus builder.
# A topic missing from this map still works - it just shows its raw key - so
# adding documents never requires editing code to keep the UI correct.
TOPIC_LABELS: dict[str, str] = {
    "workloads": "Workloads - Pods, Deployments, ReplicaSets, StatefulSets, DaemonSets, Jobs, CronJobs",
    "jobs": "Jobs & work queues - parallel processing, indexed Jobs, failure policies",
    "autoscaling": "Autoscaling - HorizontalPodAutoscaler, resource resizing, metrics",
    "configuration": "Configuration - ConfigMaps, Secrets, resource limits, probes",
    "networking": "Networking - Services, Ingress, NetworkPolicies, cluster DNS",
    "storage": "Storage - Volumes, PersistentVolumes, StorageClasses",
    "architecture": "Cluster architecture - control plane, nodes, scheduler, etcd, controllers",
    "security": "Security - RBAC, ServiceAccounts, security contexts, Pod Security Standards",
    "operations": "Operations - kubectl, debugging Pods, logging, resource monitoring",
}

SUBJECT = "Kubernetes"


@dataclass(slots=True)
class Scope:
    subject: str = SUBJECT
    topics: dict[str, int] = field(default_factory=dict)
    points: int = 0
    available: bool = False

    @property
    def topic_keys(self) -> list[str]:
        return list(self.topics)

    def bullet_list(self) -> str:
        """Multi-line description used inside prompts."""
        if not self.topics:
            return f"- {SUBJECT} concepts, configuration and operations"
        return "\n".join(
            f"- {TOPIC_LABELS.get(k, k.replace('_', ' ').title())}" for k in self.topics
        )

    def short_summary(self) -> str:
        if not self.topics:
            return f"{SUBJECT} documentation"
        return f"{SUBJECT} documentation across {len(self.topics)} areas ({self.points} passages)"

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "available": self.available,
            "points": self.points,
            "topics": [
                {
                    "key": k,
                    "label": TOPIC_LABELS.get(k, k.replace("_", " ").title()),
                    "passages": v,
                }
                for k, v in self.topics.items()
            ],
        }


_scope: Scope | None = None
_lock = threading.Lock()


def refresh() -> Scope:
    """Re-read the corpus. Called at startup and whenever scope is first needed."""
    global _scope
    with _lock:
        try:
            stats = collection_stats()
            topics = distinct_payload_values("doc_topic")
            _scope = Scope(
                subject=SUBJECT,
                topics=topics,
                points=int(stats.get("points") or 0),
                available=True,
            )
            logfire.info(
                f"Scope derived from corpus: {len(topics)} topics, {_scope.points} passages."
            )
        except RetrievalError as e:
            # Serve a degraded but honest scope rather than inventing coverage.
            logfire.error(f"Could not derive scope from corpus: {e}")
            _scope = Scope(subject=SUBJECT, topics={}, points=0, available=False)
        return _scope


def get_scope() -> Scope:
    if _scope is None:
        return refresh()
    return _scope


def reset_for_tests(scope: Scope | None = None) -> None:
    global _scope
    _scope = scope
