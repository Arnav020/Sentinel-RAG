"""
Build the Kubernetes documentation corpus from kubernetes.io.

The corpus IS the product scope: whatever is fetched here is exactly what the
assistant can answer about, and nothing else. Pages are stored as cleaned HTML
fragments (article content only, navigation and chrome removed) so the ingestion
loader still exercises real HTML structure parsing rather than pre-flattened text.

kubernetes.io documentation is CC BY 4.0. Attribution is written into every
document's front matter and survives into chunk metadata as `source_url`.

    python tools/fetch_corpus.py            # fetch into DATA/kubernetes/
    python tools/fetch_corpus.py --force    # refetch pages already on disk
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE = "https://kubernetes.io"
OUT = Path(__file__).resolve().parent.parent / "DATA" / "kubernetes"
UA = "Mozilla/5.0 (compatible; sentinel-rag-docs-ingest/2.0)"

# Curated so the corpus is coherent and self-contained: the concepts a platform
# engineer actually asks about, plus the task pages that carry the commands.
# The group becomes `doc_topic` metadata, which drives stratified evaluation and
# the scope description shown to users.
PAGES: list[tuple[str, str]] = [
    # --- workloads -----------------------------------------------------------
    ("workloads", "/docs/concepts/workloads/pods/"),
    ("workloads", "/docs/concepts/workloads/pods/pod-lifecycle/"),
    ("workloads", "/docs/concepts/workloads/controllers/deployment/"),
    ("workloads", "/docs/concepts/workloads/controllers/replicaset/"),
    ("workloads", "/docs/concepts/workloads/controllers/statefulset/"),
    ("workloads", "/docs/concepts/workloads/controllers/daemonset/"),
    ("workloads", "/docs/concepts/workloads/controllers/job/"),
    ("workloads", "/docs/concepts/workloads/controllers/cron-jobs/"),
    ("workloads", "/docs/concepts/workloads/controllers/ttlafterfinished/"),
    # --- jobs and work queues (the original demo's core subject) -------------
    ("jobs", "/docs/tasks/job/parallel-processing-expansion/"),
    ("jobs", "/docs/tasks/job/coarse-parallel-processing-work-queue/"),
    ("jobs", "/docs/tasks/job/fine-parallel-processing-work-queue/"),
    ("jobs", "/docs/tasks/job/indexed-parallel-processing-static/"),
    ("jobs", "/docs/tasks/job/automated-tasks-with-cron-jobs/"),
    ("jobs", "/docs/tasks/job/pod-failure-policy/"),
    # --- autoscaling ---------------------------------------------------------
    ("autoscaling", "/docs/tasks/run-application/horizontal-pod-autoscale/"),
    ("autoscaling", "/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/"),
    ("autoscaling", "/docs/concepts/workloads/autoscaling/"),
    ("autoscaling", "/docs/tasks/configure-pod-container/resize-container-resources/"),
    # --- configuration -------------------------------------------------------
    ("configuration", "/docs/concepts/configuration/configmap/"),
    ("configuration", "/docs/concepts/configuration/secret/"),
    ("configuration", "/docs/concepts/configuration/manage-resources-containers/"),
    ("configuration", "/docs/concepts/configuration/liveness-readiness-startup-probes/"),
    (
        "configuration",
        "/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/",
    ),
    # --- services and networking --------------------------------------------
    ("networking", "/docs/concepts/services-networking/service/"),
    ("networking", "/docs/concepts/services-networking/ingress/"),
    ("networking", "/docs/concepts/services-networking/network-policies/"),
    ("networking", "/docs/concepts/services-networking/dns-pod-service/"),
    # --- storage -------------------------------------------------------------
    ("storage", "/docs/concepts/storage/volumes/"),
    ("storage", "/docs/concepts/storage/persistent-volumes/"),
    ("storage", "/docs/concepts/storage/storage-classes/"),
    # --- cluster architecture ------------------------------------------------
    ("architecture", "/docs/concepts/overview/components/"),
    ("architecture", "/docs/concepts/architecture/nodes/"),
    ("architecture", "/docs/concepts/architecture/controller/"),
    ("architecture", "/docs/tasks/administer-cluster/configure-upgrade-etcd/"),
    ("architecture", "/docs/concepts/architecture/control-plane-node-communication/"),
    ("architecture", "/docs/concepts/scheduling-eviction/kube-scheduler/"),
    ("architecture", "/docs/concepts/scheduling-eviction/assign-pod-node/"),
    ("architecture", "/docs/concepts/scheduling-eviction/taint-and-toleration/"),
    # --- security ------------------------------------------------------------
    ("security", "/docs/reference/access-authn-authz/rbac/"),
    ("security", "/docs/concepts/security/service-accounts/"),
    ("security", "/docs/tasks/configure-pod-container/security-context/"),
    ("security", "/docs/concepts/security/pod-security-standards/"),
    # --- operations ----------------------------------------------------------
    ("operations", "/docs/reference/kubectl/quick-reference/"),
    ("operations", "/docs/tasks/debug/debug-application/debug-pods/"),
    ("operations", "/docs/tasks/debug/debug-application/determine-reason-pod-failure/"),
    ("operations", "/docs/concepts/cluster-administration/logging/"),
    ("operations", "/docs/tasks/debug/debug-cluster/resource-usage-monitoring/"),
    ("operations", "/docs/concepts/workloads/management/"),
]

# Docsy chrome that carries no documentation value.
STRIP_SELECTORS = [
    "nav",
    "header",
    "footer",
    "script",
    "style",
    "noscript",
    ".td-toc",
    ".td-sidebar",
    ".td-breadcrumbs",
    ".feedback--container",
    ".pageinfo-primary",
    "#pre-footer",
    ".copy-code-button",
    ".d-print-none",
    ".announcement",
]


def slugify(path: str) -> str:
    s = path.strip("/").replace("docs/", "", 1)
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "index"


def clean(html: str, url: str, topic: str) -> tuple[str, str] | None:
    """Reduce a full kubernetes.io page to an article-only HTML fragment."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").split("|")[0].strip() if soup.title else ""

    main = soup.select_one("div.td-content") or soup.select_one("main")
    if main is None:
        return None

    for sel in STRIP_SELECTORS:
        for el in main.select(sel):
            el.decompose()

    # Heading anchor links otherwise pollute every heading's text.
    for a in main.select("a.td-heading-self-link"):
        a.decompose()

    body = main.decode_contents()
    if len(re.sub(r"<[^>]+>", "", body).strip()) < 400:
        return None

    doc = (
        "<!--\n"
        f"title: {title}\n"
        f"source_url: {url}\n"
        f"doc_topic: {topic}\n"
        "license: CC BY 4.0 - The Kubernetes Authors\n"
        "-->\n"
        f"<h1>{title}</h1>\n{body}\n"
    )
    return title, doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch pages already on disk")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=45, follow_redirects=True, headers={"User-Agent": UA})

    ok = skipped = failed = 0
    for topic, path in PAGES:
        dest = OUT / f"{topic}__{slugify(path)}.html"
        if dest.exists() and not args.force:
            skipped += 1
            continue
        url = BASE + path
        try:
            r = client.get(url)
            r.raise_for_status()
            result = clean(r.text, url, topic)
            if result is None:
                print(f"  SKIP (no content) {path}")
                failed += 1
                continue
            title, doc = result
            dest.write_text(doc, encoding="utf-8")
            print(f"  OK  {len(doc):8,}b  {topic:13} {title[:50]}")
            ok += 1
        except Exception as e:
            print(f"  FAIL {path} -> {type(e).__name__}: {e}")
            failed += 1
        time.sleep(0.6)  # be a polite client

    client.close()
    on_disk = len(list(OUT.glob("*.html")))
    print(f"\nfetched={ok} skipped={skipped} failed={failed} total_on_disk={on_disk}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
