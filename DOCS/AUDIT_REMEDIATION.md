# Audit remediation

Every finding from the Sentinel-RAG teardown, what was changed, and where to
look. Ordered by the audit's severity, not by effort.

Two changes are not on the audit's list but forced everything else:

- **The generation model no longer exists.** `llama-3.3-70b-versatile` and
  `llama-3.1-8b-instant` were withdrawn from Groq and now return HTTP 404. The
  deployed system could not answer a single question. Every model role was
  re-selected from what the account can actually serve
  ([`app/config.py`](../app/config.py)).
- **The corpus was replaced.** The scope is now Kubernetes and only Kubernetes:
  49 documents fetched from kubernetes.io by
  [`tools/fetch_corpus.py`](../tools/fetch_corpus.py), 2,069 indexed passages.
  See [Corpus decision](#corpus-decision) for why the audit's own "keep the
  distractors" recommendation was reversed.

---

## Critical

### F01 — No relevance threshold; abstention unreachable

The cross-encoder computed a relevance score and `rerank_documents` discarded
it one line before it could be used, so retrieval always returned exactly
`top_n` passages and the responder's "no context" branch could never fire.

**Fixed in** [`app/services/retrieval/ranking_service.py`](../app/services/retrieval/ranking_service.py).
`select_relevant()` returns `(kept, top_score)`; an empty list is a meaningful
result that [`retriever.py`](../app/agents/nodes/retriever.py) turns into
`abstained: True` and [`responder.py`](../app/agents/nodes/responder.py) turns
into an explicit refusal — without calling a model at all.

The threshold was derived, not guessed
([`tools/calibrate_threshold.py`](../tools/calibrate_threshold.py)): answerable
questions score 0.943 at minimum, off-domain questions 0.15 at maximum.

**Regression guards:** `tests/unit/test_relevance.py`,
`tests/integration/test_graph_and_api.py::test_abstains_when_nothing_is_relevant`,
and an end-to-end check against real vectors in
`test_retrieval_regression.py::test_out_of_corpus_questions_abstain`.

### F02 — Eval results never persisted; 2 of 5 metrics never ran

`save_results()` had zero call sites and the only caller of `run_all_metrics`
passed neither `result_cb` nor `skip_metrics`, so the documented resumability
was implemented but unwired. No result file existed anywhere.

**Fixed in** [`evals/run_eval.py`](../evals/run_eval.py). Every run writes
`evals/runs/<timestamp>/` containing the summary, per-item rows, the git SHA,
the corpus fingerprint, every model id and the resolved package versions.
Artifacts are committed, and `evals/runs/` is deliberately not gitignored.

### F03 — No tests, no CI

**Fixed in** [`tests/`](../tests) (88 tests) and
[`.github/workflows/`](../.github/workflows). Three tiers by cost: unit (no
network, ~12 s), integration against a Qdrant service container (no LLM), and a
judged evaluation that runs nightly rather than gating a pull request.

---

## High

### F04 — Two ground-truth leakage paths in the eval harness

The old pipeline assigned `actual_contexts = relevant_contexts` on failure, and
the scorer fell back the same way, so gold context could reach the judge as
though the system had retrieved it.

**Fixed:** both fallbacks are gone.
[`ragas_eval._prepare`](../evals/harness/ragas_eval.py) excludes and *counts*
any sample without real retrieved context; `coverage` is reported beside every
metric so a partial mean can never be mistaken for a full one.

### F05 — Gold contexts were partly fabricated

10 of 19 `relevant_contexts` did not appear verbatim in the corpus; one had none
of its sentences present.

**Fixed:** the dataset is now built by *extraction*
([`evals/dataset/build.py`](../evals/dataset/build.py)) — every question is
generated from a specific indexed chunk whose id is recorded as ground truth,
verified by a different model family, and filtered by a deterministic lexical
overlap check. [`tools/verify_dataset.py`](../tools/verify_dataset.py) runs in
CI and fails if any gold chunk is missing from the live index or no longer
matches the indexed body.

### F06 — n=15 with no confidence intervals

**Fixed:** 145 items across six strata — 83 answerable (stratified over 9 topics
and 41 documents), 18 in-domain unanswerable, 10 off-domain, 14 hard negatives,
16 adversarial, 4 conversational. Every proportion is reported with a Wilson
interval and every mean with a bootstrap interval
([`evals/stats.py`](../evals/stats.py)).

### F07 / F26 — Chunker split on blank lines the loaders never emitted

**Fixed in** [`app/ingestion/chunking/splitter.py`](../app/ingestion/chunking/splitter.py),
working from the structured `Block`s the loaders now produce
([`app/ingestion/document.py`](../app/ingestion/document.py)). A chunk never
spans a section boundary, code and tables stay atomic, every chunk carries its
heading trail as an embedded prefix, and consecutive chunks overlap.

Measured: mid-sentence chunk termination fell from **58% to 4.3%**, and
gold-passage containment in the retrieved context rose from **0.81 to 0.96**.

`_split_long_text` now actually reaches its sentence-boundary strategy; the old
version returned unconditionally after its first loop iteration.

### F08 — Payload was three string fields

**Fixed:** every point now carries `body`, `filename`, `title`, `source_url`,
`doc_topic`, `heading_path`, `kind`, `chunk_index`, `char_start`, `char_end` and
`doc_sha256`, with keyword indexes on `source`, `doc_topic` and `kind`.

### F09 — Client-supplied `thread_id` keyed the conversation store

Any caller could read another user's conversation by choosing its id; every
client that omitted the field shared one thread called `default_user`.

**Fixed in** [`app/api/security.py`](../app/api/security.py). Conversation ids
are unguessable server-issued capability tokens bound to the caller's identity,
compared in constant time, expired on a TTL and capped in number. History is
bounded by turn count *and* character budget
([`app/agents/state.py`](../app/agents/state.py)).

### F10 — No auth, no rate limit

**Fixed:** `require_auth` guards `/query` and `/session` with an optional API
key and a per-identity sliding-window rate limit. When no keys are configured
auth is off (correct for local dev and CI) and `/health` reports that, so a
deployment cannot be unprotected without it being visible.

### F11 — README claimed 58 distractors; 10 were indexed

**Resolved by** replacing the corpus entirely. The README now states what is
actually indexed, and `/scope` serves it from the live collection.

### F12 — Eval stack unpinned and standing on a monkeypatch

**Fixed:** [`requirements-eval.txt`](../requirements-eval.txt) pins `ragas` and
the `langchain-google-vertexai` package its `sys.modules` shim depends on. The
shim is still required — it is documented at the point of use rather than left
as a surprise.

### F13 — Response cache made repeat eval runs non-independent

**Fixed:** `complete(..., bypass_cache=True)` on every eval path, plus
`GATEWAY_CACHE_ENABLED=false` and `USE_GATEWAY=false` in the eval workflow.

---

## Medium

| # | Finding | Fix |
|---|---|---|
| F14 | Tool Correctness was tautological — the planner always emitted the string the detector looked for | Replaced by [`behaviour_eval.py`](../evals/harness/behaviour_eval.py), which scores every category against a declared `expected_behaviour`, including categories that must *not* answer |
| F15 | Re-ingest left orphans; point ids collided across folders | Ids key on the repo-relative path; `_delete_document` runs before every upsert **and** on the emptied path |
| F16 | Per-file failures swallowed, exit 0 regardless | Manifest with per-file outcomes, count reconciliation against the index, non-zero exit on any failure |
| F17 | `EMBEDDING_DIM = 768` was validated against itself | Read from the loaded model; asserted in `test_embedding_dim_comes_from_the_model` |
| F18 | `/query` returned HTTP 200 with an apology on any error | 5xx with a correlation id; asserted in `test_internal_error_is_5xx_not_200` |
| F19 | Sources were raw undeduplicated chunks with no scores | Structured citations carrying document, section, relevance score, topic and upstream URL |
| F20 | Four prompts advertised topics the corpus did not contain | [`app/services/scope.py`](../app/services/scope.py) derives scope from indexed `doc_topic` values; the planner, topic filter, responder and web client all read from it |
| F21 | Shared non-thread-safe model behind a threadpool | `encode()` guarded by a lock; model load double-checked under a second lock |
| F22 | Gate saw only the current message | `guard(message, history=...)`; covered by `test_history_is_inspected` and a leak test |
| F23 | Dead deps; `unstructured` pulled spacy/numba to parse three files | Office loader uses `python-docx`/`python-pptx` directly — 37 ingest packages down to 4, *and* it now extracts heading structure |
| F24 | Eval suite excluded from every image | `eval` Docker target plus a compose profile |
| F25 | `JUDGE_GROQ` fell back silently while comments claimed budget isolation | Fallback is explicit and warns; the comment now says isolation comes from a different **model**, which `Settings.validate()` enforces |

## Low

| # | Finding | Fix |
|---|---|---|
| F27 | Citation map keyed on chunk text; duplicates collapsed | Identity carried end to end as `RetrievedChunk`; `test_duplicate_text_keeps_separate_identity` |
| F28 | No payload indexes; HNSW not built | Keyword indexes created at ingest. HNSW still does not engage below 10k points — noted as a known limit, not claimed as solved |
| F29 | `/graph` was unauthenticated and triggered third-party egress | Endpoint removed |
| F30 | Deprecated `on_event`, stale gateway comment, dead context cap | `lifespan` handler; comments corrected; `CONTEXT_MAX_CHARS` is now enforced and reported |

---

## Corpus decision

The audit argued *against* deleting the distractor corpus, on the grounds that
retrieval precision on a clean corpus is meaningless. That reasoning was right;
the conclusion was wrong.

Separating Kubernetes documentation from a UAP forensics report is trivial — the
measurement showed 15/15 perfect separation, which is a benchmark that cannot
fail. The distractors were never a real test.

What replaced them is a **large in-domain corpus**: 2,069 passages across nine
Kubernetes areas, where the retriever must distinguish a Job's restart policy
from a Pod's, and a StatefulSet's update strategy from a DaemonSet's. Chunk-level
hit@1 of 0.735 on that corpus is a far more honest number than 1.000 was on the
old one, and it leaves visible headroom.

Out-of-domain rejection is now tested where it belongs — in the eval set's
`off_domain` and `unanswerable_in_domain` slices — rather than by polluting the
index the product actually serves.

## Known limitations

Stated because they are real, measured, and not fixed:

1. **The relevance threshold cannot separate "in-domain and covered" from
   "in-domain but uncovered."** A cross-encoder scores topical relatedness, not
   answer presence. At candidate depth 60, 2 of 18 in-domain-uncovered questions
   still surface a passage above threshold. The generator's decline instruction
   is the second line of defence, and the `uncovered_retrieval_positive` slice
   exists to measure it.
2. **Answerable eval questions were generated from the passages they match**, so
   their relevance scores are optimistic relative to real user phrasing. This is
   why the threshold sits mid-band at 0.35 rather than tuned to the observed
   minimum of 0.943.
3. **HNSW does not engage below 10,000 points**, so current retrieval is exact.
   These numbers do not predict behaviour at larger corpus sizes.
4. **`all-mpnet-base-v2` was not benchmarked against a faster model.** It costs
   roughly 500 ms per query embed on CPU and is the largest non-LLM latency
   component; whether a smaller model would hold the retrieval numbers is
   untested.
