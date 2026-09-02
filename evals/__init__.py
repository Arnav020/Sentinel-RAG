"""
Evaluation suite.

The Streamlit app, `metrics.py`, `pipeline.py` and `guardrails_eval.py` that
used to live here have been removed. They kept results only in session state,
had two paths that substituted ground-truth contexts for retrieved ones, and
scored a 15-question dataset with no confidence intervals.

The replacement is `evals/run_eval.py`, which writes every run to
`evals/runs/<timestamp>/` as a committed artifact.

Nothing is imported here on purpose: importing ragas at package level made the
whole package unimportable whenever the eval extras were not installed.
"""
