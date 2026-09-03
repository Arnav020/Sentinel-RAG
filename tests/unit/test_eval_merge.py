"""
Judged metrics are scored one per session around a daily token cap, so several
partial runs write into one run directory. That made overwriting the dangerous
default: a run that scored `context_recall` on six samples replaced a completed
`faithfulness` measurement (n=38, coverage 1.0) with "every sample failed to
score". The aggregate was recoverable only from git.

These tests pin the merge rules that make partial runs safe.
"""

from __future__ import annotations

from evals.harness.ragas_eval import better_of, merge_per_item, merge_summaries


class TestBetterOf:
    def test_an_error_never_replaces_a_measurement(self):
        good = {"metric": "faithfulness", "value": 0.87, "n": 38}
        bad = {"metric": "faithfulness", "error": "every sample failed to score"}
        assert better_of(good, bad) == good

    def test_a_measurement_replaces_an_error(self):
        bad = {"metric": "faithfulness", "error": "every sample failed to score"}
        good = {"metric": "faithfulness", "value": 0.87, "n": 38}
        assert better_of(bad, good) == good

    def test_higher_coverage_wins(self):
        small = {"value": 1.0, "n": 6}
        large = {"value": 0.87, "n": 38}
        assert better_of(large, small) == large
        assert better_of(small, large) == large

    def test_a_rerun_of_equal_size_wins(self):
        """Equal n means the fresh run is the more current measurement."""
        old = {"value": 0.80, "n": 38}
        new = {"value": 0.87, "n": 38}
        assert better_of(old, new) == new


class TestMergeSummaries:
    def test_a_partial_run_cannot_destroy_other_metrics(self):
        previous = {
            "faithfulness": {"value": 0.8743, "n": 38, "coverage": 1.0},
            "context_recall": {"error": "every sample failed to score"},
        }
        fresh = {
            "faithfulness": {"error": "every sample failed to score"},
            "context_recall": {"value": 0.8889, "n": 6, "coverage": 0.261},
        }
        merged = merge_summaries(previous, fresh)
        assert merged["faithfulness"]["value"] == 0.8743
        assert merged["context_recall"]["value"] == 0.8889

    def test_empty_topic_breakdown_does_not_erase_the_previous_one(self):
        previous = {"faithfulness_by_topic": {"jobs": {"value": 1.0, "n": 4}}}
        merged = merge_summaries(previous, {"faithfulness_by_topic": {}})
        assert merged["faithfulness_by_topic"] == {"jobs": {"value": 1.0, "n": 4}}

    def test_scalars_pass_through(self):
        merged = merge_summaries({"duration_seconds": 10.0}, {"duration_seconds": 42.0})
        assert merged["duration_seconds"] == 42.0


class TestMergePerItem:
    def test_scores_from_separate_runs_accumulate(self):
        previous = {"A001": {"faithfulness": 0.9}}
        fresh = {"A001": {"context_recall": 1.0}, "A002": {"context_recall": 0.5}}
        merged = merge_per_item(previous, fresh)
        assert merged["A001"] == {"faithfulness": 0.9, "context_recall": 1.0}
        assert merged["A002"] == {"context_recall": 0.5}

    def test_a_null_never_overwrites_a_recorded_score(self):
        merged = merge_per_item({"A001": {"faithfulness": 0.9}}, {"A001": {"faithfulness": None}})
        assert merged["A001"]["faithfulness"] == 0.9


class TestDailyQuotaDetection:
    """
    Groq signals two very different conditions with the same 429: tokens per
    minute, which a pause clears, and tokens per day, which it does not. A run
    that cannot tell them apart keeps going after its budget is gone - one did,
    for 17 minutes and 73 failed items, and then printed the wreckage as
    behaviour metrics.
    """

    def setup_method(self):
        from app.gateway import clear_daily_quota_notes

        clear_daily_quota_notes()

    teardown_method = setup_method

    def test_a_per_day_limit_is_recorded(self):
        from app.gateway import daily_quota_exhausted
        from app.gateway.client import note_if_daily_quota

        note_if_daily_quota(
            "openai/gpt-oss-120b",
            "Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per day "
            "(TPD): Limit 200000, Used 199336, Requested 773.",
        )
        assert "openai/gpt-oss-120b" in daily_quota_exhausted()

    def test_a_per_minute_limit_is_not_recorded(self):
        from app.gateway import daily_quota_exhausted
        from app.gateway.client import note_if_daily_quota

        note_if_daily_quota(
            "openai/gpt-oss-120b",
            "Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 7900.",
        )
        assert daily_quota_exhausted() == set()

    def test_unrelated_errors_are_not_recorded(self):
        from app.gateway import daily_quota_exhausted
        from app.gateway.client import note_if_daily_quota

        note_if_daily_quota("m", "APIConnectionError: connection reset")
        assert daily_quota_exhausted() == set()


class TestTokenAccounting:
    def test_usage_accumulates_per_model_within_the_window(self):
        from app.gateway import record_usage, tokens_used_since

        before = tokens_used_since("test/model-a", 60)
        record_usage("test/model-a", 500)
        record_usage("test/model-a", 300)
        record_usage("test/model-b", 999)
        assert tokens_used_since("test/model-a", 60) == before + 800

    def test_zero_and_negative_usage_is_ignored(self):
        from app.gateway import record_usage, tokens_used_since

        before = tokens_used_since("test/model-c", 60)
        record_usage("test/model-c", 0)
        record_usage("test/model-c", -5)
        assert tokens_used_since("test/model-c", 60) == before
