"""
Judged metrics are scored one per session around a daily token cap, so several
partial runs write into one run directory. That made overwriting the dangerous
default: a run that scored `context_recall` on six samples replaced a completed
`faithfulness` measurement (n=38, coverage 1.0) with "every sample failed to
score". The aggregate was recoverable only from git.

These tests pin the merge rules that make partial runs safe.
"""

from __future__ import annotations

import pytest

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


class TestRunItemsAbortsOnDailyQuota:
    """
    The abort path itself needs a test. It shipped with a syntax error that
    every local check missed, because nothing executed it - the one code path
    whose whole job is to fire when a long, expensive run goes wrong is also the
    path least likely to be exercised by accident.
    """

    @staticmethod
    def _harness(monkeypatch, exhausted_after: int):
        from evals.harness import pipeline

        seen = {"n": 0}

        def fake_run_item(item, agent=None, with_guardrails=True):
            seen["n"] += 1
            return {"id": item["id"], "error": "", "answer": "ok"}

        def fake_exhausted(within_seconds=600):
            return {"openai/gpt-oss-120b"} if seen["n"] >= exhausted_after else set()

        monkeypatch.setattr(pipeline, "build_graph", lambda: object())
        monkeypatch.setattr(pipeline, "run_item", fake_run_item)
        monkeypatch.setattr(pipeline, "daily_quota_exhausted", fake_exhausted)
        return pipeline

    def test_it_stops_and_keeps_what_it_gathered(self, monkeypatch):
        pipeline = self._harness(monkeypatch, exhausted_after=3)
        items = [{"id": f"A{i:03d}"} for i in range(10)]

        with pytest.raises(pipeline.DailyQuotaExhausted) as excinfo:
            pipeline.run_items(items, pace_seconds=0.0)

        assert len(excinfo.value.records) == 3
        assert "openai/gpt-oss-120b" in str(excinfo.value)
        assert "3 of 10" in str(excinfo.value)

    def test_a_healthy_run_is_untouched(self, monkeypatch):
        pipeline = self._harness(monkeypatch, exhausted_after=999)
        items = [{"id": f"A{i:03d}"} for i in range(5)]
        assert len(pipeline.run_items(items, pace_seconds=0.0)) == 5


class TestTokenThrottle:
    """
    Pacing must follow measured spend, not a fixed delay. Items cost between
    ~500 and ~3,000 tokens depending on whether they were blocked, abstained or
    answered, so one sleep value is either wasteful or over the limit - a 1.5s
    pace drew 236 rate-limit errors on a 145-item run.
    """

    def test_it_waits_while_a_model_is_over_budget(self, monkeypatch):
        from evals.harness import pipeline

        spend = {"m": pipeline.TPM_LIMIT}
        slept = []
        monkeypatch.setattr(
            pipeline.time, "sleep", lambda s: (slept.append(s), spend.update(m=0))[0]
        )
        monkeypatch.setattr(pipeline, "tokens_used_since", lambda m, w: spend["m"])

        waited = pipeline._throttle(["m"], expected=100)
        assert slept, "should have waited while over budget"
        assert waited > 0

    def test_it_does_not_wait_when_there_is_room(self, monkeypatch):
        from evals.harness import pipeline

        monkeypatch.setattr(pipeline, "tokens_used_since", lambda m, w: 0)
        monkeypatch.setattr(
            pipeline.time, "sleep", lambda s: pytest.fail("throttled with budget to spare")
        )
        assert pipeline._throttle(["m"], expected=100) == 0.0

    def test_the_busiest_model_gates_the_run(self, monkeypatch):
        """One saturated model must hold the whole pipeline, not just its own stage."""
        from evals.harness import pipeline

        usage = {"cheap": 0, "busy": pipeline.TPM_LIMIT}
        monkeypatch.setattr(pipeline, "tokens_used_since", lambda m, w: usage[m])
        monkeypatch.setattr(pipeline.time, "sleep", lambda s: usage.update(busy=0))

        assert pipeline._throttle(["cheap", "busy"], expected=100) > 0

    def test_it_gives_up_rather_than_hanging_forever(self, monkeypatch):
        """A permanently saturated model must not stall the run indefinitely."""
        from evals.harness import pipeline

        monkeypatch.setattr(pipeline, "tokens_used_since", lambda m, w: pipeline.TPM_LIMIT)
        monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)

        assert pipeline._throttle(["m"], expected=100) == pipeline._THROTTLE_TIMEOUT


class TestSummaryPrinter:
    """
    A judged run that ends in a traceback reads like a run that failed.

    `print_summary` assumed every list in a tier was a list of metric dicts and
    crashed on `metrics_run: ["faithfulness"]` with AttributeError - after 25
    minutes of paid judging. The scores were already on disk, but nobody
    reading the console output would have known that.
    """

    def test_it_survives_a_list_of_plain_values(self, capsys):
        from evals.run_eval import print_summary

        print_summary(
            {
                "ragas": {
                    "faithfulness": {
                        "metric": "faithfulness",
                        "value": 0.894,
                        "ci_low": 0.833,
                        "ci_high": 0.949,
                        "n": 34,
                    },
                    "metrics_run": ["faithfulness"],
                    "judge_model": "qwen/qwen3.8-27b",
                }
            }
        )
        out = capsys.readouterr().out
        assert "0.894" in out
        assert "faithfulness" in out

    def test_it_still_renders_a_list_of_metric_dicts(self, capsys):
        from evals.run_eval import print_summary

        print_summary({"retrieval": {"retrieval": [{"metric": "hit@5", "value": 0.892, "n": 83}]}})
        assert "hit@5" in capsys.readouterr().out
