"""EvalReport / EvalMetricResult value objects and their rendering."""

from __future__ import annotations

import pytest

from agent_eval_kit.report import EvalMetricResult, EvalReport, print_report


def test_scored_derives_passed_from_threshold():
    assert EvalMetricResult.scored("m", 0.9, 0.8).passed is True
    assert EvalMetricResult.scored("m", 0.79, 0.8).passed is False
    # Exactly at the bar passes (>=).
    assert EvalMetricResult.scored("m", 0.8, 0.8).passed is True


def test_report_passes_only_when_every_metric_passes():
    ok = EvalReport(
        dataset="d",
        results=(
            EvalMetricResult.scored("a", 0.9, 0.8),
            EvalMetricResult.scored("b", 1.0, 0.99),
        ),
        n_examples=2,
    )
    assert ok.passed is True

    bad = EvalReport(
        dataset="d",
        results=(
            EvalMetricResult.scored("a", 0.9, 0.8),
            EvalMetricResult.scored("b", 0.5, 0.99),
        ),
        n_examples=2,
    )
    assert bad.passed is False


def test_empty_report_and_empty_dataset_fail_closed():
    assert EvalReport(dataset="d", results=(), n_examples=1).passed is False
    assert (
        EvalReport(
            dataset="d",
            results=(EvalMetricResult.scored("m", 1.0, 0.5),),
            n_examples=0,
        ).passed
        is False
    )


def test_print_report_renders_table_and_gate(capsys: pytest.CaptureFixture[str]):
    report = EvalReport(
        dataset="eval/golden.jsonl",
        results=(EvalMetricResult.scored("pii_safety", 1.0, 0.99),),
        n_examples=3,
    )
    print_report(report, "offline heuristic")
    out = capsys.readouterr().out
    assert "offline heuristic" in out
    assert "pii_safety" in out
    assert "GATE: PASS" in out
