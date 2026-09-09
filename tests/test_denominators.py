"""The denominator rule: a bar the corpus cannot express is a 1.0 wearing another label."""

from __future__ import annotations

import pytest

from agent_eval_kit.denominators import (
    DenominatorError,
    assert_denominator_supports,
    assert_each_denominator_supports,
    required_positives,
    supports,
)


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.0, 2), (0.5, 2), (0.8, 5), (0.85, 7), (0.90, 10), (0.95, 20), (0.99, 100), (1.0, 0)],
)
def test_required_positives_is_the_smallest_corpus_that_tolerates_one_miss(threshold, expected):
    assert required_positives(threshold) == expected


@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.85, 0.90, 0.95, 0.99])
def test_the_answer_agrees_with_the_comparison_a_scorer_actually_performs(threshold):
    """The closed form is off by one in binary floating point; the search must not be.

    ``1 / (1 - 0.9)`` is ``10.000000000000002``, so ``ceil`` returns eleven for a bar that ten
    positives clear. A rule that disagreed with the comparison the gate performs would send
    people to grow a corpus that was already big enough.
    """
    n = required_positives(threshold)
    assert (n - 1) / n >= threshold, "one miss should PASS at the reported size"
    assert (n - 2) / (n - 1) < threshold, "one miss should FAIL one below the reported size"


def test_a_thin_corpus_is_refused_and_both_ways_out_are_named():
    with pytest.raises(DenominatorError) as excinfo:
        assert_denominator_supports(0.90, 7, metric="discrepancy_recall")
    message = str(excinfo.value)
    assert "discrepancy_recall" in message
    assert "identical to 1.0" in message
    assert "10 positives" in message


def test_a_bar_of_one_needs_no_denominator():
    """1.0 already says "miss nothing", so it is never mislabelled however small the corpus."""
    assert_denominator_supports(1.0, 1, metric="ratio_reproducibility")
    assert required_positives(1.0) == 0


def test_zero_positives_is_an_absent_measurement_not_a_labelling_problem():
    with pytest.raises(DenominatorError, match="vacuous"):
        assert_denominator_supports(0.90, 0, metric="typology_recall")
    assert supports(0.90, 0) is False


def test_a_sufficient_corpus_passes():
    assert_denominator_supports(0.90, 10, metric="typology_recall")
    assert supports(0.90, 10) is True


def test_the_rule_runs_per_metric_because_aggregate_case_count_is_the_wrong_denominator():
    """Fourteen cases can carry ten positives for one metric and two for another."""
    thresholds = {"discrepancy_recall": 0.90, "citation_accuracy": 0.90}
    with pytest.raises(DenominatorError, match="citation_accuracy"):
        assert_each_denominator_supports(
            thresholds, {"discrepancy_recall": 30, "citation_accuracy": 2}, where="golden set"
        )


def test_a_metric_nobody_counted_is_refused_rather_than_skipped():
    with pytest.raises(DenominatorError, match="nobody counted"):
        assert_each_denominator_supports({"a": 0.9, "b": 0.9}, {"a": 10})


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
def test_a_threshold_outside_zero_to_one_is_not_a_quality_bar(bad):
    with pytest.raises(DenominatorError):
        required_positives(bad)


def test_booleans_are_not_numbers_here():
    with pytest.raises(DenominatorError):
        required_positives(True)  # noqa: FBT003
    with pytest.raises(DenominatorError):
        assert_denominator_supports(0.9, True, metric="m")  # noqa: FBT003
