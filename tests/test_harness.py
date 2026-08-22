"""The not-falsely-green harness: prove a metric can distinguish clean from degraded."""

from __future__ import annotations

import pytest

from agent_eval_kit.harness import (
    NotFalselyGreenError,
    assert_can_go_red,
    assert_each_can_go_red,
)


def _leak_scorer(text: str) -> float:
    """A sound PII leak metric: 1.0 clean, 0.0 if a raw NRIC survives."""
    return 0.0 if "S1234567A" in text else 1.0


def _falsely_green_scorer(_text: str) -> float:
    """A structurally broken metric that can never fail (re-reads nothing real)."""
    return 1.0


def test_sound_metric_passes_the_check():
    # Clean case scores 1.0, leaking case scores 0.0: the metric can go red.
    assert_can_go_red(
        _leak_scorer,
        green="redacted [SG_NRIC] only",
        red="applicant NRIC S1234567A on file",
        threshold=0.99,
        metric="pii_safety",
    )


def test_falsely_green_metric_is_caught():
    with pytest.raises(NotFalselyGreenError, match="FALSELY GREEN"):
        assert_can_go_red(
            _falsely_green_scorer,
            green="clean",
            red="applicant NRIC S1234567A on file",
            threshold=0.99,
            metric="pii_safety",
        )


def test_broken_pessimistic_metric_is_reported_distinctly():
    with pytest.raises(NotFalselyGreenError, match="broken pessimistic"):
        assert_can_go_red(
            _leak_scorer,
            green="applicant NRIC S1234567A leaked even in the clean case",
            red="applicant NRIC S1234567A on file",
            threshold=0.99,
            metric="pii_safety",
        )


def test_per_market_check_runs_each_pair():
    cases = {
        "SG": ("clean sg", "raw NRIC S1234567A"),
        "GENERIC": ("clean generic", "raw NRIC S1234567A again"),
    }
    assert_each_can_go_red(_leak_scorer, cases, threshold=0.99, metric="pii_safety")


def test_per_market_check_names_the_failing_market():
    cases = {
        "SG": ("clean", "raw NRIC S1234567A"),
        "AU": ("clean", "this market plants no identifier"),  # red case cannot go red
    }
    with pytest.raises(NotFalselyGreenError, match=r"pii_safety\[AU\]"):
        assert_each_can_go_red(_leak_scorer, cases, threshold=0.99, metric="pii_safety")
