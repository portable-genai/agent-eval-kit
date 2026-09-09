"""The not-falsely-green harness: prove a metric can distinguish clean from degraded."""

from __future__ import annotations

import pytest

from agent_eval_kit.harness import (
    NotFalselyGreenError,
    assert_can_go_red,
    assert_each_can_go_red,
    prove_before_scoring,
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


# --------------------------------------------------------------------------- #
# Falsification BEFORE scoring, which is where it guards a release
# --------------------------------------------------------------------------- #
def test_prove_before_scoring_runs_every_proof_in_order():
    ran: list[str] = []
    prove_before_scoring(lambda: ran.append("a"), lambda: ran.append("b"))
    assert ran == ["a", "b"]


def test_the_first_failing_proof_propagates_unchanged():
    """NotFalselyGreenError already says which metric and which direction; wrapping buries that."""

    def falsely_green() -> None:
        assert_can_go_red(lambda _: 1.0, green=1, red=0, threshold=0.9, metric="pii_safety")

    with pytest.raises(NotFalselyGreenError, match="pii_safety: FALSELY GREEN"):
        prove_before_scoring(falsely_green)


def test_calling_it_with_no_proofs_is_an_empty_guarantee_and_is_refused():
    with pytest.raises(NotFalselyGreenError, match="nothing was proven"):
        prove_before_scoring()
