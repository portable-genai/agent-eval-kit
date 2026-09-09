"""The not-falsely-green harness: make "prove this metric can go red" a one-liner.

A recurring, expensive failure mode is an eval metric that cannot fail: a redactor scored
against its own output (a closed tautology), a safety metric that re-reads the product's own
claims, or a golden set that planted no target at all, so the strictest metric is a constant 1.0
upstream of every subtlety. A metric that cannot go red proves nothing.

This module turns that check into an assertion a project runs in its test suite: give it the
metric function, a clean case that must score at/above the bar, and a degraded case that must
score below it. If the degraded case still passes, the metric is FALSELY GREEN and the assertion
fails with a message that says so. :func:`assert_each_can_go_red` runs it per market/segment,
which is the safe form: prove it per segment, not once overall.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping


class NotFalselyGreenError(AssertionError):
    """Raised when a metric cannot distinguish a clean case from a degraded one."""


def assert_can_go_red[T](
    score_fn: Callable[[T], float],
    *,
    green: T,
    red: T,
    threshold: float,
    metric: str = "metric",
) -> None:
    """Assert ``score_fn`` PASSES on ``green`` and FAILS on ``red`` at ``threshold``.

    Two independent failures are reported distinctly:

    * the clean case scoring below the bar means the metric is broken pessimistic, and
    * the degraded case scoring at/above the bar means the metric is FALSELY GREEN: it cannot
      detect the very defect it exists to catch.
    """
    green_score = score_fn(green)
    if green_score < threshold:
        raise NotFalselyGreenError(
            f"{metric}: the clean case should PASS (score >= {threshold}) but scored "
            f"{green_score}; the metric is broken pessimistic, not merely strict"
        )
    red_score = score_fn(red)
    if red_score >= threshold:
        raise NotFalselyGreenError(
            f"{metric}: FALSELY GREEN - the degraded case still scored {red_score} "
            f">= {threshold}, so the metric cannot go red and proves nothing"
        )


def assert_each_can_go_red[T](
    score_fn: Callable[[T], float],
    cases: Mapping[str, tuple[T, T]],
    *,
    threshold: float,
    metric: str = "metric",
) -> None:
    """Run :func:`assert_can_go_red` for each named ``(green, red)`` pair (per market/segment).

    A metric should be proven able to fail PER segment, each carrying its OWN target, not once
    overall: a segment missing from the config scores a vacuous 1.0 that an aggregate check hides.
    """
    for name, (green, red) in cases.items():
        assert_can_go_red(
            score_fn, green=green, red=red, threshold=threshold, metric=f"{metric}[{name}]"
        )


def prove_before_scoring(*proofs: Callable[[], None]) -> None:
    """Run every falsification proof BEFORE a single golden score is trusted.

    The ordering is the contribution, and it is the difference between a falsification proof
    that guards a release and one that decorates a test suite. Run in ``tests/``, the proof
    tells you the metric COULD have gone red on some machine at some point. Run as the first
    statement of the scored run itself, it tells you the metric that is about to score this
    corpus can go red, in this process, against this configuration, with these thresholds.

    A metric that became tautological because a scorer was refactored, a threshold was loaded
    from a stale rubric, or a pack file was emptied is caught here and nowhere else: those are
    all states a test suite green ten minutes earlier does not describe.

    Each proof is a zero-argument callable that raises on failure, so a repo writes its proofs
    as ordinary functions closing over its own scorers and thresholds. The first failure
    propagates unchanged: :class:`NotFalselyGreenError` already says which metric and which
    direction, and wrapping it would bury that.
    """
    if not proofs:
        raise NotFalselyGreenError(
            "prove_before_scoring was called with no proofs, so nothing was proven and the "
            "scored run that follows rests on an empty guarantee"
        )
    for proof in proofs:
        proof()
