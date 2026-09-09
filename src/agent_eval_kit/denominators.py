"""The denominator rule: a threshold below `1/(1-t)` positives is a 1.0 wearing another label.

A recall bar of 0.90 sounds like "you may miss one in ten". Over seven scored positives it is
not: seven times 0.90 is 6.3, so six hits score 0.857 and fail, and the only passing result is
seven out of seven. The bar is arithmetically identical to 1.0, and the label says otherwise.

That is not a rounding quibble. It changes what a reviewer believes about the system and what a
maintainer does next. A team reading 0.90 thinks there is headroom and tunes toward it; there is
none, so the first genuine near-miss breaks the build and gets "fixed" by lowering the bar. A
buyer reading 0.90 in a pitch deck thinks the corpus is big enough to express a rate. Neither is
true, and nothing in the fleet said so, because a threshold and a corpus size sit in different
files and nothing compares them.

So the comparison lives here, as an assertion a golden set can run:

    assert_denominator_supports(0.90, positives=7, metric="discrepancy_recall")

raises, and names both ways out: write the bar as 1.0, which is honest and still gates, or grow
the corpus to the ten positives a 0.90 rate needs before it can express itself.

The rule in one line: a threshold ``t`` tolerates a single miss only when the metric has at
least ``1/(1-t)`` scored positives. It is about the DENOMINATOR of the metric, not the number
of cases in the file: fourteen cases carrying seven planted findings give ``discrepancy_recall``
a denominator of seven.

Two boundaries are deliberate:

* ``t >= 1.0`` needs no denominator at all. A bar of 1.0 already says "miss nothing", so it is
  never mislabelled, and this module returns without complaint however small the corpus is. The
  separate question of whether a 1.0 metric scored anything at all is :mod:`agent_eval_kit.report`'s
  (``n_examples > 0``), not this module's.
* ``positives == 0`` is refused before the arithmetic. A metric with no positives is not badly
  labelled, it is unmeasured, and every threshold is vacuous over it.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, isfinite

__all__ = [
    "DenominatorError",
    "assert_denominator_supports",
    "assert_each_denominator_supports",
    "required_positives",
    "supports",
]


class DenominatorError(AssertionError):
    """Raised when a threshold cannot be expressed by the corpus it is measured over."""


def required_positives(threshold: float) -> int:
    """The smallest number of scored positives at which ``threshold`` can tolerate one miss.

    Defined as the smallest ``n`` for which ``(n - 1) / n >= t`` under the SAME comparison a
    scorer performs, and found by search from the closed-form ``ceil(1 / (1 - t))`` rather than
    returned from it. The closed form is right in exact arithmetic and off by one in binary
    floating point at the values that matter most: ``1 / (1 - 0.9)`` is ``10.000000000000002``,
    so a 0.90 bar would be reported as needing eleven positives when ten pass it. Searching
    against the real comparison makes the answer agree with what the gate will actually do,
    which is the only definition that helps anybody.

    Returns ``0`` for a threshold of 1.0 or above, which asks for no headroom and therefore
    needs no denominator.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise DenominatorError(f"threshold must be a number, got {threshold!r}")
    value = float(threshold)
    if not isfinite(value):
        raise DenominatorError(f"threshold must be finite, got {threshold!r}")
    if not 0.0 <= value <= 1.0:
        raise DenominatorError(f"threshold {value} is outside [0, 1] and is not a quality bar")
    if value >= 1.0:
        return 0
    candidate = max(2, ceil(1.0 / (1.0 - value)))
    while candidate > 2 and (candidate - 2) / (candidate - 1) >= value:
        candidate -= 1
    while (candidate - 1) / candidate < value:
        candidate += 1
    return candidate


def supports(threshold: float, positives: int) -> bool:
    """Whether ``positives`` is enough for ``threshold`` to mean something other than 1.0."""
    if positives <= 0:
        return False
    return positives >= required_positives(threshold)


def assert_denominator_supports(
    threshold: float,
    positives: int,
    *,
    metric: str = "metric",
) -> None:
    """Assert ``positives`` can express ``threshold``, or raise naming both ways out."""
    if not isinstance(positives, int) or isinstance(positives, bool):
        raise DenominatorError(f"{metric}: positives must be an int, got {positives!r}")
    if positives < 0:
        raise DenominatorError(f"{metric}: positives cannot be negative, got {positives}")
    if positives == 0:
        raise DenominatorError(
            f"{metric}: the metric has no scored positives, so its {threshold} bar is vacuous. "
            "This is not a labelling problem, it is an absent measurement: every threshold "
            "passes over an empty set."
        )
    needed = required_positives(threshold)
    if positives < needed:
        raise DenominatorError(
            f"{metric}: a {threshold} bar needs at least {needed} scored positives before it "
            f"can tolerate a single miss, and this metric has {positives}. Over {positives} "
            f"positives the bar is arithmetically identical to 1.0, so it is a 1.0 wearing a "
            f"{threshold} label. Write it as 1.0, which is honest and still gates, or grow the "
            f"corpus to {needed} positives."
        )


def assert_each_denominator_supports(
    thresholds: Mapping[str, float],
    positives: Mapping[str, int],
    *,
    where: str = "",
) -> None:
    """Run the rule for every metric in ``thresholds``, per metric rather than in aggregate.

    Aggregate corpus size is the wrong denominator and hides exactly the case worth catching:
    a fourteen-case golden set can carry ten positives for one metric and two for another, and
    a check over "cases" would pass both. A metric absent from ``positives`` is refused rather
    than skipped, because a metric nobody counted is a metric nobody measured.
    """
    prefix = f"{where}: " if where else ""
    missing = sorted(set(thresholds) - set(positives))
    if missing:
        raise DenominatorError(
            f"{prefix}no positive count was given for {', '.join(missing)}; a metric nobody "
            "counted is a metric whose bar nobody can check"
        )
    for metric, threshold in thresholds.items():
        assert_denominator_supports(threshold, positives[metric], metric=f"{prefix}{metric}")
