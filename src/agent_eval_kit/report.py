"""The shared evaluation report types and their console rendering.

``EvalMetricResult`` / ``EvalReport`` are the value objects an eval produces: a per-metric score
against a threshold, and the report that passes iff every metric meets its bar. They are pure
stdlib and frozen, so an offline smoke evaluator and a remote promotion-gate client both speak
the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvalMetricResult:
    """One metric's score against its threshold."""

    metric: str
    score: float
    threshold: float
    passed: bool

    @classmethod
    def scored(cls, metric: str, score: float, threshold: float) -> EvalMetricResult:
        """Build a result, deriving ``passed`` from ``score >= threshold``."""
        return cls(metric=metric, score=score, threshold=threshold, passed=score >= threshold)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """A full evaluation report over a non-empty dataset and metric set.

    ``all(())`` is mathematically true but is not evidence.  Promotion therefore fails
    closed unless at least one example and at least one metric result are present.
    """

    dataset: str
    results: tuple[EvalMetricResult, ...]
    n_examples: int = 0
    run_id: str = ""
    dataset_version: str = "v1"
    dataset_digest: str = ""
    evaluator: str = ""
    schema_version: str = "eval-run/v1"
    trace_id: str | None = None
    correlation_id: str | None = None
    artifact_refs: tuple[str, ...] = ()
    attested: bool = False

    @property
    def passed(self) -> bool:
        return self.n_examples > 0 and bool(self.results) and all(r.passed for r in self.results)


def print_report(report: EvalReport, evaluator: str) -> None:
    """Render an :class:`EvalReport` as an aligned console table with a GATE verdict."""
    name_w = max((len(r.metric) for r in report.results), default=10)
    print(f"Evaluation report  ({evaluator})")
    print(f"  dataset : {report.dataset}")
    print(f"  examples: {report.n_examples}\n")
    header = f"  {'metric'.ljust(name_w)}   score    threshold   result"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in report.results:
        verdict = "PASS" if r.passed else "FAIL"
        print(f"  {r.metric.ljust(name_w)}   {r.score:6.3f}   {r.threshold:7.2f}     {verdict}")
    print()
    print(f"  GATE: {'PASS' if report.passed else 'FAIL'}")
