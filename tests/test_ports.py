"""The gate port must be structurally checkable, and must not be satisfiable by half of it.

The drift this Protocol replaced was not a missing file: it was adapters that had `evaluate` and
no `gate`, which reads as "we have an evaluation port" while being unable to refuse a promotion.
`runtime_checkable` plus the contract suites that use it is what turns that back into a failure.
"""

from __future__ import annotations

from agent_eval_kit import EvalMetricResult, EvalReport, EvaluationGatePort


class _Complete:
    def evaluate(self, dataset_path: str) -> EvalReport:
        return EvalReport(
            dataset=dataset_path,
            results=(EvalMetricResult("citation_accuracy", 0.95, 0.90, True),),
            n_examples=1,
        )

    def gate(self, target: str) -> bool:
        return self.evaluate(target).passed


class _EvaluateOnly:
    """The shape two repositories had actually shipped."""

    def evaluate(self, dataset_path: str) -> EvalReport:
        return EvalReport(dataset=dataset_path, results=(), n_examples=0)


def test_a_complete_adapter_satisfies_the_port() -> None:
    assert isinstance(_Complete(), EvaluationGatePort)


def test_an_adapter_with_no_gate_does_NOT_satisfy_the_port() -> None:
    """The mutant. Without this the Protocol could be checking nothing and still look adopted."""
    assert not isinstance(_EvaluateOnly(), EvaluationGatePort)


def test_the_port_returns_the_kit_report_type_so_it_crosses_no_package_boundary() -> None:
    report = _Complete().evaluate("eval/datasets/golden.jsonl")
    assert isinstance(report, EvalReport)
    assert report.passed is True
