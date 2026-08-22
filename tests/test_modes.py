"""The --mode smoke|gate scaffold: dispatch and fail-closed exit codes."""

from __future__ import annotations

from pathlib import Path

from agent_eval_kit.modes import eval_main
from agent_eval_kit.report import EvalMetricResult, EvalReport


def _report(passed: bool) -> EvalReport:
    score = 1.0 if passed else 0.0
    return EvalReport(
        dataset="d",
        results=(EvalMetricResult.scored("m", score, 0.5),),
        n_examples=1,
    )


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    return path


def test_missing_dataset_returns_2(tmp_path: Path):
    code = eval_main(
        smoke=lambda _d: _report(True),
        gate=lambda _d: (_report(True), True),
        default_dataset=tmp_path / "nope.jsonl",
        argv=[],
    )
    assert code == 2


def test_smoke_pass_returns_0(tmp_path: Path):
    code = eval_main(
        smoke=lambda _d: _report(True),
        gate=lambda _d: (_report(True), True),
        default_dataset=_dataset(tmp_path),
        argv=[],
    )
    assert code == 0


def test_smoke_fail_returns_1(tmp_path: Path):
    code = eval_main(
        smoke=lambda _d: _report(False),
        gate=lambda _d: (_report(True), True),
        default_dataset=_dataset(tmp_path),
        argv=["--mode", "smoke"],
    )
    assert code == 1


def test_gate_pass_requires_both_report_and_authority(tmp_path: Path):
    dataset = _dataset(tmp_path)
    # Both green -> 0.
    assert (
        eval_main(
            smoke=lambda _d: _report(True),
            gate=lambda _d: (_report(True), True),
            default_dataset=dataset,
            argv=["--mode", "gate"],
        )
        == 0
    )
    # Report passes but the authority's verdict is False -> fail closed to 1.
    assert (
        eval_main(
            smoke=lambda _d: _report(True),
            gate=lambda _d: (_report(True), False),
            default_dataset=dataset,
            argv=["--mode", "gate"],
        )
        == 1
    )
    # Authority passes but the scored report fails -> also 1.
    assert (
        eval_main(
            smoke=lambda _d: _report(True),
            gate=lambda _d: (_report(False), True),
            default_dataset=dataset,
            argv=["--mode", "gate"],
        )
        == 1
    )


def test_explicit_dataset_arg_is_used(tmp_path: Path):
    dataset = _dataset(tmp_path)
    seen: list[Path] = []

    def smoke(d: Path) -> EvalReport:
        seen.append(d)
        return _report(True)

    eval_main(
        smoke=smoke,
        gate=lambda _d: (_report(True), True),
        default_dataset=tmp_path / "default.jsonl",
        argv=["--dataset", str(dataset)],
    )
    assert seen == [dataset]
