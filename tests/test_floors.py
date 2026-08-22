"""Per-vertical quality floors: the number, held as data, that makes UNFIT a verdict.

Every test here is a refusal. A floor table is a promotion bar, so the interesting behaviour is
what it declines to do: default a missing vertical, load an empty table, accept a floor of zero,
or let a verdict with no evidence through as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_kit.floors import (
    FLOORS_SCHEMA,
    Fitness,
    FloorDataError,
    MissingFloorError,
    QualityFloors,
    load_quality_floors,
)
from agent_eval_kit.judge import CriterionScore, JudgeVerdict
from agent_eval_kit.report import EvalReport

_DOCUMENT = {
    "schema": FLOORS_SCHEMA,
    "owner": "model risk (example)",
    "verticals": {
        "doc1-cdd-sow": {"floor": 0.70, "target": 0.88, "note": "escalation narrative"},
        "mkt6-compliance": {"floor": 0.85, "target": 0.95},
    },
}

_TOML = """
schema = "quality-floors/v1"
owner = "model risk (example)"

[verticals.doc1-cdd-sow]
floor = 0.70
target = 0.88
note = "escalation narrative"

[verticals.mkt6-compliance]
floor = 0.85
target = 0.95
"""


def _floors() -> QualityFloors:
    return QualityFloors.from_mapping(_DOCUMENT, source="example-floors")


def test_a_document_loads_its_named_verticals() -> None:
    floors = _floors()
    assert floors.verticals == ("doc1-cdd-sow", "mkt6-compliance")
    assert len(floors) == 2
    assert "doc1-cdd-sow" in floors
    assert floors.floor_for("doc1-cdd-sow").floor == pytest.approx(0.70)
    assert floors.floor_for("doc1-cdd-sow").target == pytest.approx(0.88)
    assert floors.floor_for("doc1-cdd-sow").source == "example-floors"
    assert [entry.vertical for entry in floors] == list(floors.verticals)


def test_the_three_bands_are_fit_degraded_and_unfit() -> None:
    """The band between floor and target is where "visibly lower quality" honestly lives."""
    floors = _floors()
    assert floors.assess("doc1-cdd-sow", 0.92, profile="managed").fitness is Fitness.FIT
    degraded = floors.assess("doc1-cdd-sow", 0.78, profile="reduced")
    assert degraded.fitness is Fitness.DEGRADED
    assert degraded.unfit is False
    unfit = floors.assess("doc1-cdd-sow", 0.55, profile="reduced")
    assert unfit.fitness is Fitness.UNFIT
    assert unfit.unfit is True
    assert "unfit for doc1-cdd-sow" in unfit.reason


def test_the_boundaries_belong_to_the_kinder_band() -> None:
    floors = _floors()
    assert floors.assess("doc1-cdd-sow", 0.88, profile="p").fitness is Fitness.FIT
    assert floors.assess("doc1-cdd-sow", 0.70, profile="p").fitness is Fitness.DEGRADED
    assert floors.assess("doc1-cdd-sow", 0.6999, profile="p").fitness is Fitness.UNFIT


def test_an_unnamed_vertical_raises_rather_than_inheriting_a_bar() -> None:
    """The silent-default trap: an unmeasured vertical must not borrow somebody else's floor."""
    with pytest.raises(MissingFloorError, match="no quality floor is defined"):
        _floors().floor_for("doc9-not-a-vertical")


def test_a_verdict_must_name_the_profile_it_measured() -> None:
    with pytest.raises(FloorDataError, match="must name the profile"):
        _floors().assess("doc1-cdd-sow", 0.9, profile="  ")


def test_an_empty_floor_table_refuses_to_load() -> None:
    with pytest.raises(FloorDataError, match="empty"):
        QualityFloors.from_mapping({"schema": FLOORS_SCHEMA, "verticals": {}}, source="s")


def test_a_floor_of_zero_is_refused() -> None:
    """A floor at zero passes everything, which is the same as having no floor at all."""
    document = {"schema": FLOORS_SCHEMA, "verticals": {"v": {"floor": 0.0, "target": 0.9}}}
    with pytest.raises(FloorDataError, match="cannot refuse anything"):
        QualityFloors.from_mapping(document, source="s")


def test_a_target_below_the_floor_is_refused() -> None:
    document = {"schema": FLOORS_SCHEMA, "verticals": {"v": {"floor": 0.9, "target": 0.5}}}
    with pytest.raises(FloorDataError, match="at least the floor"):
        QualityFloors.from_mapping(document, source="s")


def test_a_missing_bar_is_never_defaulted() -> None:
    document = {"schema": FLOORS_SCHEMA, "verticals": {"v": {"floor": 0.8}}}
    with pytest.raises(FloorDataError, match="target is required"):
        QualityFloors.from_mapping(document, source="s")


def test_a_misspelled_field_is_refused() -> None:
    """A bar under a misspelled key is a bar that is never applied, silently."""
    document = {"schema": FLOORS_SCHEMA, "verticals": {"v": {"floor": 0.8, "targt": 0.9}}}
    with pytest.raises(FloorDataError, match="unknown field"):
        QualityFloors.from_mapping(document, source="s")


def test_a_document_without_the_schema_tag_is_refused() -> None:
    with pytest.raises(FloorDataError, match="schema must be"):
        QualityFloors.from_mapping({"verticals": {"v": {"floor": 0.1, "target": 0.2}}})


def test_a_boolean_is_not_a_bar() -> None:
    document = {"schema": FLOORS_SCHEMA, "verticals": {"v": {"floor": True, "target": 0.9}}}
    with pytest.raises(FloorDataError, match="must be a number"):
        QualityFloors.from_mapping(document, source="s")


# --------------------------------------------------------------------------- #
# The judge verdict bridge: an absent measurement is the weakest evidence there is
# --------------------------------------------------------------------------- #
def test_a_verdict_with_no_graded_criteria_is_unfit_not_a_pass() -> None:
    """The exact class of defect that let ``EvalReport(metrics=[]).passed`` return True."""
    verdict = _floors().assess_verdict(
        "doc1-cdd-sow", JudgeVerdict(graded_by="silent/v0", scores=()), profile="reduced"
    )
    assert verdict.fitness is Fitness.UNFIT
    assert verdict.score == 0.0
    assert "graded no criteria" in verdict.reason


def test_a_graded_verdict_is_scored_against_the_floor() -> None:
    verdict = JudgeVerdict(
        graded_by="deterministic-narrative-judge/v1",
        scores=(CriterionScore(criterion="c", score=0.8),),
    )
    assessed = _floors().assess_verdict("doc1-cdd-sow", verdict, profile="reduced")
    assert assessed.fitness is Fitness.DEGRADED
    assert assessed.score == pytest.approx(0.8)


def test_the_metric_row_uses_the_floor_as_its_threshold() -> None:
    """Degradation is a visible low score, not a failed gate. The floor is the bar that refuses."""
    floors = _floors()
    degraded = floors.assess("doc1-cdd-sow", 0.78, profile="reduced").as_metric_result()
    assert degraded.threshold == pytest.approx(0.70)
    assert degraded.passed is True
    assert degraded.metric == "narrative_quality[doc1-cdd-sow/reduced]"
    unfit = floors.assess("doc1-cdd-sow", 0.55, profile="reduced").as_metric_result()
    assert unfit.passed is False


def test_a_report_built_from_an_empty_verdict_fails_closed() -> None:
    """End to end: no evidence produces a failing report, never a green one."""
    empty = _floors().assess_verdict(
        "doc1-cdd-sow", JudgeVerdict(graded_by="silent/v0", scores=()), profile="reduced"
    )
    report = EvalReport(dataset="d", results=(empty.as_metric_result(),), n_examples=1)
    assert report.passed is False
    assert EvalReport(dataset="d", results=(), n_examples=1).passed is False


# --------------------------------------------------------------------------- #
# Loading from a file, with no third-party parser
# --------------------------------------------------------------------------- #
def test_a_toml_document_loads(tmp_path: Path) -> None:
    path = tmp_path / "quality-floors.toml"
    path.write_text(_TOML, encoding="utf-8")
    floors = load_quality_floors(path)
    assert floors.verticals == ("doc1-cdd-sow", "mkt6-compliance")
    assert floors.source == str(path)


def test_a_json_document_loads(tmp_path: Path) -> None:
    path = tmp_path / "quality-floors.json"
    path.write_text(json.dumps(_DOCUMENT), encoding="utf-8")
    assert load_quality_floors(path).verticals == ("doc1-cdd-sow", "mkt6-compliance")


def test_an_unreadable_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "quality-floors.yaml"
    path.write_text("schema: quality-floors/v1\n", encoding="utf-8")
    with pytest.raises(FloorDataError, match="must be .toml or .json"):
        load_quality_floors(path)
