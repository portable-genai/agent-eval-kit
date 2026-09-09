"""The judged run: bands rather than thresholds, and a table that must itself be falsifiable."""

from __future__ import annotations

import json

import pytest

from agent_eval_kit.floors import Fitness
from agent_eval_kit.judge import JudgeUnavailableError, JudgeVerdict
from agent_eval_kit.narrative import (
    NarrativeEvalError,
    check_table_is_falsifiable,
    load_cases,
    measure,
    narrative_main,
)

PROFILES = ("managed", "reduced", "regressed")
CONTROL = "regressed"

_FLOORS = """\
schema = "quality-floors/v1"
owner = "model risk"

[verticals.advice]
floor = 0.60
target = 0.90
note = "Read by a client with nobody in between."
"""

_CASE = {
    "id": "advice-1",
    "vertical": "advice",
    "criteria": [
        {
            "name": "conveys_the_position",
            "must_cover": ["concentration in one issuer", "review at the next meeting"],
            "must_cite": ["HV-2026-03"],
            "must_not_say": ["guaranteed"],
        }
    ],
    "candidates": {
        # Covers both points, cites the house view, avoids the forbidden claim.
        "managed": (
            "There is concentration in one issuer. We review at the next meeting. See HV-2026-03."
        ),
        # Covers one point and cites; visibly worse, and the band the portability story describes.
        "reduced": "There is concentration in one issuer. See HV-2026-03.",
        # The deliberate defect: covers nothing, cites nothing, and makes the forbidden claim.
        "regressed": "Returns are guaranteed.",
    },
    "expected": {"managed": "fit", "reduced": "degraded", "regressed": "unfit"},
}


def _table(tmp_path, cases=None, name="narrative_golden.jsonl"):
    path = tmp_path / name
    rows = cases if cases is not None else [_CASE]
    path.write_text(
        "# the degradation table\n" + "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _floors(tmp_path):
    path = tmp_path / "quality-floors.toml"
    path.write_text(_FLOORS, encoding="utf-8")
    return path


def test_a_whole_judged_run_passes_offline_with_no_model_and_no_network(tmp_path, capsys):
    code = narrative_main(
        dataset=_table(tmp_path),
        floors=_floors(tmp_path),
        profiles=PROFILES,
        control=CONTROL,
        argv=[],
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "NARRATIVE GATE: PASS" in out
    assert "deterministic" in out


def test_a_band_that_moved_fails_because_it_is_a_change_nobody_reviewed(tmp_path, capsys):
    """A profile that quietly got BETTER fails too. That is the point of a table."""
    # Two cases: one calibrated, so the table stays falsifiable, and one whose middle band the
    # author predicted wrongly. Without the first, check_table_is_falsifiable fires instead and
    # this test would prove the wrong thing.
    optimistic = {
        **_CASE,
        "id": "advice-2",
        "expected": {**_CASE["expected"], "reduced": "fit"},
    }
    code = narrative_main(
        dataset=_table(tmp_path, [_CASE, optimistic]),
        floors=_floors(tmp_path),
        profiles=PROFILES,
        control=CONTROL,
        argv=[],
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "NARRATIVE GATE: FAIL" in out
    assert "the table says 'fit'" in out


def test_a_table_where_nothing_is_expected_to_degrade_is_refused_before_anything_is_graded(
    tmp_path,
):
    all_fit = {
        **_CASE,
        "candidates": {profile: _CASE["candidates"]["managed"] for profile in PROFILES},
        "expected": dict.fromkeys(PROFILES, "fit"),
    }
    with pytest.raises(NarrativeEvalError, match="middle band is unmeasured"):
        check_table_is_falsifiable([all_fit], control=CONTROL)


def test_a_control_that_is_not_expected_unfit_has_a_floor_that_refuses_nothing(tmp_path):
    lenient = {
        **_CASE,
        "expected": {"managed": "fit", "reduced": "unfit", "regressed": "degraded"},
    }
    with pytest.raises(NarrativeEvalError, match="floor that refuses nothing"):
        check_table_is_falsifiable([lenient], control=CONTROL)


def test_the_judge_itself_must_be_shown_able_to_go_red(tmp_path, capsys):
    """A judge that certifies anything is worse than no judge, because it certifies."""

    class CertifiesAnything:
        name = "certifies-anything"

        def grade(self, request):
            from agent_eval_kit.judge import CriterionScore

            return JudgeVerdict(
                graded_by=self.name,
                scores=tuple(CriterionScore(criterion=c.name, score=1.0) for c in request.criteria),
            )

    from agent_eval_kit.floors import load_quality_floors
    from agent_eval_kit.harness import NotFalselyGreenError
    from agent_eval_kit.narrative import check_judge_can_go_red

    cases = load_cases(_table(tmp_path), PROFILES)
    with pytest.raises(NotFalselyGreenError, match="FALSELY GREEN"):
        check_judge_can_go_red(
            cases,
            CertifiesAnything(),
            load_quality_floors(_floors(tmp_path)),
            reference="managed",
            control=CONTROL,
        )


def test_a_judge_that_grades_nothing_is_an_absent_measurement_not_a_low_score(tmp_path):
    class GradesNothing:
        def grade(self, request):
            return JudgeVerdict(graded_by="grades-nothing")

    from agent_eval_kit.floors import load_quality_floors

    cases = load_cases(_table(tmp_path), PROFILES)
    with pytest.raises(JudgeUnavailableError, match="absent measurement"):
        measure(
            cases,
            GradesNothing(),
            load_quality_floors(_floors(tmp_path)),
            profiles=PROFILES,
        )


def test_a_row_missing_a_profile_names_the_case_rather_than_raising_a_key_error(tmp_path):
    broken = {**_CASE, "candidates": {"managed": "a", "reduced": "b"}}
    with pytest.raises(NarrativeEvalError, match="no 'regressed' candidate"):
        load_cases(_table(tmp_path, [broken]), PROFILES)


def test_an_expectation_that_is_not_a_band_is_refused(tmp_path):
    broken = {**_CASE, "expected": {**_CASE["expected"], "reduced": "quite good"}}
    with pytest.raises(NarrativeEvalError, match="not a fitness band"):
        load_cases(_table(tmp_path, [broken]), PROFILES)


def test_an_empty_table_judges_nothing(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("# nothing here yet\n", encoding="utf-8")
    with pytest.raises(NarrativeEvalError, match="empty"):
        load_cases(path, PROFILES)


def test_the_control_must_be_one_of_the_graded_profiles(tmp_path):
    with pytest.raises(NarrativeEvalError, match="has to be graded to be a control"):
        narrative_main(
            dataset=_table(tmp_path),
            floors=_floors(tmp_path),
            profiles=PROFILES,
            control="not-a-profile",
            argv=[],
        )


def test_the_reference_and_the_control_cannot_be_the_same_profile(tmp_path):
    with pytest.raises(NarrativeEvalError, match="proved against itself"):
        narrative_main(
            dataset=_table(tmp_path),
            floors=_floors(tmp_path),
            profiles=PROFILES,
            control=CONTROL,
            reference=CONTROL,
            argv=[],
        )


def test_a_half_configured_model_judge_is_refused_rather_than_falling_back(tmp_path):
    """What it fell back to would be the thing certifying the release."""
    from agent_eval_kit.judge import JudgeConfigError

    with pytest.raises(JudgeConfigError, match="needs BOTH"):
        narrative_main(
            dataset=_table(tmp_path),
            floors=_floors(tmp_path),
            profiles=PROFILES,
            control=CONTROL,
            argv=["--judge", "local-model", "--judge-base-url", "https://x.example"],
        )


def test_the_environment_cannot_swap_the_gate_s_scorer(tmp_path, monkeypatch, capsys):
    """A gate whose scorer a stray variable could swap is not a gate."""
    from agent_eval_kit.judge import JUDGE_BACKEND_ENV

    monkeypatch.setenv(JUDGE_BACKEND_ENV, "local-model")
    code = narrative_main(
        dataset=_table(tmp_path),
        floors=_floors(tmp_path),
        profiles=PROFILES,
        control=CONTROL,
        argv=[],
    )
    assert code == 0
    assert "judge: deterministic" in capsys.readouterr().out


def test_a_vertical_with_no_floor_is_never_given_somebody_else_s_bar(tmp_path):
    from agent_eval_kit.floors import MissingFloorError, load_quality_floors

    unfloored = {**_CASE, "vertical": "a-vertical-nobody-wrote-a-floor-for"}
    cases = load_cases(_table(tmp_path, [unfloored]), PROFILES)
    with pytest.raises(MissingFloorError):
        measure(
            cases,
            __import__("agent_eval_kit").build_judge(),
            load_quality_floors(_floors(tmp_path)),
            profiles=PROFILES,
        )


def test_measured_bands_come_out_in_table_order(tmp_path):
    from agent_eval_kit import build_judge
    from agent_eval_kit.floors import load_quality_floors

    cases = load_cases(_table(tmp_path), PROFILES)
    measured = measure(
        cases, build_judge(), load_quality_floors(_floors(tmp_path)), profiles=PROFILES
    )
    assert [row.profile for row in measured] == list(PROFILES)
    assert [row.fitness for row in measured] == [Fitness.FIT, Fitness.DEGRADED, Fitness.UNFIT]
    assert all(row.matched for row in measured)
