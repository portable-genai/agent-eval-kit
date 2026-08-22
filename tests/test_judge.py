"""The offline judge, and the not-falsely-green proof turned on the judge itself.

A judge is the one scorer whose failure is invisible. A broken metric usually stops producing
numbers; a broken judge keeps producing them, they keep clearing the bar, and the certification
it issues is indistinguishable from a real one. So the tests here are mostly falsification: can
this judge go red, can an empty verdict pass, can a request that asks nothing be made at all.
"""

from __future__ import annotations

import pytest

from agent_eval_kit.harness import NotFalselyGreenError
from agent_eval_kit.judge import (
    DETERMINISTIC_BACKEND,
    JUDGE_BACKEND_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    LOCAL_MODEL_BACKEND,
    CriterionScore,
    DeterministicNarrativeJudge,
    JudgeConfigError,
    JudgeRequest,
    JudgeSelection,
    JudgeUnavailableError,
    JudgeVerdict,
    NarrativeCriterion,
    assert_judge_can_go_red,
    build_judge,
)

#: One vertical's criteria, written the way a golden set writes them: checkable data, not an
#: adjective. Identifiers are obviously fictional.
_CRITERIA = (
    NarrativeCriterion(
        name="explains_the_finding",
        must_cover=(
            "the ownership chain stops at a nominee shareholder",
            "the source of wealth is unevidenced",
        ),
        must_cite=("KYC-POL-4.2",),
    ),
    NarrativeCriterion(
        name="stays_within_its_evidence",
        must_not_say=("definitely a criminal", "guaranteed clean"),
        weight=2.0,
    ),
)

_GOOD = (
    "The ownership chain stops at a nominee shareholder, and the declared source of wealth is "
    "unevidenced in the file. Escalating under KYC-POL-4.2 for enhanced due diligence."
)
_DEGRADED = "The customer looks fine overall. Reviewer should take another look at the file."


class _AlwaysHappyJudge:
    """The failure this module exists to catch: a judge that certifies anything."""

    name = "always-happy/v0"

    def grade(self, request: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(
            graded_by=self.name,
            scores=tuple(
                CriterionScore(criterion=item.name, score=1.0, weight=item.weight)
                for item in request.criteria
            ),
        )


class _SilentJudge:
    """A judge that answers with no graded criteria: an absent measurement wearing a verdict."""

    name = "silent/v0"

    def grade(self, request: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(graded_by=self.name, scores=())


def test_the_deterministic_judge_separates_a_good_narrative_from_a_degraded_one() -> None:
    judge = DeterministicNarrativeJudge()
    good = judge.grade(JudgeRequest(candidate=_GOOD, criteria=_CRITERIA))
    degraded = judge.grade(JudgeRequest(candidate=_DEGRADED, criteria=_CRITERIA))
    assert good.score > degraded.score
    assert good.score >= 0.9
    assert degraded.score < 0.9
    assert good.graded_by == DeterministicNarrativeJudge.name
    assert {item.criterion for item in good.scores} == {c.name for c in _CRITERIA}


def test_the_deterministic_judge_is_reproducible() -> None:
    """A judge that answers differently on a rerun cannot be a promotion bar."""
    judge = DeterministicNarrativeJudge()
    request = JudgeRequest(candidate=_GOOD, criteria=_CRITERIA)
    assert judge.grade(request) == judge.grade(request)


def test_a_forbidden_claim_pulls_the_score_down() -> None:
    judge = DeterministicNarrativeJudge()
    offending = f"{_GOOD} This applicant is definitely a criminal."
    clean = judge.grade(JudgeRequest(candidate=_GOOD, criteria=_CRITERIA))
    dirty = judge.grade(JudgeRequest(candidate=offending, criteria=_CRITERIA))
    assert dirty.score < clean.score


def test_the_judge_itself_is_proven_able_to_go_red() -> None:
    """The whole point: this judge scores the good narrative above the floor and the bad below."""
    assert_judge_can_go_red(
        DeterministicNarrativeJudge(),
        criteria=_CRITERIA,
        green=_GOOD,
        red=_DEGRADED,
        floor=0.75,
        metric="narrative_quality",
    )


def test_a_judge_that_certifies_anything_is_caught() -> None:
    """The mutant control. Without this, a judge stuck at 1.0 would look like a strict one."""
    with pytest.raises(NotFalselyGreenError, match="FALSELY GREEN"):
        assert_judge_can_go_red(
            _AlwaysHappyJudge(),
            criteria=_CRITERIA,
            green=_GOOD,
            red=_DEGRADED,
            floor=0.75,
            metric="narrative_quality",
        )


def test_a_judge_that_grades_nothing_is_refused_rather_than_scored() -> None:
    with pytest.raises(JudgeUnavailableError, match="no graded criteria"):
        assert_judge_can_go_red(
            _SilentJudge(),
            criteria=_CRITERIA,
            green=_GOOD,
            red=_DEGRADED,
            floor=0.75,
        )


def test_an_empty_verdict_is_never_a_pass() -> None:
    """``all(())`` is true, and that is exactly how the fleet's last falsely-green defect read."""
    empty = JudgeVerdict(graded_by="deterministic-narrative-judge/v1", scores=())
    assert empty.has_evidence is False
    assert empty.score == 0.0
    assert empty.meets(0.0) is False
    assert empty.meets(0.75) is False


def test_a_request_with_no_criteria_is_refused() -> None:
    """The absent metric set, caught at the earliest point: you cannot even ask."""
    with pytest.raises(JudgeConfigError, match="no criteria"):
        JudgeRequest(candidate=_GOOD, criteria=())


def test_a_criterion_that_demands_nothing_is_refused() -> None:
    with pytest.raises(JudgeConfigError, match="cannot go red"):
        NarrativeCriterion(name="vibes")


def test_a_blank_phrase_is_refused() -> None:
    """A blank required point is satisfied by every narrative, including an empty one."""
    with pytest.raises(JudgeConfigError, match="blank"):
        NarrativeCriterion(name="covers", must_cover=("a real point", "   "))


def test_a_criterion_phrase_list_may_not_be_a_bare_string() -> None:
    """``must_cover="abc"`` would silently become three single-character points."""
    with pytest.raises(JudgeConfigError, match="not a single string"):
        NarrativeCriterion(name="covers", must_cover="a real point")  # type: ignore[arg-type]


def test_an_unattributed_verdict_is_refused() -> None:
    with pytest.raises(JudgeConfigError, match="must name the judge"):
        JudgeVerdict(graded_by="  ", scores=(CriterionScore(criterion="c", score=1.0),))


def test_duplicate_criterion_names_are_refused() -> None:
    duplicated = (
        NarrativeCriterion(name="same", must_cover=("one",)),
        NarrativeCriterion(name="same", must_cover=("two",)),
    )
    with pytest.raises(JudgeConfigError, match="duplicate criterion"):
        JudgeRequest(candidate=_GOOD, criteria=duplicated)


def test_a_score_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(JudgeConfigError, match="outside"):
        CriterionScore(criterion="c", score=1.5)


def test_a_non_positive_weight_is_refused() -> None:
    with pytest.raises(JudgeConfigError, match="positive"):
        NarrativeCriterion(name="c", must_cover=("x",), weight=0.0)


def test_the_verdict_score_is_weighted() -> None:
    verdict = JudgeVerdict(
        graded_by="j",
        scores=(
            CriterionScore(criterion="a", score=1.0, weight=1.0),
            CriterionScore(criterion="b", score=0.0, weight=3.0),
        ),
    )
    assert verdict.score == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Selection: offline by default, a model only when named
# --------------------------------------------------------------------------- #
def test_an_unset_environment_selects_the_offline_judge() -> None:
    """The load-bearing default: a gate runs with no server, no network, no configuration."""
    selection = JudgeSelection.from_env({})
    assert selection.backend == DETERMINISTIC_BACKEND
    assert selection.is_offline is True
    assert isinstance(build_judge(selection), DeterministicNarrativeJudge)
    assert isinstance(build_judge(), DeterministicNarrativeJudge)


def test_an_emptied_backend_variable_is_a_configuration_error() -> None:
    """Emptying a variable is a deliberate act, so it is not read as "use the default"."""
    with pytest.raises(JudgeConfigError, match="set but empty"):
        JudgeSelection.from_env({JUDGE_BACKEND_ENV: "   "})


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(JudgeConfigError, match="unknown judge backend"):
        JudgeSelection.from_env({JUDGE_BACKEND_ENV: "gpt-please"})


def test_the_model_judge_is_never_pointed_at_a_host_nobody_named() -> None:
    with pytest.raises(JudgeConfigError, match=JUDGE_BASE_URL_ENV):
        JudgeSelection.from_env({JUDGE_BACKEND_ENV: LOCAL_MODEL_BACKEND})


def test_the_model_judge_requires_the_model_id_as_evidence() -> None:
    with pytest.raises(JudgeConfigError, match=JUDGE_MODEL_ENV):
        JudgeSelection.from_env(
            {
                JUDGE_BACKEND_ENV: LOCAL_MODEL_BACKEND,
                JUDGE_BASE_URL_ENV: "http://127.0.0.1:8001",
            }
        )


def test_the_model_judge_is_selected_only_when_fully_named() -> None:
    selection = JudgeSelection.from_env(
        {
            JUDGE_BACKEND_ENV: LOCAL_MODEL_BACKEND,
            JUDGE_BASE_URL_ENV: "http://127.0.0.1:8001",
            JUDGE_MODEL_ENV: "mlx-community/example-model-4bit",
        }
    )
    assert selection.backend == LOCAL_MODEL_BACKEND
    assert selection.is_offline is False
