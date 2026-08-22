"""The offline judge harness: measure narrative quality with no model, or with one by choice.

Every eval in this fleet scores a DETERMINISTIC CORE against a deterministic fake model
adapter, so nothing measured model narrative quality at all. The portability story told buyers
that a reduced (laptop / on-prem) profile serves "visibly lower narrative quality", which is an
adjective standing where a threshold should be. Worse structurally: narrative quality was scored
only by the managed promotion authority, so the profile running the weaker local model was also
the profile with no quality measurement.

This module is the measurement, and it runs where the weak model runs:

* :class:`NarrativeCriterion` states, as data, what a good narrative for a case must convey,
  must cite, and must not claim. A criterion that demands none of the three is refused at
  construction, because a criterion nothing can violate scores a constant 1.0 forever.
* :class:`DeterministicNarrativeJudge` is the DEFAULT judge: pure standard library, no model,
  no network, no credentials, same answer every run. It is what a gate runs.
* :class:`~agent_eval_kit.local_model_judge.LocalModelJudge` is the OPT-IN judge: an
  OpenAI-compatible chat-completions server (an MLX server on a laptop serves both the subject
  model and the judge). It is selected by configuration and is never required. See
  :class:`JudgeSelection`, whose unset-environment answer is the deterministic judge.
* :func:`assert_judge_can_go_red` applies the not-falsely-green proof to the JUDGE itself. A
  judge that returns a high score for any text is worse than no judge, because it certifies.

What the deterministic judge does NOT measure is fluency or style; it measures whether the
narrative carries the reference points, cites the evidence it was told to cite, and stays off
the claims it was told to avoid. That is the checkable part of narrative quality, and it is the
part a reduced profile actually loses first. The opt-in model judge grades the rest, and the
floor it is compared against is the same either way (:mod:`agent_eval_kit.floors`).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from .harness import assert_can_go_red

#: The environment variable that names the judge backend. UNSET means the deterministic
#: offline judge, so a gate needs no configuration, no server and no network to run.
JUDGE_BACKEND_ENV = "AGENT_EVAL_JUDGE"
#: Required when, and only when, the backend is :data:`LOCAL_MODEL_BACKEND`.
JUDGE_BASE_URL_ENV = "AGENT_EVAL_JUDGE_BASE_URL"
JUDGE_MODEL_ENV = "AGENT_EVAL_JUDGE_MODEL"

#: The default: deterministic, offline, reproducible.
DETERMINISTIC_BACKEND = "deterministic"
#: Opt-in: an OpenAI-compatible chat-completions server chosen deliberately by configuration.
LOCAL_MODEL_BACKEND = "local-model"
JUDGE_BACKENDS: tuple[str, ...] = (DETERMINISTIC_BACKEND, LOCAL_MODEL_BACKEND)

#: A reference point counts as covered when at least this share of its words appears in the
#: candidate narrative. The same overlap rule the fleet's offline scorers already use.
COVERAGE_OVERLAP = 0.5

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(match.lower() for match in _TOKEN_RE.findall(text or ""))


class JudgeConfigError(ValueError):
    """Raised when the judge selection is unusable: unknown backend, or a missing setting."""


class JudgeUnavailableError(RuntimeError):
    """Raised when a judge could not produce a verdict (unreachable, or an unusable reply).

    Declared in this stdlib module rather than next to the HTTP adapter that raises it, so a
    caller can catch it without importing an HTTP client. A judge that cannot answer must
    RAISE: a silent default score would be a quality measurement invented by a failure.
    """


@dataclass(frozen=True, slots=True)
class NarrativeCriterion:
    """One graded aspect of a narrative, stated as checkable data rather than as an adjective.

    ``must_cover`` are the reference points the narrative has to convey, ``must_cite`` the
    evidence identifiers it has to name, and ``must_not_say`` the claims it must stay off
    (invented certainty, advice, an unhedged guarantee). At least one of the three must be
    non-empty: a criterion nothing can violate is a constant, and a constant proves nothing.
    """

    name: str
    must_cover: tuple[str, ...] = ()
    must_cite: tuple[str, ...] = ()
    must_not_say: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "must_cover", _clean_phrases(self.must_cover))
        object.__setattr__(self, "must_cite", _clean_phrases(self.must_cite))
        object.__setattr__(self, "must_not_say", _clean_phrases(self.must_not_say))
        if not self.name.strip():
            raise JudgeConfigError("a narrative criterion must be named")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool):
            raise JudgeConfigError(f"{self.name}: weight must be a number")
        if not isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise JudgeConfigError(f"{self.name}: weight must be a finite positive number")
        if not (self.must_cover or self.must_cite or self.must_not_say):
            raise JudgeConfigError(
                f"{self.name}: a criterion that requires no coverage, no citation and forbids "
                "nothing cannot go red, so it would score a constant 1.0 forever"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> NarrativeCriterion:
        """Build a criterion from a decoded JSON/TOML row, rejecting unknown keys."""
        known = {"name", "must_cover", "must_cite", "must_not_say", "weight"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise JudgeConfigError(f"unknown criterion field(s): {', '.join(unknown)}")
        weight = data.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise JudgeConfigError(f"criterion {data.get('name')!r}: weight must be a number")
        return cls(
            name=str(data.get("name", "")),
            must_cover=_as_phrases(data.get("must_cover", ())),
            must_cite=_as_phrases(data.get("must_cite", ())),
            must_not_say=_as_phrases(data.get("must_not_say", ())),
            weight=float(weight),
        )


def _as_phrases(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raise JudgeConfigError("criterion phrase lists must be lists, not a single string")
    if not isinstance(value, Iterable):
        raise JudgeConfigError("criterion phrase lists must be iterable")
    return tuple(str(item) for item in value)


def _clean_phrases(value: Sequence[str]) -> tuple[str, ...]:
    """Normalise to a tuple of non-blank phrases; a blank phrase is silently unfalsifiable."""
    if isinstance(value, str):
        raise JudgeConfigError("criterion phrase lists must be sequences, not a single string")
    cleaned = tuple(str(item).strip() for item in value)
    if any(not item for item in cleaned):
        raise JudgeConfigError("a criterion phrase is blank, which nothing can fail to satisfy")
    return cleaned


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """One narrative to grade, and the criteria to grade it against.

    ``criteria`` must be non-empty and uniquely named. An empty criteria list is the "absent
    metric set" failure in its earliest form: a grading request that asks nothing produces a
    verdict that can only be vacuous, so it is refused here rather than scored later.
    """

    candidate: str
    criteria: tuple[NarrativeCriterion, ...]
    reference: str = ""
    subject: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if not self.criteria:
            raise JudgeConfigError(
                "a judge request with no criteria measures nothing; an empty criteria set "
                "cannot produce a passing verdict"
            )
        names = [criterion.name for criterion in self.criteria]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise JudgeConfigError(f"duplicate criterion name(s): {', '.join(duplicates)}")


@dataclass(frozen=True, slots=True)
class CriterionScore:
    """One criterion's score in [0, 1], with the weight it carries and a short reason."""

    criterion: str
    score: float
    weight: float = 1.0
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.criterion.strip():
            raise JudgeConfigError("a criterion score must name its criterion")
        for field_name, value in (("score", self.score), ("weight", self.weight)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise JudgeConfigError(f"{self.criterion}: {field_name} must be a number")
            if not isfinite(float(value)):
                raise JudgeConfigError(f"{self.criterion}: {field_name} must be finite")
        if not 0.0 <= float(self.score) <= 1.0:
            raise JudgeConfigError(f"{self.criterion}: score {self.score} is outside [0, 1]")
        if float(self.weight) <= 0.0:
            raise JudgeConfigError(f"{self.criterion}: weight must be positive")


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """What a judge concluded: the per-criterion scores, and who graded them.

    :attr:`score` is the weighted mean, and it is ``0.0`` when there are no scores. That is
    the deliberate opposite of the defect this fleet already paid for once, where a report
    built from an empty metric list reported ``passed=True`` because ``all(())`` is true. An
    absent measurement is the weakest possible evidence here, never the strongest.
    """

    graded_by: str
    scores: tuple[CriterionScore, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", tuple(self.scores))
        if not self.graded_by.strip():
            raise JudgeConfigError(
                "a verdict must name the judge that produced it; an unattributed score cannot "
                "be reproduced or challenged"
            )

    @property
    def has_evidence(self) -> bool:
        """Whether this verdict rests on at least one graded criterion."""
        return bool(self.scores)

    @property
    def score(self) -> float:
        """The weighted mean criterion score, or ``0.0`` when nothing was graded."""
        if not self.scores:
            return 0.0
        total_weight = sum(item.weight for item in self.scores)
        if total_weight <= 0.0:  # pragma: no cover - weights are validated positive
            return 0.0
        return sum(item.score * item.weight for item in self.scores) / total_weight

    def meets(self, floor: float) -> bool:
        """Whether this verdict clears ``floor`` ON EVIDENCE. Empty is never a pass."""
        return self.has_evidence and self.score >= floor


@runtime_checkable
class JudgePort(Protocol):
    """Grades a narrative against criteria. Implemented offline by default, by a model on request.

    An implementation that cannot reach its backend, or gets a reply it cannot parse, raises
    :class:`JudgeUnavailableError`. It must never return a default score, because a score
    invented by a failure is indistinguishable from a measurement.
    """

    def grade(self, request: JudgeRequest) -> JudgeVerdict:
        """Return the verdict for ``request``, or raise :class:`JudgeUnavailableError`."""
        ...


class DeterministicNarrativeJudge:
    """The default judge: no model, no network, no credentials, same answer every run.

    Each criterion is scored from the components it actually declares, equally weighted among
    the components present: the share of ``must_cover`` points the narrative conveys, the share
    of ``must_cite`` identifiers it names, and the share of ``must_not_say`` claims it avoids.
    A reference point counts as conveyed when :data:`COVERAGE_OVERLAP` of its words appear in
    the narrative, the same overlap rule the fleet's offline scorers already use.
    """

    name = "deterministic-narrative-judge/v1"

    def grade(self, request: JudgeRequest) -> JudgeVerdict:
        candidate_tokens = _tokens(request.candidate)
        lowered = request.candidate.lower()
        scores: list[CriterionScore] = []
        for criterion in request.criteria:
            components: list[float] = []
            details: list[str] = []
            if criterion.must_cover:
                covered = sum(
                    1 for point in criterion.must_cover if _conveys(point, candidate_tokens)
                )
                components.append(covered / len(criterion.must_cover))
                details.append(f"covered {covered}/{len(criterion.must_cover)}")
            if criterion.must_cite:
                cited = sum(1 for ref in criterion.must_cite if ref.lower() in lowered)
                components.append(cited / len(criterion.must_cite))
                details.append(f"cited {cited}/{len(criterion.must_cite)}")
            if criterion.must_not_say:
                avoided = sum(
                    1 for banned in criterion.must_not_say if banned.lower() not in lowered
                )
                components.append(avoided / len(criterion.must_not_say))
                details.append(f"avoided {avoided}/{len(criterion.must_not_say)}")
            scores.append(
                CriterionScore(
                    criterion=criterion.name,
                    score=round(sum(components) / len(components), 4),
                    weight=criterion.weight,
                    detail=", ".join(details),
                )
            )
        return JudgeVerdict(graded_by=self.name, scores=tuple(scores))


def _conveys(point: str, candidate_tokens: frozenset[str]) -> bool:
    """Whether ``point`` is carried by the narrative, by word overlap."""
    point_tokens = _tokens(point)
    if not point_tokens:  # pragma: no cover - blank phrases are refused at construction
        return False
    return len(point_tokens & candidate_tokens) / len(point_tokens) >= COVERAGE_OVERLAP


@dataclass(frozen=True, slots=True)
class JudgeSelection:
    """Which judge to build, resolved EXPLICITLY, with the offline judge as the only default.

    The rule the fleet's portability proof depends on: a gate must run with no model server,
    no credentials and no network. So an unset :data:`JUDGE_BACKEND_ENV` selects the
    deterministic judge, and the model-backed judge is reached only by naming it plus the
    server and model it should use. A variable an operator EMPTIED is a configuration error
    rather than a silent fall back to the default, because emptying one is a deliberate act.
    """

    backend: str = DETERMINISTIC_BACKEND
    base_url: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if self.backend not in JUDGE_BACKENDS:
            raise JudgeConfigError(
                f"unknown judge backend {self.backend!r}; expected one of: "
                f"{', '.join(JUDGE_BACKENDS)}"
            )
        if self.backend == LOCAL_MODEL_BACKEND:
            if not self.base_url.strip():
                raise JudgeConfigError(
                    f"the {LOCAL_MODEL_BACKEND} judge needs a base URL "
                    f"({JUDGE_BASE_URL_ENV}); it is never defaulted to a host you did not name"
                )
            if not self.model.strip():
                raise JudgeConfigError(
                    f"the {LOCAL_MODEL_BACKEND} judge needs a model id ({JUDGE_MODEL_ENV}); "
                    "the model that graded a narrative is part of the evidence"
                )

    @property
    def is_offline(self) -> bool:
        """Whether this selection runs with no server and no network."""
        return self.backend == DETERMINISTIC_BACKEND

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> JudgeSelection:
        """Resolve the selection from the environment. Unset means the offline judge."""
        source = os.environ if environ is None else environ
        raw = source.get(JUDGE_BACKEND_ENV)
        if raw is None:
            return cls()
        backend = raw.strip()
        if not backend:
            raise JudgeConfigError(
                f"{JUDGE_BACKEND_ENV} is set but empty; unset it for the offline "
                f"{DETERMINISTIC_BACKEND} judge, or name one of {', '.join(JUDGE_BACKENDS)}"
            )
        return cls(
            backend=backend,
            base_url=source.get(JUDGE_BASE_URL_ENV, "").strip(),
            model=source.get(JUDGE_MODEL_ENV, "").strip(),
        )


def build_judge(selection: JudgeSelection | None = None) -> JudgePort:
    """Build the judge named by ``selection``, defaulting to the offline deterministic one.

    The model-backed adapter is imported only when it is actually selected, so a gate that
    never opts in never imports an HTTP client at all.
    """
    resolved = selection if selection is not None else JudgeSelection()
    if resolved.backend == DETERMINISTIC_BACKEND:
        return DeterministicNarrativeJudge()
    from .local_model_judge import LocalModelJudge

    return LocalModelJudge(resolved.base_url, model=resolved.model)


def assert_judge_can_go_red(
    judge: JudgePort,
    *,
    criteria: Sequence[NarrativeCriterion],
    green: str,
    red: str,
    floor: float,
    metric: str = "narrative_quality",
) -> None:
    """Prove THIS judge scores a good narrative above ``floor`` and a degraded one below it.

    The same not-falsely-green rule the rest of the fleet applies to its metrics, turned on
    the judge itself. A judge is the one scorer whose failure is invisible: it keeps returning
    numbers, they keep clearing the bar, and the certification it produces looks identical to
    a real one. :func:`~agent_eval_kit.harness.assert_can_go_red` supplies the two distinct
    failure messages (broken pessimistic, versus falsely green).
    """
    graded = tuple(criteria)

    def score(candidate: str) -> float:
        verdict = judge.grade(JudgeRequest(candidate=candidate, criteria=graded))
        if not verdict.has_evidence:
            raise JudgeUnavailableError(
                f"{metric}: the judge returned a verdict with no graded criteria, which cannot "
                "be scored for or against the floor"
            )
        return verdict.score

    assert_can_go_red(score, green=green, red=red, threshold=floor, metric=metric)


__all__ = [
    "COVERAGE_OVERLAP",
    "DETERMINISTIC_BACKEND",
    "JUDGE_BACKENDS",
    "JUDGE_BACKEND_ENV",
    "JUDGE_BASE_URL_ENV",
    "JUDGE_MODEL_ENV",
    "LOCAL_MODEL_BACKEND",
    "CriterionScore",
    "DeterministicNarrativeJudge",
    "JudgeConfigError",
    "JudgePort",
    "JudgeRequest",
    "JudgeSelection",
    "JudgeUnavailableError",
    "JudgeVerdict",
    "NarrativeCriterion",
    "assert_judge_can_go_red",
    "build_judge",
]
