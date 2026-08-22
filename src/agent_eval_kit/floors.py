"""Per-vertical quality floors: the number below which a reduced profile is UNFIT.

A portability claim that a laptop or on-prem profile serves "visibly lower narrative quality"
is an adjective standing where a threshold should be. It tells a buyer nothing about which
verticals still work on the reduced profile, and it cannot refuse anything. This module is the
missing number, and it is deliberately DATA rather than code:

* a **floor**, per named vertical, below which that vertical is UNFIT on the profile that was
  measured, meaning it must not be offered rather than merely disclaimed, and
* a **target**, at or above which the profile is FIT, so the band between the two is the
  honest home for "degraded but usable".

Fail-closed rules, all of them the same rule in different clothes: nothing is scored against a
floor nobody wrote down. An unnamed vertical raises rather than inheriting a default, an empty
floor set refuses to load, a zero floor is refused because a floor at zero cannot refuse
anything, and a verdict with no graded evidence is UNFIT rather than passing on ``all(())``.

The file format is a small TOML or JSON document, so the numbers a model-risk function owns
live in a reviewable file rather than in a Python literal somebody has to be a programmer to
change. Parsing uses ``tomllib``/``json`` from the standard library: no dependency lands in a
consumer's decision core.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path

from .judge import JudgeVerdict
from .report import EvalMetricResult

#: The schema tag every floors document must declare, so an unrelated file cannot be loaded
#: as a floor table and quietly become the bar a promotion is measured against.
FLOORS_SCHEMA = "quality-floors/v1"

_DOCUMENT_KEYS = frozenset({"schema", "verticals", "owner", "note"})
_ENTRY_KEYS = frozenset({"floor", "target", "note"})


class QualityFloorError(Exception):
    """Base class for every refusal this module makes."""


class FloorDataError(QualityFloorError, ValueError):
    """Raised when a floors document is malformed, or a score is not a usable measurement."""


class MissingFloorError(QualityFloorError, LookupError):
    """Raised when a vertical has no floor. There is no default floor, by design."""


class Fitness(StrEnum):
    """What a measured score means for the profile that produced it."""

    #: At or above the vertical's target: full quality.
    FIT = "fit"
    #: At or above the floor but below the target: usable, and visibly worse. The band the
    #: portability story was describing in words.
    DEGRADED = "degraded"
    #: Below the floor: not merely degraded. This profile must not serve this vertical.
    UNFIT = "unfit"


@dataclass(frozen=True, slots=True)
class QualityFloor:
    """One vertical's named floor and target, plus where the numbers came from."""

    vertical: str
    floor: float
    target: float
    note: str = ""
    source: str = ""


@dataclass(frozen=True, slots=True)
class FitnessVerdict:
    """One measured score, judged against one vertical's floor, for one named profile."""

    vertical: str
    profile: str
    score: float
    fitness: Fitness
    floor: QualityFloor
    reason: str

    @property
    def unfit(self) -> bool:
        """Whether this profile must not serve this vertical."""
        return self.fitness is Fitness.UNFIT

    def as_metric_result(self, metric: str = "") -> EvalMetricResult:
        """Render as an :class:`~agent_eval_kit.report.EvalMetricResult` for an eval report.

        The threshold is the FLOOR, not the target, so a metric row passes exactly when the
        profile is not unfit. Degradation belongs in the report as a visible score below
        target, not as a failed gate: the floor is the bar that refuses.
        """
        name = metric or f"narrative_quality[{self.vertical}/{self.profile}]"
        return EvalMetricResult(
            metric=name,
            score=self.score,
            threshold=self.floor.floor,
            passed=not self.unfit,
        )


def _number(value: object, *, where: str, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FloorDataError(f"{where}: {field} must be a number, got {value!r}")
    number = float(value)
    if not isfinite(number):
        raise FloorDataError(f"{where}: {field} must be finite, got {value!r}")
    return number


def _floor_from_entry(vertical: str, entry: object, source: str) -> QualityFloor:
    where = f"{source or 'floors'}: vertical {vertical!r}"
    if not vertical.strip():
        raise FloorDataError(f"{source or 'floors'}: a vertical must be named")
    if not isinstance(entry, Mapping):
        raise FloorDataError(f"{where}: each entry must be a table of floor/target")
    unknown = sorted(set(map(str, entry)) - _ENTRY_KEYS)
    if unknown:
        raise FloorDataError(
            f"{where}: unknown field(s) {', '.join(unknown)}; a misspelled bar is a bar that "
            "is not applied"
        )
    for required in ("floor", "target"):
        if required not in entry:
            raise FloorDataError(f"{where}: {required} is required and is never defaulted")
    floor = _number(entry["floor"], where=where, field="floor")
    target = _number(entry["target"], where=where, field="target")
    if not 0.0 < floor <= 1.0:
        raise FloorDataError(
            f"{where}: floor {floor} must be within (0.0, 1.0]; a floor of zero cannot refuse "
            "anything, which is the same as having no floor"
        )
    if not floor <= target <= 1.0:
        raise FloorDataError(
            f"{where}: target {target} must be at least the floor {floor} and at most 1.0"
        )
    note = entry.get("note", "")
    if not isinstance(note, str):
        raise FloorDataError(f"{where}: note must be a string")
    return QualityFloor(vertical=vertical, floor=floor, target=target, note=note, source=source)


class QualityFloors:
    """A loaded floors document: the named verticals and the bars each one carries."""

    def __init__(self, floors: Mapping[str, QualityFloor], *, source: str = "") -> None:
        if not floors:
            raise FloorDataError(
                f"{source or 'floors'}: the floor table is empty, so every vertical would be "
                "unmeasured; an empty policy is not a policy"
            )
        self._floors: dict[str, QualityFloor] = dict(floors)
        self._source = source

    @property
    def source(self) -> str:
        """Where these numbers came from, carried into every verdict for the audit trail."""
        return self._source

    @property
    def verticals(self) -> tuple[str, ...]:
        return tuple(sorted(self._floors))

    def __len__(self) -> int:
        return len(self._floors)

    def __iter__(self) -> Iterator[QualityFloor]:
        return iter(self._floors[name] for name in self.verticals)

    def __contains__(self, vertical: object) -> bool:
        return vertical in self._floors

    @classmethod
    def from_mapping(cls, data: Mapping[str, object], *, source: str = "") -> QualityFloors:
        """Build from a decoded document, refusing anything that is not a floors document."""
        where = source or "floors"
        unknown = sorted(set(map(str, data)) - _DOCUMENT_KEYS)
        if unknown:
            raise FloorDataError(f"{where}: unknown top-level key(s): {', '.join(unknown)}")
        schema = data.get("schema")
        if schema != FLOORS_SCHEMA:
            raise FloorDataError(
                f"{where}: schema must be {FLOORS_SCHEMA!r}, got {schema!r}; refusing to read "
                "an unknown document as a promotion bar"
            )
        verticals = data.get("verticals")
        if not isinstance(verticals, Mapping):
            raise FloorDataError(f"{where}: 'verticals' must be a table of named floors")
        return cls(
            {
                str(name): _floor_from_entry(str(name), entry, where)
                for name, entry in verticals.items()
            },
            source=where,
        )

    def floor_for(self, vertical: str) -> QualityFloor:
        """The named floor for ``vertical``, or raise. There is no default floor."""
        try:
            return self._floors[vertical]
        except KeyError as exc:
            known = ", ".join(self.verticals)
            raise MissingFloorError(
                f"{self._source or 'floors'}: no quality floor is defined for vertical "
                f"{vertical!r} (defined: {known}). A vertical with no floor is unmeasured, and "
                "an unmeasured vertical is not silently given somebody else's bar"
            ) from exc

    def assess(self, vertical: str, score: float, *, profile: str) -> FitnessVerdict:
        """Judge a measured ``score`` for ``profile`` against ``vertical``'s floor."""
        if not profile.strip():
            raise FloorDataError(
                "a fitness verdict must name the profile that was measured; 'which profile' "
                "is the whole question a floor answers"
            )
        floor = self.floor_for(vertical)
        measured = _number(score, where=f"{vertical}/{profile}", field="score")
        if not 0.0 <= measured <= 1.0:
            raise FloorDataError(
                f"{vertical}/{profile}: score {measured} is outside [0, 1] and is not a "
                "quality measurement"
            )
        if measured >= floor.target:
            fitness = Fitness.FIT
            reason = f"score {measured:.3f} is at or above the target {floor.target:.2f}"
        elif measured >= floor.floor:
            fitness = Fitness.DEGRADED
            reason = (
                f"score {measured:.3f} is below the target {floor.target:.2f} but at or above "
                f"the floor {floor.floor:.2f}: usable, and visibly worse"
            )
        else:
            fitness = Fitness.UNFIT
            reason = (
                f"score {measured:.3f} is below the floor {floor.floor:.2f}: this profile is "
                f"unfit for {vertical}, not merely degraded"
            )
        return FitnessVerdict(
            vertical=vertical,
            profile=profile,
            score=measured,
            fitness=fitness,
            floor=floor,
            reason=reason,
        )

    def assess_verdict(
        self, vertical: str, verdict: JudgeVerdict, *, profile: str
    ) -> FitnessVerdict:
        """Judge a :class:`~agent_eval_kit.judge.JudgeVerdict` against a vertical's floor.

        A verdict with no graded criteria is UNFIT, never a pass. That is the whole class of
        defect this fleet already paid for once, where an empty metric list reported passed.
        """
        floor = self.floor_for(vertical)
        if not verdict.has_evidence:
            return FitnessVerdict(
                vertical=vertical,
                profile=profile or "unnamed",
                score=0.0,
                fitness=Fitness.UNFIT,
                floor=floor,
                reason=(
                    f"the judge {verdict.graded_by!r} graded no criteria, so there is no "
                    "evidence; an absent measurement is unfit, never a pass"
                ),
            )
        return self.assess(vertical, verdict.score, profile=profile)


def load_quality_floors(path: str | Path) -> QualityFloors:
    """Load a floors document from a ``.toml`` or ``.json`` file, using only the stdlib."""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".toml":
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    elif suffix == ".json":
        loaded = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise FloorDataError(f"{resolved}: a floors document must be a JSON object")
        data = loaded
    else:
        raise FloorDataError(
            f"{resolved}: a floors document must be .toml or .json, so it can be read with no "
            "third-party parser"
        )
    return QualityFloors.from_mapping(data, source=str(resolved))


__all__ = [
    "FLOORS_SCHEMA",
    "Fitness",
    "FitnessVerdict",
    "FloorDataError",
    "MissingFloorError",
    "QualityFloor",
    "QualityFloorError",
    "QualityFloors",
    "load_quality_floors",
]
