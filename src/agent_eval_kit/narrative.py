"""The half a rule cannot score: is the generated prose any good, judged, against a floor.

Everything a deterministic eval measures is binary. Was the action allowed. Was the claim
grounded. Did it cite what it claimed. Did anything personal survive. Those are questions code
can answer, and code should answer them.

They leave a gap that is exactly the model's own contribution. A reply can be allowed, grounded,
correctly cited, free of personal data and still be useless, or worse: it can answer a question
nobody asked, promise something the bank has not agreed, or tell a customer who has just said
they cannot pay that there is nothing to be done. Deciding that is a judgement, so it is judged,
and the judge is held to the same standard as every other scorer: it must be shown able to fail
before anything it certifies is believed.

Four properties make this a measurement rather than a vote, and each is a defence against a
specific way a judged metric turns into decoration:

* **The judge is OFFLINE by default and chosen on the command line.** ``--judge deterministic``
  needs no model, no credentials and no network, so this runs inside the gate.
  ``--judge local-model`` sends the narratives to an OpenAI-compatible endpoint the operator
  names. No environment variable can redirect it: a gate whose scorer a stray variable could
  swap is not a gate.
* **The bar is DATA, owned by model risk**, in a floors document. A floor refuses; a target is
  full quality; between them is DEGRADED, which is the band a portability claim describes in
  adjectives and nothing measures.
* **The expectation is a TABLE, not a threshold.** Each case carries the same output written
  once per profile, with the band each should land in, so a profile that quietly got BETTER
  fails too. A band nobody predicted is a change nobody reviewed.
* **The table itself must be falsifiable.** A table where everything is expected FIT would
  certify any judge that returns high numbers, including one that returns the same high number
  for every input. So the control must be expected UNFIT, and some expectation must be DEGRADED,
  before a single narrative is graded.

Exit is ``0`` only when every measured band equals its expectation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .floors import Fitness, QualityFloors, load_quality_floors
from .judge import (
    JudgeConfigError,
    JudgePort,
    JudgeRequest,
    JudgeSelection,
    JudgeUnavailableError,
    NarrativeCriterion,
    assert_judge_can_go_red,
    build_judge,
)

__all__ = [
    "Measured",
    "NarrativeEvalError",
    "check_judge_can_go_red",
    "check_table_is_falsifiable",
    "criteria_for",
    "load_cases",
    "measure",
    "narrative_main",
    "render_table",
]


class NarrativeEvalError(RuntimeError):
    """The table is unusable, or the run cannot be trusted to have measured anything."""


@dataclass(frozen=True, slots=True)
class Measured:
    """One narrative, graded, and the band it landed in against its vertical's floor."""

    case_id: str
    vertical: str
    profile: str
    score: float
    fitness: Fitness
    expected: Fitness

    @property
    def matched(self) -> bool:
        return self.fitness is self.expected


def _validated(case: dict[str, Any], path: Path, profiles: Sequence[str]) -> dict[str, Any]:
    """One table row, checked hard on the way in, naming the case rather than raising a KeyError.

    A row missing a profile's candidate or expectation would otherwise surface as a bare
    traceback from deep inside :func:`measure`, after part of the table had already been graded
    and the operator had started believing the output.
    """
    where = f"{path}[{case.get('id') or '?'}]"
    for key in ("id", "vertical", "candidates", "expected", "criteria"):
        if not case.get(key):
            raise NarrativeEvalError(f"{where}: {key!r} is required and must not be empty")
    for profile in profiles:
        candidate = (case["candidates"] or {}).get(profile)
        if not isinstance(candidate, str) or not candidate.strip():
            raise NarrativeEvalError(f"{where}: no {profile!r} candidate, so it cannot be judged")
        band = (case["expected"] or {}).get(profile)
        try:
            Fitness(str(band))
        except ValueError:
            raise NarrativeEvalError(
                f"{where}: expected[{profile!r}] is {band!r}, which is not a fitness band"
            ) from None
    return case


def load_cases(path: Path, profiles: Sequence[str]) -> list[dict[str, Any]]:
    """Read the comment-tolerant degradation table, validating every row on the way in."""
    if not path.exists():
        raise NarrativeEvalError(f"{path}: no degradation table, so nothing is judged")
    cases: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(_validated(json.loads(line), path, profiles))
    if not cases:
        raise NarrativeEvalError(f"{path}: the degradation table is empty, so nothing is judged")
    return cases


def check_table_is_falsifiable(cases: Sequence[dict[str, Any]], *, control: str) -> None:
    """A table of passes proves nothing. Refuse one BEFORE it is run, not after.

    Three structural rules, each closing a way this could look like evidence while being none:
    the control must be expected UNFIT, some expectation must be DEGRADED, and some must be
    UNFIT. A table where everything is expected FIT certifies any judge that returns high
    numbers, including one that returns the same high number for every input.
    """
    expectations = {Fitness(str(band)) for case in cases for band in case["expected"].values()}
    if Fitness.DEGRADED not in expectations:
        raise NarrativeEvalError(
            "no case expects DEGRADED, so the middle band is unmeasured and the floor and the "
            "target are indistinguishable"
        )
    if Fitness.UNFIT not in expectations:
        raise NarrativeEvalError(
            "no case expects UNFIT, so nothing in this table would refuse a profile"
        )
    for case in cases:
        band = Fitness(str(case["expected"][control]))
        if band is not Fitness.UNFIT:
            raise NarrativeEvalError(
                f"{case['id']}: the {control!r} control is expected {band.value!r}. It is the "
                "deliberate defect; a table where it passes has a floor that refuses nothing."
            )


def criteria_for(case: dict[str, Any]) -> tuple[NarrativeCriterion, ...]:
    """The case's criteria as data. Each is refused at construction if nothing can violate it."""
    return tuple(NarrativeCriterion.from_mapping(row) for row in case["criteria"])


def measure(
    cases: Sequence[dict[str, Any]],
    judge: JudgePort,
    floors: QualityFloors,
    *,
    profiles: Sequence[str],
) -> list[Measured]:
    """Grade every candidate and place it in a band. A judge that grades nothing is an error."""
    measured: list[Measured] = []
    for case in cases:
        criteria = criteria_for(case)
        floor = floors.floor_for(case["vertical"])
        for profile in profiles:
            verdict = judge.grade(
                JudgeRequest(
                    candidate=case["candidates"][profile],
                    criteria=criteria,
                    subject=str(case.get("subject", "")),
                )
            )
            if not verdict.has_evidence:
                raise JudgeUnavailableError(
                    f"{case['id']}/{profile}: the judge returned no graded criteria. A verdict "
                    "that measured nothing is not a low score, it is an absent measurement."
                )
            score = verdict.score
            if score >= floor.target:
                fitness = Fitness.FIT
            elif score >= floor.floor:
                fitness = Fitness.DEGRADED
            else:
                fitness = Fitness.UNFIT
            measured.append(
                Measured(
                    case_id=str(case["id"]),
                    vertical=str(case["vertical"]),
                    profile=profile,
                    score=round(score, 4),
                    fitness=fitness,
                    expected=Fitness(str(case["expected"][profile])),
                )
            )
    if not measured:
        raise NarrativeEvalError("nothing was measured, which is not a pass")
    return measured


def check_judge_can_go_red(
    cases: Sequence[dict[str, Any]],
    judge: JudgePort,
    floors: QualityFloors,
    *,
    reference: str,
    control: str,
) -> None:
    """Turn falsification on the JUDGE, the one scorer whose failure is invisible.

    A broken metric returns a wrong number and something notices. A broken judge keeps returning
    numbers, they keep clearing the bar, and the certification it produces is indistinguishable
    from a real one. So before any verdict here is believed, the judge is shown to score the
    reference narrative above the floor and the deliberate control below it, per case.
    """
    for case in cases:
        floor = floors.floor_for(case["vertical"])
        assert_judge_can_go_red(
            judge,
            criteria=criteria_for(case),
            green=case["candidates"][reference],
            red=case["candidates"][control],
            floor=floor.floor,
            metric=f"narrative_quality[{case['id']}]",
        )


def render_table(measured: Sequence[Measured], *, judge_name: str, floors_path: Path) -> bool:
    """Print the graded table and return whether every band matched its expectation."""
    failures = [row for row in measured if not row.matched]
    cases = len({row.case_id for row in measured})
    print("")
    print(f"=== narrative quality (judge: {judge_name}) ===")
    print(f"  floors  : {floors_path}")
    print(f"  cases   : {cases}  narratives: {len(measured)}")
    print("")
    print("  case                   profile     score   band       expected   result")
    print("  " + "-" * 74)
    for row in measured:
        verdict = "PASS" if row.matched else "FAIL"
        print(
            f"  {row.case_id:<22} {row.profile:<11} {row.score:5.3f}   "
            f"{row.fitness.value:<10} {row.expected.value:<10} {verdict}"
        )
    print("")
    print(f"  NARRATIVE GATE: {'PASS' if not failures else 'FAIL'}")
    for row in failures:
        print(
            f"    {row.case_id}/{row.profile}: measured {row.fitness.value!r}, "
            f"the table says {row.expected.value!r}"
        )
    if failures:
        print(
            "    A band that moved is a quality change nobody reviewed. Recalibrate the table "
            "in the same commit as whatever moved it, or fix what moved."
        )
    return not failures


def _build_parser(dataset: Path, floors: Path, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--judge",
        choices=("deterministic", "local-model"),
        default="deterministic",
        help=(
            "deterministic (default): offline, no model, runs in the gate. local-model: an "
            "OpenAI-compatible endpoint you name below. Chosen HERE and never from the "
            "environment, so no stray variable can swap the gate's scorer."
        ),
    )
    parser.add_argument("--judge-base-url", default="", help="Required by --judge local-model.")
    parser.add_argument("--judge-model", default="", help="Required by --judge local-model.")
    parser.add_argument("--dataset", type=Path, default=dataset)
    parser.add_argument("--floors", type=Path, default=floors)
    return parser


def _selection_from(args: argparse.Namespace) -> JudgeSelection:
    if args.judge == "deterministic":
        return JudgeSelection(backend="deterministic")
    if not args.judge_base_url.strip() or not args.judge_model.strip():
        raise JudgeConfigError(
            "--judge local-model needs BOTH --judge-base-url and --judge-model. A model judge "
            "half-configured would fall back to something, and what it fell back to would be "
            "the thing certifying the release."
        )
    return JudgeSelection(
        backend="local-model",
        base_url=args.judge_base_url.strip(),
        model=args.judge_model.strip(),
    )


def narrative_main(
    *,
    dataset: Path,
    floors: Path,
    profiles: Sequence[str],
    control: str,
    reference: str = "",
    description: str = "Narrative quality against model-risk floors, judged.",
    argv: list[str] | None = None,
) -> int:
    """The whole judged run as one call: a repo supplies its table, its floors and its profiles.

    ``profiles`` names the ways each case is written, and they are PROFILES rather than
    adjectives: they say which deployment produced the narrative, which is what a portability
    claim is about. ``control`` is the deliberate defect, and it must be expected UNFIT
    everywhere. ``reference`` is the good narrative the judge is proved against, defaulting to
    the first profile.
    """
    if control not in profiles:
        raise NarrativeEvalError(
            f"the control profile {control!r} is not one of {list(profiles)}; the control has "
            "to be graded to be a control"
        )
    good = reference or profiles[0]
    if good == control:
        raise NarrativeEvalError(
            "the reference and the control are the same profile, so the judge would be proved "
            "against itself and could not be shown able to go red"
        )
    args = _build_parser(dataset, floors, description).parse_args(argv)
    cases = load_cases(args.dataset, profiles)
    check_table_is_falsifiable(cases, control=control)
    loaded_floors = load_quality_floors(args.floors)
    selection = _selection_from(args)
    judge = build_judge(selection)

    check_judge_can_go_red(cases, judge, loaded_floors, reference=good, control=control)
    measured = measure(cases, judge, loaded_floors, profiles=profiles)
    passed = render_table(measured, judge_name=selection.backend, floors_path=args.floors)
    return 0 if passed else 1
