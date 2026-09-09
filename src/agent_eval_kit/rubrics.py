"""Metric thresholds as reviewed data: read the bar from the rubric, never from a module dict.

Practice E1 asks that "metric thresholds are not duplicated as unlabelled constants", and most
of the fleet keeps them as a module-level ``THRESHOLDS`` dict anyway. The cost is not the
duplication. It is that a bar written as a Python literal carries no argument: a reviewer can
read that recall must clear 0.85 and cannot read WHY 0.85, who agreed it, or what it would mean
to move it. Every other bank-owned number in these services is configuration; these are the
exception, and they are the numbers a regulator asks about first.

A rubric file states one metric, its bar, and the reasoning beside it, plus any
``companion_metrics`` scored by the same run. Both are thresholds; the split is editorial,
grouping the metrics a reader should consider together.

Two symmetrical refusals, and the second is the one nobody writes by hand:

* **a metric with no reviewed bar** fails the build, because the alternative is a number
  invented at the call site;
* **a bar that names no metric** fails it too. A rubric for a metric that was renamed or deleted
  is worse than no rubric: it reads as governance, it satisfies a reviewer, and nothing scores
  it. This is the direction that rots silently, and it rots toward looking well governed.

Fails closed on every other way this can go wrong as well: a missing directory, a document that
is not a mapping, a non-numeric bar, or the same metric given two different bars in two files.
There is deliberately no fallback to a hard-coded dict, because a fallback is a second home for
a number that must have one.

``.yaml`` is the shape the fleet already uses; ``.toml`` and ``.json`` are read with the standard
library so a repo with no YAML dependency can still hold its bars as data. PyYAML is imported
lazily, inside the loader, so importing this module costs a decision core nothing.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

__all__ = [
    "Rubric",
    "RubricError",
    "Rubrics",
    "load_rubrics",
]

#: The suffixes a rubric may be written in. YAML first because it is what the fleet uses and the
#: one a reviewer is most likely to edit; the other two need no third-party parser.
RUBRIC_SUFFIXES: tuple[str, ...] = (".yaml", ".yml", ".toml", ".json")


class RubricError(ValueError):
    """Raised when the rubric set is unreadable, incomplete, or scores nothing."""


@dataclass(frozen=True, slots=True)
class Rubric:
    """One metric's reviewed bar and the argument that sits next to it."""

    metric: str
    threshold: float
    description: str = ""
    source: str = ""
    group: str = ""

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise RubricError(f"{self.source or 'rubric'}: names no metric")
        if not isinstance(self.threshold, (int, float)) or isinstance(self.threshold, bool):
            raise RubricError(
                f"{self.source or 'rubric'}: metric {self.metric!r} has a non-numeric threshold "
                f"{self.threshold!r}"
            )
        value = float(self.threshold)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise RubricError(
                f"{self.source or 'rubric'}: metric {self.metric!r} has threshold {value}, "
                "which is outside [0, 1] and is not a quality bar"
            )


def _document(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the consumer's env
            # ImportError rather than ModuleNotFoundError: a broken or shadowed install
            # fails the same way for a caller, and should get the same actionable message.
            raise RubricError(
                f"{path}: reading a YAML rubric needs PyYAML, which this package does not depend "
                "on. Add it to the consuming repo, or write the rubric as .toml or .json, which "
                "the standard library reads."
            ) from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _rubrics_in(path: Path, group: str) -> Iterator[Rubric]:
    document = _document(path)
    if not isinstance(document, Mapping):
        raise RubricError(f"{path}: a rubric must be a mapping at the top level")
    headline = str(document.get("metric") or "")
    if not headline:
        raise RubricError(
            f"{path}: names no metric. A rubric with no metric is a bar nothing is measured "
            "against, which reads as governance and gates nothing."
        )
    yield Rubric(
        metric=headline,
        threshold=document.get("threshold"),  # type: ignore[arg-type]
        description=" ".join(str(document.get("description", "")).split()),
        source=str(path),
        group=group,
    )
    companions = document.get("companion_metrics") or {}
    if not isinstance(companions, Mapping):
        raise RubricError(f"{path}: 'companion_metrics' must be a mapping of metric -> bar")
    for name, node in companions.items():
        entry = node or {}
        if not isinstance(entry, Mapping):
            raise RubricError(f"{path}: companion metric {name!r} must be a mapping")
        if "threshold" not in entry:
            raise RubricError(
                f"{path}: companion metric {name!r} has no threshold; a companion is a gated "
                "metric, not a note"
            )
        yield Rubric(
            metric=str(name),
            threshold=entry.get("threshold"),  # type: ignore[arg-type]
            description=" ".join(str(entry.get("description", "")).split()),
            source=str(path),
            group=group,
        )


class Rubrics:
    """Every reviewed bar in one rubric tree, addressable by metric name."""

    def __init__(self, rubrics: Iterable[Rubric], *, root: str = "") -> None:
        # Keyed by (group, metric), not by metric alone. A repository that gates two families
        # in one run legitimately reuses metric NAMES across them: a design review and a
        # data-residency scan both have a `citation_accuracy`, and they are different questions
        # that may carry different bars. Keying on the name alone silently collapsed the two, so
        # whichever file sorted last won and the other family's rubric vanished from the tree
        # while still sitting on disk looking authoritative.
        #
        # "One metric, one bar" still holds WITHIN a group, which is where it was always meant to
        # apply: two files in the same directory disagreeing about the same metric is a mistake.
        self._by_key: dict[tuple[str, str], Rubric] = {}
        self._root = root
        for rubric in rubrics:
            key = (rubric.group, rubric.metric)
            existing = self._by_key.get(key)
            if existing is not None and existing.threshold != rubric.threshold:
                where = f" in group {rubric.group!r}" if rubric.group else ""
                raise RubricError(
                    f"metric {rubric.metric!r} is given {rubric.threshold} in {rubric.source} "
                    f"and {existing.threshold} in {existing.source}{where}. One metric, one bar."
                )
            self._by_key[key] = rubric
        if not self._by_key:
            raise RubricError(
                f"{root or 'rubrics'}: no metric has a threshold, so nothing is gated"
            )
        #: Name-keyed view, for the common single-family case. Where a name appears in more than
        #: one group this holds the first in sorted order; a caller that gates two families
        #: should narrow with :meth:`group` first, which is what makes the bars unambiguous.
        self._by_metric: dict[str, Rubric] = {}
        for (_group, metric), rubric in sorted(self._by_key.items()):
            self._by_metric.setdefault(metric, rubric)

    @property
    def root(self) -> str:
        return self._root

    @property
    def metrics(self) -> tuple[str, ...]:
        """Every metric name in this tree, deduplicated across groups."""
        return tuple(sorted({metric for _group, metric in self._by_key}))

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def groups(self) -> tuple[str, ...]:
        """Every named group in this tree, in sorted order. ``""`` is the top level."""
        return tuple(sorted({group for group, _metric in self._by_key}))

    def __iter__(self) -> Iterator[Rubric]:
        return iter(self._by_key[key] for key in sorted(self._by_key))

    def __contains__(self, metric: object) -> bool:
        return metric in self._by_metric

    def __getitem__(self, metric: str) -> Rubric:
        try:
            return self._by_metric[metric]
        except KeyError as exc:
            known = ", ".join(self.metrics)
            raise RubricError(
                f"{self._root or 'rubrics'}: metric {metric!r} has no reviewed bar "
                f"(reviewed: {known}). A bar invented at the call site is not a reviewed bar."
            ) from exc

    def thresholds(self) -> dict[str, float]:
        """The ``{metric: threshold}`` map a runner scores against."""
        return {name: float(self._by_metric[name].threshold) for name in self.metrics}

    def group(self, name: str) -> Rubrics:
        """The sub-tree of rubrics filed under ``name``, for a repo gating two releases."""
        selected = [rubric for (group, _metric), rubric in self._by_key.items() if group == name]
        if not selected:
            named = sorted(group for group in self.groups if group)
            raise RubricError(
                f"{self._root or 'rubrics'}: no rubric group named {name!r} "
                f"(groups: {named or 'none; this tree is flat'})"
            )
        return Rubrics(selected, root=f"{self._root}/{name}")

    def assert_covers(self, scored: Iterable[str]) -> None:
        """Both directions at once: every scored metric has a bar, and every bar is scored.

        The second half is the one worth having. A rubric for a metric that was renamed or
        deleted keeps reading as governance while nothing measures it, and that is the direction
        that rots without anybody noticing, because it rots toward looking well governed.
        """
        measured = {str(item) for item in scored}
        unreviewed = sorted(measured - set(self._by_metric))
        if unreviewed:
            raise RubricError(
                f"{self._root or 'rubrics'}: metric(s) {unreviewed} are scored with no reviewed "
                "bar, so their thresholds were invented at the call site"
            )
        unscored = sorted(set(self._by_metric) - measured)
        if unscored:
            sources = ", ".join(sorted({self._by_metric[m].source for m in unscored}))
            raise RubricError(
                f"{self._root or 'rubrics'}: bar(s) {unscored} name a metric nothing scores "
                f"({sources}). A rubric with nothing behind it reads as governance and gates "
                "nothing; delete it or score it."
            )


def load_rubrics(root: Path, *, groups: Iterable[str] | None = None) -> Rubrics:
    """Load every rubric under ``root``, flat or one directory per gated release.

    ``groups``, when given, names the sub-directories that must each carry at least one rubric,
    which is how a repo with two separately gated modes refuses a half-written rubric tree. It
    is not merely a filter: a named group with no directory is an error, because the release it
    gates would otherwise ship with no bars at all.
    """
    if not root.is_dir():
        raise RubricError(
            f"{root}: no rubric directory, so no metric has a reviewed threshold and every bar "
            "in this repository is a number somebody typed at a call site"
        )
    collected: list[Rubric] = []
    if groups is None:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in RUBRIC_SUFFIXES:
                group = path.parent.name if path.parent != root else ""
                collected.extend(_rubrics_in(path, group))
    else:
        for name in groups:
            directory = root / name
            if not directory.is_dir():
                raise RubricError(f"{directory}: rubric group {name!r} has no thresholds")
            found = [
                path
                for path in sorted(directory.glob("*"))
                if path.is_file() and path.suffix.lower() in RUBRIC_SUFFIXES
            ]
            if not found:
                raise RubricError(f"{directory}: rubric group {name!r} has no rubric files")
            for path in found:
                collected.extend(_rubrics_in(path, name))
    return Rubrics(collected, root=str(root))
