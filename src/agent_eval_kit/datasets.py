"""Loading a golden set, and the three ways a load quietly turns a gate into decoration.

Every repo in the fleet re-implements the same eight lines of JSONL reading, and the drift
between the copies is not stylistic. Three failure modes recur, each of which leaves a gate
reporting a confident number over something it did not measure:

1. **A case kind no metric scores.** A rename or a typo in a ``kind`` field leaves the row in
   the file, counting toward ``n_examples``, so the report looks evidenced; meanwhile the metric
   that should have selected it selects nothing. :func:`load_jsonl` refuses the dataset when
   ``kinds=`` is given and a row names a kind outside it, which is the only point at which the
   mistake is cheap to see.
2. **An empty selection scoring 1.0.** ``sum(flags) / len(flags)`` over an empty list is a
   ZeroDivisionError, so somebody writes ``if not flags: return 1.0`` and the metric that
   measured nothing becomes the strongest number in the report. :func:`fraction` returns ``0.0``
   and :func:`warn_unmeasured` names the metric on stderr so the zero explains itself.
3. **A dataset that moved without the gate noticing.** :func:`dataset_digest` is a content hash
   the report can carry, so "which corpus produced this number" is answerable after the fact
   rather than inferred from a filename.

Comment tolerance is deliberate and load-bearing. A golden set is read by reviewers who are not
programmers, and a file that cannot explain a case in place gets explained in a separate document
that goes stale. ``#`` lines and blank lines are skipped, everywhere, so a corpus can carry its
own reasoning.

Pure standard library, with one exception resolved lazily: reviewable YAML corpora (the shape
scenario trees use) import PyYAML inside :func:`load_yaml_documents` and nowhere else, so
importing this module costs a consumer's decision core nothing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "DatasetError",
    "dataset_digest",
    "fraction",
    "load_jsonl",
    "load_yaml_documents",
    "warn_unmeasured",
]


class DatasetError(ValueError):
    """Raised when a golden set is unreadable, empty, or contains a row nothing scores."""


def load_jsonl(
    path: Path,
    *,
    kinds: Iterable[str] | None = None,
    kind_field: str = "kind",
    required: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Read a comment-tolerant JSONL golden set, refusing the three vacuous shapes.

    ``kinds``, when given, is the set of case kinds some metric actually scores. A row naming
    anything else is refused with both sides printed, because the failure it prevents is silent:
    the row keeps counting toward the example total while the metric selects nothing.

    ``required`` names fields every row must carry and must not leave empty. A missing field
    surfaces here, naming the file and the row, rather than as a bare ``KeyError`` from deep
    inside a scorer after half the corpus has already been graded.
    """
    if not path.exists():
        raise DatasetError(f"{path}: golden dataset not found")
    cases: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{number}: not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"{path}:{number}: each line must be a JSON object")
        for field in required:
            if not row.get(field):
                raise DatasetError(f"{path}:{number}: {field!r} is required and must not be empty")
        cases.append(row)
    if not cases:
        raise DatasetError(
            f"{path}: the golden set is empty, so every metric over it would be an average of "
            "nothing; an absent measurement is not a pass"
        )
    if kinds is not None:
        known = {str(item) for item in kinds}
        unknown = sorted({str(case.get(kind_field)) for case in cases} - known)
        if unknown:
            raise DatasetError(
                f"{path}: case {kind_field}(s) {unknown} are scored by no metric. Such a row "
                f"still counts toward n_examples, so the report looks evidenced while the "
                f"metric that should have scored it selects nothing. Scored kinds: "
                f"{sorted(known)}"
            )
    return cases


def load_yaml_documents(root: Path, *, pattern: str = "*.yaml") -> list[tuple[Path, Any]]:
    """Load every YAML document under ``root``, in sorted path order, with its own path.

    For the corpora a reviewer edits by hand: scenario trees, rubric sets, criteria files. The
    path travels with the document so a validation failure can name the file rather than the
    index of a list.

    PyYAML is imported HERE and nowhere else in this package. A consumer's decision core reaches
    ``agent_eval_kit`` for its report types, and a top-level ``import yaml`` would put a parser
    in that import graph for the sake of a function no domain module calls.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the consumer's env
        # ImportError rather than ModuleNotFoundError: a broken or shadowed install
        # fails the same way for a caller, and should get the same actionable message.
        raise DatasetError(
            "reading a YAML corpus needs PyYAML, which this package does not depend on. Add it "
            "to the consuming repo (every service that ships packs already has it), or keep the "
            "corpus as JSONL, which needs nothing."
        ) from exc
    if not root.is_dir():
        raise DatasetError(f"{root}: no such directory, so there is no corpus to read")
    documents: list[tuple[Path, Any]] = []
    for path in sorted(root.rglob(pattern)):
        documents.append((path, yaml.safe_load(path.read_text(encoding="utf-8"))))
    if not documents:
        raise DatasetError(f"{root}: no {pattern} document found, so nothing would be scored")
    return documents


def dataset_digest(path: Path) -> str:
    """A content hash of the golden set, ignoring comments and blank lines.

    Comments are excluded on purpose: a reviewer improving a case's explanation must not read as
    a corpus change, or the digest becomes noise and stops being checked. What it identifies is
    the DATA a number was computed over, which is the question somebody asks three months later
    when two reports disagree.
    """
    digest = hashlib.sha256()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fraction(flags: Sequence[bool]) -> float:
    """Share of ``flags`` that are true. An EMPTY selection scores 0.0, never 1.0.

    A metric whose selection was empty measured nothing, and nothing is not evidence of safety.
    Scoring 0.0 puts it below every threshold, so the named metric reports FAIL and the gate
    exits non-zero, which is what an absent measurement should do.
    """
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def warn_unmeasured(
    cases: Sequence[Mapping[str, Any]],
    metric_kinds: Mapping[str, str],
    *,
    kind_field: str = "kind",
    stream: TextIO | None = None,
) -> list[str]:
    """Name every metric whose kind selected no case, so its 0.000 explains itself.

    Returns the metric names it warned about, so a caller can fail on them explicitly rather
    than relying on the score alone. Printing is not enough on its own and is not meant to be:
    the 0.0 from :func:`fraction` is what fails the gate, and this is what tells a reader why.
    """
    present = {str(case.get(kind_field)) for case in cases}
    unmeasured = [metric for metric, kind in metric_kinds.items() if kind not in present]
    out = sys.stderr if stream is None else stream
    for metric in unmeasured:
        print(
            f"error: no {metric_kinds[metric]!r} case in the dataset, so {metric} evaluated "
            "nothing and scores 0.0",
            file=out,
        )
    return unmeasured
