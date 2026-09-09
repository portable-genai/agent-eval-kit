"""Render the golden set from the shipped demo data, and fail the build when it goes stale.

A golden set written by hand next to a demo book that is also written by hand is two corpora
claiming to describe the same system. They diverge on the first edit, and the divergence is
invisible: the demo narrates one set of clients and the gate measures another, both green.

The pattern that fixes it, and that this module generalises, has one rule that is easy to get
backwards:

* the **INPUTS** are rendered from the shipped demo data, so the cases the gate measures are the
  cases the demo shows;
* the **EXPECTATIONS** are hand-written, in their own reviewed file, and are never derived.

Deriving the expectations too would ask the code under test whether it agrees with itself. Every
metric would read 1.000 and the gate would be decoration. When the engine and the oracle
disagree, one of them is wrong and a person decides which. That is the entire value of the
arrangement.

``--check`` is what makes it hold. Rendering is a command somebody has to remember to run;
``--check`` fails the offline gate when the committed file no longer matches its two sources, so
a demo-book edit that never reached the gate is caught in the build rather than discovered later
by a reader wondering why the numbers stopped meaning anything.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

__all__ = ["GoldenRenderError", "check", "render_main", "write"]


class GoldenRenderError(RuntimeError):
    """Raised when the committed artifact is stale, missing, or renders to nothing."""


def _rendered(render: Callable[[], str]) -> str:
    text = render()
    if not text.strip():
        raise GoldenRenderError(
            "the renderer produced nothing, so writing it would empty the golden set and every "
            "metric over it would be an average of no cases"
        )
    return text if text.endswith("\n") else text + "\n"


def write(output: Path, render: Callable[[], str]) -> int:
    """Render and commit the artifact. Returns the number of non-comment lines written."""
    text = _rendered(render)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def check(output: Path, render: Callable[[], str], *, stream: TextIO | None = None) -> bool:
    """Whether the committed artifact still matches what its sources render. Prints the diff.

    A boolean plus a diff rather than an exception, because the caller is a gate step whose job
    is to print something a person can act on: the diff IS the explanation of what moved.
    """
    out = sys.stderr if stream is None else stream
    expected = _rendered(render)
    if not output.exists():
        print(f"{output}: missing; run the renderer and commit the result", file=out)
        return False
    actual = output.read_text(encoding="utf-8")
    if actual == expected:
        return True
    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=f"{output} (committed)",
        tofile=f"{output} (rendered from its sources)",
    )
    print("".join(diff), file=out)
    print(
        f"{output} is stale: its sources have changed and it has not been re-rendered, so the "
        "gate is measuring a corpus the demo no longer shows. Re-run the renderer and commit "
        "the result in the same change that moved the sources.",
        file=out,
    )
    return False


def render_main(
    *,
    output: Path,
    render: Callable[[], str],
    description: str = "Render the golden set from its sources; --check fails when it is stale.",
    argv: list[str] | None = None,
) -> int:
    """The renderer's whole CLI: write by default, verify with ``--check``.

    ``--check`` belongs INSIDE the repo's eval target rather than in a separate job. A staleness
    check that runs somewhere else is a check somebody can forget to wire up, and the failure it
    catches is one that otherwise reaches a reader as a number that quietly stopped meaning
    anything.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Exit non-zero when the committed file is stale.",
    )
    args = parser.parse_args(argv)
    if args.check:
        return 0 if check(output, render) else 1
    written = write(output, render)
    print(f"wrote {written} case(s) to {output}")
    return 0
