"""The authoring half of the replay: record once, scrub the whole batch, or write nothing.

This is the step that costs money and needs credentials, and it is deliberately the only one.
It runs by hand, against the managed profile, and writes the file
:mod:`agent_eval_kit.replay` replays offline forever after. Run it, read the diff, commit it.

The rule that makes the output safe to commit is one line long and is the whole design:

    **Nothing is written unless the whole batch is clean.**

Not a filter. A filter would write a scrubbed lie: the file would look clean, the case that
leaked would be silently absent, and the corpus would be one nobody could reason about. A
partial write is also how a scrubbed file ends up half scrubbed, because the process that
aborts halfway leaves the earlier rows on disk. So every recorded reply is scanned before any
of it is written, one hit aborts everything, and the operator fixes what reached the model
rather than the file that came out.

What each reply is scanned for is supplied by the caller, because it is repo-specific and must
be the SAME source the runtime redactor uses. A recording script that carried its own patterns
would be scoring one detector against another, which is the tautology practice E2 exists to
prevent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Recorded",
    "RecordingRefused",
    "scrub_or_refuse",
    "write_recordings",
]

_HEADER = """\
# Recorded managed-model replies, for offline replay by this repository's eval.
# Written by the recording script against the managed profile. Every row was scanned with the
# RUNTIME pattern source, for planted identifiers and for length, before ANY of it was written:
# one hit aborts the whole batch rather than dropping a row.
#
# Re-record when the corpus or the prompt changes. The replay adapter raises on a missing key
# rather than falling back, so a stale recording is loud rather than quietly wrong.
"""


class RecordingRefused(RuntimeError):
    """Something in the batch is not safe to commit, so none of it is written."""


@dataclass(frozen=True, slots=True)
class Recorded:
    """One captured call: its replay key, the case it came from, and what the model said."""

    key: str
    case_id: str
    model: str
    response: Any
    prompt_sha256: str = ""
    inputs: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "case_id": self.case_id,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "inputs": list(self.inputs),
            "response": self.response,
        }


def _text_of(recorded: Recorded, extract: Callable[[Any], str] | None) -> str:
    if extract is not None:
        return extract(recorded.response)
    response = recorded.response
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("text", "")) or json.dumps(response, ensure_ascii=False)
    return str(response)


def scrub_or_refuse(
    rows: Sequence[Recorded],
    *,
    leaks: Callable[[str], bool],
    planted: Iterable[str] = (),
    max_chars: int | None = None,
    extract: Callable[[Any], str] | None = None,
) -> None:
    """Scan every reply, and raise on the FIRST hit so that nothing is written.

    ``leaks`` is the runtime leak oracle, passed in rather than reimplemented: the same pattern
    source the product's redactor uses, so a hit means the model output carries an identifier
    and not that two detectors have drifted apart.

    ``planted`` are the identifiers a golden case deliberately put in front of the model, checked
    as literals. It is the second, independent oracle: it still fires when a pattern row is
    broken, which is precisely when the first one cannot.
    """
    tokens = [str(token) for token in planted if str(token)]
    for recorded in rows:
        text = _text_of(recorded, extract)
        if not text:
            continue
        if leaks(text):
            raise RecordingRefused(
                f"{recorded.case_id}: the recorded reply matches the runtime personal-data "
                "patterns. Nothing has been written. Fix what reached the model before "
                "recording again."
            )
        for token in tokens:
            if token in text:
                raise RecordingRefused(
                    f"{recorded.case_id}: the recorded reply contains the planted identifier "
                    f"{token!r}. Nothing has been written."
                )
        if max_chars is not None and len(text) > max_chars:
            raise RecordingRefused(
                f"{recorded.case_id}: the recorded reply is {len(text)} characters, over the "
                f"{max_chars} the validator accepts. Nothing has been written."
            )


def write_recordings(
    path: Path,
    rows: Sequence[Recorded],
    *,
    leaks: Callable[[str], bool],
    planted: Iterable[str] = (),
    max_chars: int | None = None,
    extract: Callable[[Any], str] | None = None,
    header: str = _HEADER,
) -> int:
    """Scrub the whole batch, then write it, then return the row count. Refuses an empty batch.

    The ordering is the point and is not negotiable: scrub first, over everything, and only then
    touch the file. Writing as you go is what produces a half-scrubbed corpus.
    """
    if not rows:
        raise RecordingRefused(
            "nothing was captured, so there is nothing to record. An empty recording file would "
            "make every replayed case a miss, which is loud, but writing one hides the fact that "
            "the capture itself did not run."
        )
    scrub_or_refuse(rows, leaks=leaks, planted=planted, max_chars=max_chars, extract=extract)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row.as_row(), ensure_ascii=False) for row in rows)
    path.write_text(header + body + "\n", encoding="utf-8")
    return len(rows)
